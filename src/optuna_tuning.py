#!/usr/bin/env python
"""
Optuna-based tuning for TorchMD-NET training.

This module orchestrates:
- config-driven hyperparameter sampling,
- loss-weight tuning,
- per-trial artifact persistence,
- study-level human-readable summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch

from train_physics import train_physics_informed_model
from train_standard import train_standard_model


DEFAULT_OBJECTIVE_NAME = "val_total_mse_loss"
DEFAULT_RESULTS_ROOT = Path("results") / "optuna"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.Tensor,)):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _load_json(path: Path):
    with open(path, "r") as handle:
        return json.load(handle)


def _deep_update(base: dict, override: dict):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _sample_param(trial, name: str, spec: dict):
    spec_type = spec["type"]
    if spec_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            step=spec.get("step"),
            log=bool(spec.get("log", False)),
        )
    if spec_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if spec_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if spec_type == "bool":
        return trial.suggest_categorical(name, [False, True])
    raise ValueError(f"Unsupported search-space type '{spec_type}' for parameter '{name}'.")


class OptunaPruningCallback(pl.Callback):
    """Reports validation metrics to Optuna and prunes unpromising trials."""

    def __init__(self, trial, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return

        value = metrics[self.monitor]
        if hasattr(value, "item"):
            value = float(value.item())
        else:
            value = float(value)

        # Use global_step so pruning can react to intra-epoch validation checks
        # (e.g., when val_check_interval < 1.0).
        step = int(getattr(trainer, "global_step", trainer.current_epoch))
        self.trial.report(value, step=step)
        if self.trial.should_prune():
            raise self._pruned_exception()

    @staticmethod
    def _pruned_exception():
        import optuna

        return optuna.TrialPruned()


def _build_metric_lookup(result: dict):
    metrics = {}
    metrics.update(result.get("validation_metrics", {}))

    test_results = result.get("test_results") or []
    if test_results:
        metrics.update({f"test.{key}": value for key, value in test_results[0].items()})

    config = result.get("config", {})
    if config.get("best_model_score") is not None:
        metrics["best_model_score"] = config["best_model_score"]
    return metrics


def _resolve_metric(metrics: dict, metric_name: str):
    if metric_name in metrics:
        return metrics[metric_name]
    if metric_name.startswith("validation.") and metric_name[11:] in metrics:
        return metrics[metric_name[11:]]
    if metric_name.startswith("test.") and metric_name in metrics:
        return metrics[metric_name]
    if metric_name == DEFAULT_OBJECTIVE_NAME and "best_model_score" in metrics:
        return metrics["best_model_score"]
    raise KeyError(f"Metric '{metric_name}' was not produced by this trial.")


def _compute_objective(metrics: dict, objective_cfg: dict):
    metric_name = objective_cfg.get("metric", DEFAULT_OBJECTIVE_NAME)
    metric_value = float(_resolve_metric(metrics, metric_name))
    direction = objective_cfg.get("direction", "minimize")

    score = metric_value
    for penalty in objective_cfg.get("penalties", []):
        penalty_metric = float(_resolve_metric(metrics, penalty["metric"]))
        weight = float(penalty.get("weight", 1.0))
        score += weight * penalty_metric

    if direction == "maximize":
        return -score, metric_value
    if direction == "minimize":
        return score, metric_value
    raise ValueError(f"Unsupported objective direction '{direction}'.")


def _serialize_exception(exc: Exception):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _status_payload(status: str, score=None, metric_value=None, exception=None):
    payload = {"status": status}
    if score is not None:
        payload["objective_score"] = score
    if metric_value is not None:
        payload["objective_metric_value"] = metric_value
    if exception is not None:
        payload["exception"] = _serialize_exception(exception)
    return payload


def _copy_best_checkpoint(best_trial_dir: Path, destination_dir: Path):
    metrics_path = best_trial_dir / "metrics.json"
    if not metrics_path.exists():
        return

    metrics = _load_json(metrics_path)
    best_model_path = metrics.get("best_model_path")
    if not best_model_path:
        return

    source = Path(best_model_path)
    if not source.exists():
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_dir / source.name)


def _make_summary_rows(study):
    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "status": trial.state.name.lower(),
            "objective": trial.value,
        }
        row.update({f"param.{key}": value for key, value in trial.params.items()})
        row.update({f"attr.{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    return rows


def _write_summary_csv(path: Path, rows):
    headers = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_markdown(path: Path, summary: dict, top_trials: list[dict]):
    lines = [
        f"# {summary['study_name']}",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Mode: {summary['mode']}",
        f"- Objective metric: {summary['objective']['metric']}",
        f"- Objective direction: {summary['objective']['direction']}",
        f"- Completed trials: {summary['trial_counts']['completed']}",
        f"- Pruned trials: {summary['trial_counts']['pruned']}",
        f"- Failed trials: {summary['trial_counts']['failed']}",
    ]

    best = summary.get("best_trial")
    if best:
        lines.extend(
            [
                "",
                "## Best Trial",
                "",
                f"- Trial: {best['number']}",
                f"- Objective score: {best['objective_score']}",
                f"- Objective metric value: {best['objective_metric_value']}",
                f"- Trial folder: `{best['trial_dir']}`",
                f"- Checkpoint: `{best.get('best_model_path', '')}`",
                "",
                "### Selected Parameters",
                "",
            ]
        )
        for key, value in best.get("params", {}).items():
            lines.append(f"- `{key}` = `{value}`")

    if top_trials:
        lines.extend(
            [
                "",
                "## Top Trials",
                "",
                "| rank | trial | objective | status | metric | checkpoint |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for index, trial in enumerate(top_trials, start=1):
            lines.append(
                f"| {index} | {trial['number']} | {trial.get('objective_score')} | "
                f"{trial.get('status')} | {trial.get('objective_metric_value')} | "
                f"`{trial.get('best_model_path', '')}` |"
            )

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def generate_study_summary(study, study_dir: Path, config: dict):
    rows = _make_summary_rows(study)
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    pruned = [trial for trial in study.trials if trial.state.name == "PRUNED"]
    failed = [trial for trial in study.trials if trial.state.name == "FAIL"]

    top_trials = []
    for trial in sorted(completed, key=lambda item: item.value)[:5]:
        trial_dir = study_dir / f"trial_{trial.number:04d}"
        metrics_path = trial_dir / "metrics.json"
        metrics = _load_json(metrics_path) if metrics_path.exists() else {}
        top_trials.append(
            {
                "number": trial.number,
                "status": "completed",
                "objective_score": trial.value,
                "objective_metric_value": trial.user_attrs.get("objective_metric_value"),
                "best_model_path": metrics.get("best_model_path"),
                "trial_dir": str(trial_dir),
            }
        )

    best_trial_summary = None
    if completed:
        best_trial = study.best_trial
        best_trial_dir = study_dir / f"trial_{best_trial.number:04d}"
        metrics = _load_json(best_trial_dir / "metrics.json") if (best_trial_dir / "metrics.json").exists() else {}
        best_trial_summary = {
            "number": best_trial.number,
            "objective_score": best_trial.value,
            "objective_metric_value": best_trial.user_attrs.get("objective_metric_value"),
            "trial_dir": str(best_trial_dir),
            "params": best_trial.params,
            "best_model_path": metrics.get("best_model_path"),
        }

        best_dir = study_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        _write_json(best_dir / "best_config.json", metrics.get("config", {}))
        _write_json(best_dir / "best_metrics.json", metrics)
        _copy_best_checkpoint(best_trial_dir, best_dir)

    summary = {
        "study_name": study.study_name,
        "generated_at": datetime.now().isoformat(),
        "mode": config["mode"],
        "objective": config["objective"],
        "storage": config["storage"],
        "results_root": str(study_dir),
        "trial_counts": {
            "total": len(study.trials),
            "completed": len(completed),
            "pruned": len(pruned),
            "failed": len(failed),
        },
        "best_trial": best_trial_summary,
        "search_space": config.get("search_space", {}),
        "fixed_params": config.get("fixed_params", {}),
    }

    _write_json(study_dir / "summary.json", summary)
    _write_summary_csv(study_dir / "summary.csv", rows)
    _write_summary_markdown(study_dir / "summary.md", summary, top_trials)


class StudyRunner:
    def __init__(self, config: dict):
        self.config = config
        self.study_dir = Path(config["results_root"]) / config["study_name"]
        self.study_dir.mkdir(parents=True, exist_ok=True)

    def _sample_params(self, trial):
        sampled = {}
        search_space = self.config.get("search_space", {})
        for key, spec in search_space.get("common", {}).items():
            sampled[key] = _sample_param(trial, key, spec)

        if self.config["mode"] == "physics":
            for key, spec in search_space.get("physics", {}).items():
                sampled[key] = _sample_param(trial, key, spec)

        fixed = self.config.get("fixed_params", {})
        delta_learning_enabled = sampled.get("delta_learning", fixed.get("delta_learning", False))
        if delta_learning_enabled:
            for key, spec in search_space.get("delta", {}).items():
                sampled[key] = _sample_param(trial, key, spec)

        return sampled

    def _build_training_kwargs(self, trial, trial_dir: Path):
        sampled = self._sample_params(trial)
        fixed = deepcopy(self.config.get("fixed_params", {}))
        training_kwargs = _deep_update(fixed, sampled)
        trainer_cfg = self.config.get("trainer", {})
        training_kwargs.setdefault("dataset", "MD17")
        training_kwargs.setdefault("molecule", "aspirin")
        training_kwargs.setdefault("model_type", "tensornet")
        training_kwargs.setdefault("num_epochs", int(trainer_cfg.get("max_epochs", 10)))
        training_kwargs.setdefault("checkpoint_name", f"trial_{trial.number:04d}")
        training_kwargs.setdefault("seed", int(self.config.get("seed", 42)) + trial.number)
        training_kwargs.setdefault("num_workers", int(trainer_cfg.get("num_workers", 0)))
        training_kwargs["save_dir"] = str(trial_dir / "checkpoints")
        training_kwargs["log_dir"] = str(trial_dir / "logs")
        training_kwargs["trainer_callbacks"] = [OptunaPruningCallback(trial, self.config["objective"]["metric"])]

        # Allow targeted Trainer overrides from Optuna config to improve
        # pruning responsiveness for expensive epochs.
        trainer_kwargs = {
            "enable_progress_bar": bool(trainer_cfg.get("enable_progress_bar", False)),
            "deterministic": bool(trainer_cfg.get("deterministic", False)),
        }
        for key in (
            "val_check_interval",
            "check_val_every_n_epoch",
            "num_sanity_val_steps",
            "limit_train_batches",
            "limit_val_batches",
            "max_steps",
            "accumulate_grad_batches",
            "gradient_clip_val",
            "gradient_clip_algorithm",
            "precision",
        ):
            if key in trainer_cfg:
                trainer_kwargs[key] = trainer_cfg[key]
        trainer_kwargs.update(trainer_cfg.get("trainer_kwargs", {}))
        training_kwargs["trainer_kwargs"] = trainer_kwargs
        return training_kwargs, sampled

    def _run_training(self, kwargs: dict):
        if self.config["mode"] == "physics":
            return train_physics_informed_model(**kwargs)
        if self.config["mode"] == "standard":
            return train_standard_model(**kwargs)
        raise ValueError(f"Unsupported mode '{self.config['mode']}'.")

    def objective(self, trial):
        import optuna

        trial_dir = self.study_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        kwargs, sampled = self._build_training_kwargs(trial, trial_dir)
        serializable_kwargs = dict(kwargs)
        serializable_kwargs["trainer_callbacks"] = [cb.__class__.__name__ for cb in kwargs.get("trainer_callbacks", [])]
        _write_json(trial_dir / "sampled_params.json", sampled)
        _write_json(trial_dir / "resolved_config.json", serializable_kwargs)

        try:
            result = self._run_training(kwargs)
            metrics = _build_metric_lookup(result)
            score, metric_value = _compute_objective(metrics, self.config["objective"])

            trial.set_user_attr("objective_metric_value", metric_value)
            trial.set_user_attr("best_model_path", result["best_model_path"])
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    trial.set_user_attr(key, value)

            payload = {
                "status": "completed",
                "objective_score": score,
                "objective_metric_value": metric_value,
                "best_model_path": result["best_model_path"],
                "validation_metrics": result.get("validation_metrics", {}),
                "test_results": result.get("test_results", []),
                "config": result.get("config", {}),
            }
            _write_json(trial_dir / "metrics.json", payload)
            _write_json(trial_dir / "status.json", _status_payload("completed", score, metric_value))
            return score

        except optuna.TrialPruned as exc:
            _write_json(trial_dir / "status.json", _status_payload("pruned", exception=exc))
            raise
        except torch.cuda.OutOfMemoryError as exc:
            _write_json(trial_dir / "status.json", _status_payload("failed", exception=exc))
            raise
        except Exception as exc:
            _write_json(trial_dir / "status.json", _status_payload("failed", exception=exc))
            raise


def _create_sampler(config: dict):
    import optuna

    sampler_cfg = config.get("sampler", {"name": "TPESampler", "kwargs": {}})
    name = sampler_cfg.get("name", "TPESampler")
    kwargs = sampler_cfg.get("kwargs", {})
    if name == "TPESampler":
        return optuna.samplers.TPESampler(**kwargs)
    if name == "RandomSampler":
        return optuna.samplers.RandomSampler(**kwargs)
    raise ValueError(f"Unsupported sampler '{name}'.")


def _create_pruner(config: dict):
    import optuna

    pruner_cfg = config.get("pruner", {"name": "MedianPruner", "kwargs": {}})
    name = pruner_cfg.get("name", "MedianPruner")
    kwargs = dict(pruner_cfg.get("kwargs", {}))

    # Stage configs are deep-merged with experiment defaults, so changing the
    # pruner type can otherwise leave incompatible kwargs behind from the base
    # pruner. Filter aggressively by constructor signature to keep mixed staged
    # configs robust.
    allowed_kwargs = {
        "MedianPruner": {"n_startup_trials", "n_warmup_steps", "interval_steps", "n_min_trials"},
        "SuccessiveHalvingPruner": {"min_resource", "reduction_factor", "min_early_stopping_rate", "bootstrap_count"},
        "HyperbandPruner": {"min_resource", "max_resource", "reduction_factor", "bootstrap_count"},
        "NopPruner": set(),
    }
    if name not in allowed_kwargs:
        raise ValueError(f"Unsupported pruner '{name}'.")
    kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs[name]}

    if name == "MedianPruner":
        return optuna.pruners.MedianPruner(**kwargs)
    if name == "SuccessiveHalvingPruner":
        return optuna.pruners.SuccessiveHalvingPruner(**kwargs)
    if name == "HyperbandPruner":
        return optuna.pruners.HyperbandPruner(**kwargs)
    if name == "NopPruner":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unsupported pruner '{name}'.")


def load_tuning_config(config_path: str):
    cfg = _load_json(Path(config_path))
    study_name = cfg.get("study_name", Path(config_path).stem)
    results_root = Path(cfg.get("results_root", DEFAULT_RESULTS_ROOT))
    storage = cfg.get("storage")
    if not storage:
        storage = f"sqlite:///{(results_root / study_name / 'optuna_study.db').as_posix()}"

    objective = cfg.get("objective", {})
    objective.setdefault("metric", DEFAULT_OBJECTIVE_NAME)
    objective.setdefault("direction", "minimize")

    cfg["study_name"] = study_name
    cfg["results_root"] = str(results_root)
    cfg["storage"] = storage
    cfg["objective"] = objective
    cfg.setdefault("mode", "physics")
    cfg.setdefault("fixed_params", {})
    cfg.setdefault("search_space", {})
    cfg.setdefault("trainer", {})
    cfg.setdefault("n_trials", 10)
    cfg.setdefault("timeout", None)
    cfg.setdefault("seed", 42)
    cfg.setdefault("resume", True)
    return cfg


def normalize_tuning_config(config: dict):
    cfg = deepcopy(config)
    study_name = cfg.get("study_name", "optuna_study")
    results_root = Path(cfg.get("results_root", DEFAULT_RESULTS_ROOT))
    storage = cfg.get("storage")
    if not storage:
        storage = f"sqlite:///{(results_root / study_name / 'optuna_study.db').as_posix()}"

    objective = cfg.get("objective", {})
    objective.setdefault("metric", DEFAULT_OBJECTIVE_NAME)
    objective.setdefault("direction", "minimize")

    cfg["study_name"] = study_name
    cfg["results_root"] = str(results_root)
    cfg["storage"] = storage
    cfg["objective"] = objective
    cfg.setdefault("mode", "physics")
    cfg.setdefault("fixed_params", {})
    cfg.setdefault("search_space", {})
    cfg.setdefault("trainer", {})
    cfg.setdefault("n_trials", 10)
    cfg.setdefault("timeout", None)
    cfg.setdefault("seed", 42)
    cfg.setdefault("resume", True)
    return cfg


def run_study_config(config: dict):
    import optuna

    config = normalize_tuning_config(config)
    runner = StudyRunner(config)

    study = optuna.create_study(
        study_name=config["study_name"],
        storage=config["storage"],
        direction="minimize",
        sampler=_create_sampler(config),
        pruner=_create_pruner(config),
        load_if_exists=bool(config.get("resume", True)),
    )
    study.set_user_attr("config_path", str(config.get("config_path", "<in-memory-config>")))
    study.set_user_attr("mode", config["mode"])

    study.optimize(
        runner.objective,
        n_trials=int(config.get("n_trials", 10)),
        timeout=config.get("timeout"),
        gc_after_trial=True,
        show_progress_bar=bool(config.get("show_progress_bar", False)),
    )

    generate_study_summary(study, runner.study_dir, config)
    return study


def run_study(config_path: str):
    config = load_tuning_config(config_path)
    return run_study_config(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Optuna tuning for TorchMD-NET experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to tuning config JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_study(args.config)


if __name__ == "__main__":
    main()
