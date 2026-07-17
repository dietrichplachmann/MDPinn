from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import lightning.pytorch as pl


class MetricHistoryCallback(pl.Callback):
    """Persist per-epoch metrics (plus wall-clock time and step count) for later
    plotting, inspection, and training-efficiency comparisons across conditions."""

    def __init__(self, output_dir: str | Path, run_name: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.run_name = str(run_name)
        self.rows: list[dict[str, float | int]] = []
        self._fit_start_time: float | None = None
        self._last_cumulative_wall_seconds: float = 0.0

    @staticmethod
    def _to_float(value):
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def on_fit_start(self, trainer, pl_module):
        self._fit_start_time = time.perf_counter()
        self._last_cumulative_wall_seconds = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return

        logged = {}
        for key, value in trainer.callback_metrics.items():
            scalar = self._to_float(value)
            if scalar is not None:
                logged[key] = scalar

        if not logged:
            return

        metrics = {"epoch": int(trainer.current_epoch), "global_step": int(trainer.global_step), **logged}

        # Wall-clock timing: recorded here (not on_train_epoch_end) because Lightning
        # runs validation before on_train_epoch_end fires for a given epoch, and the
        # quantity we actually want is "wall time elapsed to reach this validation
        # result" - exactly what training-efficiency/convergence-speed comparisons need.
        if self._fit_start_time is not None:
            cumulative = time.perf_counter() - self._fit_start_time
            metrics["cumulative_wall_seconds"] = cumulative
            metrics["epoch_wall_seconds"] = cumulative - self._last_cumulative_wall_seconds
            self._last_cumulative_wall_seconds = cumulative

        self.rows.append(metrics)

    def on_fit_end(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{self.run_name}_history.json"
        csv_path = self.output_dir / f"{self.run_name}_history.csv"

        self.rows = self._non_empty_rows(self.rows)

        with open(json_path, "w") as handle:
            json.dump(self.rows, handle, indent=2)

        headers = sorted({key for row in self.rows for key in row.keys()})
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        self._write_plot()
        self._write_plot(exclude_epoch0=True)

    @staticmethod
    def _non_empty_rows(rows):
        return [row for row in rows if any(key != "epoch" for key in row.keys())]

    @staticmethod
    def _series(rows, key):
        xs = []
        ys = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            xs.append(row["epoch"])
            ys.append(value)
        return xs, ys

    def _write_plot(self, exclude_epoch0=False):
        rows = self._non_empty_rows(self.rows)
        if exclude_epoch0:
            rows = [row for row in rows if float(row["epoch"]) != 0.0]
        if not rows:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        total_keys = [
            key
            for key in (
                "train_total_mse_loss",
                "val_total_mse_loss",
                "train_total_with_physics",
                "val_rollout_score",
            )
            if any(key in row for row in rows)
        ]
        energy_keys = [
            key
            for key in (
                "train_y_mse_loss",
                "val_y_mse_loss",
                "val_energy_mae",
            )
            if any(key in row for row in rows)
        ]
        force_keys = [
            key
            for key in (
                "train_neg_dy_mse_loss",
                "val_neg_dy_mse_loss",
                "val_force_mae",
            )
            if any(key in row for row in rows)
        ]
        physics_keys = [
            key
            for key in (
                "train_loss_momentum",
                "train_loss_pbc",
                "train_loss_momentum_weighted",
                "train_loss_pbc_weighted",
                "train_physics_to_supervised_ratio",
                "train_momentum_to_supervised_ratio",
                # val_rollout_* / train_short_rollout_* are diagnostic-only (not part
                # of the training loss) - see PhysicsInformedLNNP.on_validation_epoch_end
                # / on_train_epoch_end in train_physics.py.
                "val_rollout_median_mean_abs_drift_eV",
                "val_rollout_median_max_abs_drift_eV",
                "val_rollout_failure_rate",
                "train_short_rollout_mean_abs_drift_eV",
                "train_short_rollout_max_abs_drift_eV",
            )
            if any(key in row for row in rows)
        ]
        groups = [
            ("Total Loss Curves", total_keys, "Loss"),
            ("Energy Metrics", energy_keys, "Loss / metric"),
            ("Force Metrics", force_keys, "Loss / metric"),
            ("Physics / Rollout Metrics", physics_keys, "Loss / metric"),
        ]
        groups = [(title, keys, ylabel) for title, keys, ylabel in groups if keys]

        if not groups:
            return

        nrows = len(groups)
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        title_suffix = " (excluding epoch 0)" if exclude_epoch0 else ""
        for ax, (title, keys, ylabel) in zip(axes, groups):
            for key in keys:
                xs, ys = self._series(rows, key)
                if ys:
                    ax.plot(xs, ys, linewidth=1.8, label=key)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}{title_suffix}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        suffix = "_history_no_epoch0.png" if exclude_epoch0 else "_history.png"
        fig.savefig(self.output_dir / f"{self.run_name}{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def find_convergence_point(rows, metric_key: str, threshold: float, mode: str = "min"):
    """Return the first row (in epoch order) where a run's validation metric
    crosses a shared threshold, or None if it never does.

    Used to compare training efficiency across ablation conditions: given a
    target level of accuracy (typically the best value some reference
    condition reached), this finds how many epochs / gradient steps /
    wall-clock seconds each other condition needed to reach the same level.
    Reads the same row-dict format MetricHistoryCallback writes to
    `{run_name}_history.json` (each row has at least "epoch", "global_step",
    "cumulative_wall_seconds", plus whatever scalar metrics were logged).

    Args:
        rows: list of per-epoch metric dicts (e.g. json.load of a *_history.json file).
        metric_key: which logged metric to threshold on (e.g. "val_total_mse_loss").
        threshold: the target value.
        mode: "min" - first row where value <= threshold (e.g. a loss dropping to a
            target); "max" - first row where value >= threshold.

    Returns:
        The matching row dict (so callers can read row["epoch"], row["global_step"],
        row["cumulative_wall_seconds"]), or None if metric_key is never present or
        the threshold is never crossed.
    """
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

    ordered = sorted((row for row in rows if metric_key in row), key=lambda row: row["epoch"])
    for row in ordered:
        value = row[metric_key]
        if (mode == "min" and value <= threshold) or (mode == "max" and value >= threshold):
            return row
    return None
