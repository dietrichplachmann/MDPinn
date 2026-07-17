#!/usr/bin/env python
"""
Ablation-matrix runner: {loss/architecture condition} x {molecule} x {seed}.

This does not reimplement training or evaluation - it orchestrates the
functions that already exist elsewhere in this repo:
- train_physics.py's train_physics_informed_model (unified entrypoint for every
  condition; "absolute, momentum_weight=0" is a legitimate stand-in for plain
  supervised training, see train_physics.py's PhysicsInformedLNNP docstring).
- compare_models.py for static energy/force accuracy and the energy-drift proxy.
- experiment_suite.py's run_rollout_summary for real model-driven rollout drift.
- structural_metrics.py (new) for bond-length structural fidelity.
- training_history.find_convergence_point (new) for training-efficiency
  (epochs/steps/wall-clock to a shared accuracy threshold) comparisons.

Every ablation cell shares: the same training entrypoint, the same
checkpoint-selection metric (val_total_mse_loss), and the same fixed
architecture/optimizer hyperparameters - only delta_learning and
momentum_weight vary - so whatever difference shows up in the summary table is
attributable to the loss/architecture choice, not a confound.

Usage:
    python src/run_ablation_study.py                      # full matrix
    python src/run_ablation_study.py --smoke-test          # 1 molecule, 1 condition,
                                                            # 1 seed, 2 epochs - run this
                                                            # first to sanity check the
                                                            # plumbing before spending
                                                            # real GPU time.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from baseline_potential import has_analytic_baseline
from compare_models import (
    compute_metrics,
    create_dataset,
    evaluate_energy_conservation,
    evaluate_on_dataset,
    load_checkpoint,
)
from experiment_suite import run_rollout_summary
from structural_metrics import bond_length_deviation_summary, bond_length_series, infer_bonds
from train_physics import train_physics_informed_model
from training_history import find_convergence_point


# Loss/architecture conditions. "absolute" (delta_learning=False, momentum_weight=0.0)
# is the plain-supervised baseline, run through the same entrypoint as every other
# condition rather than train_standard.py's separate code path, so there is no
# code-path confound alongside the loss-composition difference.
CONDITIONS = {
    "absolute": dict(delta_learning=False, momentum_weight=0.0),
    "absolute+momentum": dict(delta_learning=False, momentum_weight=0.01),
    "delta": dict(delta_learning=True, momentum_weight=0.0),
    "delta+momentum": dict(delta_learning=True, momentum_weight=0.01),
}

# Swap freely - verify these are valid torchmdnet.datasets.MD17 molecule keys on
# your training environment first. delta/delta+momentum are automatically skipped
# for any molecule without an analytic baseline (currently aspirin only).
MOLECULES = ["aspirin", "benzene", "ethanol"]
SEEDS = [0, 1, 2]

# Fixed architecture/optimizer hyperparameters, reused from the existing best
# aspirin run (checkpoints/standard/config.json) rather than re-swept per cell -
# this study's independent variable is loss composition, not hyperparameters.
FIXED_HPARAMS = dict(
    batch_size=32,
    num_epochs=20,
    lr=1e-4,
    embedding_dimension=256,
    num_layers=6,
    num_rbf=64,
)

ROLLOUT_STEPS = 5000
ROLLOUT_DT_FS = 0.1
N_ROLLOUTS = 10
ROLLOUT_ENERGY_LOG_STRIDE = 20
REFERENCE_WINDOW_FRAMES = 200  # frames of reference MD17 trajectory used for the
                               # structural-fidelity comparison's "reference ensemble"

RESULTS_ROOT = Path("results/ablation")
CHECKPOINT_ROOT = Path("checkpoints/ablation")
LOG_ROOT = Path("logs/ablation")

RAW_RESULTS_CSV = RESULTS_ROOT / "raw_results.csv"
SUMMARY_CSV = RESULTS_ROOT / "summary_table.csv"
SUMMARY_MD = RESULTS_ROOT / "summary_table.md"

METRIC_COLUMNS = [
    "energy_mae",
    "force_mae",
    "energy_drift_mean",
    "rollout_mean_max_abs_drift_eV",
    "rollout_mean_final_drift_eV",
    "rollout_failure_rate",
    "structural_stability_score",
    "final_epoch",
    "total_wall_seconds",
    "epochs_to_threshold",
    "global_step_to_threshold",
    "wall_seconds_to_threshold",
]


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _condition_is_valid_for_molecule(condition_kwargs, molecule):
    if condition_kwargs.get("delta_learning") and not has_analytic_baseline(molecule):
        return False
    return True


def _run_training(molecule, condition_name, condition_kwargs, seed, num_epochs):
    save_dir = CHECKPOINT_ROOT / molecule / condition_name / f"seed{seed}"
    log_dir = LOG_ROOT / molecule / condition_name / f"seed{seed}"
    hparams = dict(FIXED_HPARAMS)
    hparams["num_epochs"] = num_epochs
    train_physics_informed_model(
        molecule=molecule,
        save_dir=str(save_dir),
        log_dir=str(log_dir),
        checkpoint_name="best_model",
        seed=seed,
        **condition_kwargs,
        **hparams,
    )
    return save_dir


def _structural_stability_for_rollout(row, full_dataset):
    """Bond-length structural stability score for one recorded rollout, or None
    if positions weren't recorded / the rollout failed."""
    x_traj = row.get("x_traj")
    if row.get("failed") or not x_traj:
        return None

    bonds = infer_bonds(row["z"], x_traj[0])
    if not bonds:
        return None

    rollout_lengths = bond_length_series(bonds, x_traj)

    start_idx = row["start_idx"]
    end_idx = min(start_idx + REFERENCE_WINDOW_FRAMES, len(full_dataset))
    reference_positions = [full_dataset[i].pos for i in range(start_idx, end_idx)]
    if len(reference_positions) < 2:
        return None
    reference_lengths = bond_length_series(bonds, reference_positions)

    summary = bond_length_deviation_summary(rollout_lengths, reference_lengths)
    return summary["structural_stability_score"]


def _evaluate_checkpoint(molecule, save_dir, seed):
    device = _device()
    ckpt_path = Path(save_dir) / "best_model.ckpt"
    model = load_checkpoint(str(ckpt_path), device=device)

    _, _, test_data, full_dataset = create_dataset("MD17", molecule, "./data")

    static_results = evaluate_on_dataset(model, test_data, device=device)
    static_metrics = compute_metrics(static_results)
    drift_proxy = evaluate_energy_conservation(model, full_dataset, device=device)

    rollout = run_rollout_summary(
        ckpt_path=str(ckpt_path),
        dataset="MD17",
        molecule=molecule,
        data_root="./data",
        steps=ROLLOUT_STEPS,
        dt=ROLLOUT_DT_FS,
        n_rollouts=N_ROLLOUTS,
        seed=seed,
        device=device,
        energy_log_stride=ROLLOUT_ENERGY_LOG_STRIDE,
        record_positions=True,
    )

    structural_scores = [
        score
        for row in rollout.get("rollouts", [])
        if (score := _structural_stability_for_rollout(row, full_dataset)) is not None
    ]
    structural_stability_score = (
        float(sum(structural_scores) / len(structural_scores)) if structural_scores else None
    )

    history_path = Path(save_dir) / "best_model_history.json"
    history_rows = json.loads(history_path.read_text()) if history_path.exists() else []
    final_row = history_rows[-1] if history_rows else {}

    return {
        "energy_mae": static_metrics["energy_mae"],
        "force_mae": static_metrics["force_mae"],
        "energy_drift_mean": drift_proxy["energy_drift_mean"],
        "rollout_mean_max_abs_drift_eV": rollout["mean_max_abs_drift_eV"],
        "rollout_mean_final_drift_eV": rollout["mean_final_drift_eV"],
        "rollout_failure_rate": rollout["failure_rate"],
        "structural_stability_score": structural_stability_score,
        "final_epoch": final_row.get("epoch"),
        "total_wall_seconds": final_row.get("cumulative_wall_seconds"),
        "history_rows": history_rows,
    }


def run_matrix(molecules, conditions, seeds, num_epochs):
    """Train+evaluate every valid (molecule, condition, seed) cell.

    Returns the list of raw per-run result dicts, including each run's
    `history_rows` (needed by `compute_convergence_speed`). Does not write
    the CSV itself - see `main()`, which writes it once convergence-speed
    columns have been filled in.
    """
    rows = []
    skipped = []
    for molecule in molecules:
        for condition_name, condition_kwargs in conditions.items():
            if not _condition_is_valid_for_molecule(condition_kwargs, molecule):
                skipped.append((molecule, condition_name))
                continue
            for seed in seeds:
                print(f"\n=== {molecule} / {condition_name} / seed={seed} ===")
                save_dir = _run_training(molecule, condition_name, condition_kwargs, seed, num_epochs)
                metrics = _evaluate_checkpoint(molecule, save_dir, seed)
                row = {"molecule": molecule, "condition": condition_name, "seed": seed, **metrics}
                rows.append(row)

    if skipped:
        print("\nSkipped cells (no analytic baseline for this molecule):")
        for molecule, condition_name in skipped:
            print(f"  {molecule} / {condition_name}")

    return rows


def _write_raw_results_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["molecule", "condition", "seed"] + METRIC_COLUMNS
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Wrote: {path}")


def compute_convergence_speed(rows, threshold_condition="absolute", metric_key="val_total_mse_loss"):
    """Add epochs/steps/wall-seconds-to-threshold to each row, in place.

    The shared threshold per molecule is the best `metric_key` value the
    `threshold_condition` (default: the no-physics baseline) achieved for that
    molecule, across its seeds - i.e. "how fast did each other condition reach
    what the baseline eventually reached." A condition that never crosses the
    threshold gets None (reported as "never converged to baseline level", not
    silently omitted).
    """
    thresholds = {}
    for row in rows:
        if row["condition"] != threshold_condition:
            continue
        for hist_row in row.get("history_rows", []):
            if metric_key not in hist_row:
                continue
            key = row["molecule"]
            thresholds[key] = min(thresholds.get(key, float("inf")), hist_row[metric_key])

    for row in rows:
        threshold = thresholds.get(row["molecule"])
        if threshold is None:
            row["epochs_to_threshold"] = None
            row["global_step_to_threshold"] = None
            row["wall_seconds_to_threshold"] = None
            continue
        match = find_convergence_point(row.get("history_rows", []), metric_key, threshold, mode="min")
        row["epochs_to_threshold"] = match["epoch"] if match else None
        row["global_step_to_threshold"] = match.get("global_step") if match else None
        row["wall_seconds_to_threshold"] = match.get("cumulative_wall_seconds") if match else None

    return rows


def aggregate_results(rows, summary_csv=SUMMARY_CSV, summary_md=SUMMARY_MD):
    """Group by (molecule, condition) and compute mean +/- std across seeds.

    Deliberately avoids pandas (not a confirmed dependency of this project) -
    this is plain stdlib grouping since the aggregation itself is simple
    (a handful of metrics over ~3 seeds per group).
    """
    groups = {}
    for row in rows:
        key = (row["molecule"], row["condition"])
        groups.setdefault(key, []).append(row)

    summary_rows = []
    for (molecule, condition), group_rows in sorted(groups.items()):
        summary = {"molecule": molecule, "condition": condition, "n_seeds": len(group_rows)}
        for metric in METRIC_COLUMNS:
            values = [row[metric] for row in group_rows if row.get(metric) is not None]
            if not values:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                continue
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["molecule", "condition", "n_seeds"] + [
        f"{metric}_{stat}" for metric in METRIC_COLUMNS for stat in ("mean", "std")
    ]
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote: {summary_csv}")

    _write_summary_markdown(summary_rows, Path(summary_md))
    return summary_rows


def _write_summary_markdown(summary_rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ablation summary (mean +/- std across seeds)",
        "",
        "n=3 seeds supports a coarse signal-vs-noise read, not formal significance -",
        "treat a difference as real only if it clears roughly 1 std of both conditions.",
        "",
    ]
    key_metrics = [
        "energy_mae",
        "force_mae",
        "rollout_mean_max_abs_drift_eV",
        "rollout_failure_rate",
        "structural_stability_score",
        "wall_seconds_to_threshold",
    ]
    header = "| molecule | condition | n_seeds | " + " | ".join(key_metrics) + " |"
    sep = "| --- | --- | --- | " + " | ".join("---" for _ in key_metrics) + " |"
    lines += [header, sep]
    for row in summary_rows:
        cells = []
        for metric in key_metrics:
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            cells.append("n/a" if mean is None else f"{mean:.4g} +/- {std:.2g}")
        lines.append(f"| {row['molecule']} | {row['condition']} | {row['n_seeds']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 molecule (aspirin), 1 condition (absolute), 1 seed, 2 epochs - "
        "confirm the plumbing end-to-end before running the full matrix.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        molecules, conditions, seeds, num_epochs = ["aspirin"], {"absolute": CONDITIONS["absolute"]}, [0], 2
    else:
        molecules, conditions, seeds, num_epochs = MOLECULES, CONDITIONS, SEEDS, FIXED_HPARAMS["num_epochs"]

    rows = run_matrix(molecules, conditions, seeds, num_epochs)
    rows = compute_convergence_speed(rows)
    _write_raw_results_csv(rows, RAW_RESULTS_CSV)  # rewrite with convergence-speed columns filled in
    aggregate_results(rows)


if __name__ == "__main__":
    main()
