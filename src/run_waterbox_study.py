#!/usr/bin/env python
"""
Water-box study runner: {water_absolute, water_absolute+momentum} x {seeds}.

Modeled on run_ablation_study.py's orchestration patterns (resume-if-checkpoint-
exists, incremental raw_results.csv writes, per-cell try/except fault isolation)
but scoped to this study's two conditions - no molecule axis, since WaterBox is
one fixed periodic system, not a family of molecules.

Usage:
    python src/run_waterbox_study.py                  # both conditions, all seeds
    python src/run_waterbox_study.py --smoke-test      # 1 condition, 1 seed, 2 epochs -
                                                        # run this first.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import traceback
from pathlib import Path

from evaluate_waterbox import evaluate_waterbox_checkpoint
from train_waterbox import train_waterbox_model

# water_absolute (momentum_weight=0) is the control; water_absolute+momentum
# adds the per-molecule momentum-conservation term (see train_waterbox.py /
# physics_losses.per_fragment_momentum_loss for what that actually checks).
CONDITIONS = {
    "water_absolute": dict(momentum_weight=0.0),
    "water_absolute+momentum": dict(momentum_weight=0.01),
}
SEEDS = [0, 1, 2]

# Same architecture/optimizer defaults as the aspirin ablation, reused rather
# than re-tuned - this study's independent variable is the momentum loss, not
# hyperparameters. WaterBox has far fewer total configurations (~1593) than
# MD17 aspirin (~211k), so 20 epochs here is a much smaller undertaking.
#
# batch_size (the real per-forward-pass batch, which is what drives GPU peak
# memory) is NOT reused from aspirin (was 32 there) - each WaterBox example is
# a 192-atom periodic system with real liquid-density local connectivity, vs.
# aspirin's 21-atom molecule with a sparse bond graph. TensorNet represents
# each edge as an (embedding_dimension x 3 x 3) tensor, so at a 5.0 A cutoff
# this dataset produces far more edges per example than aspirin ever did.
# Confirmed via CUDA OOM on the training box (2026-07-31, RTX 6000 Ada 48GB):
# batch_size=32 needed ~45 GB for a single pass; batch_size=4 STILL wasn't
# enough - it trained fine for 267/319 steps of epoch 0 before an unusually
# dense batch (more real neighbor pairs within cutoff than a typical one -
# max_num_neighbors=128, also unchanged from aspirin, allows this swing)
# needed 378 MiB more than was left. PyTorch's allocator never releases
# reserved memory mid-run, so the ceiling is set by the worst batch seen so
# far - not something one epoch's worth of running is guaranteed to expose the
# true worst case of. batch_size=2 leaves ~2x the margin batch_size=4 had (and
# 4 was only ~99% full), which should cover realistic tail variance.
#
# accumulate_grad_batches=16 recovers the aspirin-matched effective batch size
# (2 x 16 = 32) for the optimizer, so this is purely a memory-footprint fix,
# not a silent change to the optimization dynamics the "reused, not re-tuned"
# comment above was relying on.
FIXED_HPARAMS = dict(
    batch_size=2,
    num_epochs=20,
    lr=1e-4,
    embedding_dimension=256,
    num_layers=6,
    num_rbf=64,
    trainer_kwargs=dict(accumulate_grad_batches=16),
)

CHECKPOINT_ROOT = Path("checkpoints/waterbox_study")
LOG_ROOT = Path("logs/waterbox_study")
RESULTS_ROOT = Path("results/waterbox_study")
RAW_RESULTS_CSV = RESULTS_ROOT / "raw_results.csv"
SUMMARY_CSV = RESULTS_ROOT / "summary_table.csv"
SUMMARY_MD = RESULTS_ROOT / "summary_table.md"

METRIC_COLUMNS = [
    "energy_mae",
    "force_mae",
    "mean_per_molecule_momentum_violation",
    "max_per_molecule_momentum_violation",
    "final_epoch",
    "total_wall_seconds",
]


def _run_training(condition_name, condition_kwargs, seed, num_epochs, force_retrain=False):
    save_dir = CHECKPOINT_ROOT / condition_name / f"seed{seed}"
    log_dir = LOG_ROOT / condition_name / f"seed{seed}"
    ckpt_path = save_dir / "best_model.ckpt"
    if ckpt_path.exists() and not force_retrain:
        print(f"  Checkpoint already exists at {ckpt_path} - skipping training "
              "(pass --force-retrain to redo it anyway).")
        return save_dir

    hparams = dict(FIXED_HPARAMS)
    hparams["num_epochs"] = num_epochs
    train_waterbox_model(
        save_dir=str(save_dir),
        log_dir=str(log_dir),
        checkpoint_name="best_model",
        seed=seed,
        **condition_kwargs,
        **hparams,
    )
    return save_dir


def _evaluate_checkpoint(save_dir, seed):
    ckpt_path = Path(save_dir) / "best_model.ckpt"
    eval_result = evaluate_waterbox_checkpoint(str(ckpt_path), seed=seed)

    history_path = Path(save_dir) / "best_model_history.json"
    history_rows = json.loads(history_path.read_text()) if history_path.exists() else []
    final_row = history_rows[-1] if history_rows else {}

    return {
        "energy_mae": eval_result["energy_mae"],
        "force_mae": eval_result["force_mae"],
        "mean_per_molecule_momentum_violation": eval_result["mean_per_molecule_momentum_violation"],
        "max_per_molecule_momentum_violation": eval_result["max_per_molecule_momentum_violation"],
        "final_epoch": final_row.get("epoch"),
        "total_wall_seconds": final_row.get("cumulative_wall_seconds"),
    }


def run_matrix(conditions, seeds, num_epochs, force_retrain=False):
    """Train+evaluate every (condition, seed) cell. Resilient to a single cell
    failing (full traceback printed, cell recorded as failed, matrix continues)
    and rewrites raw_results.csv after every completed cell - same lessons as
    run_ablation_study.py's run_matrix, applied from the start here rather than
    added after a crash cost real results.

    Returns (rows, failed_cells).
    """
    rows = []
    failed = []
    for condition_name, condition_kwargs in conditions.items():
        for seed in seeds:
            cell = f"{condition_name}/seed{seed}"
            print(f"\n=== {cell} ===")
            try:
                save_dir = _run_training(condition_name, condition_kwargs, seed, num_epochs, force_retrain=force_retrain)
                metrics = _evaluate_checkpoint(save_dir, seed)
            except Exception as exc:
                print(f"FAILED: {cell}: {exc}")
                traceback.print_exc()
                failed.append(cell)
                continue
            row = {"condition": condition_name, "seed": seed, **metrics}
            rows.append(row)
            _write_raw_results_csv(rows, RAW_RESULTS_CSV)

    if failed:
        print("\nFailed cells (full tracebacks above) - fix and re-run; already-completed cells are safe:")
        for cell in failed:
            print(f"  {cell}")

    return rows, failed


def _write_raw_results_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "seed"] + METRIC_COLUMNS
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Wrote: {path}")


def aggregate_results(rows, summary_csv=SUMMARY_CSV, summary_md=SUMMARY_MD):
    """Group by condition and compute mean +/- std across seeds. Plain stdlib
    (statistics module), not pandas - matches run_ablation_study.py's choice,
    since pandas isn't a confirmed dependency anywhere in this project."""
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)

    summary_rows = []
    for condition, group_rows in sorted(groups.items()):
        summary = {"condition": condition, "n_seeds": len(group_rows)}
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
    fieldnames = ["condition", "n_seeds"] + [f"{metric}_{stat}" for metric in METRIC_COLUMNS for stat in ("mean", "std")]
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
        "# Water-box study summary (mean +/- std across seeds)",
        "",
        "Expected sanity check: water_absolute's mean_per_molecule_momentum_violation",
        "should be clearly nonzero (unlike the aspirin single-molecule study, where the",
        "whole-molecule version of this quantity was ~1e-9, floating-point noise, even",
        "with no training pressure on it). If it's already ~0 here too, that undercuts",
        "this study's premise and is worth knowing before reading anything else below.",
        "",
        "n=3 seeds supports a coarse signal-vs-noise read, not formal significance -",
        "treat a difference as real only if it clears roughly 1 std of both conditions.",
        "",
    ]
    header = "| condition | n_seeds | " + " | ".join(METRIC_COLUMNS) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in METRIC_COLUMNS) + " |"
    lines += [header, sep]
    for row in summary_rows:
        cells = []
        for metric in METRIC_COLUMNS:
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            cells.append("n/a" if mean is None else f"{mean:.4g} +/- {std:.2g}")
        lines.append(f"| {row['condition']} | {row['n_seeds']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 condition (water_absolute), 1 seed, 2 epochs - confirm the plumbing "
        "end-to-end before running the full comparison.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain every cell even if a checkpoint already exists.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        conditions = {"water_absolute": CONDITIONS["water_absolute"]}
        seeds = [0]
        num_epochs = 2
    else:
        conditions = CONDITIONS
        seeds = SEEDS
        num_epochs = FIXED_HPARAMS["num_epochs"]

    rows, failed_cells = run_matrix(conditions, seeds, num_epochs, force_retrain=args.force_retrain)
    aggregate_results(rows)
    if failed_cells:
        print(f"\n{len(failed_cells)} cell(s) failed and are NOT in the results above: {failed_cells}")
        print("Fix the underlying issue and re-run - completed cells will be skipped, not redone.")


if __name__ == "__main__":
    main()
