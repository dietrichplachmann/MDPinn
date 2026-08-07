#!/usr/bin/env python
"""Check whether the energy/force loss-weight annealing schedule
(run_waterbox_study.py's anneal_epoch/post_anneal_energy_weight/
post_anneal_force_weight) helped, for runs that already finished before
train_waterbox.py's val_checkpoint_score fix existed.

Why this is needed even after that fix: ModelCheckpoint used to monitor
val_total_mse_loss, which is w_E * val_y_mse + w_F * val_neg_dy_mse using
whatever (w_E, w_F) are active THAT epoch. Since the anneal schedule changes
those weights at anneal_epoch (0.05/0.95 force-dominant before, 0.75/0.25
energy-dominant after), the metric's own scale shifted at exactly the epoch
boundary this was supposed to test - pre-anneal totals were dominated by the
already-small, well-behaved force term; post-anneal totals were dominated by
the large, volatile energy term. That structurally favored picking a
pre-anneal epoch as "best" regardless of whether the model actually improved
post-anneal - confirmed empirically: every one of the water-box study's runs
under this metric picked its best checkpoint from before the anneal switch.

train_waterbox.py now monitors an anneal-invariant val_checkpoint_score
instead, so a fresh run doesn't need this script - its own "best" checkpoint
selection is already fair. This script exists for runs that already
completed under the old, biased monitor, where retraining from scratch just
to get a fair answer isn't worth the GPU time when the same answer is
already recoverable for free from data already on disk: val_y_l1_loss and
val_neg_dy_l1_loss are logged every epoch by MetricHistoryCallback, raw and
unweighted, so they never had the anneal-boundary bias to begin with.

Median, not mean: the first 1-2 epochs of every run have enormous, untrained-
model energy error (tens of thousands of eV) that would swamp a straight
average - the project's own training-history plots already work around this
by excluding epoch 0. Median sidesteps that without an arbitrary cutoff.

Usage:
    python src/analyze_anneal_effect.py
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from run_waterbox_study import CONDITIONS, SEEDS, CHECKPOINT_ROOT, RESULTS_ROOT, FIXED_HPARAMS

ANNEAL_EPOCH = FIXED_HPARAMS.get("anneal_epoch")

RAW_CSV = RESULTS_ROOT / "anneal_effect_raw.csv"
SUMMARY_CSV = RESULTS_ROOT / "anneal_effect_summary.csv"
SUMMARY_MD = RESULTS_ROOT / "anneal_effect_summary.md"

RAW_FIELDNAMES = [
    "condition", "seed",
    "pre_anneal_energy_median", "post_anneal_energy_median",
    "pre_anneal_force_median", "post_anneal_force_median",
    "n_pre_epochs", "n_post_epochs",
]

METRIC_KEYS = [
    "pre_anneal_energy_median", "post_anneal_energy_median",
    "pre_anneal_force_median", "post_anneal_force_median",
]


def _cell_medians(history_path: Path, anneal_epoch: int) -> dict:
    with open(history_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    pre = [r for r in rows if float(r["epoch"]) < anneal_epoch]
    post = [r for r in rows if float(r["epoch"]) >= anneal_epoch]
    if not pre or not post:
        raise ValueError(
            f"{history_path} has {len(pre)} pre-anneal and {len(post)} "
            f"post-anneal epochs (anneal_epoch={anneal_epoch}) - need at "
            "least one of each to compare. Run hasn't reached the anneal "
            "switch yet, or anneal_epoch doesn't match what this run was "
            "actually trained with."
        )

    return {
        "pre_anneal_energy_median": statistics.median(float(r["val_y_l1_loss"]) for r in pre),
        "post_anneal_energy_median": statistics.median(float(r["val_y_l1_loss"]) for r in post),
        "pre_anneal_force_median": statistics.median(float(r["val_neg_dy_l1_loss"]) for r in pre),
        "post_anneal_force_median": statistics.median(float(r["val_neg_dy_l1_loss"]) for r in post),
        "n_pre_epochs": len(pre),
        "n_post_epochs": len(post),
    }


def _write_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)

    summary_rows = []
    for condition, group_rows in sorted(groups.items()):
        summary = {"condition": condition, "n_seeds": len(group_rows)}
        for key in METRIC_KEYS:
            values = [row[key] for row in group_rows]
            summary[f"{key}_mean"] = statistics.mean(values)
            summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    fieldnames = ["condition", "n_seeds"] + [f"{k}_{stat}" for k in METRIC_KEYS for stat in ("mean", "std")]
    with open(SUMMARY_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote: {SUMMARY_CSV}")

    lines = [
        "# Anneal-effect check (median val_y_l1_loss / val_neg_dy_l1_loss, pre vs. post anneal_epoch)",
        "",
        f"anneal_epoch = {ANNEAL_EPOCH}. Unweighted, unbiased-by-the-schedule metrics - see module",
        "docstring for why val_total_mse_loss (what ModelCheckpoint used to pick 'best' by, for the",
        "run this was computed against) can't answer this question on its own.",
        "",
        "| condition | n_seeds | pre energy (eV) | post energy (eV) | pre force (eV/A) | post force (eV/A) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['n_seeds']} | "
            f"{row['pre_anneal_energy_median_mean']:.3g} +/- {row['pre_anneal_energy_median_std']:.2g} | "
            f"{row['post_anneal_energy_median_mean']:.3g} +/- {row['post_anneal_energy_median_std']:.2g} | "
            f"{row['pre_anneal_force_median_mean']:.3g} +/- {row['pre_anneal_force_median_std']:.2g} | "
            f"{row['post_anneal_force_median_mean']:.3g} +/- {row['post_anneal_force_median_std']:.2g} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {SUMMARY_MD}")


def main():
    if ANNEAL_EPOCH is None:
        raise ValueError(
            "FIXED_HPARAMS['anneal_epoch'] is None in run_waterbox_study.py - "
            "this script only makes sense for a run trained with the anneal "
            "schedule active."
        )

    rows = []
    failed = []

    for condition_name in CONDITIONS:
        for seed in SEEDS:
            cell = f"{condition_name}/seed{seed}"
            history_path = CHECKPOINT_ROOT / condition_name / f"seed{seed}" / "best_model_history.csv"
            print(f"=== {cell} ===")
            try:
                medians = _cell_medians(history_path, ANNEAL_EPOCH)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                failed.append(cell)
                continue

            print(
                f"  energy: {medians['pre_anneal_energy_median']:.3f} -> "
                f"{medians['post_anneal_energy_median']:.3f} eV   "
                f"force: {medians['pre_anneal_force_median']:.4f} -> "
                f"{medians['post_anneal_force_median']:.4f} eV/A"
            )
            rows.append({"condition": condition_name, "seed": seed, **medians})

    if failed:
        print(f"\n{len(failed)} cell(s) skipped (no history yet or bad anneal_epoch): {failed}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RAW_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nWrote: {RAW_CSV}")

    _write_summary(rows)


if __name__ == "__main__":
    main()
