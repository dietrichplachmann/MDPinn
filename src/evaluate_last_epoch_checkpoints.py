#!/usr/bin/env python
"""Re-evaluate the water-box study's six existing runs at their LAST epoch's
checkpoint, instead of the ModelCheckpoint-selected "best" (lowest
val_total_mse_loss) one that run_waterbox_study.py's summary table reports.

Why: per-epoch training histories show val_total_mse_loss oscillating by 1-3
orders of magnitude across epochs in every run, for both conditions - it does
not converge smoothly. "Best" checkpoint selection is therefore effectively
sampling from noise (e.g. one water_absolute seed's best came from epoch 3 of
18 and was never matched again). This script answers: does the
water_absolute vs. water_absolute+momentum comparison look different - and
either way, more or less noisy across seeds - once every run is evaluated at
the same fixed, non-cherry-picked point (the final epoch) instead of each
seed's own luckiest noise-dip?

Costs zero additional training: ModelCheckpoint was configured with
save_last=True, so every run should already have its final-epoch checkpoint
on disk. Only runs evaluation forward passes, not training.

Usage:
    python src/evaluate_last_epoch_checkpoints.py
"""

from __future__ import annotations

from pathlib import Path

from evaluate_waterbox import evaluate_waterbox_checkpoint
from run_waterbox_study import CONDITIONS, SEEDS, CHECKPOINT_ROOT, RESULTS_ROOT, aggregate_results, _write_raw_results_csv

LAST_RAW_CSV = RESULTS_ROOT / "last_epoch_raw_results.csv"
LAST_SUMMARY_CSV = RESULTS_ROOT / "last_epoch_summary_table.csv"
LAST_SUMMARY_MD = RESULTS_ROOT / "last_epoch_summary_table.md"


def _find_last_checkpoint(save_dir: Path) -> Path:
    """Lightning's ModelCheckpoint(save_last=True) default name is last.ckpt,
    independent of the `filename` template used for the best checkpoint - but
    this is defensive (not verified against the installed Lightning version)
    rather than assumed, since guessing wrong here once already cost real time
    this session (see the download-URL and unit-conversion detours)."""
    exact = save_dir / "last.ckpt"
    if exact.exists():
        return exact

    candidates = sorted(save_dir.glob("*last*.ckpt"))
    if candidates:
        return candidates[0]

    existing = sorted(p.name for p in save_dir.iterdir()) if save_dir.exists() else []
    raise FileNotFoundError(
        f"No last-epoch checkpoint found in {save_dir} (looked for 'last.ckpt' "
        f"and '*last*.ckpt'). Files actually present: {existing}. "
        "If save_last=True didn't produce what's expected here, check "
        "train_waterbox_model's ModelCheckpoint config and this function's "
        "assumption together before guessing again."
    )


def main():
    rows = []
    failed = []

    for condition_name in CONDITIONS:
        for seed in SEEDS:
            save_dir = CHECKPOINT_ROOT / condition_name / f"seed{seed}"
            cell = f"{condition_name}/seed{seed}"
            print(f"\n=== {cell} (last-epoch checkpoint) ===")
            try:
                ckpt_path = _find_last_checkpoint(save_dir)
                print(f"  Evaluating {ckpt_path}")
                eval_result = evaluate_waterbox_checkpoint(str(ckpt_path), seed=seed)
            except Exception as exc:
                print(f"FAILED: {cell}: {exc}")
                import traceback
                traceback.print_exc()
                failed.append(cell)
                continue

            rows.append({
                "condition": condition_name,
                "seed": seed,
                "energy_mae": eval_result["energy_mae"],
                "force_mae": eval_result["force_mae"],
                "mean_per_molecule_momentum_violation": eval_result["mean_per_molecule_momentum_violation"],
                "max_per_molecule_momentum_violation": eval_result["max_per_molecule_momentum_violation"],
            })
            _write_raw_results_csv(rows, LAST_RAW_CSV)

    if failed:
        print(f"\n{len(failed)} cell(s) failed and are NOT in the results above: {failed}")

    aggregate_results(rows, summary_csv=LAST_SUMMARY_CSV, summary_md=LAST_SUMMARY_MD)


if __name__ == "__main__":
    main()
