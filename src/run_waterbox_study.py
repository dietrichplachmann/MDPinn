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
SEEDS = [0, 1, 2, 3, 4, 5]
# Extended from 3 to 6 seeds per condition (2026-08-10), now that both reasons
# to previously hold off are resolved: checkpoint selection is anneal-invariant
# (val_checkpoint_score, train_waterbox.py) and TensorNet's periodic neighbor
# search is confirmed periodicity-aware (src/verify_periodicity.py,
# paper/main.tex Section 3.4). Seeds 0-2 already have checkpoints on disk from
# the prior run - _run_training's existing-checkpoint skip means re-running
# this only trains the 3 new seeds, not a full redo.

# Memory-footprint overrides, NOT part of the experiment design (that's
# CONDITIONS above) - kept separate so the science (momentum_weight) and the
# engineering (what fits in GPU memory) don't get conflated in one dict.
# water_absolute+momentum runs a SECOND full forward+backward pass every train
# step (WaterLNNP.step's momentum branch, gated behind momentum_weight>0) on
# top of the base supervised pass, roughly doubling per-step peak memory.
# Confirmed via CUDA OOM on the training box (2026-07-31): all 3 seeds of
# water_absolute+momentum OOM'd at FIXED_HPARAMS's batch_size=2 (which was
# only validated against water_absolute's single-pass workload). Halving to
# batch_size=1 restores roughly the same per-pass memory margin batch_size=2
# gave the single-pass condition; accumulate_grad_batches=32 (vs. 16) keeps
# the same effective batch of 32 for the optimizer.
CONDITION_HPARAM_OVERRIDES = {
    "water_absolute+momentum": dict(
        batch_size=1,
        trainer_kwargs=dict(accumulate_grad_batches=32),
    ),
}

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
    num_epochs=50,
    # Extended from the original 20 epochs (2026-08-04) to check whether the
    # wild epoch-to-epoch oscillation in val_total_mse_loss (seen across every
    # seed of both conditions) eventually settles with more training, rather
    # than being a stable feature of these dynamics. early_stop_patience=40
    # gives far more room than the 30-epoch default (which was only ever inert
    # by coincidence, since it exceeded the old 20-epoch total) - but at 40 <
    # 50 it is NOT structurally guaranteed to avoid triggering: one seed
    # already went 15 straight non-improving epochs within just a 20-epoch
    # run, so a 40-epoch plateau within 50 epochs is a real possibility, not
    # a hypothetical. If a run stops noticeably before epoch 50, check
    # final_epoch before assuming the full trajectory was observed.
    early_stop_patience=40,
    lr=1e-4,
    embedding_dimension=256,
    num_layers=6,
    num_rbf=64,
    trainer_kwargs=dict(accumulate_grad_batches=16),
    # Energy/force loss-weight annealing (added 2026-08-04), motivated by
    # this study's own results: energy oscillated wildly across every seed of
    # both conditions while force stayed comparatively well-behaved under the
    # fixed 0.05/0.95 (energy/force) weighting used for the whole run - a
    # plausible contributor, since force gets 19x the gradient signal energy
    # does, for the entire run, with nothing ever correcting for it. Current
    # practice in MACE and NequIP addresses exactly this by annealing the
    # weighting during training rather than holding it fixed: force-dominant
    # early to fix the local force/gradient shape, then switching to
    # energy-dominant later specifically to fix up absolute energy
    # calibration (Batatia et al. 2022, MACE, arXiv:2206.07697 - one reported
    # schedule uses energy:force 1:100 for the first ~75% of training, then
    # 300:100 for the rest; a related MACE recipe reports force-dominant for
    # ~60% of training then switching to energy-dominant; Batzner et al. 2022,
    # NequIP, Nat. Commun. - energy:force 40:1000 early, flipped at a fixed
    # epoch). anneal_epoch=30 is 60% of this run's 50 epochs, matching that
    # reported MACE timing convention. post_anneal weights (0.75/0.25,
    # energy:force = 3:1) mirror the ~3:1 energy:force ratio of MACE's
    # reported final-phase weights (300:100), kept in this project's existing
    # normalized-to-1 convention rather than switching to MACE's raw
    # (unnormalized) weight scale.
    anneal_epoch=30,
    post_anneal_energy_weight=0.75,
    post_anneal_force_weight=0.25,
)

# --extended-anneal override (2026-08-24): the 50-epoch run above gives the
# post-anneal (energy-focused) phase only 20 epochs to recover from the
# disruptive reweighting at anneal_epoch=30 before ModelCheckpoint's plain
# running min() locks in whatever's best-so-far. Checked across the 6-seed
# waterbox_study_zbl_bonded run: the selected epoch ranged from 17 to 47,
# with several seeds (e.g. water_absolute+momentum/seed1 at epoch 29 - ONE
# epoch before the anneal switch even fired) never getting any post-anneal
# fine-tuning at all before their pre-anneal score won by default. Comparing
# absolute vs momentum is not meaningful when one condition's "best"
# checkpoint got 16 epochs of energy-focused training and the other got zero.
# EXTENDED_NUM_EPOCHS=70 keeps anneal_epoch=30 unchanged (still the
# literature-matched 60%-of-run timing for the ORIGINAL 50-epoch schedule,
# deliberately not re-derived as 60% of 70 here - only one variable, total
# epoch budget, should change at a time) but gives 40 epochs of post-anneal
# training instead of 20. EXTENDED_ELIGIBLE_EPOCH_START=50 (anneal_epoch + 20,
# matching the same 20-epoch recovery window that already existed, just
# spent as a mandatory burn-in instead of an implicit deadline) makes every
# epoch before it ineligible for checkpoint selection (WaterLNNP's
# eligible_epoch_start, train_waterbox.py) - the remaining 20 epochs
# (50-69) are then the only ones ModelCheckpoint can ever pick from,
# guaranteeing the selected checkpoint always reflects genuine post-anneal
# convergence rather than checkpoint-selection luck.
EXTENDED_NUM_EPOCHS = 70
EXTENDED_ELIGIBLE_EPOCH_START = 50

CHECKPOINT_ROOT = Path("checkpoints/waterbox_study")
LOG_ROOT = Path("logs/waterbox_study")
RESULTS_ROOT = Path("results/waterbox_study")


def _roots(use_zbl_prior, zbl_bonded_exclusion=False, extended_anneal=False):
    """--use-zbl-prior redirects every path to a separate "_zbl"-suffixed
    root (checkpoints/waterbox_study_zbl/, etc.) rather than reusing the
    existing waterbox_study/ paths - the same pattern run_rollout_study.py
    already uses for --train-seed, and for the same reason: _run_training's
    checkpoint-exists skip means re-running this script normally wouldn't
    retrain anything (checkpoints already exist), and --force-retrain would
    silently overwrite the existing no-ZBL reference checkpoints/results
    every other part of this study (and the paper) already depends on.
    Keeping this purely additive means the existing results stay intact
    regardless of what the ZBL retrain finds.

    --zbl-bonded-exclusion adds a further "_bonded" suffix
    (checkpoints/waterbox_study_zbl_bonded/, etc.) - a separate root again,
    so the already-completed stock-ZBL negative-result comparison
    (paper/main.tex sec:q4-negative-result) is never overwritten either.

    --extended-anneal adds a further "_ext70" suffix, for the same reason:
    the already-completed waterbox_study_zbl_bonded comparison (6 seeds,
    50-epoch schedule) has a known checkpoint-selection confound (see
    EXTENDED_NUM_EPOCHS's comment above) and stays on disk untouched as a
    reference, rather than being silently overwritten by the fixed-schedule
    rerun."""
    if not use_zbl_prior:
        base_suffix = ""
    else:
        base_suffix = "_zbl_bonded" if zbl_bonded_exclusion else "_zbl"
    suffix = base_suffix + ("_ext70" if extended_anneal else "")
    if not suffix:
        return CHECKPOINT_ROOT, LOG_ROOT, RESULTS_ROOT
    return (
        Path(f"{CHECKPOINT_ROOT}{suffix}"),
        Path(f"{LOG_ROOT}{suffix}"),
        Path(f"{RESULTS_ROOT}{suffix}"),
    )

METRIC_COLUMNS = [
    "energy_mae",
    "force_mae",
    "mean_per_molecule_momentum_violation",
    "max_per_molecule_momentum_violation",
    "final_epoch",
    "total_wall_seconds",
]


def _run_training(condition_name, condition_kwargs, seed, num_epochs, checkpoint_root, log_root,
                   extra_hparams=None, force_retrain=False):
    save_dir = checkpoint_root / condition_name / f"seed{seed}"
    log_dir = log_root / condition_name / f"seed{seed}"
    ckpt_path = save_dir / "best_model.ckpt"
    if ckpt_path.exists() and not force_retrain:
        print(f"  Checkpoint already exists at {ckpt_path} - skipping training "
              "(pass --force-retrain to redo it anyway).")
        return save_dir

    hparams = dict(FIXED_HPARAMS)
    hparams["num_epochs"] = num_epochs
    hparams.update(CONDITION_HPARAM_OVERRIDES.get(condition_name, {}))
    hparams.update(extra_hparams or {})
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


def run_matrix(conditions, seeds, num_epochs, checkpoint_root, log_root, raw_results_csv,
                extra_hparams=None, force_retrain=False):
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
                save_dir = _run_training(
                    condition_name, condition_kwargs, seed, num_epochs, checkpoint_root, log_root,
                    extra_hparams=extra_hparams, force_retrain=force_retrain,
                )
                metrics = _evaluate_checkpoint(save_dir, seed)
            except Exception as exc:
                print(f"FAILED: {cell}: {exc}")
                traceback.print_exc()
                failed.append(cell)
                continue
            row = {"condition": condition_name, "seed": seed, **metrics}
            rows.append(row)
            _write_raw_results_csv(rows, raw_results_csv)

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


def aggregate_results(rows, summary_csv, summary_md):
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
    ]
    max_n_seeds = max((row.get("n_seeds") or 0) for row in summary_rows) if summary_rows else 0
    lines.append(
        f"n={max_n_seeds} seeds supports a coarse signal-vs-noise read, not formal "
        "significance - treat a difference as real only if it clears roughly 1 std "
        "of both conditions."
    )
    lines.append("")
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
    parser.add_argument(
        "--use-zbl-prior",
        action="store_true",
        help="Retrain both conditions with torchmdnet's ZBL short-range repulsive prior added "
        "(paper/main.tex sec:q4 - the fix for the both-conditions rollout instability, "
        "contingent on diagnose_short_range_collapse.py's finding). Writes to a SEPARATE "
        "checkpoints/waterbox_study_zbl//logs/waterbox_study_zbl//results/waterbox_study_zbl/ "
        "root - never touches the existing no-ZBL checkpoints/results. Stock ZBL (without "
        "--zbl-bonded-exclusion) is confirmed to substantially WORSEN rollout stability "
        "(paper/main.tex sec:q4-negative-result) - pass --zbl-bonded-exclusion too unless "
        "deliberately reproducing that already-completed negative-result comparison.",
    )
    parser.add_argument("--zbl-cutoff-distance", type=float, default=None,
                         help="Override train_waterbox.ZBL_CUTOFF_DISTANCE's default. Only used with --use-zbl-prior.")
    parser.add_argument("--zbl-max-num-neighbors", type=int, default=None,
                         help="Override train_waterbox.ZBL_MAX_NUM_NEIGHBORS's default. Only used with --use-zbl-prior.")
    parser.add_argument(
        "--zbl-bonded-exclusion",
        action="store_true",
        help="Use molecular_zbl.MolecularZBL instead of stock ZBL - excludes same-molecule "
        "atom pairs from the repulsive correction (see molecular_zbl.py and "
        "paper/literature_review_candidates.md section 0). Only takes effect with "
        "--use-zbl-prior. Writes to a further-separate checkpoints/waterbox_study_zbl_bonded/ "
        "root, so it never overwrites the stock-ZBL negative-result checkpoints either.",
    )
    parser.add_argument(
        "--extended-anneal",
        action="store_true",
        help="Retrain with EXTENDED_NUM_EPOCHS/EXTENDED_ELIGIBLE_EPOCH_START instead of "
        "FIXED_HPARAMS's num_epochs (fixes the checkpoint-selection confound described in "
        "EXTENDED_NUM_EPOCHS's comment above - some seeds' 'best' checkpoint was landing "
        "pre-anneal or with too little post-anneal recovery time to be a fair comparison). "
        "Writes to a further-separate checkpoints/waterbox_study..._ext70/ root, so the "
        "existing 50-epoch comparison is never overwritten.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list to run, overriding SEEDS (e.g. '0,1' for a quick "
        "pilot before committing the full 6-seed sweep). Ignored with --smoke-test.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        conditions = {"water_absolute": CONDITIONS["water_absolute"]}
        seeds = [0]
        num_epochs = 2
    else:
        conditions = CONDITIONS
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS
        num_epochs = EXTENDED_NUM_EPOCHS if args.extended_anneal else FIXED_HPARAMS["num_epochs"]

    checkpoint_root, log_root, results_root = _roots(
        args.use_zbl_prior, args.zbl_bonded_exclusion, args.extended_anneal,
    )
    raw_results_csv = results_root / "raw_results.csv"
    summary_csv = results_root / "summary_table.csv"
    summary_md = results_root / "summary_table.md"

    extra_hparams = {}
    if args.use_zbl_prior:
        extra_hparams["use_zbl_prior"] = True
        if args.zbl_cutoff_distance is not None:
            extra_hparams["zbl_cutoff_distance"] = args.zbl_cutoff_distance
        if args.zbl_max_num_neighbors is not None:
            extra_hparams["zbl_max_num_neighbors"] = args.zbl_max_num_neighbors
        if args.zbl_bonded_exclusion:
            extra_hparams["zbl_bonded_exclusion"] = True
    if args.extended_anneal:
        extra_hparams["eligible_epoch_start"] = EXTENDED_ELIGIBLE_EPOCH_START

    rows, failed_cells = run_matrix(
        conditions, seeds, num_epochs, checkpoint_root, log_root, raw_results_csv,
        extra_hparams=extra_hparams, force_retrain=args.force_retrain,
    )
    aggregate_results(rows, summary_csv, summary_md)
    if failed_cells:
        print(f"\n{len(failed_cells)} cell(s) failed and are NOT in the results above: {failed_cells}")
        print("Fix the underlying issue and re-run - completed cells will be skipped, not redone.")


if __name__ == "__main__":
    main()
