#!/usr/bin/env python
"""Batch multiple NVE rollouts per condition, varying one replicate axis at a
time while holding everything else fixed, to get a real distribution for the
water_absolute vs water_absolute+momentum dynamical-stability comparison
instead of trusting a single matched trial. Two axes, run independently
(--vary velocity | config) rather than crossed, to keep cost comparable
across batches:

- --vary velocity (the first batch already run): 5 different initial
  Maxwell-Boltzmann velocity draws, same starting configuration
  (TEST_CONFIG_INDEX fixed) for every replicate.
- --vary config: 5 different starting configurations (CONFIG_INDICES),
  same velocity draw (FIXED_VELOCITY_SEED) for every replicate - checks
  whether the velocity-axis result generalizes beyond the one geometry it
  was run on, rather than being specific to that particular configuration.

Why this matters: the first two matched-velocity rollouts (n=1 each) already
flipped which condition showed less energy drift the moment a velocity-draw
confound was controlled for - the same small-n fragility already seen in the
static-metric comparison (run_waterbox_study.py's summary_table.md, n=3 ->
n=6 seeds). The velocity-axis batch (n=5 each) then showed a real, mostly
non-overlapping separation - water_absolute+momentum roughly 2.3x the drift
and ~368K hotter at plateau, holding up seed-by-seed rather than being driven
by one outlier - but all 10 of those trials shared one starting geometry.
This module's --vary config batch is the direct check of whether that
separation is a property of the models, or of that one configuration.

The one finding that's already survived both matched-trial batches and
doesn't need this one to be trustworthy: every trial so far, regardless of
condition or velocity draw, heats a real ~287-335 K DFT snapshot to roughly
1900-2600 K within well under a picosecond of plain NVE, then plateaus there
rather than diverging further - see paper/main.tex Section 5.2/sec:q2 once
logged. That instability is the headline Q2 result regardless of how the
momentum-vs-absolute comparison resolves.

Modeled on run_waterbox_study.py's orchestration pattern (incremental CSV
writes, per-cell try/except fault isolation, mean+-std aggregation) but for
rollouts, not training - CHECKPOINTS below point at one fixed training seed's
checkpoint per condition (seed 0, arbitrary but held constant across every
cell in both batches), since the thing varying across cells here is the
rollout's own initial conditions, not which model was trained.

Usage:
    python src/run_rollout_study.py --smoke-test              # 1 condition, 1 replicate, 20 steps
    python src/run_rollout_study.py --vary velocity           # already run - reproduces raw_results.csv
    python src/run_rollout_study.py --vary config              # new - 5 different starting configs
"""

from __future__ import annotations

import csv
import statistics
import traceback
from pathlib import Path

from rollout_waterbox_ase import run_rollout


def checkpoints_for_seed(train_seed, checkpoint_root="checkpoints/waterbox_study"):
    """Which training seed's checkpoints to compare - default 0, matching
    every rollout run in this study so far (paper/main.tex sec:q3-progress,
    sec:q3). Seeds 1 and 5 are the ones where the STATIC per-fragment
    momentum metric already favored water_absolute+momentum
    (paper/main.tex sec:n6-update's per-seed data - seed 0, by contrast, was
    one of the three where it favored water_absolute). If the mechanism
    found for seed 0 is right, a "good" momentum seed here should show a
    smaller or reversed rollout-stability effect, not the same one.

    checkpoint_root defaults to the original no-ZBL checkpoints
    (checkpoints/waterbox_study) - pass "checkpoints/waterbox_study_zbl"
    (main's --use-zbl-prior does this) to compare the ZBL-retrained
    checkpoints instead (paper/main.tex sec:q4), matching
    run_waterbox_study.py --use-zbl-prior's own separate checkpoint root."""
    return {
        "water_absolute": f"{checkpoint_root}/water_absolute/seed{train_seed}/best_model.ckpt",
        "water_absolute+momentum":
            f"{checkpoint_root}/water_absolute+momentum/seed{train_seed}/best_model.ckpt",
    }


# Default (train_seed=0, no ZBL) - kept as a module-level constant for
# backward compatibility with anything already relying on it.
CHECKPOINTS = checkpoints_for_seed(0)

# --vary velocity: fixed starting config, 5 velocity draws (already run).
VELOCITY_SEEDS = [0, 1, 2, 3, 4]
FIXED_TEST_CONFIG_INDEX = 0

# --vary config: fixed velocity draw, 5 different starting configs. Plain
# small distinct indices, not spread out deliberately - random_split already
# shuffles the dataset before slicing off the test split (waterbox_data.py),
# so consecutive test_config_index values are already uncorrelated original
# frames, not physically nearby ones. Index 0 is skipped since the velocity
# batch already covers it.
CONFIG_INDICES = [1, 2, 3, 4, 5]
FIXED_VELOCITY_SEED = 0

# Fixed across every cell in both batches - determines the train/val/test
# split (which config any given test_config_index refers to). Must NOT vary
# within a batch (that would confound the axis actually being tested) - see
# rollout_waterbox_ase.py's run_rollout docstring on why seed and
# velocity_seed are kept as separate parameters.
DATA_SEED = 42

RESULTS_ROOT = Path("results/waterbox_rollout_study")
OUT_ROOT = RESULTS_ROOT / "runs"

METRIC_COLUMNS = [
    "drift_ev_per_atom_mev",
    "drift_fraction_pct",
    "plateau_temperature_mean",
    "plateau_temperature_std",
]


def _run_one(condition_name, ckpt, label, velocity_seed, test_config_index, steps, dt, temperature_k, out_root):
    out_dir = out_root / condition_name / label
    result = run_rollout(
        ckpt=ckpt,
        steps=steps,
        dt=dt,
        temperature_k=temperature_k,
        seed=DATA_SEED,
        velocity_seed=velocity_seed,
        test_config_index=test_config_index,
        out=str(out_dir),
    )
    return {
        "condition": condition_name,
        "velocity_seed": velocity_seed,
        "test_config_index": test_config_index,
        "drift_ev_per_atom_mev": result["drift_ev_per_atom"] * 1000,
        "drift_fraction_pct": result["drift_fraction"] * 100,
        "plateau_temperature_mean": result["plateau_temperature_mean"],
        "plateau_temperature_std": result["plateau_temperature_std"],
    }


def run_matrix(conditions, replicates, steps, dt, temperature_k, raw_csv, out_root):
    """replicates: list of (label, velocity_seed, test_config_index) tuples -
    label is just used for the per-cell output directory name and log lines.

    Resilient to a single cell failing (full traceback printed, cell recorded
    as failed, matrix continues) and rewrites raw_csv after every completed
    cell - same lessons as run_waterbox_study.py's run_matrix.

    Returns (rows, failed_cells).
    """
    rows = []
    failed = []
    for condition_name, ckpt in conditions.items():
        for label, velocity_seed, test_config_index in replicates:
            cell = f"{condition_name}/{label}"
            print(f"\n=== {cell} ===")
            try:
                row = _run_one(
                    condition_name, ckpt, label, velocity_seed, test_config_index, steps, dt, temperature_k,
                    out_root,
                )
            except Exception as exc:
                print(f"FAILED: {cell}: {exc}")
                traceback.print_exc()
                failed.append(cell)
                continue
            rows.append(row)
            _write_raw_csv(rows, raw_csv)

    if failed:
        print("\nFailed cells (full tracebacks above) - fix and re-run; already-completed cells are safe:")
        for cell in failed:
            print(f"  {cell}")

    return rows, failed


def _write_raw_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "velocity_seed", "test_config_index"] + METRIC_COLUMNS
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Wrote: {path}")


def aggregate_results(rows, summary_csv, summary_md, note):
    """Group by condition and compute mean +/- std across replicates. Plain
    stdlib (statistics module), matching run_waterbox_study.py's choice
    (pandas isn't a confirmed dependency anywhere in this project)."""
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)

    summary_rows = []
    for condition, group_rows in sorted(groups.items()):
        summary = {"condition": condition, "n_replicates": len(group_rows)}
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
    fieldnames = ["condition", "n_replicates"] + [f"{m}_{s}" for m in METRIC_COLUMNS for s in ("mean", "std")]
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote: {summary_csv}")

    _write_summary_markdown(summary_rows, Path(summary_md), note)
    return summary_rows


def _write_summary_markdown(summary_rows, path, note):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_replicates = max((row.get("n_replicates") or 0) for row in summary_rows) if summary_rows else 0
    lines = [
        "# Water-box rollout stability study summary (mean +/- std across replicates)",
        "",
        f"n={n_replicates} replicates per condition. {note}",
        "",
    ]
    header = "| condition | n_replicates | " + " | ".join(METRIC_COLUMNS) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in METRIC_COLUMNS) + " |"
    lines += [header, sep]
    for row in summary_rows:
        cells = []
        for metric in METRIC_COLUMNS:
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            cells.append("n/a" if mean is None else f"{mean:.4g} +/- {std:.2g}")
        lines.append(f"| {row['condition']} | {row['n_replicates']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vary",
        choices=["velocity", "config"],
        default="velocity",
        help="Which replicate axis to sweep - see module docstring.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=0,
        help="Which training seed's checkpoints to compare (default 0, matching every existing "
        "rollout in this study). Seeds 1 and 5 are the ones where the STATIC per-fragment "
        "momentum metric already favored water_absolute+momentum (paper/main.tex "
        "sec:n6-update) - the natural check for whether seed 0's mechanism (sec:q3) "
        "generalizes. Any --train-seed other than 0 writes to a separate "
        "results/waterbox_rollout_study_seed<N>/ directory, so it never overwrites the "
        "existing seed-0 results.",
    )
    parser.add_argument(
        "--use-zbl-prior",
        action="store_true",
        help="Compare the ZBL-retrained checkpoints (checkpoints/waterbox_study_zbl/, from "
        "run_waterbox_study.py --use-zbl-prior) instead of the original no-ZBL checkpoints "
        "(paper/main.tex sec:q4). Writes to a separate results/waterbox_rollout_study_zbl.../ "
        "root, so it never overwrites the existing no-ZBL rollout results. Combine with "
        "--zbl-bonded-exclusion to compare the bonded-exclusion variant instead of stock ZBL "
        "(paper/main.tex sec:q4-negative-result already found stock ZBL substantially worsens "
        "stability - not the comparison you want unless deliberately reproducing that).",
    )
    parser.add_argument(
        "--zbl-bonded-exclusion",
        action="store_true",
        help="Compare the bonded-exclusion ZBL checkpoints (checkpoints/waterbox_study_zbl_bonded/, "
        "from run_waterbox_study.py --use-zbl-prior --zbl-bonded-exclusion) instead of stock ZBL. "
        "Only takes effect with --use-zbl-prior. Writes to a further-separate "
        "results/waterbox_rollout_study_zbl_bonded.../ root.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 condition (water_absolute), 1 replicate, 20 steps - confirm the plumbing "
        "end-to-end before running the full comparison.",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    args = parser.parse_args()

    if not args.use_zbl_prior:
        checkpoint_root = "checkpoints/waterbox_study"
    elif args.zbl_bonded_exclusion:
        checkpoint_root = "checkpoints/waterbox_study_zbl_bonded"
    else:
        checkpoint_root = "checkpoints/waterbox_study_zbl"
    checkpoints = checkpoints_for_seed(args.train_seed, checkpoint_root=checkpoint_root)

    results_dir_name = "results/waterbox_rollout_study"
    if args.use_zbl_prior:
        results_dir_name += "_zbl_bonded" if args.zbl_bonded_exclusion else "_zbl"
    if args.train_seed != 0:
        results_dir_name += f"_seed{args.train_seed}"
    results_root = RESULTS_ROOT if results_dir_name == "results/waterbox_rollout_study" else Path(results_dir_name)
    out_root = results_root / "runs"

    if args.vary == "velocity":
        replicates = [(f"vseed{v}", v, FIXED_TEST_CONFIG_INDEX) for v in VELOCITY_SEEDS]
        # Unsuffixed filenames, matching the first batch already run and
        # analyzed - do not rename these, a prior batch's results already
        # live at this exact path (only true when results_root is the
        # default seed-0 path; a different --train-seed already lands in
        # its own directory, so no collision either way).
        raw_csv = results_root / "raw_results.csv"
        summary_csv = results_root / "summary_table.csv"
        summary_md = results_root / "summary_table.md"
        note = (
            "Identical starting geometry (DATA_SEED/test_config_index held fixed) - only the "
            "initial Maxwell-Boltzmann velocity draw differs between replicates. "
            f"train_seed={args.train_seed}, use_zbl_prior={args.use_zbl_prior}, zbl_bonded_exclusion={args.zbl_bonded_exclusion}."
        )
    else:
        replicates = [(f"cfg{c}", FIXED_VELOCITY_SEED, c) for c in CONFIG_INDICES]
        raw_csv = results_root / "raw_results_by_config.csv"
        summary_csv = results_root / "summary_table_by_config.csv"
        summary_md = results_root / "summary_table_by_config.md"
        note = (
            "Identical velocity draw (DATA_SEED/velocity_seed held fixed) - only the starting "
            "configuration (test_config_index) differs between replicates. Compare against "
            "summary_table.md's velocity-axis batch to see whether that batch's momentum-vs-"
            f"absolute separation is a property of the models or of the one configuration it was "
            f"run on. train_seed={args.train_seed}, use_zbl_prior={args.use_zbl_prior}, zbl_bonded_exclusion={args.zbl_bonded_exclusion}."
        )

    if args.smoke_test:
        conditions = {"water_absolute": checkpoints["water_absolute"]}
        replicates = replicates[:1]
        steps = 20
    else:
        conditions = checkpoints
        steps = args.steps

    rows, failed_cells = run_matrix(conditions, replicates, steps, args.dt, args.temperature_k, raw_csv, out_root)
    aggregate_results(rows, summary_csv, summary_md, note)
    if failed_cells:
        print(f"\n{len(failed_cells)} cell(s) failed and are NOT in the results above: {failed_cells}")
        print("Fix the underlying issue and re-run - completed cells are independent reruns, not resumed.")


if __name__ == "__main__":
    main()
