#!/usr/bin/env python
"""Batch multiple NVE rollouts per condition, varying ONLY the initial
velocity draw (velocity_seed) while holding the starting geometry fixed
(same DATA_SEED/TEST_CONFIG_INDEX for every replicate - see
rollout_waterbox_ase.py's seed/velocity_seed split), to get a real
distribution for the water_absolute vs water_absolute+momentum dynamical-
stability comparison instead of trusting a single matched trial.

Why this matters: the first two matched-velocity rollouts already flipped
which condition showed less energy drift the moment a velocity-draw confound
was controlled for (water_absolute: 45.4 meV/atom/ps; water_absolute+momentum:
34.2 meV/atom/ps, opposite of the earlier unmatched comparison) - the same
small-n fragility already seen in the static-metric comparison
(run_waterbox_study.py's summary_table.md, n=3 -> n=6 seeds). One more single
trial per condition isn't enough to trust either way; this runs several.

The one finding that HAS replicated across both matched trials so far and
doesn't need more replicates to state: both conditions heat a real ~287-335 K
DFT snapshot to roughly 1900-2600 K within well under a picosecond of plain
NVE, then plateau (rather than diverging further) - see paper/main.tex
Section 5.2/sec:q2 once this is logged. That instability is the headline
result regardless of how the momentum-vs-absolute comparison below resolves.

Modeled on run_waterbox_study.py's orchestration pattern (incremental CSV
writes, per-cell try/except fault isolation, mean+-std aggregation) but for
rollouts, not training - CHECKPOINTS below point at one fixed seed's
checkpoint per condition (seed 0, arbitrary but held constant), since the
thing varying across cells here is the velocity draw, not which model was
trained.

Usage:
    python src/run_rollout_study.py --smoke-test     # 1 condition, 1 velocity seed, 20 steps
    python src/run_rollout_study.py                  # both conditions x all velocity seeds, full rollout
"""

from __future__ import annotations

import csv
import statistics
import traceback
from pathlib import Path

from rollout_waterbox_ase import run_rollout

CHECKPOINTS = {
    "water_absolute": "checkpoints/waterbox_study/water_absolute/seed0/best_model.ckpt",
    "water_absolute+momentum": "checkpoints/waterbox_study/water_absolute+momentum/seed0/best_model.ckpt",
}
VELOCITY_SEEDS = [0, 1, 2, 3, 4]

# Fixed across every cell - determines the train/val/test split and which
# config TEST_CONFIG_INDEX refers to. Must NOT vary between replicates in
# this study (that's what velocity_seed is for) - see
# rollout_waterbox_ase.py's run_rollout docstring/comment on why the two are
# kept as separate parameters.
DATA_SEED = 42
TEST_CONFIG_INDEX = 0

RESULTS_ROOT = Path("results/waterbox_rollout_study")
OUT_ROOT = RESULTS_ROOT / "runs"
RAW_CSV = RESULTS_ROOT / "raw_results.csv"
SUMMARY_CSV = RESULTS_ROOT / "summary_table.csv"
SUMMARY_MD = RESULTS_ROOT / "summary_table.md"

METRIC_COLUMNS = [
    "drift_ev_per_atom_mev",
    "drift_fraction_pct",
    "plateau_temperature_mean",
    "plateau_temperature_std",
]


def _run_one(condition_name, ckpt, velocity_seed, steps, dt, temperature_k):
    out_dir = OUT_ROOT / condition_name / f"vseed{velocity_seed}"
    result = run_rollout(
        ckpt=ckpt,
        steps=steps,
        dt=dt,
        temperature_k=temperature_k,
        seed=DATA_SEED,
        velocity_seed=velocity_seed,
        test_config_index=TEST_CONFIG_INDEX,
        out=str(out_dir),
    )
    return {
        "condition": condition_name,
        "velocity_seed": velocity_seed,
        "drift_ev_per_atom_mev": result["drift_ev_per_atom"] * 1000,
        "drift_fraction_pct": result["drift_fraction"] * 100,
        "plateau_temperature_mean": result["plateau_temperature_mean"],
        "plateau_temperature_std": result["plateau_temperature_std"],
    }


def run_matrix(conditions, velocity_seeds, steps, dt, temperature_k):
    """Train+evaluate every (condition, velocity_seed) cell. Resilient to a
    single cell failing (full traceback printed, cell recorded as failed,
    matrix continues) and rewrites raw_results.csv after every completed
    cell - same lessons as run_waterbox_study.py's run_matrix.

    Returns (rows, failed_cells).
    """
    rows = []
    failed = []
    for condition_name, ckpt in conditions.items():
        for velocity_seed in velocity_seeds:
            cell = f"{condition_name}/vseed{velocity_seed}"
            print(f"\n=== {cell} ===")
            try:
                row = _run_one(condition_name, ckpt, velocity_seed, steps, dt, temperature_k)
            except Exception as exc:
                print(f"FAILED: {cell}: {exc}")
                traceback.print_exc()
                failed.append(cell)
                continue
            rows.append(row)
            _write_raw_csv(rows, RAW_CSV)

    if failed:
        print("\nFailed cells (full tracebacks above) - fix and re-run; already-completed cells are safe:")
        for cell in failed:
            print(f"  {cell}")

    return rows, failed


def _write_raw_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "velocity_seed"] + METRIC_COLUMNS
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Wrote: {path}")


def aggregate_results(rows, summary_csv=SUMMARY_CSV, summary_md=SUMMARY_MD):
    """Group by condition and compute mean +/- std across velocity-seed
    replicates. Plain stdlib (statistics module), matching
    run_waterbox_study.py's choice (pandas isn't a confirmed dependency
    anywhere in this project)."""
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

    _write_summary_markdown(summary_rows, Path(summary_md))
    return summary_rows


def _write_summary_markdown(summary_rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_replicates = max((row.get("n_replicates") or 0) for row in summary_rows) if summary_rows else 0
    lines = [
        "# Water-box rollout stability study summary (mean +/- std across velocity-draw replicates)",
        "",
        f"n={n_replicates} velocity-seed replicates per condition, identical starting geometry "
        "(DATA_SEED/TEST_CONFIG_INDEX held fixed - only the initial Maxwell-Boltzmann velocity "
        "draw differs between replicates). A single matched trial already flipped which condition "
        "showed less drift once a velocity-draw confound was controlled for - read a difference "
        "here as a coarse signal, not a formal significance claim, the same way the static-metric "
        "study treats n=3-6 training seeds.",
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
        "--smoke-test",
        action="store_true",
        help="1 condition (water_absolute), 1 velocity seed, 20 steps - confirm the plumbing "
        "end-to-end before running the full comparison.",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    args = parser.parse_args()

    if args.smoke_test:
        conditions = {"water_absolute": CHECKPOINTS["water_absolute"]}
        velocity_seeds = [0]
        steps = 20
    else:
        conditions = CHECKPOINTS
        velocity_seeds = VELOCITY_SEEDS
        steps = args.steps

    rows, failed_cells = run_matrix(conditions, velocity_seeds, steps, args.dt, args.temperature_k)
    aggregate_results(rows)
    if failed_cells:
        print(f"\n{len(failed_cells)} cell(s) failed and are NOT in the results above: {failed_cells}")
        print("Fix the underlying issue and re-run - completed cells are independent reruns, not resumed.")


if __name__ == "__main__":
    main()
