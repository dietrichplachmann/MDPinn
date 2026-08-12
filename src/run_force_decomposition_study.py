#!/usr/bin/env python
"""Checks whether the pattern found in the first trajectory-frame
cross-evaluation (src/analyze_force_decomposition.py --mode trajectory) -
water_absolute+momentum's own rollout visiting geometries with a
consistently HIGHER net per-molecule force than water_absolute's, at matched
elapsed time, even though the two models predict nearly identical forces on
any given geometry (candidates 1 and 3 in paper/main.tex Section 5.3,
sec:q3, both showed no point-wise signal) - replicates across the velocity-
and config-axis rollout pairs already collected by run_rollout_study.py, or
was specific to that one pair.

Reuses the trajectories already on disk (results/waterbox_rollout_study/
runs/<condition>/<label>/rollout.xyz) - no new rollouts needed, this is pure
post-hoc analysis of existing data. For each replicate label, cross-evaluates
both models on both conditions' saved trajectory at FRAME_INDICES (same
indices as the first single-pair check, for direct comparability), then
aggregates the momentum-minus-absolute net-force gap at each frame across
all replicates.

Modeled on run_waterbox_study.py/run_rollout_study.py's orchestration
pattern (incremental CSV writes, per-cell try/except fault isolation, mean
+-std aggregation).

Usage:
    python src/run_force_decomposition_study.py --smoke-test   # 1 replicate, 1 frame
    python src/run_force_decomposition_study.py                # both axes, all replicates
"""

from __future__ import annotations

import csv
import statistics
import traceback
from pathlib import Path

from analyze_force_decomposition import analyze_trajectory_frames, load_models

CHECKPOINTS = {
    "water_absolute": "checkpoints/waterbox_study/water_absolute/seed0/best_model.ckpt",
    "water_absolute+momentum": "checkpoints/waterbox_study/water_absolute+momentum/seed0/best_model.ckpt",
}

RUNS_ROOT = Path("results/waterbox_rollout_study/runs")
# Same indices as the first single-pair check (results/waterbox_rollout,
# results/waterbox_rollout_momentum) - steps 0/200/800/1800 at
# energy_log_stride=10 - for direct comparability with that result.
FRAME_INDICES = [0, 20, 80, 180]

VELOCITY_LABELS = [f"vseed{v}" for v in range(5)]
CONFIG_LABELS = [f"cfg{c}" for c in range(1, 6)]

RESULTS_ROOT = Path("results/force_decomposition_study")
RAW_CSV = RESULTS_ROOT / "raw_results.csv"
SUMMARY_CSV = RESULTS_ROOT / "summary_by_frame.csv"
SUMMARY_MD = RESULTS_ROOT / "summary_by_frame.md"


def _trajectory_paths_for(label):
    return {
        "water_absolute": str(RUNS_ROOT / "water_absolute" / label / "rollout.xyz"),
        "water_absolute+momentum": str(RUNS_ROOT / "water_absolute+momentum" / label / "rollout.xyz"),
    }


def _run_one(axis_name, label, models, frame_indices):
    """Cross-evaluate both models on one replicate's trajectory pair, then
    collapse the per-eval_model rows into a single momentum-vs-absolute
    net-force gap per frame - averaging over which model did the evaluating,
    since the two models were already shown to agree closely point-wise
    (analyze_force_decomposition.py's first result), so averaging gives a
    more robust per-geometry estimate than picking one arbitrarily.
    """
    traj_paths = _trajectory_paths_for(label)
    cell_rows = analyze_trajectory_frames(
        trajectory_paths=traj_paths,
        frame_indices=frame_indices,
        models=models,
        verbose=False,
    )

    net_by_traj_frame = {}
    for row in cell_rows:
        key = (row["trajectory"], row["frame"])
        net_by_traj_frame.setdefault(key, []).append(row["net_force_mag_mean"])

    rows = []
    for frame in frame_indices:
        abs_net = statistics.mean(net_by_traj_frame[("water_absolute", frame)])
        mom_net = statistics.mean(net_by_traj_frame[("water_absolute+momentum", frame)])
        rows.append({
            "axis": axis_name,
            "label": label,
            "frame": frame,
            "absolute_traj_net_force": abs_net,
            "momentum_traj_net_force": mom_net,
            "gap_momentum_minus_absolute": mom_net - abs_net,
        })
    return rows


def run_matrix(replicate_specs, models, frame_indices):
    """replicate_specs: list of (axis_name, label) tuples. Resilient to a
    single replicate's trajectory being missing/unreadable (full traceback
    printed, replicate recorded as failed, matrix continues) and rewrites
    raw_results.csv after every completed replicate - same lessons as
    run_waterbox_study.py's run_matrix.

    Returns (rows, failed_replicates).
    """
    rows = []
    failed = []
    for axis_name, label in replicate_specs:
        cell = f"{axis_name}/{label}"
        print(f"\n=== {cell} ===")
        try:
            cell_rows = _run_one(axis_name, label, models, frame_indices)
        except Exception as exc:
            print(f"FAILED: {cell}: {exc}")
            traceback.print_exc()
            failed.append(cell)
            continue
        rows.extend(cell_rows)
        _write_raw_csv(rows, RAW_CSV)
        for row in cell_rows:
            print(
                f"  frame={row['frame']:<4d} absolute_net={row['absolute_traj_net_force']:.4f}  "
                f"momentum_net={row['momentum_traj_net_force']:.4f}  "
                f"gap={row['gap_momentum_minus_absolute']:+.4f}"
            )

    if failed:
        print("\nFailed replicates (full tracebacks above):")
        for cell in failed:
            print(f"  {cell}")

    return rows, failed


def _write_raw_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["axis", "label", "frame", "absolute_traj_net_force",
                  "momentum_traj_net_force", "gap_momentum_minus_absolute"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Wrote: {path}")


def aggregate_by_frame(rows, summary_csv=SUMMARY_CSV, summary_md=SUMMARY_MD):
    """Group by frame index and compute mean +/- std of the momentum-minus-
    absolute net-force gap across every replicate (both axes pooled) - the
    direct test of whether the single-pair finding replicates: if the gap is
    consistently positive (mean clearing its own std, or at least not
    straddling zero) at every frame, that's a real, repeated pattern, not an
    artifact of the one trajectory pair it was first noticed in.
    """
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame"], []).append(row["gap_momentum_minus_absolute"])

    summary_rows = []
    for frame in sorted(by_frame):
        gaps = by_frame[frame]
        n_positive = sum(1 for g in gaps if g > 0)
        summary_rows.append({
            "frame": frame,
            "n_replicates": len(gaps),
            "gap_mean": statistics.mean(gaps),
            "gap_std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            "n_positive": n_positive,
        })

    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "n_replicates", "gap_mean", "gap_std", "n_positive"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Wrote: {summary_csv}")

    _write_summary_markdown(summary_rows, Path(summary_md))
    return summary_rows


def _write_summary_markdown(summary_rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Force-decomposition replication study: momentum-minus-absolute net-force gap by frame",
        "",
        "gap = mean per-molecule net force on water_absolute+momentum's own trajectory geometry "
        "minus the same on water_absolute's own trajectory geometry, at matched elapsed time "
        "(averaged over which model did the evaluating - the two models agree closely "
        "point-wise, see analyze_force_decomposition.py's equilibrium/trajectory results). "
        "n_positive = how many replicates (out of n_replicates, pooled across the velocity and "
        "config axes) showed momentum's geometry with the higher net force at that frame - the "
        "direct replication check for the pattern first noticed in a single trajectory pair.",
        "",
        "| frame | n_replicates | gap_mean | gap_std | n_positive |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['frame']} | {row['n_replicates']} | {row['gap_mean']:.4g} | "
            f"{row['gap_std']:.3g} | {row['n_positive']}/{row['n_replicates']} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 replicate (velocity axis, vseed0), 1 frame - confirm the plumbing "
        "end-to-end before running the full replication check.",
    )
    args = parser.parse_args()

    models = load_models(CHECKPOINTS["water_absolute"], CHECKPOINTS["water_absolute+momentum"])

    if args.smoke_test:
        replicate_specs = [("velocity", VELOCITY_LABELS[0])]
        frame_indices = FRAME_INDICES[:1]
    else:
        replicate_specs = [("velocity", label) for label in VELOCITY_LABELS] + \
                           [("config", label) for label in CONFIG_LABELS]
        frame_indices = FRAME_INDICES

    rows, failed = run_matrix(replicate_specs, models, frame_indices)
    aggregate_by_frame(rows)
    if failed:
        print(f"\n{len(failed)} replicate(s) failed and are NOT in the results above: {failed}")


if __name__ == "__main__":
    main()
