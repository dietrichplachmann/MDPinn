#!/usr/bin/env python
"""Recompute time-windowed RDFs for the existing all-seed NVE study.

The original per-run rdf.csv averages the complete 1 ps trajectory, mixing
the initial state with the heating transient and hot tail.  This script uses
the saved rollout.xyz frames and energy-history times to calculate separate
windows, then writes replicate, seed-level, and condition-level metrics.

Example:
    python src/postprocess_rdfs.py \
      --results-prefix results/waterbox_rollout_study_zbl_bonded_ext70 \
      --expected-seeds 0,1,2,3,4,5 --require-complete
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import numpy as np


PAIR_DEFINITIONS = {
    "O-O": ((8, 8), (2.2, 3.5), (3.0, 4.5)),
    "O-H": ((8, 1), (0.7, 1.2), (1.05, 1.7)),
    "H-H": ((1, 1), (1.2, 2.0), (1.5, 2.6)),
}


def _parse_windows(value: str) -> list[tuple[float, float]]:
    windows = []
    for item in value.split(","):
        match = re.fullmatch(r"\s*([0-9.]+)\s*:\s*([0-9.]+)\s*", item)
        if not match:
            raise argparse.ArgumentTypeError(f"invalid window {item!r}; expected START:END")
        start, end = map(float, match.groups())
        if not 0 <= start < end:
            raise argparse.ArgumentTypeError(f"invalid window {item!r}")
        windows.append((start, end))
    return windows


def _seed_from_root(root: Path, base_name: str) -> int | None:
    if root.name == base_name:
        return 0
    match = re.fullmatch(re.escape(base_name) + r"_seed(\d+)", root.name)
    return int(match.group(1)) if match else None


def _discover_roots(prefix: Path) -> dict[int, Path]:
    candidates = [prefix, *prefix.parent.glob(f"{prefix.name}_seed*")]
    roots = {}
    for candidate in candidates:
        seed = _seed_from_root(candidate, prefix.name)
        if candidate.is_dir() and seed is not None:
            roots[seed] = candidate
    return dict(sorted(roots.items()))


def _read_times(path: Path) -> list[float]:
    with path.open(newline="") as handle:
        return [float(row["time_fs"]) for row in csv.DictReader(handle)]


def _read_reference(path: Path, pair_name: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["r_angstrom"]) for row in rows]),
        np.asarray([float(row[f"{pair_name}_reference"]) for row in rows]),
    )


def _range_indices(r: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.flatnonzero((r >= bounds[0]) & (r <= bounds[1]))


def _curve_metrics(
    r: np.ndarray, rdf: np.ndarray, reference: np.ndarray, pair_name: str,
    oxygen_number_density: float,
) -> dict[str, float]:
    _, peak_bounds, minimum_bounds = PAIR_DEFINITIONS[pair_name]
    peak_candidates = _range_indices(r, peak_bounds)
    if len(peak_candidates) == 0:
        raise ValueError(f"RDF grid does not cover peak bounds for {pair_name}")
    peak_index = int(peak_candidates[np.argmax(rdf[peak_candidates])])
    minimum_candidates = _range_indices(r, (max(r[peak_index], minimum_bounds[0]), minimum_bounds[1]))
    if len(minimum_candidates) == 0:
        raise ValueError(f"RDF grid does not cover minimum bounds for {pair_name}")
    minimum_index = int(minimum_candidates[np.argmin(rdf[minimum_candidates])])
    metrics = {
        "integrated_absolute_error": float(np.trapezoid(np.abs(rdf - reference), r)),
        "peak_r_angstrom": float(r[peak_index]),
        "peak_height": float(rdf[peak_index]),
        "first_minimum_r_angstrom": float(r[minimum_index]),
        "first_minimum_height": float(rdf[minimum_index]),
    }
    if pair_name == "O-O":
        mask = r <= r[minimum_index]
        metrics["coordination_number"] = float(
            4.0 * math.pi * oxygen_number_density * np.trapezoid(rdf[mask] * r[mask] ** 2, r[mask])
        )
    else:
        metrics["coordination_number"] = float("nan")
    return metrics


def _mean_std_rows(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for group, members in sorted(groups.items()):
        result = dict(zip(group_keys, group))
        result["n"] = len(members)
        for metric in metric_keys:
            values = [float(row[metric]) for row in members if math.isfinite(float(row[metric]))]
            result[f"{metric}_mean"] = statistics.mean(values) if values else float("nan")
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows available for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-prefix", type=Path, required=True)
    parser.add_argument("--windows-fs", type=_parse_windows,
                        default=_parse_windows("0:100,100:300,700:1000"))
    parser.add_argument("--expected-seeds", default="0,1,2,3,4,5")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        from ase.io import read as ase_read
        from rollout_waterbox_ase import _averaged_rdf
    except ImportError as exc:
        raise SystemExit("ASE is required; run this script in the remote training environment.") from exc

    roots = _discover_roots(args.results_prefix)
    expected = {int(item) for item in args.expected_seeds.split(",") if item.strip()}
    if args.require_complete and set(roots) != expected:
        raise ValueError(f"found seeds {sorted(roots)}, expected {sorted(expected)}")

    rows = []
    for train_seed, root in roots.items():
        for trajectory_path in sorted((root / "runs").glob("*/*/rollout.xyz")):
            condition = trajectory_path.parent.parent.name
            replicate = trajectory_path.parent.name
            axis = "velocity" if replicate.startswith("vseed") else "config"
            history_path = trajectory_path.with_name("energy_history.csv")
            rdf_path = trajectory_path.with_name("rdf.csv")
            if not history_path.exists() or not rdf_path.exists():
                if args.require_complete:
                    raise FileNotFoundError(f"missing history or RDF beside {trajectory_path}")
                continue
            frames = ase_read(str(trajectory_path), index=":")
            times = _read_times(history_path)
            if len(frames) != len(times):
                raise ValueError(f"{trajectory_path}: {len(frames)} frames but {len(times)} times")
            n_oxygen = sum(int(z) == 8 for z in frames[0].get_atomic_numbers())
            oxygen_density = n_oxygen / frames[0].get_volume()

            for window_start, window_end in args.windows_fs:
                selected = [frame for frame, time in zip(frames, times) if window_start <= time <= window_end]
                if len(selected) < 2:
                    if args.require_complete:
                        raise ValueError(f"{trajectory_path}: fewer than two frames in {window_start}:{window_end} fs")
                    continue
                for pair_name, (elements, _, _) in PAIR_DEFINITIONS.items():
                    reference_r, reference = _read_reference(rdf_path, pair_name)
                    spacing = reference_r[1] - reference_r[0]
                    rmax = float(reference_r[-1] + 0.5 * spacing)
                    rdf, r = _averaged_rdf(selected, rmax, len(reference_r), elements)
                    reference_on_grid = np.interp(r, reference_r, reference)
                    metrics = _curve_metrics(r, rdf, reference_on_grid, pair_name, oxygen_density)
                    rows.append({
                        "train_seed": train_seed,
                        "condition": condition,
                        "replicate_axis": axis,
                        "replicate": replicate,
                        "window_start_fs": window_start,
                        "window_end_fs": window_end,
                        "pair": pair_name,
                        "n_frames": len(selected),
                        **metrics,
                    })

    metric_keys = [
        "integrated_absolute_error", "peak_r_angstrom", "peak_height",
        "first_minimum_r_angstrom", "first_minimum_height", "coordination_number",
    ]
    seed_summary = _mean_std_rows(
        rows,
        ["train_seed", "condition", "replicate_axis", "window_start_fs", "window_end_fs", "pair"],
        metric_keys,
    )
    seed_points = [
        {
            **{key: row[key] for key in (
                "train_seed", "condition", "replicate_axis", "window_start_fs", "window_end_fs", "pair"
            )},
            **{metric: row[f"{metric}_mean"] for metric in metric_keys},
        }
        for row in seed_summary
    ]
    condition_summary = _mean_std_rows(
        seed_points,
        ["condition", "replicate_axis", "window_start_fs", "window_end_fs", "pair"],
        metric_keys,
    )
    output_dir = args.output_dir or args.results_prefix / "rdf_analysis"
    _write_csv(output_dir / "rdf_replicate_metrics.csv", rows)
    _write_csv(output_dir / "rdf_seed_summary.csv", seed_summary)
    _write_csv(output_dir / "rdf_condition_summary.csv", condition_summary)


if __name__ == "__main__":
    main()
