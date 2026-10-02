#!/usr/bin/env python
"""Summarize NVT/NPT thermodynamic histories with seed-aware aggregation.

Replicates are averaged within each trained seed first.  Condition summaries
then use the trained-seed means, preventing velocity replicates from being
misreported as independent model replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ("temperature_k", "density_g_cm3", "pressure_bar", "volume_a3")
DEFAULT_CONDITIONS = ("water_absolute", "water_absolute+momentum")


def _finite(values):
    return [value for value in values if math.isfinite(value)]


def _block_sem(values: list[float], n_blocks: int = 10) -> float:
    values = _finite(values)
    if len(values) < 2:
        return float("nan")
    n_blocks = min(n_blocks, len(values))
    block_size = len(values) // n_blocks
    if block_size == 0:
        return float("nan")
    block_means = [
        statistics.mean(values[index * block_size:(index + 1) * block_size])
        for index in range(n_blocks)
    ]
    return statistics.stdev(block_means) / math.sqrt(n_blocks) if n_blocks > 1 else float("nan")


def _read_production_history(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["phase"] != "production":
                continue
            rows.append({key: float(raw[key]) for key in ("time_fs", *METRICS)})
    return rows


def _replicate_rows(root: Path, ensembles: set[str]) -> list[dict]:
    output = []
    for manifest_path in sorted(root.glob("*/*/seed*/vseed*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        relative = manifest_path.relative_to(root).parts
        ensemble, condition, seed_label, velocity_label = relative[:4]
        if ensemble not in ensembles:
            continue
        history_path = manifest_path.parent / "thermodynamic_history.csv"
        row = {
            "ensemble": ensemble,
            "condition": condition,
            "train_seed": int(seed_label.removeprefix("seed")),
            "velocity_seed": int(velocity_label.removeprefix("vseed")),
            "status": manifest.get("status"),
            "abort_reason": manifest.get("abort_reason") or "",
            "temperature_target_k": manifest.get("temperature_k"),
            "pressure_target_bar": manifest.get("pressure_bar"),
            "dt_fs": manifest.get("dt_fs"),
            "production_ps_requested": manifest.get("production_ps_requested"),
            "steps_completed": manifest.get("steps_completed"),
            "n_production_samples": 0,
        }
        history = _read_production_history(history_path) if history_path.exists() else []
        row["n_production_samples"] = len(history)
        for metric in METRICS:
            values = _finite([sample[metric] for sample in history])
            row[f"{metric}_mean"] = statistics.mean(values) if values else float("nan")
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"{metric}_block_sem"] = _block_sem(values)
        output.append(row)
    return output


def _aggregate(rows: list[dict], group_keys: tuple[str, ...], source_suffix: str) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for group, members in sorted(groups.items()):
        result = dict(zip(group_keys, group))
        result["n"] = len(members)
        result["n_complete"] = sum(row.get("status", "complete") == "complete" for row in members)
        for metric in METRICS:
            source = f"{metric}_{source_suffix}"
            values = _finite([float(row[source]) for row in members])
            result[f"{metric}_mean"] = statistics.mean(values) if values else float("nan")
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(result)
    return output


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows found for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {path}")


def _paired_seed_differences(seed_rows: list[dict]) -> list[dict]:
    by_key = {
        (row["ensemble"], row["train_seed"], row["condition"]): row
        for row in seed_rows
    }
    output = []
    keys = sorted({(row["ensemble"], row["train_seed"]) for row in seed_rows})
    for ensemble, train_seed in keys:
        absolute = by_key.get((ensemble, train_seed, "water_absolute"))
        momentum = by_key.get((ensemble, train_seed, "water_absolute+momentum"))
        if absolute is None or momentum is None:
            continue
        row = {"ensemble": ensemble, "train_seed": train_seed}
        for metric in METRICS:
            row[f"{metric}_momentum_minus_absolute"] = (
                float(momentum[f"{metric}_mean"]) - float(absolute[f"{metric}_mean"])
            )
        output.append(row)
    return output


def _paired_summary(rows: list[dict]) -> list[dict]:
    output = []
    for ensemble in sorted({row["ensemble"] for row in rows}):
        members = [row for row in rows if row["ensemble"] == ensemble]
        result = {"ensemble": ensemble, "n_paired_seeds": len(members)}
        for metric in METRICS:
            key = f"{metric}_momentum_minus_absolute"
            values = _finite([float(row[key]) for row in members])
            result[f"{key}_mean"] = statistics.mean(values) if values else float("nan")
            result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            result[f"{key}_sem"] = (
                statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
            )
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/waterbox_property_study_zbl_bonded_ext70"))
    parser.add_argument("--ensembles", default="nvt,npt")
    parser.add_argument("--expected-seeds", default="0,1,2,3,4,5")
    parser.add_argument("--expected-velocity-seeds", default="0,1")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    ensembles = {item.strip() for item in args.ensembles.split(",") if item.strip()}
    expected_seeds = {int(item) for item in args.expected_seeds.split(",") if item.strip()}
    expected_velocity_seeds = {
        int(item) for item in args.expected_velocity_seeds.split(",") if item.strip()
    }
    replicate_rows = _replicate_rows(args.results_root, ensembles)
    if args.require_complete:
        actual = {
            (row["ensemble"], row["condition"], row["train_seed"], row["velocity_seed"])
            for row in replicate_rows
        }
        expected = {
            (ensemble, condition, train_seed, velocity_seed)
            for ensemble in ensembles
            for condition in DEFAULT_CONDITIONS
            for train_seed in expected_seeds
            for velocity_seed in expected_velocity_seeds
        }
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"property-run matrix mismatch; missing={missing}, extra={extra}")
        incomplete = [row for row in replicate_rows if row["status"] != "complete"]
        if incomplete:
            raise ValueError(f"{len(incomplete)} property runs are not complete")
    seed_rows = _aggregate(
        replicate_rows, ("ensemble", "condition", "train_seed"), "mean"
    )
    condition_rows = _aggregate(
        [{**row, "status": "complete" if row["n_complete"] == row["n"] else "incomplete"}
         for row in seed_rows],
        ("ensemble", "condition"), "mean",
    )
    paired_rows = _paired_seed_differences(seed_rows)
    paired_summary = _paired_summary(paired_rows)
    output_dir = args.results_root / "summary"
    _write(output_dir / "thermodynamic_replicate_metrics.csv", replicate_rows)
    _write(output_dir / "thermodynamic_seed_summary.csv", seed_rows)
    _write(output_dir / "thermodynamic_condition_summary.csv", condition_rows)
    if paired_rows:
        _write(output_dir / "thermodynamic_paired_seed_differences.csv", paired_rows)
        _write(output_dir / "thermodynamic_paired_summary.csv", paired_summary)


if __name__ == "__main__":
    main()
