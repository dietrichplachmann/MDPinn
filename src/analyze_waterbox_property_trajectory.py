#!/usr/bin/env python
"""Measure molecular distortion and close contacts in an NVT/NPT run.

The molecule assignment is inferred once from the first saved equilibration
frame and then held fixed.  This is important: re-inferring connectivity after
a collapse could silently redefine an unphysical close contact as a bond.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from diagnose_short_range_collapse import molecule_group_ids


PAIR_TYPES = (("O-O", 8, 8), ("O-H", 8, 1), ("H-H", 1, 1))


def _minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = delta @ np.linalg.inv(cell)
    return (fractional - np.rint(fractional)) @ cell


def _distance(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> float:
    return float(np.linalg.norm(_minimum_image(b - a, cell)))


def _load_frames(run_dir: Path):
    frames = []
    for name in ("equilibration.xyz", "production.xyz"):
        path = run_dir / name
        if path.exists():
            frames.extend(ase_read(str(path), index=":"))
    if not frames:
        raise FileNotFoundError(f"no equilibration.xyz or production.xyz under {run_dir}")
    return frames


def _water_groups(numbers: np.ndarray, positions: np.ndarray, cell: np.ndarray):
    group_ids = molecule_group_ids(numbers, positions, np.diag(cell).copy())
    groups = [np.flatnonzero(group_ids == group) for group in np.unique(group_ids)]
    bad = [Counter(numbers[group].tolist()) for group in groups
           if Counter(numbers[group].tolist()) != Counter({8: 1, 1: 2})]
    if bad:
        raise ValueError(
            "first-frame connectivity did not resolve exclusively into H2O molecules; "
            f"bad compositions={bad[:5]}"
        )
    return np.asarray(group_ids), groups


def _frame_metrics(atoms, group_ids: np.ndarray, groups: list[np.ndarray]) -> dict:
    numbers = np.asarray(atoms.numbers)
    positions = np.asarray(atoms.positions)
    cell = np.asarray(atoms.cell)
    oh_lengths, hh_lengths, angles = [], [], []
    for group in groups:
        oxygen = group[numbers[group] == 8][0]
        hydrogens = group[numbers[group] == 1]
        vectors = [_minimum_image(positions[h] - positions[oxygen], cell) for h in hydrogens]
        lengths = [float(np.linalg.norm(vector)) for vector in vectors]
        oh_lengths.extend(lengths)
        hh_lengths.append(_distance(positions[hydrogens[0]], positions[hydrogens[1]], cell))
        cosine = np.dot(vectors[0], vectors[1]) / (lengths[0] * lengths[1])
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))

    delta = positions[:, None, :] - positions[None, :, :]
    fractional = delta @ np.linalg.inv(cell)
    distance_matrix = np.linalg.norm((fractional - np.rint(fractional)) @ cell, axis=-1)
    same_molecule = group_ids[:, None] == group_ids[None, :]
    intermolecular_minima = {}
    for label, z1, z2 in PAIR_TYPES:
        indices_1 = np.flatnonzero(numbers == z1)
        indices_2 = np.flatnonzero(numbers == z2)
        distances = distance_matrix[np.ix_(indices_1, indices_2)]
        different = ~same_molecule[np.ix_(indices_1, indices_2)]
        intermolecular_minima[label] = float(np.min(distances[different]))

    return {
        "step": int(atoms.info.get("step", -1)),
        "time_fs": float(atoms.info.get("time_fs", "nan")),
        "phase": atoms.info.get("phase", "unknown"),
        "oh_mean_a": float(np.mean(oh_lengths)),
        "oh_std_a": float(np.std(oh_lengths)),
        "oh_min_a": min(oh_lengths),
        "oh_max_a": max(oh_lengths),
        "hoh_angle_mean_deg": float(np.mean(angles)),
        "hoh_angle_std_deg": float(np.std(angles)),
        "hoh_angle_min_deg": min(angles),
        "hoh_angle_max_deg": max(angles),
        "intramolecular_hh_min_a": min(hh_lengths),
        "n_molecules_hh_below_1a": sum(value < 1.0 for value in hh_lengths),
        "intermolecular_oo_min_a": intermolecular_minima["O-O"],
        "intermolecular_oh_min_a": intermolecular_minima["O-H"],
        "intermolecular_hh_min_a": intermolecular_minima["H-H"],
    }


def _first_event(rows: list[dict], predicate):
    return next((row for row in rows if predicate(row)), None)


def _thermodynamic_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    times = np.asarray([float(row["time_fs"]) for row in rows])
    temperatures = np.asarray([float(row["temperature_k"]) for row in rows])
    potential = np.asarray([float(row["epot_ev"]) for row in rows])
    total = np.asarray([float(row["etot_ev"]) for row in rows])
    tail = times >= times[-1] - 100.0
    return {
        "initial_temperature_k": float(temperatures[0]),
        "final_temperature_k": float(temperatures[-1]),
        "maximum_temperature_k": float(np.max(temperatures)),
        "final_100fs_temperature_mean_k": float(np.mean(temperatures[tail])),
        "final_100fs_temperature_std_k": float(np.std(temperatures[tail], ddof=1)),
        "potential_energy_change_ev": float(potential[-1] - potential[0]),
        "total_energy_change_ev": float(total[-1] - total[0]),
        "final_100fs_potential_energy_change_ev": float(
            potential[-1] - potential[np.flatnonzero(tail)[0]]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    frames = _load_frames(args.run_dir)
    first = frames[0]
    group_ids, groups = _water_groups(
        np.asarray(first.numbers), np.asarray(first.positions), np.asarray(first.cell)
    )
    rows = [_frame_metrics(frame, group_ids, groups) for frame in frames]
    output_path = args.run_dir / "structural_history.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_frames": len(rows),
        "n_water_molecules": len(groups),
        "initial": rows[0],
        "final": rows[-1],
        "minimum_hoh_angle_deg": min(row["hoh_angle_min_deg"] for row in rows),
        "minimum_intramolecular_hh_a": min(row["intramolecular_hh_min_a"] for row in rows),
        "maximum_molecules_hh_below_1a": max(row["n_molecules_hh_below_1a"] for row in rows),
        "minimum_intermolecular_oo_a": min(row["intermolecular_oo_min_a"] for row in rows),
        "minimum_intermolecular_oh_a": min(row["intermolecular_oh_min_a"] for row in rows),
        "minimum_intermolecular_hh_a": min(row["intermolecular_hh_min_a"] for row in rows),
        "first_hoh_angle_below_80_deg": _first_event(
            rows, lambda row: row["hoh_angle_min_deg"] < 80.0
        ),
        "first_intramolecular_hh_below_1_2a": _first_event(
            rows, lambda row: row["intramolecular_hh_min_a"] < 1.2
        ),
        "first_intramolecular_hh_below_1_0a": _first_event(
            rows, lambda row: row["n_molecules_hh_below_1a"] > 0
        ),
        "first_intermolecular_oo_below_2_3a": _first_event(
            rows, lambda row: row["intermolecular_oo_min_a"] < 2.3
        ),
        "thermodynamics": _thermodynamic_summary(
            args.run_dir / "thermodynamic_history.csv"
        ),
    }
    summary_path = args.run_dir / "structural_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {output_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
