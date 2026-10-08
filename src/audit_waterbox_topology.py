#!/usr/bin/env python
"""Audit fixed water-molecule identities across held-out WaterBox configs.

Water-box training infers molecule membership once from dataset sample 0 and
reuses that atom grouping for every configuration.  This audit applies the
same fixed grouping to each requested training seed's test split and flags
starting structures in which an assigned O-H bond is unusually long, a
cross-molecule O-H contact is unusually short, or a hydrogen is closer to a
different oxygen than to its assigned oxygen.

The thresholds are explicit configuration-selection rules, not claims about
chemical bond orders.  The full continuous metrics are always written so the
selection can be revisited without rerunning model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from diagnose_short_range_collapse import molecule_group_ids, pairwise_min_image_distances
from waterbox_data import load_waterbox_dataset, random_split


WATER_MASS_AMU = 18.01528
AMU_PER_A3_TO_G_CM3 = 1.66053906660


def _parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _fixed_topology(dataset):
    reference = dataset[0]
    numbers = np.asarray(reference.z.detach().cpu(), dtype=int)
    positions = np.asarray(reference.pos.detach().cpu(), dtype=float)
    box = np.asarray(reference.box.detach().cpu(), dtype=float).reshape(3, 3)
    group_ids = molecule_group_ids(numbers, positions, np.diag(box).copy())
    groups = [np.flatnonzero(group_ids == group) for group in np.unique(group_ids)]
    bad = [Counter(numbers[group].tolist()) for group in groups
           if Counter(numbers[group].tolist()) != Counter({8: 1, 1: 2})]
    if bad or len(groups) != 64:
        raise ValueError(
            "dataset sample 0 did not reproduce the training topology of 64 H2O groups; "
            f"n_groups={len(groups)}, bad_compositions={bad[:5]}"
        )
    assigned_oxygen = np.full(numbers.shape[0], -1, dtype=int)
    assigned_oh_pairs = []
    for group in groups:
        oxygen = int(group[numbers[group] == 8][0])
        hydrogens = group[numbers[group] == 1]
        assigned_oxygen[hydrogens] = oxygen
        assigned_oh_pairs.extend((oxygen, int(hydrogen)) for hydrogen in hydrogens)
    return numbers, np.asarray(group_ids), assigned_oxygen, assigned_oh_pairs


def _config_metrics(sample, reference_numbers, group_ids, assigned_oxygen, assigned_oh_pairs):
    numbers = np.asarray(sample.z.detach().cpu(), dtype=int)
    if not np.array_equal(numbers, reference_numbers):
        raise ValueError("configuration atom count/order differs from dataset sample 0")
    positions = np.asarray(sample.pos.detach().cpu(), dtype=float)
    box = np.asarray(sample.box.detach().cpu(), dtype=float).reshape(3, 3)
    box_lengths = np.diag(box).copy()
    distances = pairwise_min_image_distances(positions, box_lengths)

    assigned_oh = np.asarray([distances[o, h] for o, h in assigned_oh_pairs])
    oxygen_indices = np.flatnonzero(numbers == 8)
    hydrogen_indices = np.flatnonzero(numbers == 1)
    cross_oh = distances[np.ix_(oxygen_indices, hydrogen_indices)]
    cross_mask = group_ids[oxygen_indices, None] != group_ids[hydrogen_indices][None, :]
    minimum_cross_oh = float(np.min(cross_oh[cross_mask]))

    n_h_closer_to_other_o = 0
    largest_reassignment_margin = 0.0
    for hydrogen in hydrogen_indices:
        assigned = assigned_oxygen[hydrogen]
        oxygen_distances = distances[oxygen_indices, hydrogen]
        nearest_offset = int(np.argmin(oxygen_distances))
        nearest = int(oxygen_indices[nearest_offset])
        if nearest != assigned:
            n_h_closer_to_other_o += 1
            margin = float(distances[assigned, hydrogen] - distances[nearest, hydrogen])
            largest_reassignment_margin = max(largest_reassignment_margin, margin)

    volume = float(abs(np.linalg.det(box)))
    density = 64.0 * WATER_MASS_AMU / volume * AMU_PER_A3_TO_G_CM3
    return {
        "volume_a3": volume,
        "density_g_cm3": density,
        "assigned_oh_min_a": float(np.min(assigned_oh)),
        "assigned_oh_mean_a": float(np.mean(assigned_oh)),
        "assigned_oh_max_a": float(np.max(assigned_oh)),
        "intermolecular_oh_min_a": minimum_cross_oh,
        "n_h_closer_to_other_o": n_h_closer_to_other_o,
        "largest_reassignment_margin_a": largest_reassignment_margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--seeds", type=_parse_ints, default=_parse_ints("0,1,2,3,4,5"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/waterbox_property_diagnostics_zbl_bonded_ext70/topology_audit"),
    )
    parser.add_argument(
        "--maximum-assigned-oh-a", type=float, default=1.3,
        help="Maximum fixed-topology O-H length permitted for a clean start.",
    )
    parser.add_argument(
        "--minimum-intermolecular-oh-a", type=float, default=1.3,
        help="Minimum cross-topology O-H distance permitted for a clean start.",
    )
    parser.add_argument(
        "--recommended-per-seed", type=int, default=5,
        help="Number of clean split-local indices to retain per seed in the JSON summary.",
    )
    args = parser.parse_args()

    dataset = load_waterbox_dataset(args.data_root)
    topology = _fixed_topology(dataset)
    rows = []
    recommendations = {}
    for seed in args.seeds:
        _, _, test_data = random_split(dataset, seed=seed)
        clean_for_seed = []
        for split_index, sample in enumerate(test_data):
            metrics = _config_metrics(sample, *topology)
            clean = (
                metrics["assigned_oh_max_a"] <= args.maximum_assigned_oh_a
                and metrics["intermolecular_oh_min_a"] >= args.minimum_intermolecular_oh_a
                and metrics["n_h_closer_to_other_o"] == 0
            )
            raw_index = int(test_data.indices[split_index])
            row = {
                "train_seed": seed,
                "test_config_index": split_index,
                "raw_dataset_index": raw_index,
                "topology_clean": int(clean),
                **metrics,
            }
            rows.append(row)
            if clean and len(clean_for_seed) < args.recommended_per_seed:
                clean_for_seed.append({
                    "test_config_index": split_index,
                    "raw_dataset_index": raw_index,
                    "density_g_cm3": metrics["density_g_cm3"],
                    "assigned_oh_max_a": metrics["assigned_oh_max_a"],
                    "intermolecular_oh_min_a": metrics["intermolecular_oh_min_a"],
                })
        recommendations[str(seed)] = clean_for_seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "topology_audit.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "topology_source": "dataset_sample_0_matching_training",
        "data_root": args.data_root,
        "selection_rule": {
            "maximum_assigned_oh_a": args.maximum_assigned_oh_a,
            "minimum_intermolecular_oh_a": args.minimum_intermolecular_oh_a,
            "require_n_h_closer_to_other_o": 0,
        },
        "n_test_configs_by_seed": {
            str(seed): sum(row["train_seed"] == seed for row in rows) for seed in args.seeds
        },
        "n_clean_by_seed": {
            str(seed): sum(row["train_seed"] == seed and row["topology_clean"] for row in rows)
            for seed in args.seeds
        },
        "recommended_clean_configs": recommendations,
    }
    json_path = args.output_dir / "topology_audit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
