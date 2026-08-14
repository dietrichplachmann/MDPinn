#!/usr/bin/env python
"""Free diagnostic for Q4 (paper/main.tex Section 5.4, sec:q4): scans rollout
trajectories already saved on disk for anomalously short non-bonded
interatomic distances, and checks whether those events occur at or before
each rollout's heating-transient onset - the direct test of the "missing
short-range repulsion" hypothesis (Fu et al. 2022, arXiv:2210.07237) before
committing to the cost of enabling torchmdnet's ZBL prior and retraining.

Every rollout in this study heats a real ~300K DFT snapshot to 1000-2600K
within under 1ps, regardless of condition/seed/starting configuration
(sec:rollout-results). The literature's proposed mechanism: a purely learned
potential has no reliable training signal at anomalously short interatomic
distances (real AIMD data rarely samples them), so once a rollout's own
dynamics wander into a too-close configuration, the network can predict an
arbitrarily wrong (even attractive) force there instead of the strong
repulsion that should exist, releasing energy and compounding the
instability that shows up downstream as heating. If that is the real
mechanism, anomalously short non-bonded distances should appear AT OR BEFORE
each trajectory's heating-transient onset - not only after the system is
already hot, which would instead suggest short contacts are a symptom of
instability from some other cause, not the trigger.

"Anomalously short" is defined empirically, not from a hardcoded literature
number (this project's own history - the Bohr/Hartree unit mixup in
waterbox_data.py - is reason enough not to trust a constant without checking
it against real data first): a sample of reference configurations from the
raw WaterBox dataset (real DFT liquid water, never overheated) establishes a
per-element-pair-type (O-O/O-H/H-H) floor - the closest a real ~300K liquid
water configuration in this dataset actually gets.

Two calibration bugs were caught (and fixed here) by actually inspecting a
first run's output rather than trusting a plausible-looking summary table:
every single one of 42 trajectories reported its first sub-floor event at
frame 0 (t=0), the real, unheated DFT starting snapshot itself, before any
dynamics happened at all - impossible if the floor genuinely meant "an
anomaly caused by the model's own dynamics." Root causes:

1. Excluding only literal covalent bonds isn't enough for H-H: a water
   molecule's own two hydrogens sit ~1.5 A apart purely from the H-O-H bond
   angle, with no bond between them - a normal geometric fact, not a close
   contact. Fixed by excluding same-MOLECULE pairs (via
   structural_metrics.infer_molecule_groups), not just literal bonds -
   correctly excludes intramolecular H-H as well as the O-H covalent bonds,
   in one consistent step.
2. The floor itself was originally a percentile over every distance pooled
   across ~200 reference configs (each with thousands of pairs) - then each
   rollout frame's own MINIMUM (over its own thousands of pairs) was checked
   against that pooled percentile. That's an apples-to-oranges comparison:
   the minimum of a few thousand draws will land below a threshold that only
   excludes 0.1% of a much larger pool almost regardless of whether anything
   is actually wrong. Fixed by computing each reference config's OWN minimum
   distance first, pooling THOSE ~200 per-config minima, and taking a
   percentile of that - the same statistic (per-configuration minimum) on
   both sides of the comparison.

Periodicity: distances use this project's own minimum-image convention
(delta - box_lengths * round(delta / box_lengths), mirroring
baseline_potential._infer_bonds_from_positions exactly, not a fresh
convention invented for this script) rather than ase.geometry.get_distances -
this keeps the core numeric logic (pairwise_min_image_distances,
molecule_group_ids, element_pair_intermolecular_distances,
analyze_trajectory's onset math) plain numpy/torch, exercisable with
synthetic arrays even without ase/torchmdnet installed (this checkout has
neither - see CLAUDE.md). Only the I/O layer (ase.io.read on rollout.xyz,
load_waterbox_dataset for the reference floor) needs the training box.

Frame/history alignment: rollout_waterbox_ase.py's run_rollout appends to
both `history` (-> energy_history.csv) and `trajectory_frames` (->
rollout.xyz) inside the same _record() call at the same energy_log_stride -
so frame index i in rollout.xyz and row i in energy_history.csv are always
the same simulation step, no separate step-alignment bookkeeping needed
(lengths are still checked defensively below in case a future rollout ever
decouples the two).

Usage (training box only - reads real dataset + real rollout.xyz files):
    python src/diagnose_short_range_collapse.py --smoke-test
    python src/diagnose_short_range_collapse.py

IMPORTANT - written without ase/torchmdnet installed locally (same caveat as
every other water-box script in this repo). The pure numpy/torch functions
below have been synthetically tested locally (see scratchpad); the ase.io.read
/ load_waterbox_dataset-dependent orchestration has not been executed yet.
"""

from __future__ import annotations

import csv
import traceback
from pathlib import Path

import numpy as np
import torch

# (name, atomic_number_a, atomic_number_b) - O=8, H=1.
ELEMENT_PAIRS = [("O-O", 8, 8), ("O-H", 8, 1), ("H-H", 1, 1)]


def pairwise_min_image_distances(positions: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """(N,3) positions, (3,) orthorhombic box side lengths -> (N,N) distance
    matrix. Mirrors baseline_potential._infer_bonds_from_positions' minimum-
    image convention (delta - box_lengths * round(delta / box_lengths))
    exactly - the same convention already verified periodicity-aware in
    verify_periodicity.py, not a fresh one invented for this script."""
    delta = positions[:, None, :] - positions[None, :, :]
    delta = delta - box_lengths * np.round(delta / box_lengths)
    return np.linalg.norm(delta, axis=-1)


def molecule_group_ids(z: np.ndarray, positions: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """Per-atom molecule id via structural_metrics.infer_molecule_groups
    (reused, not reinvented) on one geometry - intended for frame 0 of a
    rollout (the real DFT starting configuration) or an individual reference
    config, since distance-based bond inference isn't expected to stay
    meaningful once a rollout is already deep in its overheated plateau (a
    real water molecule doesn't dissociate on this timescale, so treating
    topology as fixed from a trustworthy frame is the right simplification,
    matching how per_fragment_momentum_loss elsewhere in this project
    already treats "per-molecule" as an assignment fixed at grouping time).
    Returns an (N,) int array of group ids."""
    from structural_metrics import infer_molecule_groups

    z_t = torch.as_tensor(z, dtype=torch.long)
    pos_t = torch.as_tensor(positions, dtype=torch.float32)
    box_t = torch.diag(torch.as_tensor(box_lengths, dtype=torch.float32))
    return infer_molecule_groups(z_t, pos_t, box=box_t).cpu().numpy()


def same_molecule_mask(group_ids: np.ndarray) -> np.ndarray:
    """(N,N) boolean mask, True where atoms i,j belong to the same molecule
    group - covers both the O-H covalent bonds AND each molecule's own
    (non-bonded) H...H pair in one consistent check, unlike excluding only
    literal bonds (see module docstring, calibration bug 1). Also covers
    self-pairs (i==j is always "same molecule" as itself), so no separate
    diagonal-exclusion step is needed."""
    return group_ids[:, None] == group_ids[None, :]


def element_pair_intermolecular_distances(z: np.ndarray, dist_matrix: np.ndarray, same_molecule: np.ndarray) -> dict:
    """Returns {pair_name: 1-D array of cross-molecule distances for that
    element-pair type in this one frame} - i.e. distances between atoms in
    DIFFERENT molecules, the physically relevant quantity for "did this
    configuration get anomalously close in a way missing short-range
    repulsion would matter for" (an intramolecular O-H bond or H...H isn't
    what ZBL-style priors are meant to fix). Vectorized (no Python loop over
    atoms/pairs - see this project's own "vectorize anything called once per
    [hot loop]" lesson, which applies just as much to a per-frame scan over
    every trajectory as to a training batch). Each unordered pair appears
    twice in the returned array (i,j and j,i) for same-element pairs -
    harmless for min()/count-below-threshold, so no dedup step is needed."""
    out = {}
    for name, za, zb in ELEMENT_PAIRS:
        idx_a = np.where(z == za)[0]
        idx_b = np.where(z == zb)[0]
        sub = dist_matrix[np.ix_(idx_a, idx_b)]
        mask = same_molecule[np.ix_(idx_a, idx_b)]
        out[name] = sub[~mask]
    return out


def compute_reference_floors(
    data_root: str = "./data",
    n_reference_configs: int = 200,
    seed: int = 42,
    floor_percentile: float = 1.0,
):
    """Empirical "how close does real liquid water ever get" floor per
    element-pair type, from a random sample of the raw WaterBox dataset's own
    test-split configurations (never overheated). For each reference config,
    computes only that config's OWN minimum cross-molecule distance per
    element-pair type (not every pairwise distance in the config) - the
    floor is then a percentile over these ~n_reference_configs per-config
    minima. This matters (calibration bug 2, module docstring): checking a
    rollout frame's own minimum against a percentile of every individual
    pairwise distance ever seen conflates "population percentile" with
    "extreme-value/order-statistic" - the minimum of a few thousand pairs
    almost always sits below a threshold defined to exclude only the bottom
    0.1% of a much larger pool, regardless of whether anything is actually
    anomalous. Comparing per-config minimum to per-config minimum keeps both
    sides of the comparison the same statistic. floor_percentile defaults to
    1.0 (1st percentile) rather than the old 0.1, appropriate now that the
    pool is only ~n_reference_configs values (was thousands per config
    before) - a lower percentile than that leaves too few reference samples
    below it to be a stable estimate. Needs torchmdnet - training-box only.
    """
    from waterbox_data import load_waterbox_dataset, random_split

    full_dataset = load_waterbox_dataset(data_root=data_root)
    _, _, test_data = random_split(full_dataset, seed=seed)
    n = min(n_reference_configs, len(test_data))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_data), size=n, replace=False)

    per_config_minima = {name: [] for name, _, _ in ELEMENT_PAIRS}
    for idx in indices:
        sample = test_data[int(idx)]
        z = sample.z.detach().cpu().numpy()
        pos = sample.pos.detach().cpu().numpy()
        box_lengths = np.asarray(sample.box).reshape(3, 3).diagonal()
        dist = pairwise_min_image_distances(pos, box_lengths)
        same_molecule = same_molecule_mask(molecule_group_ids(z, pos, box_lengths))
        for name, arr in element_pair_intermolecular_distances(z, dist, same_molecule).items():
            if arr.size:
                per_config_minima[name].append(float(arr.min()))

    floors, stats = {}, {}
    for name, minima in per_config_minima.items():
        minima = np.asarray(minima)
        floors[name] = float(np.percentile(minima, floor_percentile))
        stats[name] = {
            "min": float(minima.min()),
            "floor_value": floors[name],
            "n_reference_configs": int(minima.size),
        }
    return floors, stats


def _read_history(path):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "step": int(row["step"]),
                "time_fs": float(row["time_fs"]),
                "temperature_k": float(row["temperature_k"]),
            })
    return rows


def heating_onset_frame(temperatures: np.ndarray, plateau_fraction: float = 0.3, onset_fraction: float = 0.5):
    """First frame index where temperature crosses halfway (onset_fraction)
    between the starting temperature and the plateau temperature (mean over
    the tail plateau_fraction of frames - same convention run_rollout already
    uses for its own reported plateau_temperature_mean, recomputed here since
    energy_history.csv doesn't persist that summary value per-run). Returns
    (onset_idx or None, t0, t_plateau)."""
    t0 = float(temperatures[0])
    tail_n = max(1, int(len(temperatures) * plateau_fraction))
    t_plateau = float(temperatures[-tail_n:].mean())
    if t_plateau <= t0:
        return None, t0, t_plateau
    threshold = t0 + onset_fraction * (t_plateau - t0)
    crossed = np.where(temperatures >= threshold)[0]
    onset_idx = int(crossed[0]) if crossed.size else None
    return onset_idx, t0, t_plateau


def analyze_trajectory(traj_path, history_path, floors, plateau_fraction=0.3, onset_fraction=0.5):
    """Full per-trajectory scan: reads rollout.xyz + energy_history.csv,
    infers bonds once from frame 0, computes per-frame non-bonded min
    distance + sub-floor counts per element-pair type, and locates both the
    heating-transient onset and the first sub-floor-distance frame. Needs
    ase - training-box only."""
    from ase.io import read as ase_read

    frames = ase_read(str(traj_path), index=":")
    history = _read_history(history_path)
    if len(frames) != len(history):
        print(
            f"WARNING: {traj_path} has {len(frames)} frames but {history_path} has "
            f"{len(history)} rows - truncating both to the shorter length."
        )
    n = min(len(frames), len(history))
    frames, history = frames[:n], history[:n]

    z = frames[0].get_atomic_numbers()
    pos0 = frames[0].get_positions()
    box_lengths0 = np.array(frames[0].get_cell()).diagonal()
    same_molecule = same_molecule_mask(molecule_group_ids(z, pos0, box_lengths0))

    rows = []
    for frame, hist_row in zip(frames, history):
        pos = frame.get_positions()
        box_lengths = np.array(frame.get_cell()).diagonal()
        dist = pairwise_min_image_distances(pos, box_lengths)
        per_type = element_pair_intermolecular_distances(z, dist, same_molecule)
        row = dict(hist_row)
        for name, arr in per_type.items():
            row[f"min_dist_{name}"] = float(arr.min()) if arr.size else float("nan")
            row[f"n_below_floor_{name}"] = int((arr < floors[name]).sum()) if arr.size else 0
        rows.append(row)

    temperatures = np.array([r["temperature_k"] for r in rows])
    onset_idx, t0, t_plateau = heating_onset_frame(temperatures, plateau_fraction, onset_fraction)

    first_subfloor_idx = {}
    for name, _, _ in ELEMENT_PAIRS:
        col = f"n_below_floor_{name}"
        idxs = [i for i, r in enumerate(rows) if r[col] > 0]
        first_subfloor_idx[name] = idxs[0] if idxs else None
    candidates = [v for v in first_subfloor_idx.values() if v is not None]
    first_any_subfloor_idx = min(candidates) if candidates else None

    return {
        "rows": rows,
        "t0": t0,
        "t_plateau": t_plateau,
        "onset_idx": onset_idx,
        "onset_time_fs": rows[onset_idx]["time_fs"] if onset_idx is not None else None,
        "first_subfloor_idx": first_subfloor_idx,
        "first_any_subfloor_idx": first_any_subfloor_idx,
        "first_any_subfloor_time_fs": (
            rows[first_any_subfloor_idx]["time_fs"] if first_any_subfloor_idx is not None else None
        ),
    }


def verdict(result, tolerance_frames=3):
    """Compares first_any_subfloor_idx against onset_idx within +/-
    tolerance_frames (default 3, ~15 fs at the standard dt=0.5/stride=10
    logging - a small window since energy_log_stride already coarsens frame
    resolution). PRECEDES/COINCIDES both count as evidence FOR the missing-
    repulsion mechanism (a short-range collapse detectable at or before the
    heating transient is consistent with being its trigger); FOLLOWS or no
    violation at all count as evidence against it for that trajectory."""
    onset = result["onset_idx"]
    subfloor = result["first_any_subfloor_idx"]
    if onset is None:
        return "no heating transient detected (unexpected given sec:rollout-results - re-check this trajectory)"
    if subfloor is None:
        return "no anomalously-short non-bonded distance observed despite heating - argues AGAINST the missing-repulsion mechanism for this trajectory"
    if subfloor <= onset - tolerance_frames:
        return "short-range collapse PRECEDES heating onset - consistent with being the trigger"
    if subfloor <= onset + tolerance_frames:
        return "short-range collapse COINCIDES with heating onset - consistent with being the trigger"
    return "short-range collapse FOLLOWS heating onset - looks like a downstream symptom, not the trigger"


def discover_trajectories(results_root="results"):
    """Finds every rollout.xyz under results/waterbox_rollout*/ with a
    sibling energy_history.csv (the pairing run_rollout always writes them
    with), covering results/waterbox_rollout/, results/waterbox_rollout_momentum/,
    results/waterbox_rollout_study*/runs/**/, and
    results/waterbox_rollout_study_seed*/runs/**/ - the exact paths named in
    CLAUDE.md's Q4 next-step note - without hardcoding any one of them."""
    root = Path(results_root)
    pairs = []
    for traj_path in sorted(root.rglob("rollout.xyz")):
        rel = traj_path.relative_to(root)
        if not rel.parts[0].startswith("waterbox_rollout"):
            continue
        history_path = traj_path.parent / "energy_history.csv"
        if not history_path.exists():
            print(f"Skipping {traj_path}: no sibling energy_history.csv found.")
            continue
        label = str(rel.parent) if str(rel.parent) != "." else rel.parts[0]
        condition = "momentum" if "momentum" in str(rel) else "absolute"
        pairs.append({"label": label, "traj_path": traj_path, "history_path": history_path, "condition": condition})
    return pairs


def _write_summary_markdown(summary_rows, path, floors, floor_stats, floor_percentile):
    n_precedes = sum(1 for r in summary_rows if "PRECEDES" in r["verdict"])
    n_coincides = sum(1 for r in summary_rows if "COINCIDES" in r["verdict"])
    n_follows = sum(1 for r in summary_rows if "FOLLOWS" in r["verdict"])
    n_none = sum(1 for r in summary_rows if "AGAINST" in r["verdict"])
    lines = [
        "# Short-range collapse diagnostic (Q4, paper/main.tex sec:q4)",
        "",
        f"Empirical non-bonded distance floors (p{floor_percentile} over reference DFT configs):",
    ]
    for name, stat in floor_stats.items():
        lines.append(
            f"- {name}: floor = {floors[name]:.4f} A (true min of per-config minima = {stat['min']:.4f} A, "
            f"n_reference_configs={stat['n_reference_configs']})"
        )
    lines += [
        "",
        f"n={len(summary_rows)} trajectories analyzed. {n_precedes} precede heating onset, "
        f"{n_coincides} coincide, {n_follows} follow, {n_none} show no sub-floor violation at all.",
        "",
        "| label | condition | onset (fs) | first sub-floor (fs) | verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['label']} | {r['condition']} | {r['onset_time_fs']} | "
            f"{r['first_any_subfloor_time_fs']} | {r['verdict']} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {path}")


def run_diagnostic(
    results_root="results",
    data_root="./data",
    out=None,
    n_reference_configs=200,
    floor_percentile=1.0,
    plateau_fraction=0.3,
    onset_fraction=0.5,
    tolerance_frames=3,
    seed=42,
    limit=None,
):
    out_dir = Path(out) if out else Path("results/short_range_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing reference non-bonded distance floors from raw WaterBox dataset configs...")
    floors, floor_stats = compute_reference_floors(
        data_root=data_root, n_reference_configs=n_reference_configs, seed=seed, floor_percentile=floor_percentile,
    )
    for name, stat in floor_stats.items():
        print(
            f"  {name}: floor (p{floor_percentile} of per-config minima) = {floors[name]:.4f} A, "
            f"true min of per-config minima = {stat['min']:.4f} A, n_reference_configs={stat['n_reference_configs']}"
        )

    trajectories = discover_trajectories(results_root)
    if limit:
        trajectories = trajectories[:limit]
    if not trajectories:
        print(f"No rollout.xyz + energy_history.csv pairs found under {results_root}/waterbox_rollout*/")
        return []

    summary_rows = []
    for entry in trajectories:
        print(f"\n=== {entry['label']} ===")
        try:
            result = analyze_trajectory(
                entry["traj_path"], entry["history_path"], floors,
                plateau_fraction=plateau_fraction, onset_fraction=onset_fraction,
            )
        except Exception as exc:
            print(f"FAILED: {entry['label']}: {exc}")
            traceback.print_exc()
            continue

        per_traj_dir = out_dir / entry["label"].replace("/", "_")
        per_traj_dir.mkdir(parents=True, exist_ok=True)
        per_traj_csv = per_traj_dir / "short_range_diagnostic.csv"
        with open(per_traj_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["rows"])
        print(f"Wrote: {per_traj_csv}")

        v = verdict(result, tolerance_frames=tolerance_frames)
        print(
            f"Onset: frame {result['onset_idx']} ({result['onset_time_fs']} fs), "
            f"T0={result['t0']:.1f}K -> plateau {result['t_plateau']:.1f}K"
        )
        print(
            f"First sub-floor distance: frame {result['first_any_subfloor_idx']} "
            f"({result['first_any_subfloor_time_fs']} fs)"
        )
        print(f"Verdict: {v}")

        summary_rows.append({
            "label": entry["label"],
            "condition": entry["condition"],
            "t0_k": result["t0"],
            "t_plateau_k": result["t_plateau"],
            "onset_idx": result["onset_idx"],
            "onset_time_fs": result["onset_time_fs"],
            "first_subfloor_idx_OO": result["first_subfloor_idx"]["O-O"],
            "first_subfloor_idx_OH": result["first_subfloor_idx"]["O-H"],
            "first_subfloor_idx_HH": result["first_subfloor_idx"]["H-H"],
            "first_any_subfloor_idx": result["first_any_subfloor_idx"],
            "first_any_subfloor_time_fs": result["first_any_subfloor_time_fs"],
            "verdict": v,
        })

    if not summary_rows:
        print("\nEvery discovered trajectory failed - see tracebacks above.")
        return []

    summary_csv = out_dir / "summary.csv"
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote: {summary_csv}")

    _write_summary_markdown(summary_rows, out_dir / "summary.md", floors, floor_stats, floor_percentile)
    return summary_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--n-reference-configs", type=int, default=200)
    parser.add_argument("--floor-percentile", type=float, default=1.0)
    parser.add_argument("--plateau-fraction", type=float, default=0.3)
    parser.add_argument("--onset-fraction", type=float, default=0.5)
    parser.add_argument("--tolerance-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Small reference sample (20 configs) and only the first discovered trajectory - "
        "confirm the plumbing end-to-end before the full scan.",
    )
    args = parser.parse_args()

    run_diagnostic(
        results_root=args.results_root,
        data_root=args.data_root,
        out=args.out,
        n_reference_configs=20 if args.smoke_test else args.n_reference_configs,
        floor_percentile=args.floor_percentile,
        plateau_fraction=args.plateau_fraction,
        onset_fraction=args.onset_fraction,
        tolerance_frames=args.tolerance_frames,
        seed=args.seed,
        limit=1 if args.smoke_test else None,
    )
