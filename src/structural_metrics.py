#!/usr/bin/env python
"""Structural-fidelity checks for MD rollouts.

Rollout evaluation elsewhere in this repo (rollout_nve.py, compare_rollouts.py,
compare_models.py) is purely energetic (drift in eV). These functions add a
geometric sanity check on top - does the molecule stay chemically valid, not
just energy-conserving - by comparing bond-length distributions between a
model-driven rollout and the reference MD17 ensemble.
"""

from __future__ import annotations

import torch

from baseline_potential import _infer_bonds_from_positions


def infer_bonds(z: torch.Tensor, pos: torch.Tensor, box: torch.Tensor | None = None) -> list[tuple[int, int]]:
    """Return a bond list (atom index pairs) inferred from one equilibrium geometry.

    Thin wrapper around baseline_potential._infer_bonds_from_positions, which is
    distance/valence based (not topology-file based), so this works for any MD17
    molecule, not just aspirin.

    box: optional (3,3) or (1,3,3) periodic box matrix (e.g. WaterBox's `box`
    field). When given, bond distances use the minimum-image convention -
    required for periodic systems, where a bonded pair can straddle a box
    face. Only the diagonal is used (this codebase's periodic data is
    orthorhombic); pass None for non-periodic systems like MD17.
    """
    z_signature = tuple(int(v) for v in z.detach().cpu().tolist())
    box_lengths = None
    if box is not None:
        box_lengths = torch.as_tensor(box).reshape(3, 3).diagonal().detach().cpu()
    return _infer_bonds_from_positions(z_signature, pos.detach().cpu(), box_lengths=box_lengths)


def bond_length_series(bonds: list[tuple[int, int]], pos_traj) -> torch.Tensor:
    """Per-bond, per-frame lengths from a trajectory of positions.

    Args:
        bonds: list of (i, j) atom index pairs, e.g. from `infer_bonds`.
        pos_traj: (T, N, 3) tensor, or a list/tuple of (N, 3) tensors (as produced
            by rollout_nve.velocity_verlet_rollout's series["x"] when
            record_positions=True).

    Returns:
        (T, n_bonds) tensor of bond lengths in the same length units as pos_traj
        (Angstroms for MD17).
    """
    if isinstance(pos_traj, (list, tuple)):
        pos_traj = torch.stack([p if torch.is_tensor(p) else torch.as_tensor(p) for p in pos_traj])
    if not bonds:
        return torch.zeros((pos_traj.shape[0], 0))

    idx_i = torch.tensor([i for i, _ in bonds], dtype=torch.long)
    idx_j = torch.tensor([j for _, j in bonds], dtype=torch.long)
    delta = pos_traj[:, idx_i, :] - pos_traj[:, idx_j, :]
    return torch.linalg.norm(delta, dim=-1)


def bond_length_deviation_summary(
    rollout_lengths: torch.Tensor,
    reference_lengths: torch.Tensor,
    k: float = 4.0,
) -> dict:
    """Compare a rollout's per-bond length distribution against a reference ensemble.

    Args:
        rollout_lengths: (T_rollout, n_bonds) bond lengths from the model-driven rollout.
        reference_lengths: (T_ref, n_bonds) bond lengths from the reference MD17 ensemble
            (same bond ordering as rollout_lengths - pass the same `bonds` list to both
            calls of `bond_length_series`).
        k: number of reference standard deviations used as the "still chemically
            plausible" band for the out-of-band frame fraction below.

    Returns:
        dict with:
        - per_bond: list of per-bond reference/rollout mean+std and fractional deviation.
        - max_fractional_mean_deviation: largest |rollout_mean - ref_mean| / ref_mean
          over all bonds - a quick "did any bond's typical length shift" check.
        - frac_frames_any_bond_out_of_band: fraction of rollout frames where at least
          one bond length falls outside [ref_mean - k*ref_std, ref_mean + k*ref_std]
          for that bond - a "did the molecule visibly distort" check.
        - structural_stability_score: alias for frac_frames_any_bond_out_of_band, the
          single scalar meant to sit alongside energy drift and force/energy MAE in a
          summary table (0.0 = every frame stayed within the reference's typical
          bond-length band, 1.0 = every frame had at least one bond out of band).
    """
    if rollout_lengths.shape[1] != reference_lengths.shape[1]:
        raise ValueError("rollout_lengths and reference_lengths must have the same number of bonds.")

    n_bonds = rollout_lengths.shape[1]
    if n_bonds == 0:
        return {
            "per_bond": [],
            "max_fractional_mean_deviation": 0.0,
            "frac_frames_any_bond_out_of_band": 0.0,
            "structural_stability_score": 0.0,
        }

    ref_mean = reference_lengths.mean(dim=0)
    ref_std = reference_lengths.std(dim=0)
    rollout_mean = rollout_lengths.mean(dim=0)
    rollout_std = rollout_lengths.std(dim=0)

    frac_dev = torch.abs(rollout_mean - ref_mean) / ref_mean.clamp(min=1e-8)
    max_fractional_mean_deviation = float(frac_dev.max().item())

    lower = ref_mean - k * ref_std
    upper = ref_mean + k * ref_std
    out_of_band = (rollout_lengths < lower) | (rollout_lengths > upper)  # (T, n_bonds)
    any_bond_out = out_of_band.any(dim=1)  # (T,)
    frac_frames_any_bond_out_of_band = float(any_bond_out.float().mean().item())

    per_bond = [
        {
            "bond_index": idx,
            "reference_mean_A": float(ref_mean[idx].item()),
            "reference_std_A": float(ref_std[idx].item()),
            "rollout_mean_A": float(rollout_mean[idx].item()),
            "rollout_std_A": float(rollout_std[idx].item()),
            "fractional_mean_deviation": float(frac_dev[idx].item()),
        }
        for idx in range(n_bonds)
    ]

    return {
        "per_bond": per_bond,
        "max_fractional_mean_deviation": max_fractional_mean_deviation,
        "frac_frames_any_bond_out_of_band": frac_frames_any_bond_out_of_band,
        "structural_stability_score": frac_frames_any_bond_out_of_band,
    }


def infer_molecule_groups(z: torch.Tensor, pos: torch.Tensor, box: torch.Tensor | None = None) -> torch.Tensor:
    """Assign each atom to a molecule via connected components over inferred bonds.

    For a periodic multi-molecule system (e.g. a water box), this is what makes
    a *per-molecule* momentum-conservation check possible: `infer_bonds` (via
    `baseline_potential._infer_bonds_from_positions`) is distance/valence based
    and molecule-agnostic, so running connected-components over its output
    groups atoms into individual molecules regardless of what order the dataset
    happens to store atoms in - it does NOT assume atoms are stored in
    contiguous per-molecule blocks, which is not guaranteed by any dataset here.

    box: optional periodic box (see `infer_bonds`) - pass the sample's `box`
    for periodic multi-molecule systems so bonds crossing a box face are still
    detected via the minimum-image convention.

    Returns a (N,) long tensor of group ids in [0, num_groups). Atoms with no
    inferred bond to anything else form their own singleton group.
    """
    n_atoms = z.shape[0]
    bonds = infer_bonds(z, pos, box=box)

    parent = list(range(n_atoms))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i, j in bonds:
        union(i, j)

    roots = [find(i) for i in range(n_atoms)]
    unique_roots = sorted(set(roots))
    root_to_group = {root: idx for idx, root in enumerate(unique_roots)}
    group_ids = [root_to_group[r] for r in roots]

    return torch.tensor(group_ids, dtype=torch.long)


def summarize_molecule_groups(z: torch.Tensor, group_ids: torch.Tensor) -> dict:
    """Per-group atomic-number composition, for sanity-checking a grouping before
    trusting it anywhere else (e.g. asserting every water molecule works out to
    exactly {O: 1, H: 2}, not some other split caused by a missed/spurious bond).
    """
    from collections import Counter

    n_groups = int(group_ids.max().item()) + 1 if group_ids.numel() else 0
    compositions = []
    for g in range(n_groups):
        mask = group_ids == g
        zs = z[mask].tolist()
        compositions.append(dict(Counter(zs)))
    return {"n_groups": n_groups, "compositions": compositions}
