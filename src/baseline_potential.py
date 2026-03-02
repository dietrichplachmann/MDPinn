#!/usr/bin/env python
"""
Analytic baseline potential used by delta-learning.

Current baseline is intentionally minimal: Lennard-Jones 12-6 with cutoff,
implemented for single molecules and batched PyG graphs.

Units:
- positions: Angstrom
- energies: eV
- forces: eV / Angstrom
"""

from __future__ import annotations

import torch


def lj_energy_forces(
    pos: torch.Tensor,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise LJ energy and forces for one molecule (no PBC)."""
    device = pos.device
    dtype = pos.dtype

    n_atoms = pos.shape[0]
    if n_atoms <= 1:
        return torch.zeros((), device=device, dtype=dtype), torch.zeros_like(pos)

    # Pairwise displacement matrix r_ij = r_i - r_j.
    rij = pos[:, None, :] - pos[None, :, :]
    r2 = (rij * rij).sum(dim=-1)

    # Keep only unique i<j pairs inside cutoff.
    triu = torch.triu(torch.ones((n_atoms, n_atoms), device=device, dtype=torch.bool), diagonal=1)
    r = torch.sqrt(torch.clamp(r2, min=1e-12))
    mask = triu & (r < float(r_cut_A))

    if not mask.any():
        return torch.zeros((), device=device, dtype=dtype), torch.zeros_like(pos)

    r_sel = r[mask]
    rij_sel = rij[mask]

    sig = torch.as_tensor(float(sigma_A), device=device, dtype=dtype)
    eps = torch.as_tensor(float(epsilon_eV), device=device, dtype=dtype)

    inv_r = 1.0 / r_sel
    sr = sig * inv_r
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6

    # LJ pair energy and summed molecular energy.
    u_pairs = 4.0 * eps * (sr12 - sr6)
    u_total = u_pairs.sum()

    # Force on pair i-j from analytic derivative.
    coef = -24.0 * eps * (2.0 * sr12 - sr6) * (inv_r * inv_r)
    fij = coef[:, None] * rij_sel

    idx = mask.nonzero(as_tuple=False)
    i_idx = idx[:, 0]
    j_idx = idx[:, 1]

    forces = torch.zeros_like(pos)
    forces.index_add_(0, i_idx, fij)
    forces.index_add_(0, j_idx, -fij)

    return u_total, forces


def lj_energy_forces_batched(
    z: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply `lj_energy_forces` to each graph in a PyG batch.

    Returns:
    - U_per_graph: shape (B,)
    - F_all_atoms: shape (N,3)
    """
    _ = z  # Placeholder: current baseline is species-agnostic.

    device = pos.device
    dtype = pos.dtype
    unique_graphs = torch.unique(batch)

    u_per_graph = torch.zeros((int(unique_graphs.numel()),), device=device, dtype=dtype)
    f_all_atoms = torch.zeros_like(pos)

    for graph_idx, graph_id in enumerate(unique_graphs.tolist()):
        mask = batch == graph_id
        u_graph, f_graph = lj_energy_forces(
            pos[mask],
            epsilon_eV=epsilon_eV,
            sigma_A=sigma_A,
            r_cut_A=r_cut_A,
        )
        u_per_graph[graph_idx] = u_graph
        f_all_atoms[mask] = f_graph

    return u_per_graph, f_all_atoms
