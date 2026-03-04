#!/usr/bin/env python
"""
Analytic baseline potential used by delta-learning.

Current baseline is intentionally minimal: Lennard-Jones 12-6 with cutoff,
implemented for single molecules and batched PyG graphs.

Units:
- positions: Angstrom
- energies: eV
- forces: eV / Angstrom

How this connects to your advisor's outline:
- In the full target setup, U_ref would include bonded + nonbonded analytic terms.
- This file is the place where those analytic terms belong.
- Right now it is a minimal LJ baseline so the delta-learning wiring is working.
"""

from __future__ import annotations

import torch


def lj_energy_forces(
    pos: torch.Tensor,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise LJ energy and forces for one molecule (no PBC).

    Physical picture:
    - For each atom pair i,j we evaluate a 12-6 LJ interaction.
    - Summing all pair energies gives U_ref(R).
    - Pairwise force vectors are added to atoms i and j with opposite signs
      (Newton's third law), so total pair force contribution is balanced.
    """
    device = pos.device
    dtype = pos.dtype

    n_atoms = pos.shape[0]
    if n_atoms <= 1:
        return torch.zeros((), device=device, dtype=dtype), torch.zeros_like(pos)

    # Pairwise displacement matrix r_ij = r_i - r_j.
    # Each row/column pair corresponds to one interatomic distance vector.
    rij = pos[:, None, :] - pos[None, :, :]
    r2 = (rij * rij).sum(dim=-1)

    # Keep only unique i<j pairs inside cutoff.
    # i<j avoids double counting pair energy.
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
    # This is the baseline potential surface used in delta-learning:
    #   U_hyb(R) = U_ref(R) + DeltaU_NN(R)
    u_pairs = 4.0 * eps * (sr12 - sr6)
    u_total = u_pairs.sum()

    # Force on pair i-j from analytic derivative.
    # Sign convention here is consistent with F = - dU/dr.
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
    # In a production FF baseline this would use atom types/charges/parameters.

    device = pos.device
    dtype = pos.dtype
    unique_graphs = torch.unique(batch)

    u_per_graph = torch.zeros((int(unique_graphs.numel()),), device=device, dtype=dtype)
    f_all_atoms = torch.zeros_like(pos)

    for graph_idx, graph_id in enumerate(unique_graphs.tolist()):
        # Solve each molecule independently, then stitch results back into
        # the PyG-collated batch tensors.
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
