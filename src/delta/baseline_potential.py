#!/usr/bin/env python
"""Analytic baseline potential for Δ-learning.

This is intentionally simple and self-contained so you can swap it later.
Current baseline: single-parameter Lennard-Jones (12-6) with a cutoff.

Units (consistent with MD17 + TorchMD-Net conventions):
- positions: Å
- energy: eV
- forces: eV/Å
"""

from __future__ import annotations

import torch


def lj_energy_forces(
    pos: torch.Tensor,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute LJ energy and forces for a single molecule (no PBC).

    Args:
        pos: (N,3) positions in Å
        epsilon_eV: ε in eV
        sigma_A: σ in Å
        r_cut_A: cutoff in Å (smoothness not enforced)

    Returns:
        U: scalar tensor (eV)
        F: (N,3) tensor (eV/Å)
    """
    device = pos.device
    dtype = pos.dtype

    N = pos.shape[0]
    if N <= 1:
        return torch.zeros((), device=device, dtype=dtype), torch.zeros_like(pos)

    # pairwise displacement r_ij = r_i - r_j
    rij = pos[:, None, :] - pos[None, :, :]  # (N,N,3)
    r2 = (rij * rij).sum(dim=-1)  # (N,N)

    # mask: i<j, r>0, within cutoff
    # (avoid sqrt until after masking)
    triu = torch.triu(torch.ones((N, N), device=device, dtype=torch.bool), diagonal=1)
    r = torch.sqrt(torch.clamp(r2, min=1e-12))
    within = r < float(r_cut_A)
    mask = triu & within

    if not mask.any():
        return torch.zeros((), device=device, dtype=dtype), torch.zeros_like(pos)

    r_m = r[mask]  # (P,)
    rij_m = rij[mask]  # (P,3)

    sig = torch.as_tensor(float(sigma_A), device=device, dtype=dtype)
    eps = torch.as_tensor(float(epsilon_eV), device=device, dtype=dtype)

    inv_r = 1.0 / r_m
    sr = sig * inv_r
    sr2 = sr * sr
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6

    # Energy: 4 ε (sr12 - sr6)
    U_pairs = 4.0 * eps * (sr12 - sr6)  # (P,)
    U = U_pairs.sum()

    # Force magnitude factor:
    # dU/dr = 24 ε (2 sr12 - sr6) / r
    # Force on i from j: F_ij = - dU/dr * (r_i - r_j)/r
    # => F_ij = -24 ε (2 sr12 - sr6) * (r_ij) / r^2
    coef = -24.0 * eps * (2.0 * sr12 - sr6) * (inv_r * inv_r)  # (P,)
    fij = coef[:, None] * rij_m  # (P,3)

    # Scatter-add to atoms: for each pair (i,j), add fij to i and -fij to j
    # Reconstruct indices for masked pairs
    idx = mask.nonzero(as_tuple=False)  # (P,2)
    i = idx[:, 0]
    j = idx[:, 1]

    F = torch.zeros_like(pos)
    F.index_add_(0, i, fij)
    F.index_add_(0, j, -fij)

    return U, F


def lj_energy_forces_batched(
    z: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    epsilon_eV: float = 0.01,
    sigma_A: float = 1.0,
    r_cut_A: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched wrapper for multiple molecules in one PyG batch.

    Args:
        z: (N,) atomic numbers (unused for this simple baseline)
        pos: (N,3)
        batch: (N,) graph id per atom

    Returns:
        U_per_graph: (B,) energies in eV
        F: (N,3) forces in eV/Å
    """
    device = pos.device
    dtype = pos.dtype
    unique = torch.unique(batch)
    B = int(unique.numel())

    U = torch.zeros((B,), device=device, dtype=dtype)
    F = torch.zeros_like(pos)

    for bi, g in enumerate(unique.tolist()):
        m = (batch == g)
        U_g, F_g = lj_energy_forces(
            pos[m],
            epsilon_eV=epsilon_eV,
            sigma_A=sigma_A,
            r_cut_A=r_cut_A,
        )
        U[bi] = U_g
        F[m] = F_g

    return U, F
