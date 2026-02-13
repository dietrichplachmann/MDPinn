#!/usr/bin/env python
"""
Physics-Informed Loss Functions for TorchMD-NET
Adapted from MDPinn custom loss functions
"""

import torch
import torch.nn.functional as F


def momentum_symmetry_loss(R, F_pred):
    """
    Enforce momentum conservation constraints:
        L_sym = || sum_i F_i ||^2 + || sum_i r_i × F_i ||^2

    This ensures:
    - Linear momentum conservation: sum of forces = 0
    - Angular momentum conservation: sum of torques = 0

    Args:
        R: (N, 3) atomic positions
        F_pred: (N, 3) predicted forces

    Returns:
        Scalar loss tensor
    """
    # Linear momentum (total force should be zero)
    F_sum = F_pred.sum(dim=0)  # (3,)
    linear_term = (F_sum ** 2).sum()

    # Angular momentum (total torque about origin should be zero)
    torque_i = torch.cross(R, F_pred, dim=1)  # (N, 3)
    T_sum = torque_i.sum(dim=0)  # (3,)
    angular_term = (T_sum ** 2).sum()

    return linear_term + angular_term


def periodic_bc_loss_improved(model, R, Z, box_L, F_pred, batch_size_single):
    """
    Enforce periodic boundary conditions:
        L_BC = ||U(r) - U(r+L)||^2 + ||F(r) - F(r+L)||^2

    Checks periodicity in all 3 box directions

    Args:
        model: The representation model (callable that takes z, pos, batch)
        R: (N, 3) atomic positions with requires_grad=True
        Z: (N,) atomic numbers
        box_L: (3,) box lengths
        F_pred: (N, 3) forces at R
        batch_size_single: Batch tensor for single molecule

    Returns:
        Scalar loss tensor
    """
    device = R.device
    N = R.shape[0]

    # Create batch tensor (all atoms in same molecule)
    if batch_size_single.numel() == 1:
        batch = torch.zeros(N, dtype=torch.long, device=device)
    else:
        batch = batch_size_single

    L_bc = torch.tensor(0.0, device=device)

    # Check periodicity in each box direction
    for dim in range(3):
        # Create shift vector
        shift = torch.zeros(N, 3, device=device)
        shift[:, dim] = box_L[dim]

        # Shifted coordinates (detach and re-require grad for clean autograd)
        R_shift = (R + shift).clone().detach().requires_grad_(True)

        # Compute energy and forces at shifted position
        E_shift = model(Z, R_shift, batch=batch)

        # Forces at shifted position
        F_shift = -torch.autograd.grad(
            E_shift.sum(),  # Sum for batched computation
            R_shift,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Energy at original position
        E_orig = model(Z, R, batch=batch)

        # Energy periodicity term
        L_bc += torch.mean((E_orig - E_shift) ** 2)

        # Force periodicity term
        L_bc += torch.mean((F_pred - F_shift) ** 2)

    # Average over 3 dimensions
    return L_bc / 3.0


def nve_loss_from_md17_trajectory(model, traj_batch, device, dt=0.5):
    """
    NVE (energy conservation) loss for MD17-style trajectory data

    Penalizes drift in predicted energy along a trajectory:
        L_NVE = mean((E_pred(t) - E_pred(0))^2)

    Args:
        model: The representation model
        traj_batch: dict with:
            'R_traj': (T, N, 3) positions
            'Z': (N,) atomic numbers
            'E_ref_traj': (T,) reference energies (optional)
        device: torch device
        dt: timestep in fs (for velocity computation if needed)

    Returns:
        Scalar loss tensor
    """
    R_traj = traj_batch['R_traj'].to(device)  # (T, N, 3)
    Z = traj_batch['Z'].to(device)  # (N,)

    T, N, _ = R_traj.shape

    # Create batch tensor
    batch = torch.zeros(N, dtype=torch.long, device=device)

    # Compute predicted energies along trajectory
    E_pred_list = []
    for t in range(T):
        R_t = R_traj[t]  # (N, 3)
        E_t = model(Z, R_t, batch=batch)
        E_pred_list.append(E_t.squeeze())

    E_pred_traj = torch.stack(E_pred_list)  # (T,)

    # Penalize energy drift from initial value
    E0 = E_pred_traj[0]
    drift = E_pred_traj - E0
    L_drift = torch.mean(drift ** 2)

    # Optional: also penalize deviation from reference energies if available
    L_ref = torch.tensor(0.0, device=device)
    if 'E_ref_traj' in traj_batch:
        E_ref_traj = traj_batch['E_ref_traj'].to(device)
        L_ref = torch.mean((E_pred_traj - E_ref_traj) ** 2)

    # Combined loss (drift is more important for NVE)
    return L_drift + 0.1 * L_ref


def compute_velocities_from_positions(R_traj, dt=0.5):
    """
    Compute velocities from position trajectory using finite differences

    Args:
        R_traj: (T, N, 3) position trajectory
        dt: timestep in femtoseconds

    Returns:
        V_traj: (T, N, 3) velocity trajectory in Angstrom/fs
    """
    V_traj = torch.zeros_like(R_traj)

    # Forward difference for first frame
    V_traj[0] = (R_traj[1] - R_traj[0]) / dt

    # Central difference for middle frames
    if R_traj.shape[0] > 2:
        V_traj[1:-1] = (R_traj[2:] - R_traj[:-2]) / (2 * dt)

    # Backward difference for last frame
    V_traj[-1] = (R_traj[-1] - R_traj[-2]) / dt

    return V_traj


def nve_loss_with_kinetic_energy(model, traj_batch, device, masses, dt=0.5):
    """
    Full NVE loss including kinetic energy:
        E_total = K + U_pred
        L_NVE = mean((E_total(t) - E_total(0))^2)

    Args:
        model: The representation model
        traj_batch: dict with 'R_traj' and 'Z'
        device: torch device
        masses: (N,) atomic masses in amu
        dt: timestep in fs

    Returns:
        Scalar loss tensor
    """
    R_traj = traj_batch['R_traj'].to(device)  # (T, N, 3)
    Z = traj_batch['Z'].to(device)  # (N,)
    masses = masses.to(device)  # (N,)

    T, N, _ = R_traj.shape

    # Compute velocities from positions
    V_traj = compute_velocities_from_positions(R_traj, dt)  # (T, N, 3)

    # Compute kinetic energy at each timestep
    # K = 0.5 * m * v^2 (in appropriate units)
    v2 = (V_traj ** 2).sum(dim=-1)  # (T, N)
    m_broadcast = masses.view(1, N).expand(T, N)
    K_traj = 0.5 * (m_broadcast * v2).sum(dim=1)  # (T,)

    # Compute potential energy at each timestep
    batch = torch.zeros(N, dtype=torch.long, device=device)
    U_pred_list = []
    for t in range(T):
        R_t = R_traj[t]
        U_t = model(Z, R_t, batch=batch)
        U_pred_list.append(U_t.squeeze())

    U_pred_traj = torch.stack(U_pred_list)  # (T,)

    # Total energy
    E_total_traj = K_traj + U_pred_traj  # (T,)

    # Penalize drift from initial total energy
    E0 = E_total_traj[0]
    drift = E_total_traj - E0
    L_NVE = torch.mean(drift ** 2)

    return L_NVE


def translation_invariance_loss(model, R, Z, batch):
    """
    Enforce translation invariance:
        L_trans = ||U(r) - U(r + c)||^2

    where c is a random translation vector

    NOTE: This is redundant for equivariant models like TensorNet and ET,
    but included for completeness

    Args:
        model: The representation model
        R: (N, 3) positions
        Z: (N,) atomic numbers
        batch: batch indices

    Returns:
        Scalar loss tensor
    """
    device = R.device

    # Random translation
    c = torch.randn(1, 3, device=device) * 0.1  # Small random shift

    # Energy at original position
    E_orig = model(Z, R, batch=batch)

    # Energy at translated position
    R_trans = R + c
    E_trans = model(Z, R_trans, batch=batch)

    # Should be identical for translationally invariant model
    return torch.mean((E_orig - E_trans) ** 2)


def rotation_invariance_loss(model, R, Z, batch):
    """
    Enforce rotation invariance:
        L_rot = ||U(r) - U(Q @ r)||^2

    where Q is a random rotation matrix

    NOTE: This is redundant for equivariant models

    Args:
        model: The representation model
        R: (N, 3) positions
        Z: (N,) atomic numbers
        batch: batch indices

    Returns:
        Scalar loss tensor
    """
    device = R.device

    # Generate random rotation matrix (simplified - just rotate around z-axis)
    theta = torch.rand(1, device=device) * 2 * 3.14159
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    rotation_matrix = torch.tensor([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)

    # Energy at original position
    E_orig = model(Z, R, batch=batch)

    # Energy at rotated position
    R_rot = R @ rotation_matrix.T
    E_rot = model(Z, R_rot, batch=batch)

    # Should be identical for rotationally invariant model
    return torch.mean((E_orig - E_rot) ** 2)


# Atomic masses (amu) for common elements
ATOMIC_MASSES = {
    1: 1.008,  # H
    6: 12.011,  # C
    7: 14.007,  # N
    8: 15.999,  # O
    9: 18.998,  # F
    15: 30.974,  # P
    16: 32.06,  # S
    17: 35.45,  # Cl
}


def get_atomic_masses(Z):
    """
    Get atomic masses for given atomic numbers

    Args:
        Z: (N,) tensor of atomic numbers

    Returns:
        masses: (N,) tensor of atomic masses in amu
    """
    masses = torch.zeros_like(Z, dtype=torch.float32)
    for i, z in enumerate(Z):
        z_val = z.item()
        masses[i] = ATOMIC_MASSES.get(z_val, 1.0)  # Default to 1.0 if unknown

    return masses