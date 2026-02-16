#!/usr/bin/env python
"""
Physics-Informed Loss Functions for TorchMD-NET
Complete implementation with NVE loss for MD17 trajectories
"""

import torch
import torch.nn.functional as F


# ============================================================================
# MOMENTUM CONSERVATION LOSS (Works with any batch)
# ============================================================================

def momentum_symmetry_loss(R, F_pred):
    """
    Enforce momentum conservation constraints:
        L_mom = || sum_i F_i ||^2 + || sum_i r_i × F_i ||^2

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


# ============================================================================
# NVE ENERGY CONSERVATION LOSS (Requires trajectory data)
# ============================================================================

def build_trajectory_batch(dataset, start_idx, traj_length, device):
    """
    Build a trajectory batch from consecutive MD17 frames

    Args:
        dataset: MD17 dataset object
        start_idx: Starting frame index
        traj_length: Number of consecutive frames
        device: torch device

    Returns:
        dict with 'R_traj', 'Z', 'E_ref_traj'
    """
    R_traj = []
    E_ref_traj = []

    # Collect consecutive frames
    for t in range(traj_length):
        idx = start_idx + t
        if idx >= len(dataset):
            # Wrap around if we exceed dataset length
            idx = idx % len(dataset)

        sample = dataset[idx]
        R_traj.append(sample.pos)
        E_ref_traj.append(sample.y)

    # Get atomic numbers from first frame
    Z = dataset[start_idx].z

    return {
        'R_traj': torch.stack(R_traj).to(device),  # (T, N, 3)
        'Z': Z.to(device),  # (N,)
        'E_ref_traj': torch.stack(E_ref_traj).to(device),  # (T,)
    }


def nve_loss_from_trajectory(model, traj_batch, device, dt=0.5):
    """
    NVE (energy conservation) loss for trajectory data

    Penalizes drift in predicted energy along a trajectory:
        L_NVE = mean((E_pred(t) - E_pred(0))^2)

    This enforces that total energy should remain constant in NVE ensemble.

    Args:
        model: The representation model (callable: model(z, pos, batch) -> energy)
        traj_batch: dict with:
            'R_traj': (T, N, 3) positions over time
            'Z': (N,) atomic numbers
            'E_ref_traj': (T,) reference energies (optional)
        device: torch device
        dt: timestep in fs (default 0.5 for MD17)

    Returns:
        Scalar loss tensor
    """
    R_traj = traj_batch['R_traj']  # (T, N, 3)
    Z = traj_batch['Z']  # (N,)

    T, N, _ = R_traj.shape

    # Create batch tensor (all atoms belong to same molecule)
    batch = torch.zeros(N, dtype=torch.long, device=device)

    # Compute predicted energies along trajectory
    E_pred_list = []
    for t in range(T):
        R_t = R_traj[t]  # (N, 3)

        # Forward pass through model
        E_t = model(Z, R_t, batch=batch)

        # Handle different output formats
        if isinstance(E_t, tuple):
            E_t = E_t[0]  # Some models return (energy, forces)

        E_pred_list.append(E_t.squeeze())

    E_pred_traj = torch.stack(E_pred_list)  # (T,)

    # Penalize energy drift from initial value
    # In NVE ensemble, E(t) should equal E(0) for all t
    E0 = E_pred_traj[0].detach()  # Detach to only penalize drift, not absolute value
    drift = E_pred_traj - E0
    L_drift = torch.mean(drift ** 2)

    # Optional: also penalize deviation from reference energies
    # This helps the model learn correct absolute energies
    L_ref = torch.tensor(0.0, device=device)
    if 'E_ref_traj' in traj_batch:
        E_ref_traj = traj_batch['E_ref_traj']
        # Weight reference loss less than drift (drift is more important)
        L_ref = 0.1 * torch.mean((E_pred_traj - E_ref_traj) ** 2)

    return L_drift + L_ref


def nve_loss_with_kinetic_energy(model, traj_batch, device, masses, dt=0.5):
    """
    Full NVE loss including kinetic energy:
        E_total = K + U_pred
        L_NVE = mean((E_total(t) - E_total(0))^2)

    This is more physically accurate but requires computing velocities
    from positions via finite differences.

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

    # Compute velocities from positions using central differences
    V_traj = compute_velocities_from_positions(R_traj, dt)  # (T, N, 3)

    # Compute kinetic energy at each timestep
    # K = 0.5 * m * v^2
    # Units: amu * (Å/fs)^2 -> need conversion factor for eV
    # 1 amu * (Å/fs)^2 = 0.01036427 eV
    conversion = 0.01036427

    v2 = (V_traj ** 2).sum(dim=-1)  # (T, N) - squared velocity magnitude
    m_broadcast = masses.view(1, N).expand(T, N)
    K_traj = 0.5 * conversion * (m_broadcast * v2).sum(dim=1)  # (T,)

    # Compute potential energy at each timestep
    batch = torch.zeros(N, dtype=torch.long, device=device)
    U_pred_list = []
    for t in range(T):
        R_t = R_traj[t]
        U_t = model(Z, R_t, batch=batch)
        if isinstance(U_t, tuple):
            U_t = U_t[0]
        U_pred_list.append(U_t.squeeze())

    U_pred_traj = torch.stack(U_pred_list)  # (T,)

    # Total energy
    E_total_traj = K_traj + U_pred_traj  # (T,)

    # Penalize drift from initial total energy
    E0 = E_total_traj[0].detach()
    drift = E_total_traj - E0
    L_NVE = torch.mean(drift ** 2)

    return L_NVE


def compute_velocities_from_positions(R_traj, dt=0.5):
    """
    Compute velocities from position trajectory using finite differences

    Uses central differences for interior points:
        v(t) = [r(t+dt) - r(t-dt)] / (2*dt)

    Args:
        R_traj: (T, N, 3) position trajectory
        dt: timestep in femtoseconds

    Returns:
        V_traj: (T, N, 3) velocity trajectory in Angstrom/fs
    """
    T = R_traj.shape[0]
    V_traj = torch.zeros_like(R_traj)

    # Forward difference for first frame
    V_traj[0] = (R_traj[1] - R_traj[0]) / dt

    # Central difference for middle frames (more accurate)
    if T > 2:
        V_traj[1:-1] = (R_traj[2:] - R_traj[:-2]) / (2 * dt)

    # Backward difference for last frame
    V_traj[-1] = (R_traj[-1] - R_traj[-2]) / dt

    return V_traj


# ============================================================================
# PERIODIC BOUNDARY CONDITIONS LOSS (Only for periodic systems)
# ============================================================================

def periodic_bc_loss_improved(model, R, Z, box_L, F_pred, batch):
    """
    Enforce periodic boundary conditions:
        L_PBC = ||U(r) - U(r+L)||^2 + ||F(r) - F(r+L)||^2

    Checks periodicity in all 3 box directions.

    NOTE: Only use this for systems with periodic boundary conditions!
    MD17 single molecules do NOT have PBC - skip this loss for them.

    Args:
        model: The representation model
        R: (N, 3) atomic positions with requires_grad=True
        Z: (N,) atomic numbers
        box_L: (3,) box lengths [Lx, Ly, Lz]
        F_pred: (N, 3) forces at R
        batch: batch indices

    Returns:
        Scalar loss tensor
    """
    device = R.device
    N = R.shape[0]

    L_pbc = torch.tensor(0.0, device=device)

    # Check periodicity in each box direction
    for dim in range(3):
        # Create shift vector in this direction
        shift = torch.zeros(N, 3, device=device)
        shift[:, dim] = box_L[dim]

        # Shifted coordinates
        R_shift = (R + shift).clone().detach().requires_grad_(True)

        # Energy at shifted position
        E_shift = model(Z, R_shift, batch=batch)
        if isinstance(E_shift, tuple):
            E_shift = E_shift[0]

        # Forces at shifted position
        F_shift = -torch.autograd.grad(
            E_shift.sum(),
            R_shift,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Energy at original position
        E_orig = model(Z, R, batch=batch)
        if isinstance(E_orig, tuple):
            E_orig = E_orig[0]

        # Energy should be identical (periodic)
        L_pbc += torch.mean((E_orig - E_shift) ** 2)

        # Forces should be identical (periodic)
        L_pbc += torch.mean((F_pred - F_shift) ** 2)

    # Average over 3 dimensions
    return L_pbc / 3.0


# ============================================================================
# SYMMETRY LOSSES (Redundant for equivariant models but included)
# ============================================================================

def translation_invariance_loss(model, R, Z, batch):
    """
    Enforce translation invariance: U(r) = U(r + c)

    NOTE: This is redundant for SE(3)-equivariant models like TensorNet.
    Only useful for testing or non-equivariant models.
    """
    device = R.device

    # Random translation
    c = torch.randn(1, 3, device=device) * 0.1

    # Energy at original position
    E_orig = model(Z, R, batch=batch)
    if isinstance(E_orig, tuple):
        E_orig = E_orig[0]

    # Energy at translated position
    R_trans = R + c
    E_trans = model(Z, R_trans, batch=batch)
    if isinstance(E_trans, tuple):
        E_trans = E_trans[0]

    # Should be identical
    return torch.mean((E_orig - E_trans) ** 2)


def rotation_invariance_loss(model, R, Z, batch):
    """
    Enforce rotation invariance: U(r) = U(Q @ r)

    NOTE: This is redundant for SE(3)-equivariant models.
    """
    device = R.device

    # Generate random rotation (simplified - z-axis rotation)
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
    if isinstance(E_orig, tuple):
        E_orig = E_orig[0]

    # Energy at rotated position
    R_rot = R @ rotation_matrix.T
    E_rot = model(Z, R_rot, batch=batch)
    if isinstance(E_rot, tuple):
        E_rot = E_rot[0]

    # Should be identical
    return torch.mean((E_orig - E_rot) ** 2)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    35: 79.904,  # Br
    53: 126.90,  # I
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    """
    Example showing how to use these losses
    """

    # Example 1: Momentum loss (works on any batch)
    print("Example 1: Momentum Conservation Loss")
    N = 5
    R = torch.randn(N, 3)
    F = torch.randn(N, 3)

    loss_mom = momentum_symmetry_loss(R, F)
    print(f"  Momentum loss: {loss_mom.item():.6f}")

    # Forces that conserve momentum should give low loss
    F_conserved = torch.randn(N, 3)
    F_conserved[-1] = -F_conserved[:-1].sum(dim=0)  # Last force balances others
    loss_mom_good = momentum_symmetry_loss(R, F_conserved)
    print(f"  Momentum loss (conserved): {loss_mom_good.item():.6f}")

    print("\nExample 2: NVE Loss")
    print("  (Requires actual model and dataset - see train_physics_FIXED.py)")

    print("\nExample 3: Building trajectory from MD17")
    print("""
    from torchmdnet.datasets import MD17

    dataset = MD17(root='./data', molecules='aspirin')
    traj_batch = build_trajectory_batch(
        dataset, 
        start_idx=0, 
        traj_length=100, 
        device='cpu'
    )

    # Now compute NVE loss
    loss_nve = nve_loss_from_trajectory(model, traj_batch, device='cpu')
    """)

    print("\n✓ Physics losses module ready!")
    print("\nAvailable losses:")
    print("  - momentum_symmetry_loss(R, F)")
    print("  - nve_loss_from_trajectory(model, traj_batch, device)")
    print("  - nve_loss_with_kinetic_energy(model, traj_batch, device, masses)")
    print("  - periodic_bc_loss_improved(model, R, Z, box_L, F, batch)")
    print("  - build_trajectory_batch(dataset, start_idx, traj_length, device)")