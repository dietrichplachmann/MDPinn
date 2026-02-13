import torch

def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def nve_loss_from_traj(model, nve_batch, device):
    """
    nve_batch is a dict with:
        R_traj_np:   (T, N, 3)
        V_traj_np:   (T, N, 3)
        Z_np:        (N,)
        box_L_np:    (3,)        # not actually needed for NVE, but handy for later
        U_ref_traj_np: (T,)      # analytic bonded + long-range energy
        masses_np:   (N,)        # atomic masses

    Returns:
        L_NVE (scalar tensor)
    """
    # unpack
    R_traj_np     = nve_batch["R_traj_np"]
    V_traj_np     = nve_batch["V_traj_np"]
    Z_np          = nve_batch["Z_np"]
    box_L_np      = nve_batch["box_L_np"]
    U_ref_traj_np = nve_batch["U_ref_traj_np"]
    masses_np     = nve_batch["masses_np"]

    # convert to tensors
    R_traj = torch.tensor(R_traj_np, dtype=torch.float32, device=device)   # (T, N, 3)
    V_traj = torch.tensor(V_traj_np, dtype=torch.float32, device=device)   # (T, N, 3)
    Z      = torch.tensor(Z_np,      dtype=torch.long,   device=device)    # (N,)
    box_L  = torch.tensor(box_L_np,  dtype=torch.float32, device=device)   # (3,)
    U_ref_traj = torch.tensor(U_ref_traj_np, dtype=torch.float32, device=device)  # (T,)
    masses = torch.tensor(masses_np, dtype=torch.float32, device=device)   # (N,)

    T, N, _ = R_traj.shape

    # ----- kinetic energy K(t) -----
    # v^2 per atom: (T, N)
    v2 = (V_traj ** 2).sum(dim=-1)
    # masses broadcast: (T, N)
    m_broadcast = masses.view(1, N).expand(T, N)
    # K(t): (T,)
    K_traj = 0.5 * (m_broadcast * v2).sum(dim=1)

    # ----- NN short-range correction U_phi(t) -----
    U_phi_list = []
    for t in range(T):
        R_t = R_traj[t]  # (N, 3), no grad wrt R needed for NVE
        # We only need grad wrt model params, so no requires_grad_ on R_t
        E_t = model(R_t, Z)  # scalar or (1,)
        U_phi_list.append(E_t.squeeze())

    U_phi_traj = torch.stack(U_phi_list, dim=0)  # (T,)

    # ----- total hybrid energy along the trajectory -----
    E_tot_traj = K_traj + U_ref_traj + U_phi_traj  # (T,)

    # ----- NVE loss: penalize drift from t=0 -----
    E0 = E_tot_traj[0]
    diff = E_tot_traj - E0
    L_NVE = torch.mean(diff ** 2)

    return L_NVE

def momentum_symmetry_loss(R, F_pred):
    """
    Implements:
        L_sym = || sum_i F_i ||^2 + || sum_i r_i × F_i ||^2

    R: (N, 3) positions
    F_pred: (N, 3) predicted forces
    """
    # total force (linear momentum conservation)
    F_sum = F_pred.sum(dim=0)               # (3,)
    # total torque about origin (angular momentum conservation)
    torque_i = torch.cross(R, F_pred, dim=1)  # (N, 3)
    T_sum = torque_i.sum(dim=0)             # (3,)

    L_sym = (F_sum ** 2).sum() + (T_sum ** 2).sum()
    return L_sym


def periodic_bc_loss(model, R, Z, box_L, F_pred):
    """
    Implements:
        L_BC = ||U_phi(r) - U_phi(r+L)||^2 + ||∇ U_phi(r) - ∇ U_phi(r+L)||^2

    R:      (N, 3) with requires_grad=True
    Z:      (N,)
    box_L:  (3,)
    F_pred: forces at R, shape (N, 3)
    """
    device = R.device
    L_vec = box_L.view(1, 3)  # (1, 3)

    # Periodic image coordinates: make them a separate leaf so we can take grads
    R_shift = (R + L_vec).clone().detach().requires_grad_(True)

    # Energies
    E_R      = model(R,       Z)  # scalar or (1,)
    E_shift  = model(R_shift, Z)

    # Forces at the shifted coordinates
    F_shift = -torch.autograd.grad(
        E_shift, R_shift,
        grad_outputs=torch.ones_like(E_shift),
        create_graph=True
    )[0]  # (N, 3)

    # Energy periodicity term
    L_bc_E = torch.mean((E_R - E_shift) ** 2)

    # Force periodicity term (note: F = -∇U, so they should match)
    L_bc_F = torch.mean((F_pred - F_shift) ** 2)

    return L_bc_E + L_bc_F


def nve_loss(E_tot_traj):
    """
    Implements:
        L_NVE = (1 / (|B| T)) sum_{b,t} (E_tot^{(b)}(t) - E_tot^{(b)}(0))^2

    E_tot_traj can be:
      - (T,) for a single trajectory, or
      - (B, T) for a batch of trajectories.
    """
    if E_tot_traj is None:
        return None

    if E_tot_traj.dim() == 1:
        E0 = E_tot_traj[0]
        diff = E_tot_traj - E0
        return torch.mean(diff ** 2)

    # (B, T)
    E0 = E_tot_traj[:, 0:1]      # (B, 1), broadcast over T
    diff = E_tot_traj - E0       # (B, T)
    return torch.mean(diff ** 2)


def ic_loss(r0_pred, v0_pred, r0_true, v0_true):
    """
    Implements:
        L_IC = (1/N) sum_i [ ||r_i(0) - r_{i,0}||^2 + ||v_i(0) - v_{i,0}||^2 ]

    All arguments shape: (N, 3)
    """
    if any(x is None for x in (r0_pred, v0_pred, r0_true, v0_true)):
        return None

    pos_term = torch.mean((r0_pred - r0_true) ** 2)
    vel_term = torch.mean((v0_pred - v0_true) ** 2)
    return pos_term + vel_term

def total_loss(model, batch, weights, device, nve_batch=None):
    """
    batch = (R_np, Z_np, F_ref_np, E_ref_np, box_L_np)

    weights dict now expected to contain:
        "wF", "wE", "wS", "wBC", "wNVE"
    nve_batch: optional dict for NVE, see nve_loss_from_traj
    """

    # unpack batch
    R_np, Z_np, F_ref_np, E_ref_np, box_L_np = batch

    # convert to torch tensors
    R     = to_tensor(R_np, device).clone().detach().requires_grad_(True)  # (N, 3)
    Z     = torch.tensor(Z_np, dtype=torch.long, device=device)            # (N,)
    F_ref = to_tensor(F_ref_np, device)                                    # (N, 3)
    E_ref = torch.tensor(E_ref_np, dtype=torch.float32, device=device)     # scalar
    box_L = to_tensor(box_L_np, device)                                    # (3,)

    # compute energy
    E_pred = model(R, Z)

    if not E_pred.requires_grad:
        raise RuntimeError("E_pred lost its grad_fn — R did not enter computation graph.")

    # forces from autograd
    F_pred = -torch.autograd.grad(E_pred, R, create_graph=True)[0]

    # force + energy losses
    L_F = torch.mean((F_pred - F_ref)**2)
    L_E = torch.mean((E_pred - E_ref)**2)

    # symmetry (translation invariance)
    c = torch.randn(1, 3, device=device)
    L_sym = torch.mean((model(R + c, Z) - E_pred)**2)

    # periodic BC (energy only, as before)
    L_vec = box_L.view(1, 3)
    L_bc = torch.mean((model(R + L_vec, Z) - E_pred)**2)

    # ----- NVE term -----
    if nve_batch is not None and weights.get("wNVE", 0.0) != 0.0:
        L_NVE = nve_loss_from_traj(model, nve_batch, device)
    else:
        L_NVE = torch.tensor(0.0, device=device)

    total = (
        weights["wF"] * L_F +
        weights["wE"] * L_E +
        weights["wS"] * L_sym +
        weights["wBC"] * L_bc +
        weights["wNVE"] * L_NVE
    )

    logs = {
        "F": L_F.item(),
        "E": L_E.item(),
        "Sym": L_sym.item(),
        "PBC": L_bc.item(),
        "NVE": L_NVE.item(),
        "Total": total.item(),
    }

    return total, logs

'''
def total_loss(model, batch, weights, device):

    # unpack batch
    R_np, Z_np, F_ref_np, E_ref_np, box_L_np = batch

    # convert to torch tensors
    R     = to_tensor(R_np, device).clone().detach().requires_grad_(True)  # (N, 3)
    Z     = torch.tensor(Z_np, dtype=torch.long, device=device)             # (N,)
    F_ref = to_tensor(F_ref_np, device)                                    # (N, 3)
    E_ref = torch.tensor(E_ref_np, dtype=torch.float32, device=device)     # scalar
    box_L = to_tensor(box_L_np, device)                                    # (3,)

    # compute energy
    E_pred = model(R, Z)

    if not E_pred.requires_grad:
        raise RuntimeError("E_pred lost its grad_fn — R did not enter computation graph.")

    # forces from autograd
    F_pred = -torch.autograd.grad(E_pred, R, create_graph=True)[0]

    # losses
    L_F = torch.mean((F_pred - F_ref)**2)
    L_E = torch.mean((E_pred - E_ref)**2)

    # symmetry (translation invariance)
    c = torch.randn(1, 3, device=device)
    L_sym = torch.mean((model(R + c, Z) - E_pred)**2)

    # periodic BC
    L_vec = box_L.view(1, 3)
    L_bc = torch.mean((model(R + L_vec, Z) - E_pred)**2)

    total = (
        weights["wF"] * L_F +
        weights["wE"] * L_E +
        weights["wS"] * L_sym +
        weights["wBC"] * L_bc
    )

    logs = {
        "F": L_F.item(),
        "E": L_E.item(),
        "Sym": L_sym.item(),
        "PBC": L_bc.item(),
        "Total": total.item(),
    }

    return total, logs
'''