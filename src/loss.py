import torch

def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


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

def total_loss(model, batch, weights, device, traj=None, ic=None):
    """
    batch = (R_np, Z_np, F_ref_np, E_ref_np, box_L_np)

    Optional:
      traj: dict, may contain
          - "E_tot_traj_np": np.ndarray with shape (T,) or (B, T)
      ic: dict, may contain
          - "r0_pred", "v0_pred", "r0_true", "v0_true" (all np arrays (N,3))

    weights should contain:
      "wF", "wE", "wS", "wNVE", "wIC", "wBC"
      (and you can add more, like "wTrans" if you keep the random-translation term).
    """

    # unpack batch
    R_np, Z_np, F_ref_np, E_ref_np, box_L_np = batch

    # convert to torch tensors
    R     = to_tensor(R_np, device).clone().detach().requires_grad_(True)  # (N, 3)
    Z     = torch.tensor(Z_np, dtype=torch.long, device=device)            # (N,)
    F_ref = to_tensor(F_ref_np, device)                                    # (N, 3)
    E_ref = torch.tensor(E_ref_np, dtype=torch.float32, device=device)     # scalar or (B,)
    box_L = to_tensor(box_L_np, device)                                    # (3,)

    # --- energy ---
    E_pred = model(R, Z)  # U_phi(R,Z)

    if not E_pred.requires_grad:
        raise RuntimeError("E_pred lost its grad_fn — R did not enter computation graph.")

    # --- forces from autograd ---
    F_pred = -torch.autograd.grad(
        E_pred, R,
        grad_outputs=torch.ones_like(E_pred),
        create_graph=True
    )[0]  # (N, 3)

    # =========================
    #  L_F: force matching term
    # =========================
    L_F = torch.mean((F_pred - F_ref) ** 2)

    # ===================================
    #  L_E: energy matching (hybrid vs ref)
    # ===================================
    L_E = torch.mean((E_pred - E_ref) ** 2)

    # ==========================================
    #  L_sym: linear + angular momentum symmetry
    # ==========================================
    L_sym = momentum_symmetry_loss(R, F_pred)

    # (OPTIONAL) keep your original translation invariance term
    # if you still like it; give it its own weight "wTrans".
    # c = torch.randn(1, 3, device=device)
    # L_trans = torch.mean((model(R + c, Z) - E_pred) ** 2)

    # =====================
    #  L_BC: periodic BCs
    # =====================
    L_BC = periodic_bc_loss(model, R, Z, box_L, F_pred)

    # ======================
    #  L_NVE: energy drift
    # ======================
    L_NVE = torch.tensor(0.0, device=device)
    if traj is not None and "E_tot_traj_np" in traj:
        E_tot_traj = to_tensor(traj["E_tot_traj_np"], device)
        L_NVE_val = nve_loss(E_tot_traj)
        if L_NVE_val is not None:
            L_NVE = L_NVE_val

    # ==============================
    #  L_IC: initial conditions term
    # ==============================
    L_IC = torch.tensor(0.0, device=device)
    if ic is not None:
        r0_pred = ic.get("r0_pred", None)
        v0_pred = ic.get("v0_pred", None)
        r0_true = ic.get("r0_true", None)
        v0_true = ic.get("v0_true", None)

        # convert if provided as numpy
        def _maybe_to_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.float32)
            return to_tensor(x, device)

        r0_pred = _maybe_to_tensor(r0_pred)
        v0_pred = _maybe_to_tensor(v0_pred)
        r0_true = _maybe_to_tensor(r0_true)
        v0_true = _maybe_to_tensor(v0_true)

        L_IC_val = ic_loss(r0_pred, v0_pred, r0_true, v0_true)
        if L_IC_val is not None:
            L_IC = L_IC_val

    # ===================
    #  TOTAL LOSS
    # ===================
    total = (
        weights["wF"]   * L_F   +
        weights["wE"]   * L_E   +
        weights["wS"]   * L_sym +
        weights["wNVE"] * L_NVE +
        weights["wIC"]  * L_IC  +
        weights["wBC"]  * L_BC
        # + weights.get("wTrans", 0.0) * L_trans   # if you keep translation invariance separately
    )

    logs = {
        "F":     L_F.item(),
        "E":     L_E.item(),
        "Sym":   L_sym.item(),
        "NVE":   L_NVE.item(),
        "IC":    L_IC.item(),
        "PBC":   L_BC.item(),
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