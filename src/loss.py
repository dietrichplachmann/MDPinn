import torch

def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

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
