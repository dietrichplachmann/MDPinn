import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import CorrectionPotential


def load_split(npz_path, device):
    data = np.load(npz_path)

    R_all = torch.tensor(data["R"], dtype=torch.float32, device=device)       # (B, N, 3)
    Z_all = torch.tensor(data["Z"], dtype=torch.int64,  device=device)        # (B, N)
    F_all = torch.tensor(data["F_ref"], dtype=torch.float32, device=device)   # (B, N, 3)
    E_all = torch.tensor(data["E_ref"], dtype=torch.float32, device=device)   # (B,)
    box_L = torch.tensor(data["box_L"], dtype=torch.float32, device=device)   # (B, 3)

    return R_all, Z_all, F_all, E_all, box_L


def compute_train_loss_for_index(model, idx,
                                 R_all, Z_all, F_all, E_all, box_all,
                                 weights, device):
    """
    Training loss for a single configuration index.
    Uses forces + energies + symmetry + PBC.
    This mirrors train_minimal.py which works.
    """
    R = R_all[idx].clone().detach().requires_grad_(True)  # (N, 3)
    Z = Z_all[idx]                                        # (N,)
    F_ref = F_all[idx]                                    # (N, 3)
    E_ref = E_all[idx]                                    # scalar
    box_L = box_all[idx]                                  # (3,)

    # Energy from model
    E_pred = model(R, Z)
    if not E_pred.requires_grad:
        raise RuntimeError("E_pred does not require grad in TRAINING; R not in graph.")

    # Forces from autograd
    F_pred = -torch.autograd.grad(E_pred, R, create_graph=True)[0]

    # Force loss
    L_F = torch.mean((F_pred - F_ref) ** 2)
    # Energy loss
    L_E = torch.mean((E_pred - E_ref) ** 2)

    # Symmetry: translation invariance
    c = torch.randn(1, 3, device=device)
    E_shift = model(R + c, Z)
    L_sym = torch.mean((E_pred - E_shift) ** 2)

    # Periodic BC: energy periodic with box length
    L_vec = box_L.view(1, 3)
    E_img = model(R + L_vec, Z)
    L_bc = torch.mean((E_pred - E_img) ** 2)

    total = (
        weights["wF"]  * L_F +
        weights["wE"]  * L_E +
        weights["wS"]  * L_sym +
        weights["wBC"] * L_bc
    )

    logs = {
        "F":   L_F.detach().cpu().item(),
        "E":   L_E.detach().cpu().item(),
        "Sym": L_sym.detach().cpu().item(),
        "PBC": L_bc.detach().cpu().item(),
        "Total": total.detach().cpu().item(),
    }

    return total, logs


def compute_valid_loss_for_index(model, idx,
                                 R_all, Z_all, F_all, E_all, box_all,
                                 weights, device):
    """
    Validation loss for a single configuration index.
    DOES NOT use forces or autograd.grad (no gradients in validation).
    Uses only energy + symmetry + PBC.
    """

    # No requires_grad here, we are in eval mode
    R = R_all[idx]      # (N, 3)
    Z = Z_all[idx]      # (N,)
    E_ref = E_all[idx]  # scalar
    box_L = box_all[idx]

    # Energy prediction (no grad required)
    E_pred = model(R, Z)

    # Pure energy loss
    L_E = torch.mean((E_pred - E_ref) ** 2)

    # Symmetry: translation invariance
    c = torch.randn(1, 3, device=device)
    E_shift = model(R + c, Z)
    L_sym = torch.mean((E_pred - E_shift) ** 2)

    # Periodic BC: energy periodic with box length
    L_vec = box_L.view(1, 3)
    E_img = model(R + L_vec, Z)
    L_bc = torch.mean((E_pred - E_img) ** 2)

    # We ignore force loss in validation (no F_pred)
    L_F = torch.tensor(0.0, device=device)

    total = (
        weights["wF"]  * L_F +  # = 0, effectively
        weights["wE"]  * L_E +
        weights["wS"]  * L_sym +
        weights["wBC"] * L_bc
    )

    logs = {
        "F":   L_F.detach().cpu().item(),
        "E":   L_E.detach().cpu().item(),
        "Sym": L_sym.detach().cpu().item(),
        "PBC": L_bc.detach().cpu().item(),
        "Total": total.detach().cpu().item(),
    }

    return total, logs


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --- Load splits ---
    train_npz = os.path.join("data", "configs_train.npz")
    valid_npz = os.path.join("data", "configs_valid.npz")

    R_tr, Z_tr, F_tr, E_tr, box_tr = load_split(train_npz, device)
    R_va, Z_va, F_va, E_va, box_va = load_split(valid_npz, device)

    B_tr, N, _ = R_tr.shape
    B_va = R_va.shape[0]

    print(f"Train: {B_tr} configs, {N} atoms each")
    print(f"Valid: {B_va} configs")

    # --- Model & optimizer ---
    model = CorrectionPotential(hidden_dim=64).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # Loss weights
    weights = {
        "wF":  1.0,
        "wE":  0.1,
        "wS":  0.01,
        "wBC": 0.01,
    }

    n_epochs = 20
    steps_per_epoch = 50  # number of random train configs per epoch

    for epoch in range(1, n_epochs + 1):
        # ----- TRAIN -----
        model.train()
        train_loss_values = []

        for _ in range(steps_per_epoch):
            idx = torch.randint(0, B_tr, (1,)).item()
            loss, logs = compute_train_loss_for_index(
                model, idx,
                R_tr, Z_tr, F_tr, E_tr, box_tr,
                weights, device
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_values.append(logs["Total"])

        mean_train = sum(train_loss_values) / len(train_loss_values)

        # ----- VALIDATION -----
        model.eval()
        valid_loss_values = []
        with torch.no_grad():
            for idx in range(B_va):
                loss, logs = compute_valid_loss_for_index(
                    model, idx,
                    R_va, Z_va, F_va, E_va, box_va,
                    weights, device
                )
                valid_loss_values.append(logs["Total"])

        mean_valid = sum(valid_loss_values) / len(valid_loss_values)

        print(f"Epoch {epoch:03d} | Train {mean_train:.6f} | Valid {mean_valid:.6f}")

    torch.save(model.state_dict(), "trained_model.pth")
    print("Training complete → saved to trained_model.pth")


if __name__ == "__main__":
    train()
