import os
import numpy as np
import torch

from src.depreciated.model import CorrectionPotential


def load_split(npz_path, device):
    data = np.load(npz_path)

    R_all = torch.tensor(data["R"], dtype=torch.float32, device=device)       # (B, N, 3)
    Z_all = torch.tensor(data["Z"], dtype=torch.int64,  device=device)        # (B, N)
    F_all = torch.tensor(data["F_ref"], dtype=torch.float32, device=device)   # (B, N, 3)
    E_all = torch.tensor(data["E_ref"], dtype=torch.float32, device=device)   # (B,)
    box_L = torch.tensor(data["box_L"], dtype=torch.float32, device=device)   # (B, 3)

    return R_all, Z_all, F_all, E_all, box_L


def evaluate_on_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ---- Load test data ----
    test_npz = os.path.join("data", "configs_test.npz")
    R_te, Z_te, F_te, E_te, box_te = load_split(test_npz, device)
    B, N, _ = R_te.shape
    print(f"Test set: {B} configs, {N} atoms each")

    # ---- Load trained model ----
    model = CorrectionPotential(hidden_dim=64).to(device)
    state_path = "trained_model.pth"
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Could not find {state_path}. Train the model first.")
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()

    # ---- Metrics accumulators ----
    # Energies
    energy_abs_errors = []
    energy_sq_errors  = []

    # Forces: per-component
    force_abs_errors = []
    force_sq_errors  = []

    # ---- Loop over test configs ----
    for idx in range(B):
        # Positions must require grad to compute forces
        R = R_te[idx].clone().detach().requires_grad_(True)  # (N, 3)
        Z = Z_te[idx]                                        # (N,)
        F_ref = F_te[idx]                                    # (N, 3)
        E_ref = E_te[idx]                                    # scalar

        # Energy prediction
        E_pred = model(R, Z)

        # Forces from autograd
        F_pred = -torch.autograd.grad(E_pred, R)[0]  # (N, 3)

        # ---- Energy errors ----
        e_err = (E_pred - E_ref).detach().cpu().item()
        energy_abs_errors.append(abs(e_err))
        energy_sq_errors.append(e_err**2)

        # ---- Force errors (per component) ----
        f_diff = (F_pred - F_ref).detach().cpu().numpy().reshape(-1)  # all components
        force_abs_errors.extend(np.abs(f_diff))
        force_sq_errors.extend(f_diff**2)

    # ---- Compute metrics ----
    energy_abs_errors = np.array(energy_abs_errors)
    energy_sq_errors  = np.array(energy_sq_errors)
    force_abs_errors  = np.array(force_abs_errors)
    force_sq_errors   = np.array(force_sq_errors)

    E_MAE  = energy_abs_errors.mean()
    E_RMSE = np.sqrt(energy_sq_errors.mean())

    F_MAE  = force_abs_errors.mean()
    F_RMSE = np.sqrt(force_sq_errors.mean())

    print("\n=== Test set metrics ===")
    print(f"Energy MAE  : {E_MAE:.6f}")
    print(f"Energy RMSE : {E_RMSE:.6f}")
    print(f"Force MAE   : {F_MAE:.6f}")
    print(f"Force RMSE  : {F_RMSE:.6f}")


if __name__ == "__main__":
    evaluate_on_test()
