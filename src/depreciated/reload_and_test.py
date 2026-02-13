import os
import torch
import numpy as np

from model import CorrectionPotential
from checkpointing import load_checkpoint
from test_model import load_split


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Path to a specific checkpoint
    # edit this to change run
    ckpt_path = os.path.join("checkpoints", "run_20251202_153045.pt")

    ckpt = load_checkpoint(ckpt_path, map_location=device)

    config = ckpt["config"]
    mp = config["model_params"]

    # Rebuild model
    model = CorrectionPotential(
        emb_dim=mp["emb_dim"],
        hidden_dim=mp["hidden_dim"],
        n_rbf=mp["n_rbf"],
        cutoff=mp["cutoff"],
        max_Z=mp["max_Z"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # --- Evaluate on test set, similar to test_model.py ---
    from test_model import load_split as load_split_test

    test_npz = config["data"]["test_npz"]
    R_te, Z_te, F_te, E_te, box_te = load_split_test(test_npz, device)
    B, N, _ = R_te.shape
    print(f"Test set: {B} configs, {N} atoms each")

    energy_abs_errors = []
    energy_sq_errors = []
    force_abs_errors = []
    force_sq_errors = []

    for idx in range(B):
        R = R_te[idx].clone().detach().requires_grad_(True)
        Z = Z_te[idx]
        F_ref = F_te[idx]
        E_ref = E_te[idx]

        E_pred = model(R, Z)
        F_pred = -torch.autograd.grad(E_pred, R)[0]

        e_err = (E_pred - E_ref).detach().cpu().item()
        energy_abs_errors.append(abs(e_err))
        energy_sq_errors.append(e_err**2)

        f_diff = (F_pred - F_ref).detach().cpu().numpy().reshape(-1)
        force_abs_errors.extend(np.abs(f_diff))
        force_sq_errors.extend(f_diff**2)

    energy_abs_errors = np.array(energy_abs_errors)
    energy_sq_errors = np.array(energy_sq_errors)
    force_abs_errors = np.array(force_abs_errors)
    force_sq_errors = np.array(force_sq_errors)

    E_MAE  = energy_abs_errors.mean()
    E_RMSE = np.sqrt(energy_sq_errors.mean())
    F_MAE  = force_abs_errors.mean()
    F_RMSE = np.sqrt(force_sq_errors.mean())

    print("\n=== Test set metrics for this checkpoint ===")
    print("Checkpoint:", ckpt_path)
    print(f"Energy MAE  : {E_MAE:.6f}")
    print(f"Energy RMSE : {E_RMSE:.6f}")
    print(f"Force MAE   : {F_MAE:.6f}")
    print(f"Force RMSE  : {F_RMSE:.6f}")


if __name__ == "__main__":
    main()
