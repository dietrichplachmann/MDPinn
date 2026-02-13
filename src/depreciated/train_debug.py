import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

# -----------------------------
# Model: same debug-safe version
# -----------------------------
class CorrectionPotential(nn.Module):
    """
    Minimal NN potential that DEFINITELY depends on atomic positions R.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.energy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, R, Z):
        # Z ignored for now; we just want E(R) with gradients
        h = self.coord_mlp(R)           # (N, hidden_dim)
        per_atom_E = self.energy_mlp(h) # (N, 1)
        return per_atom_E.sum()         # scalar


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------------
    # 1. Load NPZ data directly
    # -----------------------------
    npz_path = os.path.join("data", "configs_train.npz")
    data = np.load(npz_path)

    R_all = data["R"]       # (B, N, 3)
    Z_all = data["Z"]       # (B, N)
    F_all = data["F_ref"]   # (B, N, 3)
    E_all = data["E_ref"]   # (B,)
    box_all = data["box_L"] # (B, 3)

    B, N, _ = R_all.shape
    print(f"Loaded {B} configurations with {N} atoms each.")

    # Turn them into torch tensors
    R_all = torch.tensor(R_all, dtype=torch.float32, device=device)
    Z_all = torch.tensor(Z_all, dtype=torch.int64, device=device)
    F_all = torch.tensor(F_all, dtype=torch.float32, device=device)
    E_all = torch.tensor(E_all, dtype=torch.float32, device=device)
    box_all = torch.tensor(box_all, dtype=torch.float32, device=device)

    # -----------------------------
    # 2. Set up model and optimizer
    # -----------------------------
    model = CorrectionPotential(hidden_dim=64).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # Loss weights
    wF = 1.0   # forces
    wE = 0.1   # energies
    wS = 0.01  # symmetry
    wBC = 0.01 # periodic

    # -----------------------------
    # 3. Training loop (random configs)
    # -----------------------------
    n_epochs = 10
    steps_per_epoch = 50

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []

        for step in range(steps_per_epoch):
            # sample a random configuration index
            idx = torch.randint(0, B, (1,)).item()

            R = R_all[idx].clone().detach().requires_grad_(True)  # (N, 3)
            Z = Z_all[idx]                                        # (N,)
            F_ref = F_all[idx]                                    # (N, 3)
            E_ref = E_all[idx]                                    # scalar
            box_L = box_all[idx]                                  # (3,)

            # ----- energy and forces -----
            E_pred = model(R, Z)  # scalar tensor, must require grad
            if not E_pred.requires_grad:
                raise RuntimeError("E_pred does not require grad; something is wrong.")

            F_pred = -torch.autograd.grad(E_pred, R, create_graph=True)[0]  # (N, 3)

            # ----- losses -----
            L_F = torch.mean((F_pred - F_ref) ** 2)
            L_E = torch.mean((E_pred - E_ref) ** 2)

            # Symmetry: translation invariance
            c = torch.randn(1, 3, device=device)
            E_shift = model(R + c, Z)
            L_sym = torch.mean((E_pred - E_shift) ** 2)

            # PBC: periodic images
            L_vec = box_L.view(1, 3)
            E_img = model(R + L_vec, Z)
            L_bc = torch.mean((E_pred - E_img) ** 2)

            total = wF * L_F + wE * L_E + wS * L_sym + wBC * L_bc

            opt.zero_grad()
            total.backward()
            opt.step()

            epoch_losses.append(total.detach().cpu().item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch:03d} | Mean loss: {mean_loss:.6f}")

    torch.save(model.state_dict(), "trained_model_minimal.pth")
    print("Saved model to trained_model_minimal.pth")

if __name__ == "__main__":
    main()
