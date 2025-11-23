import torch
import torch.nn as nn

class CorrectionPotential(nn.Module):
    """
    NN potential that depends on both:
      - atomic positions R
      - atomic types Z
    Outputs a scalar energy.
    """

    def __init__(self, hidden_dim=64, max_Z=100):
        super().__init__()

        # Map atomic number (Z) to a learned embedding vector
        self.embedding = nn.Embedding(max_Z, hidden_dim)

        # Map Cartesian coords to hidden_dim
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Per-atom output head
        self.energy_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, R, Z):
        """
        R: tensor (N, 3)
        Z: tensor (N,)  ints like [8,1,1,...]
        """

        # Atom-type embedding → (N, hidden_dim)
        h_Z = self.embedding(Z)

        # Position embedding → (N, hidden_dim)
        h_R = self.coord_mlp(R)

        # Combine atom identity + positional info
        h = h_Z + h_R

        # Per-atom energies
        per_atom_E = self.energy_mlp(h)  # (N, 1)

        # Total energy
        return per_atom_E.sum()
