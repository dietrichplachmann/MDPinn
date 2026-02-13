import torch
import torch.nn as nn


class CorrectionPotential(nn.Module):
    """
    Pairwise RBF NN potential with cutoff:

        E = sum_{i<j, r_ij < r_c} f_theta( e(Z_i), e(Z_j), RBF(r_ij) )

    - Z: atomic numbers (1 = H, 8 = O, etc.)
    - R: positions in nm
    - Uses a radial cutoff and RBFs for stable learning.
    """

    def __init__(
        self,
        emb_dim: int = 16,
        hidden_dim: int = 64,
        n_rbf: int = 32,
        #cutoff: float = 0.6,   # nm
        cutoff: float = 6.0,    # angstroms
        max_Z: int = 100,
    ):
        super().__init__()

        self.cutoff = cutoff
        self.n_rbf = n_rbf

        # Atom-type embedding
        self.embedding = nn.Embedding(max_Z, emb_dim)

        # RBF centers between 0 and cutoff
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("rbf_centers", centers)

        # Width (gamma) for Gaussians
        # Choose so neighboring RBFs overlap reasonably
        self.gamma = 10.0 / (cutoff ** 2)

        # Pairwise MLP input: e_i, e_j, RBF(r_ij)
        pair_in_dim = emb_dim * 2 + n_rbf

        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # scalar pair energy
        )

    def _rbf(self, r: torch.Tensor) -> torch.Tensor:
        """
        Compute radial basis expansion of distances.

        r: (...,) distances
        returns: (..., n_rbf)
        """
        # (..., 1) - (..., n_rbf) -> (..., n_rbf)
        diff = r.unsqueeze(-1) - self.rbf_centers  # broadcast over centers
        return torch.exp(-self.gamma * diff ** 2)

    def forward(self, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        R: (N, 3) positions
        Z: (N,)   atomic numbers
        Returns: scalar total energy
        """
        device = R.device
        N = R.shape[0]

        # Atom embeddings
        e = self.embedding(Z)  # (N, emb_dim)

        # Pairwise vectors and distances
        Ri = R.unsqueeze(1)    # (N, 1, 3)
        Rj = R.unsqueeze(0)    # (1, N, 3)
        dR = Ri - Rj           # (N, N, 3)
        r_ij = torch.norm(dR, dim=-1)  # (N, N)

        # Mask: i < j and r_ij < cutoff
        tri_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
        cutoff_mask = r_ij < self.cutoff
        mask = tri_mask & cutoff_mask  # (N, N) boolean
        num_pairs = mask.sum().item()

        if not mask.any():
            # No pairs in cutoff -> define zero energy that still depends on R
            # so autograd can compute dE/dR = 0 without error.
            print("WARNING: no pairs within cutoff; num_pairs =", num_pairs)
            return (R.sum() * 0.0)
        #else:
        #    print("num_pairs within cutoff:", num_pairs)

        # Select valid pairs
        r_valid = r_ij[mask]  # (P,)
        ei = e.unsqueeze(1).expand(N, N, -1)[mask]  # (P, emb_dim)
        ej = e.unsqueeze(0).expand(N, N, -1)[mask]  # (P, emb_dim)

        # RBF features for distances
        rbf_feat = self._rbf(r_valid)  # (P, n_rbf)

        # Concatenate features: [e_i, e_j, RBF(r_ij)]
        pair_feat = torch.cat([ei, ej, rbf_feat], dim=-1)  # (P, 2*emb_dim + n_rbf)

        # Pair energies
        pair_E = self.pair_mlp(pair_feat)  # (P, 1)

        # Total energy
        E_total = pair_E.sum()

        return E_total
