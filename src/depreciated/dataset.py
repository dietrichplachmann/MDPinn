import numpy as np
from torch.utils.data import Dataset

class WaterConfigDataset(Dataset):
    """
    Returns numpy arrays, NOT torch tensors.
    This avoids autograd-breaking conversions inside DataLoader.
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)

        self.R = data["R"]           # (B, N, 3) numpy
        self.Z = data["Z"]           # (B, N)
        self.F_ref = data["F_ref"]   # (B, N, 3)
        self.E_ref = data["E_ref"]   # (B,)
        self.box_L = data["box_L"]   # (B, 3)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, idx):
        return (
            self.R[idx],        # numpy
            self.Z[idx],        # numpy
            self.F_ref[idx],    # numpy
            self.E_ref[idx],    # numpy scalar
            self.box_L[idx]     # numpy
        )
