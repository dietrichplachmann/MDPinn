#!/usr/bin/env python
"""Dataset loading for the periodic liquid-water study.

Loads torchmdnet.datasets.WaterBox (Cheng et al. revPBE0-D3 liquid water,
1593 configurations, periodic box vectors included - see the project plan for
citation details) and splits it randomly rather than contiguously.

This is deliberately NOT built on data_splits.contiguous_split: that helper
assumes one long single-molecule trajectory where dataset index order is time
order (true for MD17's aspirin/benzene/ethanol trajectories, which is what it
was built for). WaterBox is reported to mix ~1000 classical-MD configurations
with ~593 path-integral-MD configurations; a contiguous split would risk
putting most/all of one kind in a single split. A seeded random split spreads
both kinds across train/val/test with high probability without needing to
hardcode the exact classical/PIMD boundary index, which isn't exposed as
metadata on the dataset object.

IMPORTANT: this module was written without torchmdnet installed locally (this
Windows checkout has no torchmdnet/CUDA - see the project's own notes on this).
The constructor signature and attribute names below (`WaterBox(root=...)`,
per-sample z/pos/y/neg_dy/box with box shape (1,3,3), len(dataset) == 1593)
are taken from torchmd-net's public source on GitHub, not verified against an
actual installed copy. Run `python src/waterbox_data.py` directly on the
training box first - it just loads the dataset and prints shapes - before
relying on anything downstream.
"""

from __future__ import annotations

import random

from torch.utils.data import Subset


def load_waterbox_dataset(data_root: str = "./data"):
    """Load the WaterBox dataset. Raises ImportError with a clear message if
    the installed torchmdnet version doesn't expose this dataset class."""
    try:
        from torchmdnet.datasets import WaterBox
    except ImportError as exc:
        raise ImportError(
            "torchmdnet.datasets.WaterBox not found. This module expects a "
            "torchmd-net version that includes the WaterBox dataset class "
            "(see torchmdnet/datasets/water.py in the torchmd-net source) - "
            "check the installed version supports it before debugging further."
        ) from exc

    # torchmdnet's shipped download URL (Materials Cloud's old record_id= API)
    # 404s - that endpoint was retired when Materials Cloud migrated to
    # short-ID URLs. Same dataset, same record (2018.0020/v1, Cheng et al.,
    # DOI 10.24435/materialscloud:2018.0020/v1), current address instead of
    # patching the installed package.
    WaterBox.url = "https://www.materialscloud.org/records/eg3pn-1fw83/files/training-set.zip?download=1"

    return WaterBox(root=data_root)


def random_split(dataset, train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42):
    """Seeded random train/val/test split (not contiguous - see module docstring).

    Returns (train_subset, val_subset, test_subset), each a torch.utils.data.Subset.
    """
    n_samples = len(dataset)
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)

    train_end = int(train_frac * n_samples)
    val_end = train_end + int(val_frac * n_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


if __name__ == "__main__":
    dataset = load_waterbox_dataset()
    print(f"len(dataset) = {len(dataset)}")
    sample = dataset[0]
    print(f"sample.z.shape = {tuple(sample.z.shape)}")
    print(f"sample.pos.shape = {tuple(sample.pos.shape)}")
    print(f"sample.y = {sample.y}")
    print(f"sample.neg_dy.shape = {tuple(sample.neg_dy.shape)}")
    box = getattr(sample, "box", None)
    print(f"sample.box.shape = {tuple(box.shape) if box is not None else None}")

    train_data, val_data, test_data = random_split(dataset)
    print(f"train/val/test sizes: {len(train_data)}/{len(val_data)}/{len(test_data)}")
