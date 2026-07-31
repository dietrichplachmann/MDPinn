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

# CODATA 2018 atomic-unit conversion factors. The raw dataset_1593.xyz shipped
# in the WaterBox download is a hand-serialized dump of native CP2K/Quickstep
# AIMD output (Bohr/Hartree throughout) wearing a quippy-style extended-xyz
# header (Properties=/Lattice=/nneightol=) - it was never actually converted
# to the Angstrom/eV convention that header style normally implies. Confirmed
# empirically (2026-07-31): the raw Lattice diagonal (23.465) only matches the
# physically expected ~12.42 A box size for 64 liquid-density water molecules
# after x0.529177 (Bohr->A); the same factor turns the nearest O-H neighbor
# distances into real bond lengths (~0.98-1.12 A). Force magnitudes only land
# in the physically normal ~1-3 eV/A range for a thermally-sampled AIMD frame
# after the matching Hartree/Bohr->eV/A conversion - read directly as eV/A
# they're implausibly small (looks like a relaxed structure, not a liquid
# ensemble). See project_mdpinn.md for the full derivation.
_BOHR_TO_ANGSTROM = 0.529177210903
_HARTREE_TO_EV = 27.211386245988
_HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = _HARTREE_TO_EV / _BOHR_TO_ANGSTROM


def _convert_atomic_units_to_ev_angstrom(data):
    """Per-sample transform: rescale a raw WaterBox Data object from the
    dataset's native Bohr/Hartree units to the Angstrom/eV convention every
    other part of this codebase (MD17, structural_metrics, physics_losses)
    assumes."""
    data.pos = data.pos * _BOHR_TO_ANGSTROM
    box = getattr(data, "box", None)
    if box is not None:
        data.box = box * _BOHR_TO_ANGSTROM
    data.y = data.y * _HARTREE_TO_EV
    data.neg_dy = data.neg_dy * _HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    return data


def load_waterbox_dataset(data_root: str = "./data"):
    """Load the WaterBox dataset, converted to Angstrom/eV. Raises ImportError
    with a clear message if the installed torchmdnet version doesn't expose
    this dataset class."""
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
    WaterBox.url = "https://archive.materialscloud.org/records/eg3pn-1fw83/files/training-set.zip?download=1"

    return WaterBox(root=data_root, transform=_convert_atomic_units_to_ev_angstrom)


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
