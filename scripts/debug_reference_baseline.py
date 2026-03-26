#!/usr/bin/env python
"""Debug one-frame aspirin baseline energy components and force residuals."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torchmdnet.datasets import MD17

from baseline_potential import (
    debug_aspirin_reference_components,
    load_reference_energy_offset_eV,
    reference_energy_forces,
)
from data_splits import contiguous_split

# PyTorch 2.6+ compatibility for TorchMD-Net processed dataset files.
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


def _subset_for_split(dataset, split_name):
    train_data, val_data, test_data = contiguous_split(dataset)
    return {"train": train_data, "val": val_data, "test": test_data}[split_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--index", type=int, default=0, help="Index within the chosen split.")
    args = parser.parse_args()

    dataset = MD17(root="./data", molecules=args.molecule)
    subset = _subset_for_split(dataset, args.split)
    sample = subset[args.index]

    pos = sample.pos.detach().clone()
    z = sample.z.detach().clone()
    energy_true = float(sample.y.view(-1)[0].item())
    force_true = sample.neg_dy.detach().clone()
    energy_offset = load_reference_energy_offset_eV(args.molecule)

    components = debug_aspirin_reference_components(
        pos,
        z,
        box_l=sample.box if "box" in sample else None,
        energy_offset_eV=energy_offset,
    )
    energy_ref, force_ref = reference_energy_forces(
        z=z,
        pos=pos,
        molecule=args.molecule,
        box_l=sample.box if "box" in sample else None,
        energy_offset_eV=energy_offset,
    )

    force_residual = force_ref - force_true
    print("Reference components (eV):")
    for key, value in components.items():
        print(f"  {key}: {float(value.item())}")

    print("Reference vs target:")
    print(f"  energy_ref_eV: {float(energy_ref.item())}")
    print(f"  energy_true_eV: {energy_true}")
    print(f"  energy_residual_eV: {float(energy_ref.item()) - energy_true}")
    print(f"  force_mae_eV_per_A: {float(force_residual.abs().mean().item())}")
    print(f"  force_rmse_eV_per_A: {float(torch.sqrt((force_residual ** 2).mean()).item())}")
    print(f"  force_max_abs_eV_per_A: {float(force_residual.abs().max().item())}")


if __name__ == "__main__":
    main()
