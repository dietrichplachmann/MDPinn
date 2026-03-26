#!/usr/bin/env python
"""Print baseline-only energy/force statistics on MD17 aspirin frames."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

from torchmdnet.datasets import MD17

from baseline_potential import reference_energy_forces_batched
from data_splits import contiguous_split

# PyTorch 2.6+ compatibility for TorchMD-Net processed dataset files.
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


def summarize_errors(values):
    values = [float(v) for v in values]
    abs_values = [abs(v) for v in values]
    squared = [v * v for v in values]
    return {
        "mean": mean(values),
        "mae": mean(abs_values),
        "rmse": (mean(squared)) ** 0.5,
        "max_abs": max(abs_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()

    dataset = MD17(root="./data", molecules=args.molecule)
    train_data, val_data, test_data = contiguous_split(dataset)
    split_map = {"train": train_data, "val": val_data, "test": test_data}
    subset = split_map[args.split]

    loader = GeometricDataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    energy_errors = []
    force_errors = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        batch.pos = batch.pos.detach().clone()
        u_ref, f_ref = reference_energy_forces_batched(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch,
            molecule=args.molecule,
            box_l=batch.box if "box" in batch else None,
        )
        energy_errors.extend((u_ref.detach().cpu() - batch.y.view(-1).detach().cpu()).tolist())
        force_errors.extend((f_ref.detach().cpu() - batch.neg_dy.detach().cpu()).reshape(-1).tolist())

    print("Energy baseline residual stats:", summarize_errors(energy_errors))
    print("Force baseline residual stats:", summarize_errors(force_errors))


if __name__ == "__main__":
    main()
