#!/usr/bin/env python
"""Investigates Q3 (paper/main.tex Section 5.3, sec:q3): why does
water_absolute+momentum produce hotter NVE rollouts than water_absolute,
despite being trained to REDUCE per-molecule net force/torque - which
naively should make dynamics more stable, not less (src/run_rollout_study.py
found momentum's plateau temperature exceeds absolute's in 10/10 replicate
trials across two independent axes).

Compares the two conditions' predicted forces directly (not accuracy against
DFT ground truth - that's evaluate_waterbox.py's job) at the same real
held-out configurations, decomposed per molecule into:
  - F_net: the molecule's net force (sum over its 3 atoms) - exactly the
    quantity physics_losses.per_fragment_momentum_loss's linear term
    penalizes, and what the momentum loss is directly trained to shrink.
  - F_internal = F_i - F_net/3 for each atom i in the molecule - the
    "shape-distorting" component, which sums to zero across the molecule's
    own 3 atoms by construction (pure algebraic decomposition, not a model
    property) and is therefore invisible to the momentum loss entirely: a
    model can drive F_net to ~0 while F_internal stays large, and the loss
    would never know. That's exactly the kind of force that injects kinetic
    energy into internal (bond-stretch/bend) vibrational modes without
    violating momentum conservation at all - candidate mechanism 3 in the
    paper's Q3 discussion.

If water_absolute+momentum shows smaller F_net (expected - that's the
trained objective) but a LARGER internal-force share of its total force
magnitude than water_absolute at the same configurations, that's direct,
mechanistic evidence for candidate 3: the momentum loss gets satisfied by
redistributing force within a molecule rather than by producing genuinely
gentler forces overall - and that redistributed force is exactly what would
show up as extra heating once integrated forward in a real rollout.

This does NOT test the other two candidates in the paper (a checkpoint-
selection artifact specific to these seed-0 checkpoints; a systematic
force-accuracy tradeoff invisible to static force MAE) - those need
different evidence (re-running with different seeds; comparing against
ground-truth DFT forces rather than the two models against each other).

Usage:
    python src/analyze_force_decomposition.py \\
        --ckpt-absolute checkpoints/waterbox_study/water_absolute/seed0/best_model.ckpt \\
        --ckpt-momentum "checkpoints/waterbox_study/water_absolute+momentum/seed0/best_model.ckpt"

IMPORTANT - written without torchmdnet installed locally (same caveat as
every other water-box script in this repo). Nothing here has been executed
yet.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluate_waterbox import load_waterbox_checkpoint
from structural_metrics import infer_molecule_groups
from waterbox_data import load_waterbox_dataset, random_split


def _predict_forces(model, z, pos, box, device):
    pos_req = pos.detach().clone().requires_grad_(True)
    batch = torch.zeros(z.shape[0], dtype=torch.long, device=device)
    with torch.enable_grad():
        _, forces = model(z, pos_req, batch=batch, box=box)
    return forces.detach()


def decompose_forces(forces, group_ids, num_molecules):
    """Per molecule m with atoms A_m: F_net_m = sum_{i in A_m} F_i;
    F_internal_i = F_i - F_net_m / |A_m| for each atom i in A_m. Sums to
    zero across each molecule's own atoms by construction - a pure algebraic
    split of the force field, not a model property, so it applies equally
    to either condition's predictions.

    Returns (net_force_mag_per_molecule, internal_force_mag_per_atom) as
    numpy arrays.
    """
    device = forces.device
    counts = torch.zeros(num_molecules, device=device)
    counts.index_add_(0, group_ids, torch.ones(len(group_ids), device=device))

    net_force = torch.zeros(num_molecules, 3, device=device)
    net_force.index_add_(0, group_ids, forces)
    net_force_mag = net_force.norm(dim=1)

    internal_force = forces - (net_force / counts.unsqueeze(1))[group_ids]
    internal_force_mag = internal_force.norm(dim=1)
    return net_force_mag.cpu().numpy(), internal_force_mag.cpu().numpy()


def analyze(ckpt_absolute, ckpt_momentum, data_root="./data", n_configs=6, seed=42, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "water_absolute": load_waterbox_checkpoint(ckpt_absolute, device=device),
        "water_absolute+momentum": load_waterbox_checkpoint(ckpt_momentum, device=device),
    }

    full_dataset = load_waterbox_dataset(data_root=data_root)
    _, _, test_data = random_split(full_dataset, seed=seed)
    n_configs = min(n_configs, len(test_data))

    raw_by_condition = {name: [] for name in models}
    net_by_condition = {name: [] for name in models}
    internal_by_condition = {name: [] for name in models}

    for idx in range(n_configs):
        sample = test_data[idx].to(device)
        z, pos, box = sample.z, sample.pos.float(), getattr(sample, "box", None)
        group_ids = infer_molecule_groups(z, pos, box=box).to(device)
        num_molecules = int(group_ids.max().item()) + 1

        for name, model in models.items():
            forces = _predict_forces(model, z, pos, box, device)
            net_mag, internal_mag = decompose_forces(forces, group_ids, num_molecules)

            raw_by_condition[name].append(forces.norm(dim=1).cpu().numpy())
            net_by_condition[name].append(net_mag)
            internal_by_condition[name].append(internal_mag)

    print(
        f"{'condition':28s} {'raw |F| (eV/A)':20s} {'net |F_mol| (eV/A)':22s} "
        f"{'internal |F| (eV/A)':22s} internal/raw"
    )
    summary = {}
    for name in models:
        raw = np.concatenate(raw_by_condition[name])
        net = np.concatenate(net_by_condition[name])
        internal = np.concatenate(internal_by_condition[name])
        share = float(internal.mean() / raw.mean())
        summary[name] = {
            "raw_force_mag_mean": float(raw.mean()),
            "raw_force_mag_std": float(raw.std()),
            "net_force_mag_mean": float(net.mean()),
            "net_force_mag_std": float(net.std()),
            "internal_force_mag_mean": float(internal.mean()),
            "internal_force_mag_std": float(internal.std()),
            "internal_force_share": share,
        }
        print(
            f"{name:28s} {raw.mean():7.4f}+/-{raw.std():6.4f}   "
            f"{net.mean():9.4f}+/-{net.std():7.4f}   "
            f"{internal.mean():9.4f}+/-{internal.std():7.4f}   {share:.4f}"
        )

    print(
        "\nIf water_absolute+momentum shows a smaller net-force magnitude (expected - that's "
        "the trained objective) but a HIGHER internal-force share than water_absolute, that "
        "supports candidate mechanism 3 (paper/main.tex Section 5.3, sec:q3): momentum "
        "conservation satisfied by redistributing force within molecules rather than by "
        "reducing force error overall - exactly the kind of force that would inject kinetic "
        "energy into internal vibrational modes during a real rollout, invisible to any "
        "per-molecule net-force metric. If the internal-force share is similar or lower for "
        "momentum despite the hotter rollouts, this mechanism isn't it - look at the other two "
        "candidates in sec:q3 instead."
    )

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-absolute", type=str, required=True)
    parser.add_argument("--ckpt-momentum", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--n-configs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analyze(
        ckpt_absolute=args.ckpt_absolute,
        ckpt_momentum=args.ckpt_momentum,
        data_root=args.data_root,
        n_configs=args.n_configs,
        seed=args.seed,
    )
