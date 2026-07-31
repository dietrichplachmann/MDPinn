#!/usr/bin/env python
"""Evaluation for the periodic water-box study.

Primary metric: per-molecule momentum violation on held-out configurations -
this is the actual test of the hypothesis (does the per-fragment momentum loss
reduce a genuinely nonzero quantity), which the aspirin study couldn't test
because the answer there was trivially "already ~0" for a single isolated
molecule (see physics_losses.per_fragment_momentum_loss's docstring for why a
multi-molecule periodic system is different).

Secondary: ordinary energy/force accuracy, to check the per-fragment
constraint isn't costing accuracy relative to water_absolute.

Expected sanity check before trusting any comparison: running this against a
water_absolute (momentum_weight=0) checkpoint should show a clearly nonzero
mean_per_molecule_momentum_violation - if it's already ~0 with no training
pressure on it, that would undercut the premise of this whole study the same
way it did for the aspirin single-molecule case, and is worth knowing before
running the full comparison.

IMPORTANT - written without torchmdnet installed locally, same caveat as
train_waterbox.py/waterbox_data.py - the checkpoint-loading and box-shape
assumptions below should be checked on the training box before trusting them.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from torchmdnet.module import LNNP

from physics_losses import per_fragment_momentum_loss
from structural_metrics import infer_molecule_groups, summarize_molecule_groups
from waterbox_data import load_waterbox_dataset, random_split


# PyTorch 2.7 checkpoint compatibility (matches the rest of this repo).
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


def load_waterbox_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "hyper_parameters" in checkpoint:
        hparams = checkpoint["hyper_parameters"]
    elif "hparams" in checkpoint:
        hparams = checkpoint["hparams"]
    else:
        raise ValueError("No hyperparameters found in checkpoint")

    model = LNNP(hparams)
    if "state_dict" not in checkpoint:
        raise ValueError("No state_dict found in checkpoint")
    # strict=False: WaterLNNP's extra `local_molecule_ids` buffer will be in the
    # state_dict; loading into a plain LNNP for evaluation only needs the base
    # model weights, so tolerate that one extra/missing key rather than fail.
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model.eval().to(device)


def _predict_forces(model, z, pos, box, device):
    pos_req = pos.detach().clone().requires_grad_(True)
    batch = torch.zeros(z.shape[0], dtype=torch.long, device=device)
    with torch.enable_grad():
        out = model(z, pos_req, batch=batch, box=box)
    if isinstance(out, tuple) and len(out) >= 2:
        energy_pred, force_pred = out[0], out[1]
    else:
        raise RuntimeError("Model forward did not return force predictions.")
    return energy_pred.detach(), force_pred.detach()


def evaluate_waterbox_checkpoint(checkpoint_path, data_root="./data", device=None, max_samples=200, seed=42):
    """Evaluate a water-box checkpoint on its held-out test split (same seed as
    training => same split via waterbox_data.random_split).

    Returns a dict with energy_mae, force_mae, and the per-molecule momentum
    violation (mean and max across evaluated configurations) - see module
    docstring for why the momentum number is the metric that actually matters
    here, not just a diagnostic.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_waterbox_checkpoint(checkpoint_path, device=device)

    full_dataset = load_waterbox_dataset(data_root=data_root)
    _, _, test_data = random_split(full_dataset, seed=seed)

    sample0 = full_dataset[0]
    local_molecule_ids = infer_molecule_groups(sample0.z, sample0.pos)
    summary = summarize_molecule_groups(sample0.z, local_molecule_ids)
    num_molecules_per_system = summary["n_groups"]
    local_molecule_ids = local_molecule_ids.to(device)

    energy_errors = []
    force_errors = []
    momentum_violations = []

    n_eval = min(max_samples, len(test_data))
    for idx in tqdm(range(n_eval), desc="Evaluating water-box checkpoint"):
        sample = test_data[idx].to(device)
        box = getattr(sample, "box", None)

        energy_pred, force_pred = _predict_forces(model, sample.z, sample.pos.float(), box, device)

        energy_true = sample.y.squeeze()
        force_true = sample.neg_dy

        energy_errors.append(abs(float((energy_pred.squeeze() - energy_true).item())))
        force_errors.append(float(torch.abs(force_pred - force_true).mean().item()))

        # Single-graph sample (batch index 0 for every atom), so the global
        # molecule id is just the local one - see build_global_molecule_ids for
        # the batched-training-time version this specializes.
        loss_mom = per_fragment_momentum_loss(
            sample.pos.float(), force_pred, local_molecule_ids, num_molecules_per_system
        )
        momentum_violations.append(float(loss_mom.item()))

    return {
        "n_evaluated": n_eval,
        "energy_mae": sum(energy_errors) / len(energy_errors),
        "force_mae": sum(force_errors) / len(force_errors),
        "mean_per_molecule_momentum_violation": sum(momentum_violations) / len(momentum_violations),
        "max_per_molecule_momentum_violation": max(momentum_violations),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = evaluate_waterbox_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
