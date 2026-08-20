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

from diagnose_short_range_collapse import (
    element_pair_intermolecular_distances,
    molecule_group_ids,
    pairwise_min_image_distances,
    same_molecule_mask,
)
from molecular_zbl import register_molecular_zbl_prior
from physics_losses import per_fragment_momentum_loss
from structural_metrics import infer_molecule_groups, summarize_molecule_groups
from waterbox_data import load_waterbox_dataset, random_split


# PyTorch 2.7 checkpoint compatibility (matches the rest of this repo).
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


def load_waterbox_checkpoint(checkpoint_path, device="cpu"):
    # Idempotent, unconditional: this is the ONE shared reload path used by
    # this module, waterbox_ase.py's TensorNetCalculator (every rollout
    # step), and analyze_force_decomposition.py - a checkpoint saved with
    # prior_model="MolecularZBL" needs this name registered into
    # torchmdnet.priors's own namespace before LNNP(hparams) below can
    # reconstruct it, exactly as train_waterbox.py must do at construction
    # time. Cheap no-op for checkpoints that don't use it.
    register_molecular_zbl_prior()

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


def _min_intermolecular_distance_angstrom(z_cpu_numpy, pos_cpu_numpy, box):
    """Cheapest possible per-config summary for spotting whether an
    anomalously close pair (Q4, paper/main.tex sec:q4) is what's behind an
    otherwise-inexplicable per-config error spike under the ZBL prior (its
    1/distance singularity means one such config could dominate a whole
    test-set mean, even with every other config totally normal) - reuses
    diagnose_short_range_collapse.py's own primitives rather than a second,
    separately-trusted distance computation."""
    box_lengths = torch.as_tensor(box).reshape(3, 3).diagonal().cpu().numpy()
    dist = pairwise_min_image_distances(pos_cpu_numpy, box_lengths)
    same_molecule = same_molecule_mask(molecule_group_ids(z_cpu_numpy, pos_cpu_numpy, box_lengths))
    per_type = element_pair_intermolecular_distances(z_cpu_numpy, dist, same_molecule)
    all_d = torch.cat([torch.as_tensor(arr) for arr in per_type.values() if arr.size])
    return float(all_d.min().item()) if all_d.numel() else float("nan")


def evaluate_waterbox_checkpoint(
    checkpoint_path, data_root="./data", device=None, max_samples=200, seed=42, return_per_sample=False,
):
    """Evaluate a water-box checkpoint on its held-out test split (same seed as
    training => same split via waterbox_data.random_split).

    Returns a dict with energy_mae, force_mae, and the per-molecule momentum
    violation (mean and max across evaluated configurations) - see module
    docstring for why the momentum number is the metric that actually matters
    here, not just a diagnostic.

    return_per_sample=True additionally includes a "per_sample" list (one
    dict per evaluated test config: test_config_index, energy_error,
    force_error, momentum_violation, min_intermolecular_distance_angstrom) -
    off by default (extra compute for the distance check, and most callers
    just want the aggregate), turn on when a mean looks implausible and you
    need to know whether it's one outlier config or a uniformly bad model
    (paper/main.tex sec:q4 - this is exactly what caught the water_absolute
    ZBL seed-0 anomaly: energy_mae=16404 eV where every other cell was
    2-5 eV).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_waterbox_checkpoint(checkpoint_path, device=device)

    full_dataset = load_waterbox_dataset(data_root=data_root)
    _, _, test_data = random_split(full_dataset, seed=seed)

    sample0 = full_dataset[0]
    local_molecule_ids = infer_molecule_groups(sample0.z, sample0.pos, box=getattr(sample0, "box", None))
    summary = summarize_molecule_groups(sample0.z, local_molecule_ids)
    num_molecules_per_system = summary["n_groups"]
    local_molecule_ids = local_molecule_ids.to(device)

    energy_errors = []
    force_errors = []
    momentum_violations = []
    per_sample = []

    n_eval = min(max_samples, len(test_data))
    for idx in tqdm(range(n_eval), desc="Evaluating water-box checkpoint"):
        sample = test_data[idx].to(device)
        box = getattr(sample, "box", None)

        energy_pred, force_pred = _predict_forces(model, sample.z, sample.pos.float(), box, device)

        energy_true = sample.y.squeeze()
        force_true = sample.neg_dy

        energy_error = abs(float((energy_pred.squeeze() - energy_true).item()))
        force_error = float(torch.abs(force_pred - force_true).mean().item())
        energy_errors.append(energy_error)
        force_errors.append(force_error)

        # Single-graph sample (batch index 0 for every atom), so the global
        # molecule id is just the local one - see build_global_molecule_ids for
        # the batched-training-time version this specializes.
        loss_mom = per_fragment_momentum_loss(
            sample.pos.float(), force_pred, local_molecule_ids, num_molecules_per_system
        )
        momentum_violation = float(loss_mom.item())
        momentum_violations.append(momentum_violation)

        if return_per_sample:
            min_dist = _min_intermolecular_distance_angstrom(
                sample.z.cpu().numpy(), sample.pos.float().cpu().numpy(), box,
            )
            per_sample.append({
                "test_config_index": idx,
                "energy_error": energy_error,
                "force_error": force_error,
                "momentum_violation": momentum_violation,
                "min_intermolecular_distance_angstrom": min_dist,
            })

    result = {
        "n_evaluated": n_eval,
        "energy_mae": sum(energy_errors) / len(energy_errors),
        "force_mae": sum(force_errors) / len(force_errors),
        "mean_per_molecule_momentum_violation": sum(momentum_violations) / len(momentum_violations),
        "max_per_molecule_momentum_violation": max(momentum_violations),
    }
    if return_per_sample:
        result["per_sample"] = per_sample
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--per-sample", action="store_true",
        help="Also report the per-test-config breakdown, sorted worst-energy-error-first - use "
        "when the aggregate energy_mae looks implausible (e.g. thousands of eV) to check whether "
        "it's one/few outlier configs (a ZBL 1/distance singularity hitting an anomalously close "
        "pair, paper/main.tex sec:q4) rather than a uniformly bad model.",
    )
    args = parser.parse_args()

    result = evaluate_waterbox_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        max_samples=args.max_samples,
        seed=args.seed,
        return_per_sample=args.per_sample,
    )

    if args.per_sample:
        per_sample = result.pop("per_sample")
        print(json.dumps(result, indent=2))
        worst = sorted(per_sample, key=lambda row: row["energy_error"], reverse=True)
        print(f"\n{'idx':5s} {'energy_error':>14s} {'force_error':>12s} {'momentum_viol':>14s} "
              f"{'min_intermol_dist_A':>20s}")
        for row in worst[:10]:
            print(
                f"{row['test_config_index']:<5d} {row['energy_error']:14.4f} {row['force_error']:12.4f} "
                f"{row['momentum_violation']:14.4f} {row['min_intermolecular_distance_angstrom']:20.4f}"
            )
        print(
            "\nIf the top row(s) are enormously worse than the rest AND show an unusually small "
            "min_intermol_dist_A relative to the others, that directly supports a ZBL-singularity "
            "outlier config, not a uniformly bad model - the mean above is then misleading; report "
            "the outlier-excluded mean instead."
        )
    else:
        print(json.dumps(result, indent=2))
