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

This first mode (analyze(), --mode equilibrium) found no signal: momentum's
net-force magnitude wasn't smaller than absolute's at these configurations
(if anything slightly larger), and the internal-force share was nearly
identical between conditions (~97% for both). That result is specific to
near-equilibrium held-out DFT configurations, though - both models were
trained on data close to that distribution, so a mechanism that's absent
there could still emerge once geometries drift into configuration space
neither model saw during training, which is exactly what a real overheated
rollout does (heating to ~1000-2600K within under 1ps, per
src/run_rollout_study.py). The second mode below
(analyze_trajectory_frames(), --mode trajectory) tests that directly:
candidate mechanism 1 in the paper's Q3 discussion (a force-accuracy
difference invisible to static force MAE but that matters once the system
evolves into higher-displacement, more energetic configurations).

--mode trajectory reads real frames back from already-saved rollout.xyz
trajectories (src/rollout_waterbox_ase.py writes one per run) and
CROSS-evaluates: both models are run on frames from BOTH conditions'
trajectories, not just the model that produced them. This separates two
different questions that a same-model-on-own-trajectory comparison would
conflate: "does a given model's force response change at hot/distorted
geometries" vs. "are the hot geometries one condition's own rollout reaches
qualitatively different from the other condition's" - either could produce
a naive-looking correlation on its own.

This does NOT test the remaining candidate in the paper (a checkpoint-
selection artifact specific to these seed-0 checkpoints) - that needs
different evidence (repeating either analysis across more training seeds).

Usage:
    # Mode 1 (already run): equilibrium held-out configs, no signal found.
    python src/analyze_force_decomposition.py --mode equilibrium \\
        --ckpt-absolute checkpoints/waterbox_study/water_absolute/seed0/best_model.ckpt \\
        --ckpt-momentum "checkpoints/waterbox_study/water_absolute+momentum/seed0/best_model.ckpt"

    # Mode 2 (new): real rollout trajectory frames, cross-evaluated.
    python src/analyze_force_decomposition.py --mode trajectory \\
        --ckpt-absolute checkpoints/waterbox_study/water_absolute/seed0/best_model.ckpt \\
        --ckpt-momentum "checkpoints/waterbox_study/water_absolute+momentum/seed0/best_model.ckpt" \\
        --traj-absolute results/waterbox_rollout/rollout.xyz \\
        --traj-momentum results/waterbox_rollout_momentum/rollout.xyz \\
        --frame-indices 0,20,80,180

IMPORTANT - written without torchmdnet/ase installed locally (same caveat as
every other water-box script in this repo). Mode 1 has been run and its
result is summarized above; mode 2 has not been executed yet.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.io import read as ase_read

from evaluate_waterbox import load_waterbox_checkpoint
from structural_metrics import infer_molecule_groups
from waterbox_ase import tensors_from_atoms
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


def net_torque_mag(pos, forces, group_ids, num_molecules):
    """Per-molecule net torque magnitude ||sum_i (r_i - centroid_m) x F_i||.

    Deliberately mirrors physics_losses.py's per_fragment_momentum_loss
    centroid/cross-product computation line-for-line (same centroid
    calculation, same torch.cross call, same index_add_ scatter) rather than
    writing new torque logic - that function's angular term is exactly
    ||T_sum||^2 before the final mean-over-molecules; this returns ||T_sum||
    per molecule (not squared, not meaned) so it's directly comparable in
    kind to decompose_forces's net_force_mag output.
    """
    device = pos.device
    counts = torch.zeros(num_molecules, device=device)
    counts.index_add_(0, group_ids, torch.ones(len(group_ids), device=device))

    pos_sum = torch.zeros(num_molecules, 3, device=device)
    pos_sum.index_add_(0, group_ids, pos)
    centroid = pos_sum / counts.unsqueeze(1)

    pos_centered = pos - centroid[group_ids]
    torque_i = torch.cross(pos_centered, forces, dim=1)
    torque_sum = torch.zeros(num_molecules, 3, device=device)
    torque_sum.index_add_(0, group_ids, torque_i)
    return torque_sum.norm(dim=1).cpu().numpy()


def load_models(ckpt_absolute, ckpt_momentum, device=None):
    """Shared by analyze() and analyze_trajectory_frames() - also lets a
    caller running many trajectory pairs (run_force_decomposition_study.py)
    load each checkpoint once and reuse it, instead of reloading from disk
    on every replicate."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "water_absolute": load_waterbox_checkpoint(ckpt_absolute, device=device),
        "water_absolute+momentum": load_waterbox_checkpoint(ckpt_momentum, device=device),
    }


def analyze(ckpt_absolute, ckpt_momentum, data_root="./data", n_configs=6, seed=42, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(ckpt_absolute, ckpt_momentum, device)

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


def analyze_trajectory_frames(
    trajectory_paths,
    frame_indices,
    ckpt_absolute=None,
    ckpt_momentum=None,
    models=None,
    device=None,
    verbose=True,
):
    """Cross-evaluate both models on real trajectory snapshots (not held-out
    static DFT configs) - tests candidate mechanism 1 (paper/main.tex Section
    5.3, sec:q3): does either model's force behavior diverge specifically at
    the hot, distorted geometries a real NVE rollout actually reaches, in a
    way invisible at equilibrium (analyze()'s result found no signal there).

    trajectory_paths: {label: path to a rollout.xyz} - e.g. one per
    condition's own rollout. Each labeled trajectory's frames are evaluated
    under BOTH models, not just the one that produced them - see module
    docstring for why that cross-evaluation matters here.

    frame_indices: which frames (by index into the saved trajectory, same
    stride as --energy-log-stride used when the rollout was run - index i
    is step i*energy_log_stride) to pull and evaluate. Pick a spread from
    early (still near the starting geometry) to late (deep in the overheated
    plateau) to see whether any divergence is present from the start or only
    emerges as the geometry drifts.

    Pass models=... (from load_models()) to reuse already-loaded checkpoints
    across many calls (run_force_decomposition_study.py calls this once per
    replicate trajectory pair) instead of reloading from disk every time;
    otherwise ckpt_absolute/ckpt_momentum load fresh. verbose=False
    suppresses the per-row prints, useful when a caller is aggregating many
    calls' worth of rows itself.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if models is None:
        if not ckpt_absolute or not ckpt_momentum:
            raise ValueError("Provide either models=... (pre-loaded) or both ckpt_absolute and ckpt_momentum.")
        models = load_models(ckpt_absolute, ckpt_momentum, device)

    if verbose:
        print(
            f"{'trajectory':16s} {'frame':6s} {'eval_model':28s} {'raw |F|':10s} "
            f"{'net |F_mol|':12s} {'internal |F|':13s} internal/raw"
        )
    rows = []
    for traj_label, traj_path in trajectory_paths.items():
        for frame_idx in frame_indices:
            atoms = ase_read(traj_path, index=frame_idx)
            z, pos, box = tensors_from_atoms(atoms, device)
            group_ids = infer_molecule_groups(z, pos, box=box).to(device)
            num_molecules = int(group_ids.max().item()) + 1

            for eval_name, model in models.items():
                forces = _predict_forces(model, z, pos, box, device)
                net_mag, internal_mag = decompose_forces(forces, group_ids, num_molecules)
                raw_mag = forces.norm(dim=1).cpu().numpy()
                share = float(internal_mag.mean() / raw_mag.mean())

                row = {
                    "trajectory": traj_label,
                    "frame": frame_idx,
                    "eval_model": eval_name,
                    "raw_force_mag_mean": float(raw_mag.mean()),
                    "net_force_mag_mean": float(net_mag.mean()),
                    "internal_force_mag_mean": float(internal_mag.mean()),
                    "internal_force_share": share,
                }
                rows.append(row)
                if verbose:
                    print(
                        f"{traj_label:16s} {frame_idx:<6d} {eval_name:28s} "
                        f"{row['raw_force_mag_mean']:8.4f}   {row['net_force_mag_mean']:10.4f}   "
                        f"{row['internal_force_mag_mean']:11.4f}   {share:.4f}"
                    )

    if verbose:
        print(
            "\nRead this by fixing eval_model and comparing across frame (does a model's own "
            "force behavior change as the geometry heats up?), then by fixing trajectory+frame "
            "and comparing across eval_model (do the two models react differently to the SAME "
            "geometry?). A frame-dependent divergence that's absent at frame 0 but grows at "
            "later frames would support candidate mechanism 1 (paper/main.tex Section 5.3, "
            "sec:q3) - a force-accuracy difference that only matters once the system leaves the "
            "training distribution, invisible to both the static force MAE and the "
            "equilibrium-config check above."
        )

    return rows


def paired_bias_test(ckpt_absolute=None, ckpt_momentum=None, models=None, data_root="./data",
                      n_configs=100, seed=42, device=None):
    """Statistically rigorous version of the point-wise comparisons in
    analyze()/analyze_trajectory_frames(): those eyeballed MEANS over a
    handful of configs and found "the two models agree closely" - this
    tests whether a small but SYSTEMATIC, sign-consistent bias is hiding in
    that noise, using many more configs/atoms/molecules and a proper paired
    significance test rather than comparing aggregate means. This is the
    direct test paper/main.tex Section 5.3 (sec:q3) calls for, implied by
    run_force_decomposition_study.py's 10/10-unanimous trajectory-divergence
    direction but not yet directly detected in any point-wise comparison.

    For each of n_configs held-out configs, forms the PAIRED difference
    (momentum minus absolute) at every individual atom/molecule - not
    averaged first - for four quantities: raw per-atom force magnitude,
    per-molecule net force magnitude and net torque magnitude
    (decompose_forces / net_torque_mag), and per-atom internal force
    magnitude. Net torque is included specifically to check a candidate
    reconciliation: the n=6 static study found momentum's AGGREGATE
    per_fragment_momentum_loss (net force squared + net torque squared,
    averaged across 6 training seeds) was lower for momentum, while net
    force magnitude alone (this test, seed 0 only) comes out higher for
    momentum - if net torque shows a comparably-sized bias in the opposite
    direction, that resolves the apparent tension (the combined squared
    metric could be dominated by a torque improvement masking a linear-force
    regression) rather than the two results actually contradicting.

    For each quantity, reports:
      - mean paired difference, standard error, and a two-sided z-test
        p-value (normal approximation - valid here since n is in the
        thousands, so the CLT applies regardless of the underlying
        per-atom/per-molecule distribution shape; no scipy dependency
        needed, just math.erfc)
      - a sign test: what fraction of individual pairs have momentum >
        absolute, tested against the null of exactly 50/50 (binomial normal
        approximation) - this is the direct "is the sign consistently
        one-directional" test, not just "is the mean nonzero"
    """
    import math

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if models is None:
        if not ckpt_absolute or not ckpt_momentum:
            raise ValueError("Provide either models=... (pre-loaded) or both ckpt_absolute and ckpt_momentum.")
        models = load_models(ckpt_absolute, ckpt_momentum, device)

    full_dataset = load_waterbox_dataset(data_root=data_root)
    _, _, test_data = random_split(full_dataset, seed=seed)
    n_configs = min(n_configs, len(test_data))

    raw_diffs, net_diffs, internal_diffs, torque_diffs = [], [], [], []
    for idx in range(n_configs):
        sample = test_data[idx].to(device)
        z, pos, box = sample.z, sample.pos.float(), getattr(sample, "box", None)
        group_ids = infer_molecule_groups(z, pos, box=box).to(device)
        num_molecules = int(group_ids.max().item()) + 1

        forces_abs = _predict_forces(models["water_absolute"], z, pos, box, device)
        forces_mom = _predict_forces(models["water_absolute+momentum"], z, pos, box, device)

        raw_diffs.append((forces_mom.norm(dim=1) - forces_abs.norm(dim=1)).cpu().numpy())

        net_abs, internal_abs = decompose_forces(forces_abs, group_ids, num_molecules)
        net_mom, internal_mom = decompose_forces(forces_mom, group_ids, num_molecules)
        net_diffs.append(net_mom - net_abs)
        internal_diffs.append(internal_mom - internal_abs)

        torque_abs = net_torque_mag(pos, forces_abs, group_ids, num_molecules)
        torque_mom = net_torque_mag(pos, forces_mom, group_ids, num_molecules)
        torque_diffs.append(torque_mom - torque_abs)

    def _report(name, diffs):
        diffs = np.concatenate(diffs)
        n = len(diffs)
        mean = float(diffs.mean())
        se = float(diffs.std(ddof=1) / np.sqrt(n))
        z = mean / se if se > 0 else float("nan")
        p_mean = math.erfc(abs(z) / math.sqrt(2))

        frac_positive = float((diffs > 0).sum()) / n
        se_frac = 0.5 / np.sqrt(n)
        z_sign = (frac_positive - 0.5) / se_frac
        p_sign = math.erfc(abs(z_sign) / math.sqrt(2))

        print(
            f"{name:20s} n={n:<7d} mean_diff={mean:+.5f}  z={z:+7.2f}  p={p_mean:.2e}   "
            f"frac_positive={frac_positive:.4f}  z_sign={z_sign:+7.2f}  p_sign={p_sign:.2e}"
        )
        return {
            "n": n, "mean_diff": mean, "z_mean": z, "p_mean": p_mean,
            "frac_positive": frac_positive, "z_sign": z_sign, "p_sign": p_sign,
        }

    print(f"Paired momentum-minus-absolute force differences across {n_configs} held-out configs:")
    results = {
        "raw_force_mag": _report("raw_force_mag", raw_diffs),
        "net_force_mag": _report("net_force_mag", net_diffs),
        "internal_force_mag": _report("internal_force_mag", internal_diffs),
        "net_torque_mag": _report("net_torque_mag", torque_diffs),
    }

    print(
        "\nA small |z| / large p means no statistically detectable systematic bias in that "
        "quantity. A significant, consistent-sign bias (large |z|, tiny p, frac_positive far "
        "from 0.5) in net_force_mag specifically is the systematic component behind the rollout "
        "heating (sec:q3). net_torque_mag is the reconciliation check: if it shows a "
        "comparably-sized bias in the OPPOSITE direction (momentum lower), that explains why the "
        "n=6 study's aggregate per_fragment_momentum_loss (net force^2 + net torque^2, averaged "
        "across 6 seeds) came out lower for momentum despite this seed's net force alone being "
        "higher - a torque improvement masking a linear-force regression, not a contradiction."
    )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["equilibrium", "trajectory", "paired-bias"],
        default="equilibrium",
        help="equilibrium (already run): held-out static DFT configs, eyeballed means. "
        "trajectory (already run): real rollout.xyz frames, cross-evaluated under both models. "
        "paired-bias (new): rigorous paired significance test across many more configs.",
    )
    parser.add_argument("--ckpt-absolute", type=str, required=True)
    parser.add_argument("--ckpt-momentum", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--n-configs", type=int, default=None,
                         help="held-out configs to use. Defaults to 6 for --mode equilibrium "
                         "(matches the original eyeballed check) or 100 for --mode paired-bias "
                         "(needs many more samples for a well-powered test) if not given.")
    parser.add_argument("--seed", type=int, default=42, help="equilibrium/paired-bias modes only")
    parser.add_argument("--traj-absolute", type=str, default=None,
                         help="trajectory mode: path to water_absolute's rollout.xyz")
    parser.add_argument("--traj-momentum", type=str, default=None,
                         help="trajectory mode: path to water_absolute+momentum's rollout.xyz")
    parser.add_argument("--frame-indices", type=str, default="0,20,80,180",
                         help="trajectory mode: comma-separated frame indices (index i = step "
                         "i*energy_log_stride the rollout was logged at)")
    args = parser.parse_args()

    if args.mode == "equilibrium":
        analyze(
            ckpt_absolute=args.ckpt_absolute,
            ckpt_momentum=args.ckpt_momentum,
            data_root=args.data_root,
            n_configs=args.n_configs if args.n_configs is not None else 6,
            seed=args.seed,
        )
    elif args.mode == "trajectory":
        if not args.traj_absolute or not args.traj_momentum:
            parser.error("--mode trajectory requires both --traj-absolute and --traj-momentum")
        frame_indices = [int(x) for x in args.frame_indices.split(",")]
        analyze_trajectory_frames(
            trajectory_paths={
                "water_absolute": args.traj_absolute,
                "water_absolute+momentum": args.traj_momentum,
            },
            frame_indices=frame_indices,
            ckpt_absolute=args.ckpt_absolute,
            ckpt_momentum=args.ckpt_momentum,
        )
    else:
        paired_bias_test(
            ckpt_absolute=args.ckpt_absolute,
            ckpt_momentum=args.ckpt_momentum,
            data_root=args.data_root,
            n_configs=args.n_configs if args.n_configs is not None else 100,
            seed=args.seed,
        )
