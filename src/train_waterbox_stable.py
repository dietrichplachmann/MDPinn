#!/usr/bin/env python
"""StABlE-style stability-aware fine-tuning, step 3 of the plan (paper/main.tex
sec:q4-stable-plan) - ties boltzmann_estimator.py's pseudo-loss and
waterbox_langevin.py's sampler/rewind mechanism together into the actual
alternating simulate/learn loop, fine-tuning an already-trained ext70
checkpoint rather than training from scratch (paper/main.tex
sec:q4-stable-path: "fine-tuning is explicitly a continuation of
pretraining").

One StABlE outer iteration:
  1. Every replica runs a fresh Langevin sampling phase from wherever it
     currently is (waterbox_langevin.run_stable_sampling_phase), rewinding
     on a stability trip, and returns a subsampled window of Snapshots.
  2. All replicas' collected Snapshots are pooled into one batch (the
     estimator only needs "samples from P_theta", not which replica they
     came from - paper/main.tex sec:q4-stable-step1).
  3. For each pooled sample: g_i = its O-O/O-H/H-H RDF (pure geometry, no
     grad - reuses rollout_waterbox_ase.py's already-validated RDF
     machinery), U_i = the model's predicted total energy there (a fresh
     forward pass WITH grad enabled, tracing back to theta).
  4. boltzmann_estimator_pseudo_loss(g, U, g_target, kT) gives L_obs;
     L_QM is the ordinary supervised energy/force MSE on a small batch of
     real labeled TRAINING-split data (never on the sampled/simulated
     configurations - see boltzmann_estimator.py's docstring on why L_QM
     is a pure regularizer here, unlike active learning).
  5. One optimizer.step() on L_obs + lambda_qm * L_QM.

Deliberately NOT batched into one multi-graph forward pass per gradient
step: each pooled sample's energy is computed with its own single-graph
forward call, exactly the pattern evaluate_waterbox.py and waterbox_ase.py
already use and trust everywhere else in this project, rather than a new,
unverified multi-graph batching path whose correctness can't be checked
without a real torchmdnet install. The number of forward passes per
gradient step here is small (n_replicas * learn_window_steps /
subsample_stride, tens not thousands) - a real but modest efficiency cost,
flagged here as a future optimization once this simpler, definitely-correct
version is confirmed working end to end, not attempted blind.

Checkpoint I/O: load_waterbox_checkpoint (evaluate_waterbox.py) returns a
real LightningModule with its own `.hparams` - this loop is a raw PyTorch
loop, not run through Lightning's Trainer, so on save this module
constructs the minimal checkpoint dict load_waterbox_checkpoint actually
reads (`state_dict` + `hyper_parameters`) directly, rather than the fuller
Trainer-produced format (optimizer state, epoch counters, etc. - never
read by anything in this project's evaluation/rollout tooling, so not
worth reproducing).

IMPORTANT - written without ase/torchmdnet installed locally (same caveat
as every other water-box script in this repo). Nothing in this module has
been executed yet - see this module's Usage section for the exact
--smoke-test command to run on the training box first (2-3 outer
iterations, confirms checkpoints save and the loss doesn't diverge/crash -
build order item 3, paper/main.tex sec:q4-stable-plan), before a real run.

Usage (training box only, NOT yet run):
    python src/train_waterbox_stable.py \\
        --ckpt checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed1/best_model.ckpt \\
        --out checkpoints/waterbox_study_zbl_bonded_ext70_stable/water_absolute/seed1 \\
        --smoke-test
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from boltzmann_estimator import boltzmann_estimator_pseudo_loss, observable_loss_value
from diagnose_short_range_collapse import (
    compute_reference_floors,
    molecule_group_ids,
    same_molecule_mask,
)
from evaluate_waterbox import load_waterbox_checkpoint
from waterbox_ase import TensorNetCalculator, atoms_from_waterbox_sample, tensors_from_atoms
from waterbox_data import load_waterbox_dataset, random_split
from waterbox_langevin import ReplicaState, Snapshot, run_stable_sampling_phase

# Reused directly from rollout_waterbox_ase.py rather than duplicated -
# these are its module-private helpers (leading underscore), imported
# anyway since this is exactly the kind of cross-module reuse this
# project's own discipline prefers over a second, separately-trusted copy
# of "compute a reference RDF from real DFT configs" (see e.g.
# diagnose_short_range_collapse.py's frame_min_distances, added for the
# identical reason). A future cleanup could rename them public in that
# module; not done here to avoid touching an already-validated file more
# than necessary.
from rollout_waterbox_ase import ELEMENT_PAIRS, _averaged_rdf, _safe_rmax, _sample_reference_frames

K_BOLTZMANN_EV_PER_K = 8.617333262e-5  # eV/K


def _stacked_rdf(atoms, rmax, nbins) -> np.ndarray:
    """Concatenates the O-O/O-H/H-H RDF (rollout_waterbox_ase.ELEMENT_PAIRS
    order, each nbins long) for ONE frame into a single flat (3*nbins,)
    array - this is g(Gamma) for boltzmann_estimator_pseudo_loss's
    vector-observable path. Pure geometry (ase.geometry.rdf.get_rdf on
    positions only) - no model involved, no gradient needed, matching
    boltzmann_estimator.py's requirement that g never carry a grad path
    through the sampler."""
    parts = []
    for _, elems in ELEMENT_PAIRS:
        rdf, _ = _averaged_rdf([atoms], rmax, nbins, elems)
        parts.append(rdf)
    return np.concatenate(parts)


def _config_energy(model, z, pos, box, device):
    """Single-configuration forward pass with grad enabled, returning ONLY
    the predicted total energy (still attached to theta) - mirrors
    evaluate_waterbox.py's _predict_forces pattern (pos must have
    requires_grad=True for this model's derivative=True forward pass to
    run at all, per that function's own established convention - not
    optional, the model's internal force computation needs it regardless
    of whether the caller wants forces), but keeps the energy attached
    rather than detaching it, and discards the (unneeded here) force
    output rather than returning it."""
    pos_req = pos.detach().clone().requires_grad_(True)
    batch = torch.zeros(z.shape[0], dtype=torch.long, device=device)
    with torch.enable_grad():
        energy, _forces = model(z, pos_req, batch=batch, box=box)
    return energy.squeeze()


def _snapshot_to_atoms(snapshot: Snapshot, template_atoms):
    """Builds a throwaway ase.Atoms for one Snapshot, reusing the
    replica's own template Atoms object for atomic numbers/cell (fixed
    throughout one replica's dynamics - only positions/velocities change
    snapshot to snapshot)."""
    atoms = template_atoms.copy()
    atoms.set_positions(snapshot.positions)
    return atoms


def _qm_sample_indices_and_weights(model, train_data, batch_size, rng):
    y_weight = float(model.hparams.y_weight)
    neg_dy_weight = float(model.hparams.neg_dy_weight)
    indices = rng.choice(len(train_data), size=min(batch_size, len(train_data)), replace=False)
    return indices, y_weight, neg_dy_weight


def _qm_per_sample_loss(model, train_data, idx, device, y_weight, neg_dy_weight):
    """One training-split sample's weighted energy/force MSE, still
    attached to theta (grad enabled) - the shared per-sample computation
    both L_QM variants below build on, so the forward-pass/weighting logic
    exists in exactly one place."""
    sample = train_data[int(idx)].to(device)
    z, pos, box = sample.z, sample.pos.float(), getattr(sample, "box", None)
    pos_req = pos.detach().clone().requires_grad_(True)
    batch = torch.zeros(z.shape[0], dtype=torch.long, device=device)
    with torch.enable_grad():
        energy_pred, force_pred = model(z, pos_req, batch=batch, box=box)
    y_mse = (energy_pred.squeeze() - sample.y.squeeze()) ** 2
    dy_mse = torch.mean((force_pred - sample.neg_dy) ** 2)
    return y_weight * y_mse + neg_dy_weight * dy_mse


def _qm_regularizer_loss(model, train_data, batch_size, device, rng):
    """L_QM as a single differentiable tensor, summed across the batch
    before the caller calls .backward() once - the original design. Kept
    (not removed) specifically for verify_qm_gradient.py: that script
    needs to call .backward() itself and re-evaluate the loss at perturbed
    parameter values with no side effects, which requires a plain
    differentiable return value, not a function that triggers its own
    backward pass. Safe here because verify_qm_gradient.py only ever uses
    small batch sizes (2) where accumulating a few create_graph=True
    graphs at once was never the problem - see
    _qm_regularizer_loss_and_backward below for why THIS pattern is unsafe
    at the batch sizes fine_tune_stable actually needs."""
    indices, y_weight, neg_dy_weight = _qm_sample_indices_and_weights(model, train_data, batch_size, rng)
    total = torch.zeros((), device=device)
    for idx in indices:
        total = total + _qm_per_sample_loss(model, train_data, idx, device, y_weight, neg_dy_weight)
    return total / len(indices)


def _qm_regularizer_loss_and_backward(model, train_data, batch_size, device, rng, lambda_qm):
    """L_QM for the real fine-tuning loop: calls .backward() ITSELF, per
    sample, immediately - accumulating into model.parameters()' .grad
    exactly as fine_tune_stable's optimizer expects, rather than returning
    one differentiable summed loss for the caller to backward() once
    (_qm_regularizer_loss above). This is not a style choice: the model's
    internal derivative=True forward pass computes forces via
    autograd.grad(energy, pos, create_graph=True), which keeps that
    sample's ENTIRE first-order backward graph alive (needed to be
    differentiable a second time) until something actually calls
    backward() on it. Accumulating 8 such graphs into one running sum
    before a single backward() - the original design, fine at
    verify_qm_gradient.py's batch size of 2 but not here - held all 8
    alive simultaneously and, in the first real (non-smoke) fine-tuning
    run, was enough to exhaust a 47.5 GiB GPU on its own (this project's
    own established batch_size for ordinary supervised training on this
    exact model is 1-2, not 8 - a real, now-confirmed-not-hypothetical
    memory cost of create_graph=True the original design underestimated).
    Per-sample immediate backward() means at most ONE such graph is ever
    alive at a time, regardless of batch_size. Gradients from repeated
    .backward() calls accumulate into .grad by default (PyTorch's normal
    behavior, reset only by the caller's own optimizer.zero_grad()) -
    mathematically identical to a single backward() on the summed loss,
    just without ever materializing every sample's graph at once.

    Returns the (detached) scalar loss VALUE only, for logging - there is
    no loss tensor to return since backward() already ran."""
    indices, y_weight, neg_dy_weight = _qm_sample_indices_and_weights(model, train_data, batch_size, rng)
    n = len(indices)
    total_value = 0.0
    for idx in indices:
        per_sample_loss = _qm_per_sample_loss(model, train_data, idx, device, y_weight, neg_dy_weight) / n
        (lambda_qm * per_sample_loss).backward()
        total_value += float(per_sample_loss.item())
    return total_value


def fine_tune_stable(
    ckpt,
    out_dir,
    data_root="./data",
    n_replicas=4,
    n_outer_iterations=200,
    check_interval=20,
    learn_window_steps=100,
    subsample_stride=2,
    temperature_k=300.0,
    friction_fs_inv=0.01,
    dt_fs=0.5,
    lambda_qm=1.0,
    lr=1e-6,
    qm_batch_size=8,
    rdf_rmax=6.0,
    rdf_nbins=200,
    n_reference_configs=200,
    floor_percentile=1.0,
    save_every=20,
    seed=42,
    device=None,
    max_pooled_samples=5,
):
    """The alternating StABlE fine-tuning loop (module docstring). Writes a
    checkpoint every save_every outer iterations plus a final one, and a
    history.json log (per-iteration L_obs/L_QM/n_rewinds), to out_dir.

    lr defaults to 1e-6, two orders of magnitude below the original
    training run's 1e-4 (paper/main.tex sec:q4-stable-plan: "well below the
    original 1e-4 - this is fine-tuning a converged model, not training
    from scratch") - a starting point, not an empirically verified value;
    watch the loss history for divergence (too high) or no movement at all
    (too low) and adjust. lambda_qm=1.0 is similarly a starting point, not
    tuned - see this function's history.json output plus a post-hoc
    evaluate_waterbox.py --per-sample check (module docstring) to confirm
    static accuracy hasn't regressed before trusting a real run's result.

    max_pooled_samples=5 bounds how many collected Snapshots actually get
    forwarded through _config_energy (each holding a create_graph=True graph
    simultaneously, unlike L_QM's per-sample-immediate-backward - the
    observable loss needs the whole pooled batch's covariance jointly, so
    this accumulation can't be avoided the same way) per outer iteration,
    regardless of n_replicas. Unlike n_replicas, this is NOT a knob to scale
    up for more sample diversity - it exists purely as a memory ceiling.

    CORRECTED after a second real OOM, not a fresh guess this time: the
    first cap attempt used 12 (a THEORETICAL worst case - 4 replicas x
    ceil(5/subsample_stride)=3 samples/replica at iteration 0, when every
    replica is still freshly-equilibrated and stable) reasoned to be safe
    because the 4-replica pilot supposedly already "validated" it. It
    hadn't: the pilot's own logged history never actually pooled more than
    5 samples in any real iteration (instability naturally kept the true
    count well under the 12-sample worst case) - 12 was never actually
    exercised, so "already validated" was false, and n_replicas=8 capped at
    12 OOM'd again at nearly identical memory usage (46.54 of 47.5 GiB) to
    the uncapped n_replicas=8 crash before it. Lesson (CLAUDE.md's own,
    re-learned the hard way here): calibrate a memory ceiling from what was
    actually MEASURED to run, never from a theoretical worst-case
    calculation on paper. 5 is the actually-observed pilot maximum, with
    zero headroom assumed - the loop below also logs per-sample
    torch.cuda.memory_allocated() growth on outer_iter==0 specifically so
    a future increase to this cap is calibrated from real numbers, not
    another guess.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kT = K_BOLTZMANN_EV_PER_K * temperature_k

    print(f"Loading checkpoint: {ckpt}")
    model = load_waterbox_checkpoint(ckpt, device=device)
    model.train()

    full_dataset = load_waterbox_dataset(data_root=data_root)
    train_data, _val_data, _test_data = random_split(full_dataset, seed=seed)

    print("Sampling reference DFT configurations for the RDF target...")
    reference_frames = _sample_reference_frames(full_dataset, n_samples=n_reference_configs, seed=seed)

    print("Computing Q4 short-range-collapse stability floors (diagnose_short_range_collapse.py)...")
    floors, _floor_stats = compute_reference_floors(
        data_root=data_root, n_reference_configs=n_reference_configs, seed=seed, floor_percentile=floor_percentile,
    )

    print(f"Building {n_replicas} replicas from distinct training-split starting configurations...")
    replica_indices = rng.choice(len(train_data), size=n_replicas, replace=False)
    replica_atoms, replica_states, same_molecule_masks = [], [], []
    for replica_i, idx in enumerate(replica_indices):
        sample = train_data[int(idx)]
        atoms = atoms_from_waterbox_sample(sample)
        # model=model (not checkpoint_path=ckpt): every replica MUST share
        # the SAME live model object the optimizer below updates, not its
        # own independently-loaded copy - see waterbox_ase.py's
        # TensorNetCalculator docstring for why this matters (a separately-
        # loaded copy would silently never reflect any fine-tuning
        # progress, breaking the on-policy premise this whole estimator
        # depends on).
        atoms.calc = TensorNetCalculator(model=model, device=device)
        # Maxwell-Boltzmann at temperature_k, not a cold (zero-velocity)
        # start - matches the already-verified convention used everywhere
        # else in this project (waterbox_langevin._smoke_test_thermostat,
        # rollout_waterbox_ase.run_rollout). A cold start isn't obviously
        # wrong, but it's an unjustified departure from the one convention
        # already empirically confirmed to equilibrate sanely (paper/main.tex
        # sec:q4-stable-step2's thermostat smoke-test result) - no reason to
        # introduce a second, unverified regime for the real fine-tuning
        # loop specifically. Explicit RandomState per replica (this
        # project's own established discipline - CLAUDE.md's Lessons on
        # MaxwellBoltzmannDistribution's global-RNG pitfall).
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=np.random.RandomState(seed + replica_i))
        Stationary(atoms)

        z = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        box_lengths = np.array(atoms.get_cell()).diagonal()
        group_ids = molecule_group_ids(z, positions, box_lengths)
        same_molecule_masks.append(same_molecule_mask(group_ids))

        initial_snapshot = Snapshot(positions=positions.copy(), velocities=atoms.get_velocities().copy(), step=0)
        replica_states.append(ReplicaState(initial_snapshot))
        replica_atoms.append(atoms)

    # rmax computed from BOTH the reference frames AND the replicas' own
    # starting boxes, not just the reference sample - rollout_waterbox_ase.py
    # learned this the hard way (its own comment: "the reference sample is
    # what actually triggered CellTooSmall on the training box"). Box size
    # is fixed per replica for the whole run (constant-volume Langevin, no
    # barostat), so each replica's starting atoms object's cell is the only
    # one that matters here, not every later Snapshot's.
    rmax = _safe_rmax(reference_frames + replica_atoms, rdf_rmax)
    g_target_parts = [
        _averaged_rdf(reference_frames, rmax, rdf_nbins, elems)[0] for _, elems in ELEMENT_PAIRS
    ]
    g_target = torch.tensor(np.concatenate(g_target_parts), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    def _save(tag):
        path = out_dir / f"stable_{tag}.ckpt"
        torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}, path)
        print(f"Wrote: {path}")
        return str(path)

    for outer_iter in range(n_outer_iterations):
        # Every replica's TensorNetCalculator holds the SAME `model` object
        # constructed above (not a separate loaded copy - see the model=model
        # comment where replicas were built). optimizer.step() mutates that
        # object's parameters IN PLACE every iteration (standard PyTorch
        # behavior - the optimizer never replaces the parameter tensors,
        # only their .data), so every replica automatically samples under
        # the latest weights on this iteration's sampling phase below, with
        # no extra plumbing needed (paper/main.tex sec:q4-stable-plan).
        collected_per_replica = run_stable_sampling_phase(
            replica_atoms=replica_atoms,
            replica_states=replica_states,
            same_molecule_masks=same_molecule_masks,
            floors=floors,
            dt_fs=dt_fs,
            temperature_k=temperature_k,
            friction_fs_inv=friction_fs_inv,
            check_interval=check_interval,
            learn_window_steps=learn_window_steps,
            subsample_stride=subsample_stride,
            rng_seed=seed + outer_iter,
        )
        n_rewinds_this_iter = [state.n_rewinds for state in replica_states]

        # Flatten first (cheap - plain numpy Snapshots, no grad yet) so the
        # max_pooled_samples cap can be applied BEFORE any _config_energy
        # call, not after - see fine_tune_stable's own docstring on why an
        # uncapped pool scales directly (and dangerously) with n_replicas.
        all_snapshots = [
            (atoms, snap)
            for atoms, snapshots in zip(replica_atoms, collected_per_replica)
            for snap in snapshots
        ]
        n_collected_this_iter = len(all_snapshots)
        if n_collected_this_iter > max_pooled_samples:
            keep_idx = rng.choice(n_collected_this_iter, size=max_pooled_samples, replace=False)
            all_snapshots = [all_snapshots[i] for i in keep_idx]

        # Per-sample memory growth, logged only on outer_iter==0 - gives a
        # real, measured per-sample GPU cost for calibrating max_pooled_samples
        # in a future run, rather than repeating the same "reason about it on
        # paper first" mistake that got max_pooled_samples=12 wrong (see
        # fine_tune_stable's docstring on the second real OOM this caused).
        log_mem = outer_iter == 0 and device.startswith("cuda")
        pooled_g, pooled_U = [], []
        for sample_i, (atoms, snap) in enumerate(all_snapshots):
            frame_atoms = _snapshot_to_atoms(snap, atoms)
            pooled_g.append(_stacked_rdf(frame_atoms, rmax, rdf_nbins))
            z_t, pos_t, box_t = tensors_from_atoms(frame_atoms, device)
            pooled_U.append(_config_energy(model, z_t, pos_t, box_t, device))
            if log_mem:
                allocated_gb = torch.cuda.memory_allocated(device) / 1e9
                print(
                    f"  [mem] iter 0, pooled_U sample {sample_i + 1}/{len(all_snapshots)}: "
                    f"{allocated_gb:.2f} GB allocated"
                )

        if len(pooled_U) < 2:
            print(
                f"[iter {outer_iter}] only {len(pooled_U)} sample(s) survived this phase "
                f"(collected={n_collected_this_iter}, rewinds={n_rewinds_this_iter}) - skipping this gradient "
                "step, boltzmann_estimator_pseudo_loss needs N>=2."
            )
            history.append({
                "iter": outer_iter, "skipped": True, "n_rewinds": n_rewinds_this_iter,
                "n_collected": n_collected_this_iter, "n_samples": len(pooled_U),
            })
            continue

        g_tensor = torch.tensor(np.stack(pooled_g), dtype=torch.float32, device=device)
        U_tensor = torch.stack(pooled_U)

        # zero_grad ONCE up front, then two SEPARATE backward() calls
        # (L_obs's own, then L_QM's per-sample ones inside
        # _qm_regularizer_loss_and_backward) rather than summing both into
        # one loss tensor first - gradients accumulate into .grad across
        # repeated backward() calls by default, so this is mathematically
        # identical to backward() on (L_obs + lambda_qm*L_QM) together,
        # just without ever holding L_obs's graph and every L_QM sample's
        # graph in memory simultaneously (see
        # _qm_regularizer_loss_and_backward's docstring for why that
        # exhausted a 47.5 GiB GPU in the first real run attempt).
        optimizer.zero_grad()
        loss_obs_pseudo = boltzmann_estimator_pseudo_loss(g_tensor, U_tensor, g_target, kT)
        loss_obs_pseudo.backward()
        qm_loss_value = _qm_regularizer_loss_and_backward(model, train_data, qm_batch_size, device, rng, lambda_qm)
        optimizer.step()

        with torch.no_grad():
            real_obs_value = float(observable_loss_value(g_tensor, g_target).item())
        record = {
            "iter": outer_iter,
            "skipped": False,
            "n_rewinds": n_rewinds_this_iter,
            "n_collected": n_collected_this_iter,
            "n_samples": len(pooled_U),
            "observable_loss_value": real_obs_value,
            "qm_loss_value": qm_loss_value,
        }
        history.append(record)
        print(
            f"[iter {outer_iter}] n_samples={len(pooled_U)} (collected={n_collected_this_iter}) "
            f"rewinds={n_rewinds_this_iter} L_obs(real)={real_obs_value:.6f} L_QM={record['qm_loss_value']:.6f}"
        )

        if (outer_iter + 1) % save_every == 0:
            _save(f"iter{outer_iter + 1}")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    final_path = _save("final")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"final_checkpoint": final_path, "history_path": str(out_dir / "history.json")}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--n-replicas", type=int, default=4)
    parser.add_argument("--n-outer-iterations", type=int, default=200)
    parser.add_argument("--check-interval", type=int, default=20)
    parser.add_argument("--learn-window-steps", type=int, default=100)
    parser.add_argument("--subsample-stride", type=int, default=2)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--lambda-qm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--qm-batch-size", type=int, default=8)
    parser.add_argument(
        "--max-pooled-samples", type=int, default=5,
        help="Memory ceiling, NOT a diversity knob - see fine_tune_stable's docstring. Caps how many "
        "collected snapshots get forwarded through _config_energy per iteration regardless of "
        "n_replicas, since that computation (unlike L_QM) must hold every sample's create_graph=True "
        "graph simultaneously. Default 5 is the actually-OBSERVED pilot maximum (zero headroom) - a "
        "first attempt at 12 (a theoretical, never-actually-exercised worst case) still OOM'd at "
        "n_replicas=8. Only raise this once iter-0's per-sample [mem] log lines from a real run show "
        "clear headroom below the GPU's capacity.",
    )
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="3 replicas, 3 outer iterations, short-but-not-tiny windows - confirm the whole loop "
        "runs end to end AND actually exercises a real gradient step (checkpoints save, loss "
        "doesn't crash/NaN) before a real run (build order item 3, paper/main.tex "
        "sec:q4-stable-plan). NOT a claim that 3 iterations does anything useful. Sized so a "
        "SINGLE well-behaved replica alone clears the N>=2 sample floor (4 checks/phase, "
        "stride=2 -> 2 samples) even if every other replica happens to be chronically unstable "
        "(a real, expected outcome for some starting configs, not a bug - see paper/main.tex "
        "sec:q4-stable-step3's smoke-test-run discussion) - the original 2-replica/2-check "
        "sizing had no headroom for that and starved on the very first real run.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        fine_tune_stable(
            ckpt=args.ckpt, out_dir=args.out, data_root=args.data_root,
            n_replicas=3, n_outer_iterations=3, check_interval=10, learn_window_steps=40,
            subsample_stride=2, temperature_k=args.temperature_k, friction_fs_inv=args.friction_fs_inv,
            dt_fs=args.dt, lambda_qm=args.lambda_qm, lr=args.lr, qm_batch_size=2,
            n_reference_configs=20, save_every=1, seed=args.seed,
            max_pooled_samples=args.max_pooled_samples,
        )
    else:
        fine_tune_stable(
            ckpt=args.ckpt, out_dir=args.out, data_root=args.data_root,
            n_replicas=args.n_replicas, n_outer_iterations=args.n_outer_iterations,
            check_interval=args.check_interval, learn_window_steps=args.learn_window_steps,
            subsample_stride=args.subsample_stride, temperature_k=args.temperature_k,
            friction_fs_inv=args.friction_fs_inv, dt_fs=args.dt, lambda_qm=args.lambda_qm,
            lr=args.lr, qm_batch_size=args.qm_batch_size, save_every=args.save_every, seed=args.seed,
            max_pooled_samples=args.max_pooled_samples,
        )
