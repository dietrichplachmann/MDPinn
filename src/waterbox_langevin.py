#!/usr/bin/env python
"""StABlE-style thermostatted sampler with instability rewind, step 2 of the
stability-aware fine-tuning plan (paper/main.tex sec:q4-stable-plan/
sec:q4-stable-path). Generates the on-policy configuration samples
boltzmann_estimator.py's pseudo-loss needs: run each replica's own Langevin
dynamics under the CURRENT model, watch for the same short-range collapse
this project already diagnosed for Q4 (diagnose_short_range_collapse.py),
and rewind to the last known-good state on a trip rather than let a replica
wander into an unphysical, uninformative region of phase space.

Split into two halves, matching diagnose_short_range_collapse.py's own
established pattern (see that module's docstring) precisely so the same
"what's testable here vs. what needs the training box" boundary applies:

  - Snapshot / ReplicaState (below): plain numpy bookkeeping, no ase/torch
    dependency at all - exercisable synthetically on this Windows checkout.
    See src/verify_waterbox_langevin.py for the synthetic test.
  - run_stable_sampling_phase: the actual ASE-driven dynamics. Needs a real
    ase + torchmdnet install - NOT executable on this checkout, and NOT yet
    smoke-tested anywhere (see this module's Usage section for the exact
    command to run on the training box before trusting it).

Design decisions checked against ASE's actual documented behavior rather
than assumed (this project's own established discipline - see
verify_zbl_units.py/verify_periodicity.py for why a plausible-looking
default isn't trusted without checking):

  - `friction` is passed to ase.md.langevin.Langevin already divided by
    ase.units.fs (ASE's own documented convention, e.g. `0.01 / units.fs`
    for a physical 0.01 fs^-1 rate) - the OPPOSITE direction from
    `timestep=dt*units.fs` already used elsewhere in this project
    (rollout_waterbox_ase.py). Getting this backwards would silently make
    the thermostat couple ~10x too weakly or too strongly (units.fs is
    ASE's internal-time-unit-per-fs conversion factor, ~0.0982), not crash -
    exactly the kind of unit mistake this project has been bitten by before
    (waterbox_data.py's Bohr/Hartree mixup). This module's own
    `friction_fs_inv` parameter is named to make the physical (fs^-1) value
    the caller supplies unambiguous, with the /units.fs conversion done
    once, internally, at the single call site.
  - After every rewind (an external, out-of-band reset of atoms' positions
    and velocities via set_positions/set_velocities), a FRESH Langevin
    object is constructed rather than continuing to call .run() on the old
    one. ASE's own documentation does not clearly state whether Langevin
    tolerates an externally-mutated Atoms state between .run() calls
    without carrying stale internal bookkeeping (e.g. cached half-step
    random-force draws) - rather than trust an unverified assumption about
    undocumented internal continuity, this sidesteps the question entirely.
    Reconstructing costs one cheap Python object per rewind, negligible
    next to the simulation cost itself.
  - `rng=np.random.RandomState(seed)` explicitly, not ASE's own
    `rng=np.random` default - this project already learned the hard way
    (CLAUDE.md's Lessons) that ASE's MaxwellBoltzmannDistribution silently
    draws from whatever state numpy's GLOBAL rng happens to be in unless
    told otherwise; the same discipline is applied here for Langevin's own
    random force draws and for the post-rewind velocity redraw.

Usage (training box only, once ase/torchmdnet are installed - NOT yet run):
    python src/waterbox_langevin.py --ckpt checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed1/best_model.ckpt --smoke-test
  This is specifically a THERMOSTAT sanity check, not yet a full StABlE
  phase: runs one replica, logs temperature over a short Langevin
  equilibration, and prints whether it settles near --temperature-k. If the
  friction unit convention above were wrong, this would show up directly
  here (barely any coupling -> temperature keeps drifting like plain NVE;
  wildly overdamped -> temperature collapses/oscillates nonphysically) -
  check this BEFORE trusting anything built on top of this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Snapshot:
    """One replica's full dynamical state at a given simulation step -
    enough to exactly resume or rewind to it (positions AND velocities;
    velocities matter because ase.md dynamics objects read live momenta
    off the Atoms object, not just positions)."""

    positions: np.ndarray  # (N, 3)
    velocities: np.ndarray  # (N, 3)
    step: int


class ReplicaState:
    """Per-replica bookkeeping for the StABlE simulate/rewind/collect loop
    (paper/main.tex sec:q4-stable-path). Pure numpy, no ase/torch dependency
    - see verify_waterbox_langevin.py for a synthetic test of this class in
    isolation from any real dynamics.

    Usage from run_stable_sampling_phase's per-check loop:
        if stability check passes: state.record_stable_frame(snapshot)
        else: snapshot_to_reload = state.rewind()
        ... at the end of a learn-window's worth of steps:
        samples_for_this_gradient_step = state.take_learn_window(stride)
    """

    def __init__(self, initial_snapshot: Snapshot):
        self.last_good: Snapshot = initial_snapshot
        self.n_rewinds: int = 0
        self.collected: list = []

    def record_stable_frame(self, snapshot: Snapshot) -> None:
        """Call after a step that PASSED the stability check."""
        self.last_good = snapshot
        self.collected.append(snapshot)

    def rewind(self) -> Snapshot:
        """Call after a step that FAILED the stability check. Discards
        whatever was collected since the last rewind (that window is now
        invalidated - it ended in an unphysical state) and returns the last
        known-good snapshot for the caller to reload into the live Atoms
        object. The caller is responsible for redrawing fresh velocities
        there (module docstring) - this class only tracks state, it doesn't
        touch ase objects itself, keeping it dependency-free and testable
        here."""
        self.n_rewinds += 1
        self.collected = []
        return self.last_good

    def take_learn_window(self, stride: int) -> list:
        """Subsample every stride-th collected frame for one StABlE
        gradient step ("sampling every S-th state to obtain uncorrelated
        samples", paper/main.tex sec:q4-stable-path), then clear the
        window - each gradient step uses a FRESH simulation window, not a
        growing/reused buffer (see boltzmann_estimator.py's docstring on
        why this project's estimator is deliberately on-policy, unlike
        DiffTRe's multi-step importance-reweighting variant). May return
        fewer than expected (even zero) frames if instability consumed
        most or all of the window via repeated rewinds - the caller (via
        boltzmann_estimator_pseudo_loss's own N>=2 check) is responsible
        for skipping a gradient step when too few samples survived, rather
        than this class silently padding or fabricating samples."""
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        window = self.collected[::stride]
        self.collected = []
        return window


def run_stable_sampling_phase(
    replica_atoms,
    replica_states,
    same_molecule_masks,
    floors,
    dt_fs,
    temperature_k,
    friction_fs_inv,
    check_interval,
    learn_window_steps,
    subsample_stride,
    rng_seed,
):
    """Advances every replica's Langevin dynamics learn_window_steps steps,
    checking the Q4 short-range-collapse stability criterion
    (diagnose_short_range_collapse.frame_violates_floors) every
    check_interval steps and rewinding on a trip. Returns a list (one entry
    per replica) of subsampled Snapshot lists - hand these to whatever
    builds boltzmann_estimator_pseudo_loss's g/U tensors (re-evaluate the
    CURRENT model with grad enabled at each Snapshot's positions to get U;
    compute the target observable, e.g. RDF, from the positions to get g;
    not yet built - train_waterbox_stable.py, step 3 of the plan).

    replica_atoms: list of ASE Atoms, one per replica, .calc already
        attached (a TensorNetCalculator wrapping the model currently being
        fine-tuned - see this module's docstring on why a single
        Calculator instance holding a live reference to the model object
        automatically reflects later optimizer.step() updates with no
        extra plumbing, since PyTorch mutates parameters in place).
    replica_states: list of ReplicaState, one per replica, already
        initialized from each atoms object's starting Snapshot.
    same_molecule_masks: list of (N, N) bool arrays, one per replica - each
        replica's own fixed molecule-membership mask (computed once from
        ITS starting frame via diagnose_short_range_collapse.
        molecule_group_ids + same_molecule_mask), matching
        analyze_trajectory's own established convention of treating
        topology as fixed rather than re-inferring bonds every frame.
    floors: dict from diagnose_short_range_collapse.compute_reference_floors.
    friction_fs_inv: Langevin friction coefficient in PHYSICAL fs^-1 units
        (this function does the /ase.units.fs conversion internally - see
        module docstring). Typical literature range 1e-4 to 1e-2 fs^-1 per
        ASE's own docstring; not yet empirically tuned for this system -
        verify via this module's --smoke-test before trusting a specific
        value in a real fine-tuning run.

    NOT unit-testable without ase/torchmdnet (needs a real Langevin object
    and a real Calculator) - run this module's --smoke-test on the training
    box before trusting anything here (see module docstring's Usage).
    """
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    from diagnose_short_range_collapse import frame_min_distances, frame_violates_floors

    all_collected = []
    for replica_idx, (atoms, state, same_molecule) in enumerate(
        zip(replica_atoms, replica_states, same_molecule_masks)
    ):
        rng = np.random.RandomState(rng_seed + replica_idx)  # distinct stream per replica, still reproducible
        dyn = Langevin(
            atoms,
            timestep=dt_fs * units.fs,
            temperature_K=temperature_k,
            friction=friction_fs_inv / units.fs,
            rng=rng,
        )

        steps_done = 0
        while steps_done < learn_window_steps:
            n = min(check_interval, learn_window_steps - steps_done)
            dyn.run(n)
            steps_done += n

            z = atoms.get_atomic_numbers()
            positions = atoms.get_positions()
            box_lengths = np.array(atoms.get_cell()).diagonal()
            min_dists = frame_min_distances(z, positions, box_lengths, same_molecule)

            if frame_violates_floors(min_dists, floors):
                good = state.rewind()
                atoms.set_positions(good.positions)
                atoms.set_velocities(good.velocities)
                # Redraw fresh velocities rather than reusing the rewound
                # snapshot's own (module docstring: a rewound replica
                # shouldn't deterministically repeat the same failure).
                MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
                Stationary(atoms)
                # Fresh Langevin object - see module docstring on why this
                # avoids relying on unverified internal-state continuity
                # across an external hard reset.
                dyn = Langevin(
                    atoms,
                    timestep=dt_fs * units.fs,
                    temperature_K=temperature_k,
                    friction=friction_fs_inv / units.fs,
                    rng=rng,
                )
            else:
                snapshot = Snapshot(
                    positions=positions.copy(),
                    velocities=atoms.get_velocities().copy(),
                    step=dyn.nsteps,
                )
                state.record_stable_frame(snapshot)

        all_collected.append(state.take_learn_window(subsample_stride))

    return all_collected


def _smoke_test_thermostat(ckpt, data_root, temperature_k, friction_fs_inv, dt_fs, steps, energy_log_stride, seed):
    """Thermostat-only sanity check (module docstring's Usage) - deliberately
    NOT a full StABlE phase (no stability check, no rewind, single replica):
    the one thing to verify before anything else is that Langevin actually
    equilibrates temperature toward temperature_k at a physically sensible
    rate, which would directly expose a friction-unit mistake (module
    docstring)."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    from waterbox_ase import TensorNetCalculator, atoms_from_waterbox_sample
    from waterbox_data import load_waterbox_dataset, random_split

    full_dataset = load_waterbox_dataset(data_root=data_root)
    _, _, test_data = random_split(full_dataset, seed=seed)
    sample = test_data[0]
    atoms = atoms_from_waterbox_sample(sample)
    atoms.calc = TensorNetCalculator(ckpt)

    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=temperature_k,
        friction=friction_fs_inv / units.fs,
        rng=rng,
    )

    temps = []

    def _record():
        t = atoms.get_temperature()
        temps.append(t)
        print(f"  step {dyn.nsteps:6d}  T = {t:8.2f} K")

    dyn.attach(_record, interval=energy_log_stride)
    print(
        f"Running {steps} steps of {dt_fs} fs Langevin (friction={friction_fs_inv} fs^-1, "
        f"target T={temperature_k} K) - watch for equilibration, not drift or collapse..."
    )
    dyn.run(steps)

    tail = np.array(temps[-max(1, len(temps) // 3):])
    print(
        f"\nTail mean T = {tail.mean():.1f} +/- {tail.std():.1f} K "
        f"(target {temperature_k} K). If this is far off target or still trending "
        f"monotonically at the end of the run, the friction magnitude/unit convention "
        f"needs revisiting before trusting this module further."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--energy-log-stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Thermostat-only sanity check (see module docstring) - NOT a full StABlE phase.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test_thermostat(
            ckpt=args.ckpt, data_root=args.data_root, temperature_k=args.temperature_k,
            friction_fs_inv=args.friction_fs_inv, dt_fs=args.dt, steps=args.steps,
            energy_log_stride=args.energy_log_stride, seed=args.seed,
        )
    else:
        print(
            "No standalone full-phase entrypoint yet - run_stable_sampling_phase is called "
            "from train_waterbox_stable.py (not yet built, step 3 of the plan). Use "
            "--smoke-test to sanity-check the Langevin thermostat alone first."
        )
