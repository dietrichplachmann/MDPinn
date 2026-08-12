#!/usr/bin/env python
"""Real-dynamics evaluation for water-box checkpoints: NVE rollout via ASE's
VelocityVerlet (periodic-boundary-aware, unlike rollout_nve.py - see
waterbox_ase.py's docstring for why ASE rather than hand-rolled PBC) plus the
two standard checks for whether an NNIP produces stable, physically realistic
MD (paper/main.tex Section 5.2/sec:q2; Morrow/Gardner/Deringer, cited there):
total-energy drift over the trajectory, and O-O/O-H/H-H radial distribution
functions compared against a reference computed from real DFT configurations.

NVE, not NVT: a thermostat would mask exactly the instability this is meant
to detect (a bad potential's energy drift gets corrected away by the
thermostat instead of showing up in the diagnostic).

Starting configuration: a real held-out WaterBox test configuration's
positions (same test split as evaluate_waterbox.py, seed=42) - a physically
reasonable starting geometry, not a synthetic/relaxed one. Initial velocities
sampled from a Maxwell-Boltzmann distribution at --temperature-k (default
300 K, a standard liquid-water benchmark temperature - the exact AIMD
sampling temperature of Cheng et al. 2019's dataset is NOT confirmed here;
override if that matters for your comparison). Net momentum is zeroed
afterward (ase.md.velocitydistribution.Stationary) so reported temperature
reflects internal/thermal motion, not center-of-mass drift - doesn't affect
RDF (translation-invariant) or total energy conservation either way.

RDF reference: computed from a random sample of the raw WaterBox dataset's
own configurations (independent DFT snapshots, not a trajectory - valid for
RDF, which is just a pair-distance histogram, no time-correlation needed),
via the same ase.geometry.rdf.get_rdf call used on the rollout trajectory, so
both are computed identically and are directly comparable.

--rdf-rmax defaults to 6.0 A: this dataset's box edge is ~12.4-13.7 A
(verify_periodicity.py), and get_rdf's minimum-image convention requires
rmax <= (box edge)/2 - same constraint TorchMD-Net's own docs state for the
model's cutoff (Section 3.4, label sec:periodicity), so 6.0 A stays safely under
that bound across every config in this dataset.

Usage:
    python src/rollout_waterbox_ase.py --ckpt checkpoints/waterbox_study/water_absolute/seed0/best_model.ckpt --steps 20
    # confirm the plumbing end-to-end (a few seconds) before a real rollout:
    python src/rollout_waterbox_ase.py --ckpt checkpoints/waterbox_study/water_absolute/seed0/best_model.ckpt --steps 2000

IMPORTANT - written without ase/torchmdnet installed locally (same caveat as
waterbox_ase.py and every other water-box script in this repo). Nothing here
has been executed yet - run the --steps 20 smoke test first.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from ase import units
from ase.geometry.rdf import get_rdf
from ase.io import write as ase_write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet

from waterbox_ase import TensorNetCalculator, atoms_from_waterbox_sample
from waterbox_data import load_waterbox_dataset, random_split

RESULTS_ROOT = Path("results/waterbox_rollout")

# (name, (atomic_number_1, atomic_number_2)) - O=8, H=1.
ELEMENT_PAIRS = [("O-O", (8, 8)), ("O-H", (8, 1)), ("H-H", (1, 1))]


def _averaged_rdf(frames, rmax, nbins, elements):
    """get_rdf accepting an iterable of Atoms to average over is an ASE
    >=3.28.0 feature; fall back to per-frame get_rdf + manual averaging on
    older installs. Confirmed on the training box's installed ASE (<3.28):
    it doesn't reject a list up front with TypeError - it accepts it, then
    fails inside with AttributeError ('list' object has no attribute
    'cell') because it unconditionally tries atoms.cell.volume. Catch both
    rather than just the TypeError originally guessed at."""
    try:
        rdf, rr = get_rdf(frames, rmax=rmax, nbins=nbins, elements=elements)
        return np.asarray(rdf), np.asarray(rr)
    except (TypeError, AttributeError):
        rdfs = []
        rr = None
        for frame in frames:
            single_rdf, rr = get_rdf(frame, rmax=rmax, nbins=nbins, elements=elements)
            rdfs.append(single_rdf)
        return np.mean(rdfs, axis=0), rr


def _sample_reference_frames(full_dataset, n_samples=200, seed=42):
    """n_samples raw dataset configurations (not the rollout's trajectory) -
    the "what does real liquid water actually look like" comparison point.
    Sampled once and reused across all element pairs, rather than resampled
    per pair - also lets the caller inspect box sizes across the same set
    used for the RDF, needed for _safe_rmax below."""
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, len(full_dataset))
    indices = rng.choice(len(full_dataset), size=n_samples, replace=False)
    return [atoms_from_waterbox_sample(full_dataset[int(i)]) for i in indices]


def _safe_rmax(frames, requested_rmax):
    """get_rdf's minimum-image convention requires rmax < (smallest box
    edge)/2 - box size varies configuration to configuration in this dataset
    (confirmed on the training box: as low as ~11.94 A, versus the ~12.4-13.7
    A range verify_periodicity.py happened to sample), so a single fixed
    constant isn't safe across an arbitrary batch of configs. Compute the
    real limit from whichever frames are actually in play instead of
    guessing a smaller constant and risking the same failure on a still
    smaller box elsewhere in the 1593 configs."""
    min_edge = min(float(min(frame.cell.lengths())) for frame in frames)
    safe = min(requested_rmax, 0.45 * min_edge)
    if safe < requested_rmax:
        print(
            f"Reducing rdf_rmax from {requested_rmax} to {safe:.3f} A - smallest box edge "
            f"among the frames used ({min_edge:.3f} A) doesn't support the requested cutoff "
            "under minimum-image convention (needs rmax < edge/2)."
        )
    return safe


def run_rollout(
    ckpt,
    data_root="./data",
    steps=2000,
    dt=0.5,
    temperature_k=300.0,
    seed=42,
    test_config_index=0,
    energy_log_stride=10,
    rdf_rmax=6.0,
    rdf_nbins=200,
    out=None,
):
    out_dir = Path(out) if out else RESULTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    full_dataset = load_waterbox_dataset(data_root=data_root)
    _, _, test_data = random_split(full_dataset, seed=seed)
    sample = test_data[test_config_index]

    atoms = atoms_from_waterbox_sample(sample)
    atoms.calc = TensorNetCalculator(ckpt)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k)
    Stationary(atoms)

    dyn = VelocityVerlet(atoms, timestep=dt * units.fs)

    history = []
    trajectory_frames = []

    def _record():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        history.append({
            # nsteps is a plain attribute, not get_number_of_steps() (that
            # method doesn't exist on ASE's MolecularDynamics - checked
            # against ase/md/md.py's source rather than assumed).
            "step": dyn.nsteps,
            "time_fs": dyn.nsteps * dt,
            "epot_ev": epot,
            "ekin_ev": ekin,
            "etot_ev": epot + ekin,
            "temperature_k": atoms.get_temperature(),
        })
        trajectory_frames.append(atoms.copy())

    # No manual pre-call needed here: ase/md/md.py's irun() already calls
    # every attached observer once at nsteps==0 before the first integration
    # step ("for historical reasons", per its own comment) - an explicit
    # _record() call here duplicated that, producing an exact-duplicate
    # step-0 row/frame (confirmed on the training box's first real output).
    dyn.attach(_record, interval=energy_log_stride)

    print(f"Running {steps} steps of {dt} fs NVE (VelocityVerlet), "
          f"logging every {energy_log_stride} steps...")
    dyn.run(steps)

    history_path = out_dir / "energy_history.csv"
    with open(history_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Wrote: {history_path}")

    trajectory_path = out_dir / "rollout.xyz"
    ase_write(str(trajectory_path), trajectory_frames)
    print(f"Wrote: {trajectory_path} ({len(trajectory_frames)} frames)")

    etot = np.array([row["etot_ev"] for row in history])
    n_atoms = len(atoms)
    drift_ev_per_atom = (etot[-1] - etot[0]) / n_atoms
    drift_fraction = (etot[-1] - etot[0]) / abs(etot[0]) if etot[0] != 0 else float("nan")
    total_ps = steps * dt / 1000
    print(
        f"Total energy drift: {drift_ev_per_atom * 1000:.4f} meV/atom over {total_ps:.3f} ps "
        f"({drift_fraction * 100:.4f}% of total starting energy)"
    )

    reference_frames = _sample_reference_frames(full_dataset, seed=seed)
    # One rmax shared by every RDF call below, so every column in rdf.csv
    # lands on the same r-grid - computed from BOTH the rollout's (fixed,
    # single) box and the reference sample's (varying) box sizes, not just
    # the rollout's, since the reference sample is what actually triggered
    # CellTooSmall on the training box.
    rmax = _safe_rmax(trajectory_frames + reference_frames, rdf_rmax)

    rdf_path = out_dir / "rdf.csv"
    header = ["r_angstrom"]
    columns = {}
    for name, elems in ELEMENT_PAIRS:
        rollout_rdf, rr = _averaged_rdf(trajectory_frames, rmax, rdf_nbins, elems)
        ref_rdf, _ = _averaged_rdf(reference_frames, rmax, rdf_nbins, elems)
        columns[f"{name}_rollout"] = rollout_rdf
        columns[f"{name}_reference"] = ref_rdf
        header += [f"{name}_rollout", f"{name}_reference"]

    with open(rdf_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, r in enumerate(rr):
            writer.writerow([r] + [columns[col][i] for col in header[1:]])
    print(f"Wrote: {rdf_path}")

    return {
        "history_path": str(history_path),
        "trajectory_path": str(trajectory_path),
        "rdf_path": str(rdf_path),
        "drift_ev_per_atom": float(drift_ev_per_atom),
        "drift_fraction": float(drift_fraction),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-config-index", type=int, default=0)
    parser.add_argument("--energy-log-stride", type=int, default=10)
    parser.add_argument("--rdf-rmax", type=float, default=6.0)
    parser.add_argument("--rdf-nbins", type=int, default=200)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    run_rollout(
        ckpt=args.ckpt,
        data_root=args.data_root,
        steps=args.steps,
        dt=args.dt,
        temperature_k=args.temperature_k,
        seed=args.seed,
        test_config_index=args.test_config_index,
        energy_log_stride=args.energy_log_stride,
        rdf_rmax=args.rdf_rmax,
        rdf_nbins=args.rdf_nbins,
        out=args.out,
    )
