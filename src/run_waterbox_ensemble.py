#!/usr/bin/env python
"""Run one documented NVT or isotropic-NPT WaterBox property trajectory.

This is separate from rollout_waterbox_ase.py by design.  The existing NVE
workflow is a stability/energy-conservation diagnostic; this workflow uses a
thermostat or thermostat+barostat to estimate equilibrium structural and
thermophysical properties.  NPT additionally requires TensorNetCalculator's
stress implementation to pass verify_waterbox_stress.py on the remote GPU.

Examples:
    python src/run_waterbox_ensemble.py --ensemble nvt --ckpt CKPT --out OUT \
      --data-seed 0 --equilibration-ps 1 --production-ps 2
    python src/run_waterbox_ensemble.py --ensemble npt --ckpt CKPT --out OUT \
      --data-seed 0 --equilibration-ps 20 --production-ps 100
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import numpy as np
from ase import units
from ase.io import write as ase_write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from waterbox_ase import TensorNetCalculator, atoms_from_waterbox_sample
from waterbox_data import load_waterbox_dataset, random_split


AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


class PropertyRunAbort(RuntimeError):
    pass


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("ase", "numpy", "torch", "torchmd-net"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-found"
    return versions


def _build_dynamics(atoms, ensemble, dt_fs, temperature_k, pressure_bar, tdamp_fs, pdamp_fs):
    if ensemble == "nvt":
        try:
            from ase.md.nose_hoover_chain import NoseHooverChainNVT
        except ImportError as exc:
            raise ImportError(
                "This workflow requires ASE's NoseHooverChainNVT. Upgrade ASE on the remote "
                "machine rather than silently switching thermostat algorithms."
            ) from exc
        return NoseHooverChainNVT(
            atoms,
            timestep=dt_fs * units.fs,
            temperature_K=temperature_k,
            tdamp=tdamp_fs * units.fs,
        )

    try:
        from ase.md.nose_hoover_chain import IsotropicMTKNPT
    except ImportError as exc:
        raise ImportError(
            "NPT requires ASE's IsotropicMTKNPT. Upgrade ASE and run "
            "verify_waterbox_stress.py before starting NPT."
        ) from exc
    return IsotropicMTKNPT(
        atoms,
        timestep=dt_fs * units.fs,
        temperature_K=temperature_k,
        pressure_au=pressure_bar * units.bar,
        tdamp=tdamp_fs * units.fs,
        pdamp=pdamp_fs * units.fs,
    )


def _density_g_cm3(atoms) -> float:
    return float(atoms.get_masses().sum() / atoms.get_volume() * AMU_PER_A3_TO_G_PER_CM3)


def _pressure_bar(atoms) -> float:
    # Match the pressure used by an NPT integrator by including the kinetic
    # (ideal-gas) contribution, not just the configurational calculator stress.
    stress = np.asarray(atoms.get_stress(voigt=False, include_ideal_gas=True), dtype=float)
    return float(-np.trace(stress) / 3.0 / units.bar)


def run_ensemble(
    *, checkpoint: str, ensemble: str, out: str, data_root: str = "./data",
    data_seed: int, test_config_index: int = 0, velocity_seed: int = 0,
    temperature_k: float = 300.0, pressure_bar: float = 1.0,
    dt_fs: float = 0.1, equilibration_ps: float = 20.0, production_ps: float = 100.0,
    tdamp_fs: float = 100.0, pdamp_fs: float = 1000.0,
    log_interval_steps: int = 10, trajectory_interval_steps: int = 100,
    maximum_temperature_k: float = 5000.0, minimum_density_g_cm3: float = 0.2,
    maximum_density_g_cm3: float = 2.0,
    stress_fd_epsilon: float = 0.003,
) -> dict:
    if ensemble not in {"nvt", "npt"}:
        raise ValueError(f"ensemble must be 'nvt' or 'npt', got {ensemble!r}")
    for name, value in (("dt_fs", dt_fs), ("equilibration_ps", equilibration_ps),
                        ("production_ps", production_ps), ("tdamp_fs", tdamp_fs)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if ensemble == "npt" and pdamp_fs <= 0:
        raise ValueError(f"pdamp_fs must be positive, got {pdamp_fs}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "thermodynamic_history.csv"
    trajectory_path = out_dir / "production.xyz"
    manifest_path = out_dir / "run_manifest.json"

    dataset = load_waterbox_dataset(data_root)
    _, _, test_data = random_split(dataset, seed=data_seed)
    if not 0 <= test_config_index < len(test_data):
        raise IndexError(f"test_config_index={test_config_index} outside 0..{len(test_data) - 1}")
    atoms = atoms_from_waterbox_sample(test_data[test_config_index])
    atoms.calc = TensorNetCalculator(
        checkpoint,
        stress_mode="hydrostatic_fd" if ensemble == "npt" else "analytic",
        stress_fd_epsilon=stress_fd_epsilon,
    )
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_k, rng=np.random.RandomState(velocity_seed)
    )
    Stationary(atoms)

    dynamics = _build_dynamics(
        atoms, ensemble, dt_fs, temperature_k, pressure_bar, tdamp_fs, pdamp_fs
    )
    equilibration_steps = int(round(equilibration_ps * 1000.0 / dt_fs))
    production_steps = int(round(production_ps * 1000.0 / dt_fs))
    total_steps = equilibration_steps + production_steps
    if equilibration_steps < 1 or production_steps < 1:
        raise ValueError("equilibration and production must each contain at least one MD step")

    history = []
    production_frames = []
    aborted_reason = None

    def phase() -> str:
        return "equilibration" if dynamics.nsteps < equilibration_steps else "production"

    def record_history() -> None:
        nonlocal aborted_reason
        epot = float(atoms.get_potential_energy())
        ekin = float(atoms.get_kinetic_energy())
        temperature = float(atoms.get_temperature())
        density = _density_g_cm3(atoms)
        pressure = _pressure_bar(atoms) if ensemble == "npt" else float("nan")
        row = {
            "step": dynamics.nsteps,
            "time_fs": dynamics.nsteps * dt_fs,
            "phase": phase(),
            "epot_ev": epot,
            "ekin_ev": ekin,
            "etot_ev": epot + ekin,
            "temperature_k": temperature,
            "volume_a3": float(atoms.get_volume()),
            "density_g_cm3": density,
            "pressure_bar": pressure,
        }
        history.append(row)
        finite_values = [epot, ekin, temperature, density, atoms.get_volume()]
        if not all(math.isfinite(value) for value in finite_values):
            aborted_reason = "non-finite thermodynamic value"
        elif temperature > maximum_temperature_k:
            aborted_reason = f"temperature exceeded {maximum_temperature_k:g} K"
        elif density < minimum_density_g_cm3 or density > maximum_density_g_cm3:
            aborted_reason = (
                f"density {density:.6g} g/cm^3 outside "
                f"[{minimum_density_g_cm3:g}, {maximum_density_g_cm3:g}]"
            )
        if aborted_reason:
            raise PropertyRunAbort(aborted_reason)

    def record_trajectory() -> None:
        if phase() == "production":
            production_frames.append(atoms.copy())

    dynamics.attach(record_history, interval=log_interval_steps)
    dynamics.attach(record_trajectory, interval=trajectory_interval_steps)

    status = "complete"
    try:
        dynamics.run(total_steps)
    except PropertyRunAbort:
        status = "aborted"
    except Exception as exc:
        status = "failed"
        aborted_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if history:
            with history_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
        if production_frames:
            ase_write(str(trajectory_path), production_frames)
        manifest = {
            "status": status,
            "abort_reason": aborted_reason,
            "checkpoint": checkpoint,
            "ensemble": ensemble,
            "data_root": data_root,
            "data_seed": data_seed,
            "test_config_index": test_config_index,
            "velocity_seed": velocity_seed,
            "temperature_k": temperature_k,
            "pressure_bar": pressure_bar if ensemble == "npt" else None,
            "dt_fs": dt_fs,
            "equilibration_ps_requested": equilibration_ps,
            "production_ps_requested": production_ps,
            "tdamp_fs": tdamp_fs,
            "pdamp_fs": pdamp_fs if ensemble == "npt" else None,
            "stress_mode": "hydrostatic_fd" if ensemble == "npt" else None,
            "stress_fd_epsilon": stress_fd_epsilon if ensemble == "npt" else None,
            "log_interval_steps": log_interval_steps,
            "trajectory_interval_steps": trajectory_interval_steps,
            "steps_completed": dynamics.nsteps,
            "last_time_fs": dynamics.nsteps * dt_fs,
            "history_path": str(history_path),
            "trajectory_path": str(trajectory_path) if production_frames else None,
            "n_production_frames": len(production_frames),
            "software_versions": _software_versions(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if status == "aborted":
        print(f"ABORTED: {aborted_reason}")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {history_path}")
    if production_frames:
        print(f"Wrote: {trajectory_path} ({len(production_frames)} frames)")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", choices=("nvt", "npt"), required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--data-seed", type=int, required=True,
                        help="Use the training seed here so the starting structure comes from that model's test split.")
    parser.add_argument("--test-config-index", type=int, default=0)
    parser.add_argument("--velocity-seed", type=int, default=0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--equilibration-ps", type=float, default=20.0)
    parser.add_argument("--production-ps", type=float, default=100.0)
    parser.add_argument("--tdamp-fs", type=float, default=100.0)
    parser.add_argument("--pdamp-fs", type=float, default=1000.0)
    parser.add_argument("--stress-fd-epsilon", type=float, default=0.003)
    parser.add_argument("--log-interval-steps", type=int, default=10)
    parser.add_argument("--trajectory-interval-steps", type=int, default=100)
    parser.add_argument("--maximum-temperature-k", type=float, default=5000.0)
    parser.add_argument("--minimum-density-g-cm3", type=float, default=0.2)
    parser.add_argument("--maximum-density-g-cm3", type=float, default=2.0)
    args = parser.parse_args()
    run_ensemble(
        checkpoint=args.ckpt, ensemble=args.ensemble, out=args.out,
        data_root=args.data_root, data_seed=args.data_seed,
        test_config_index=args.test_config_index, velocity_seed=args.velocity_seed,
        temperature_k=args.temperature_k, pressure_bar=args.pressure_bar,
        dt_fs=args.dt_fs, equilibration_ps=args.equilibration_ps,
        production_ps=args.production_ps, tdamp_fs=args.tdamp_fs, pdamp_fs=args.pdamp_fs,
        log_interval_steps=args.log_interval_steps,
        trajectory_interval_steps=args.trajectory_interval_steps,
        maximum_temperature_k=args.maximum_temperature_k,
        minimum_density_g_cm3=args.minimum_density_g_cm3,
        maximum_density_g_cm3=args.maximum_density_g_cm3,
        stress_fd_epsilon=args.stress_fd_epsilon,
    )


if __name__ == "__main__":
    main()
