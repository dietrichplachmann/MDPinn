#!/usr/bin/env python
"""Paired, resumable WaterBox NVT -> NVE evaluation of existing checkpoints.

Default: six training seeds x three intact held-out starts x two velocity
draws x two conditions. Runs are sequential on one GPU. No training, NPT,
rewinds, or velocity resets at the phase boundary. Structural flags are
operational diagnostics, not universal definitions of chemical validity.
They do not stop integration; numerical failure/overheating does.

The saved study plan freezes settings, configuration selection, dataset,
checkpoint hashes and source hashes. Completed or numerically aborted runs
are retained on restart. Interrupted/error attempts are preserved and rerun
from their original start in a new attempt directory.

Use --prepare-only to validate all twelve models before launching the matrix;
--analyze-only regenerates tables without importing ASE or TorchMD-Net.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path


CONDITIONS = ("water_absolute", "water_absolute+momentum")
TERMINAL = {"complete", "aborted"}
RULES = {
    "minimum_hoh_angle_deg": 80.0,
    "minimum_intramolecular_hh_a": 1.2,
    "minimum_intermolecular_oo_a": 2.3,
    "maximum_assigned_oh_a": 1.3,
    "minimum_intermolecular_oh_a": 1.3,
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def structural_flags(row, rules=RULES):
    checks = (
        ("angle_below_floor", "hoh_angle_min_deg", "minimum_hoh_angle_deg", False),
        ("intramolecular_hh_below_floor", "intramolecular_hh_min_a", "minimum_intramolecular_hh_a", False),
        ("intermolecular_oo_below_floor", "intermolecular_oo_min_a", "minimum_intermolecular_oo_a", False),
        ("assigned_oh_above_ceiling", "oh_max_a", "maximum_assigned_oh_a", True),
        ("intermolecular_oh_below_floor", "intermolecular_oh_min_a", "minimum_intermolecular_oh_a", False),
    )
    return [name for name, metric, limit, above in checks
            if (row[metric] > rules[limit] if above else row[metric] < rules[limit])]


def choose_starts(candidates, count):
    """First eligible test-permutation entries; never inspect model outputs."""
    eligible = [r for r in sorted(candidates, key=lambda r: r["test_config_index"])
                if r["topology_clean"] and not r["initial_flags"]]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible starts; need {count}.")
    return eligible[:count]


def last_attempt(case_dir):
    paths = sorted(Path(case_dir).glob("attempt_*/run_manifest.json"))
    return json.loads(paths[-1].read_text()) if paths else None


def versions():
    result = {"python": platform.python_version()}
    for name in ("numpy", "ase", "torch", "torchmd-net", "torchmdnet", "lightning"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-found"
    return result


def prepare(args):
    import numpy as np
    import torch
    import torchmdnet
    from analyze_waterbox_property_trajectory import _frame_metrics
    from audit_waterbox_topology import _fixed_topology, _config_metrics
    from waterbox_ase import atoms_from_waterbox_sample
    from waterbox_data import load_waterbox_dataset, random_split

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate the remote training environment.")
    root = Path(args.checkpoint_root)
    checkpoints = {}
    for seed in args.seeds:
        for condition in CONDITIONS:
            p = root / condition / f"seed{seed}" / "best_model.ckpt"
            if not p.is_file():
                raise FileNotFoundError(p)
            config = json.loads(p.with_name("config.json").read_text())
            if config["training"]["seed"] != seed:
                raise ValueError(f"Training seed mismatch: {p}")
            if config["model_args"]["prior_model"] != "MolecularZBL":
                raise ValueError(f"Expected the bonded-exclusion checkpoint: {p}")
            checkpoints[f"{seed}/{condition}"] = {
                "path": str(p.resolve()), "sha256": digest(p),
                "config_sha256": digest(p.with_name("config.json")),
            }

    dataset = load_waterbox_dataset(args.data_root)
    if len(dataset) != 1593:
        raise ValueError(f"Expected 1593 WaterBox configurations, got {len(dataset)}")
    topology = _fixed_topology(dataset)
    numbers, group_ids, _, _ = topology
    groups = [np.flatnonzero(group_ids == g) for g in np.unique(group_ids)]
    # Include labels as well as geometry, ordering and cells in the dataset ID.
    h = hashlib.sha256()
    for i in range(len(dataset)):
        sample = dataset[i]
        for name in ("z", "pos", "box", "y", "neg_dy"):
            array = getattr(sample, name).detach().cpu().contiguous().numpy()
            h.update(str((i, name, array.shape, str(array.dtype))).encode())
            h.update(array.tobytes())

    starts, audit = {}, []
    for seed in args.seeds:
        train, val, test = random_split(dataset, seed=seed)
        for index, raw_index in enumerate(test.indices):
            sample = dataset[int(raw_index)]
            metrics = _config_metrics(sample, *topology)
            structure = _frame_metrics(atoms_from_waterbox_sample(sample), group_ids, groups)
            clean = (metrics["assigned_oh_max_a"] <= 1.3
                     and metrics["intermolecular_oh_min_a"] >= 1.3
                     and metrics["n_h_closer_to_other_o"] == 0)
            audit.append({"train_seed": seed, "test_config_index": index,
                          "raw_dataset_index": int(raw_index), "topology_clean": clean,
                          "initial_flags": structural_flags(structure), **metrics,
                          "initial_min_angle_deg": structure["hoh_angle_min_deg"],
                          "initial_min_hh_a": structure["intramolecular_hh_min_a"],
                          "initial_min_oo_a": structure["intermolecular_oo_min_a"]})
        selected = choose_starts([r for r in audit if r["train_seed"] == seed], args.configs_per_seed)
        for row in selected:
            if row["raw_dataset_index"] in train.indices or row["raw_dataset_index"] in val.indices:
                raise ValueError("Selected start is not held out from this checkpoint pair.")
        starts[str(seed)] = selected

    source_names = (Path(__file__).name, "waterbox_ase.py", "evaluate_waterbox.py",
                    "molecular_zbl.py", "waterbox_data.py", "structural_metrics.py",
                    "baseline_potential.py", "diagnose_short_range_collapse.py",
                    "audit_waterbox_topology.py", "analyze_waterbox_property_trajectory.py",
                    "run_waterbox_ensemble.py")
    settings = {k: getattr(args, k) for k in (
        "seeds", "velocity_seeds", "configs_per_seed", "temperature_k", "dt_fs",
        "nvt_ps", "nve_ps", "tdamp_fs", "log_stride", "structure_stride",
        "trajectory_stride", "maximum_temperature_k", "device")}
    installed_sources = hashlib.sha256()
    package_root = Path(torchmdnet.__file__).resolve().parent
    for p in sorted(package_root.rglob("*.py")):
        installed_sources.update(p.relative_to(package_root).as_posix().encode())
        installed_sources.update(bytes.fromhex(digest(p)))
    runtime = {"torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
               "default_dtype": str(torch.get_default_dtype()),
               "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
               "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
               "torchmdnet_python_source_sha256": installed_sources.hexdigest()}
    plan = {"schema": 1, "settings": settings, "rules": RULES,
            "selection": "First test-permutation entries passing fixed topology and initial structural rules; no model output used.",
            "interpretation": "Exploratory paired evaluation on previously used dataset; diagnostic thresholds are not universal chemistry criteria. NVT is a fixed preparation budget, not certified equilibrium.",
            "dataset_sha256": h.hexdigest(), "data_root": str(Path(args.data_root).resolve()),
            "topology_ids": group_ids.tolist(), "checkpoints": checkpoints,
            "starts": starts, "software": versions(), "runtime": runtime,
            "source_sha256": {name: digest(Path(__file__).parent / name) for name in source_names}}
    path = args.out / "study_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Study plan/settings/data/software differ from the saved plan. Use a new --out directory.")
    write_json(path, plan)
    write_csv(args.out / "starting_configuration_audit.csv", [
        {**r, "initial_flags": ";".join(r["initial_flags"])} for r in audit])
    write_csv(args.out / "selected_starts.csv", [
        {**r, "initial_flags": ";".join(r["initial_flags"]), "configuration_slot": slot}
        for seed in args.seeds for slot, r in enumerate(starts[str(seed)])])
    return plan, dataset, group_ids, groups


def checked_calculator(checkpoint, group_ids, device):
    import numpy as np
    import torch
    from waterbox_ase import TensorNetCalculator
    calc = TensorNetCalculator(checkpoint, device=device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    expected = set(calc.model.state_dict())
    if expected - set(state) or set(state) - expected - {"local_molecule_ids"}:
        raise ValueError(f"Checkpoint/model state-key mismatch: {checkpoint}")
    buffers = [v.detach().cpu().numpy() for k, v in calc.model.named_buffers()
               if k.endswith("local_molecule_ids")]
    if not buffers or any(not np.array_equal(v[:, None] == v, group_ids[:, None] == group_ids)
                          for v in buffers):
        raise ValueError("Loaded prior topology differs from the fixed analysis topology.")
    return calc


def preflight(plan, dataset, group_ids):
    import gc
    import numpy as np
    import torch
    from waterbox_ase import atoms_from_waterbox_sample
    rows = []
    for seed in plan["settings"]["seeds"]:
        sample = dataset[plan["starts"][str(seed)][0]["raw_dataset_index"]]
        for condition in CONDITIONS:
            entry = plan["checkpoints"][f"{seed}/{condition}"]
            calc = checked_calculator(entry["path"], group_ids, plan["settings"]["device"])
            atoms = atoms_from_waterbox_sample(sample)
            atoms.calc = calc
            energy, forces = float(atoms.get_potential_energy()), atoms.get_forces()
            if not math.isfinite(energy) or forces.shape != (192, 3) or not np.isfinite(forces).all():
                raise ValueError(f"Invalid energy/forces: seed {seed}, {condition}")
            rows.append({"train_seed": seed, "condition": condition, "energy_ev": energy,
                         "maximum_abs_force_ev_a": float(np.max(np.abs(forces)))})
            print(f"PREFLIGHT seed={seed} {condition}: finite energy/forces; state keys and topology checked", flush=True)
            del atoms, calc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def run_case(plan, dataset, group_ids, groups, seed, start, velocity_seed, condition, case_dir):
    import gc
    import numpy as np
    import torch
    from ase import units
    from ase.io import write as ase_write
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from analyze_waterbox_property_trajectory import _frame_metrics
    from run_waterbox_ensemble import _build_dynamics
    from waterbox_ase import atoms_from_waterbox_sample

    case_dir.mkdir(parents=True, exist_ok=True)
    attempt = case_dir / f"attempt_{len(list(case_dir.glob('attempt_*'))) + 1:04d}"
    attempt.mkdir()
    settings = plan["settings"]
    checkpoint = plan["checkpoints"][f"{seed}/{condition}"]
    result = {"train_seed": seed, "condition": condition,
              "raw_dataset_index": start["raw_dataset_index"],
              "test_config_index": start["test_config_index"], "velocity_seed": velocity_seed,
              "status": "running", "attempt_dir": str(attempt.resolve()),
              "checkpoint_sha256": checkpoint["sha256"], "error": None,
              "first_structural_event": None, "first_nve_structural_event": None}
    write_json(attempt / "run_manifest.json", result)
    start_wall = time.monotonic()
    therm, structures = [], []
    try:
        atoms = atoms_from_waterbox_sample(dataset[start["raw_dataset_index"]])
        atoms.calc = checked_calculator(checkpoint["path"], group_ids, settings["device"])
        MaxwellBoltzmannDistribution(atoms, temperature_K=settings["temperature_k"],
                                    rng=np.random.RandomState(velocity_seed))
        Stationary(atoms)
        nvt_steps = round(settings["nvt_ps"] * 1000 / settings["dt_fs"])
        nve_steps = round(settings["nve_ps"] * 1000 / settings["dt_fs"])
        # Both conditions must have byte-identical physical starting states.
        initial = hashlib.sha256()
        for value in (atoms.numbers, atoms.positions, np.asarray(atoms.cell), atoms.get_momenta()):
            initial.update(np.ascontiguousarray(value).tobytes())
        result["initial_state_sha256"] = initial.hexdigest()
        paths = (attempt / "thermodynamic_history.csv", attempt / "structural_history.csv")
        with paths[0].open("w", newline="", buffering=1) as th, paths[1].open("w", newline="", buffering=1) as sh:
            writers = [None, None]
            for phase, steps in (("nvt", nvt_steps), ("nve", nve_steps)):
                offset = 0 if phase == "nvt" else nvt_steps
                if phase == "nvt":
                    dyn = _build_dynamics(atoms, "nvt", settings["dt_fs"], settings["temperature_k"],
                                          1.0, settings["tdamp_fs"], 1000.0)
                else:
                    # New integrator, SAME Atoms and calculator. No momenta resampling/rescaling.
                    before = (atoms.positions.copy(), atoms.get_momenta().copy(), np.array(atoms.cell))
                    dyn = VelocityVerlet(atoms, timestep=settings["dt_fs"] * units.fs)
                    if not all(np.array_equal(a, b) for a, b in zip(before,
                               (atoms.positions, atoms.get_momenta(), np.array(atoms.cell)))):
                        raise RuntimeError("NVE construction changed the physical state")
                    result["handoff_preserved_state"] = True
                    ase_write(str(attempt / "nvt_final.xyz"), atoms.copy(), format="extxyz")
                trajectory = attempt / ("equilibration.xyz" if phase == "nvt" else "production.xyz")
                with trajectory.open("w") as xyz:
                    def record():
                        step = int(dyn.nsteps)
                        time_fs = (offset + step) * settings["dt_fs"]
                        header = {"step": offset + step, "phase_step": step, "time_fs": time_fs,
                                  "phase_time_fs": step * settings["dt_fs"], "phase": phase}
                        temperature = float(atoms.get_temperature())
                        potential, kinetic = float(atoms.get_potential_energy()), float(atoms.get_kinetic_energy())
                        finite = all(math.isfinite(v) for v in (temperature, potential, kinetic))
                        row = {**header, "epot_ev": potential if finite else None,
                               "ekin_ev": kinetic if finite else None,
                               "etot_ev": potential + kinetic if finite else None,
                               "temperature_k": temperature if finite else None}
                        therm.append(row)
                        if writers[0] is None:
                            writers[0] = csv.DictWriter(th, fieldnames=list(row)); writers[0].writeheader()
                        writers[0].writerow(row)
                        if not finite or temperature > settings["maximum_temperature_k"]:
                            raise FloatingPointError("non-finite energy/temperature or numerical temperature ceiling exceeded")
                        atoms.info.update(header)
                        if step % settings["structure_stride"] == 0 or step == steps:
                            structural = {**_frame_metrics(atoms, group_ids, groups), **header}
                            if not all(math.isfinite(v) for v in structural.values() if isinstance(v, (int, float))):
                                raise FloatingPointError("non-finite structural metric")
                            flags = structural_flags(structural, plan["rules"])
                            structural["flags"] = ";".join(flags)
                            structures.append(structural)
                            if writers[1] is None:
                                writers[1] = csv.DictWriter(sh, fieldnames=list(structural)); writers[1].writeheader()
                            writers[1].writerow(structural)
                            if flags:
                                event = {**header, "flags": flags}
                                if result["first_structural_event"] is None:
                                    result["first_structural_event"] = event
                                if phase == "nve" and result["first_nve_structural_event"] is None:
                                    result["first_nve_structural_event"] = event
                        if step % settings["trajectory_stride"] == 0 or step == steps:
                            ase_write(xyz, atoms.copy(), format="extxyz")
                            xyz.flush()
                        if step % max(settings["log_stride"], round(100 / settings["dt_fs"])) == 0:
                            print(f"  {phase} {step * settings['dt_fs']:.0f} fs T={temperature:.1f} K", flush=True)

                    dyn.attach(record, interval=settings["log_stride"])
                    try:
                        dyn.run(steps)
                    finally:
                        dyn.close()
                nphase = [r for r in therm if r["phase"] == phase]
                if not nphase or nphase[0]["phase_step"] != 0 or nphase[-1]["phase_step"] != steps:
                    raise RuntimeError("Missing phase-boundary records; check ASE observer behavior")
            result["status"] = "complete"
    except FloatingPointError as exc:
        result.update(status="aborted", error=str(exc))
    except KeyboardInterrupt:
        result.update(status="interrupted", error="interrupted by user")
        raise
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        result["wall_seconds"] = time.monotonic() - start_wall
        result.update(summarize(therm, structures, result["status"]))
        write_json(attempt / "run_manifest.json", result)
        if "atoms" in locals():
            atoms.calc = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def summarize(therm, structures, status):
    result = {"structural_survival": int(status == "complete" and not any(r.get("flags") for r in structures)),
              "last_time_fs": therm[-1]["time_fs"] if therm else 0.0}
    for phase in ("nvt", "nve"):
        rows = [r for r in therm if r["phase"] == phase and r.get("temperature_k") is not None]
        geom = [r for r in structures if r["phase"] == phase]
        if not rows:
            continue
        tail = [r for r in rows if r["phase_time_fs"] >= rows[-1]["phase_time_fs"] - 200]
        result[f"{phase}_tail_temperature_k"] = statistics.mean(r["temperature_k"] for r in tail)
        result[f"{phase}_potential_change_ev"] = rows[-1]["epot_ev"] - rows[0]["epot_ev"]
        result[f"{phase}_energy_change_ev"] = rows[-1]["etot_ev"] - rows[0]["etot_ev"]
        result[f"{phase}_maximum_energy_deviation_ev"] = max(abs(r["etot_ev"] - rows[0]["etot_ev"]) for r in rows)
        result[f"{phase}_flagged_frame_fraction"] = statistics.mean(bool(r["flags"]) for r in geom) if geom else None
        for metric in ("hoh_angle_min_deg", "intramolecular_hh_min_a", "intermolecular_oo_min_a"):
            result[f"{phase}_{metric}"] = min(r[metric] for r in geom) if geom else None
        if phase == "nvt" and len(tail) > 1:
            midpoint = (tail[0]["phase_time_fs"] + tail[-1]["phase_time_fs"]) / 2
            early = [r for r in tail if r["phase_time_fs"] <= midpoint]
            late = [r for r in tail if r["phase_time_fs"] > midpoint]
            result["nvt_tail_potential_half_difference_ev"] = (statistics.mean(r["epot_ev"] for r in late)
                                                                          - statistics.mean(r["epot_ev"] for r in early))
    return result


def tasks(plan, root):
    s = plan["settings"]
    for slot in range(s["configs_per_seed"]):
        for velocity in s["velocity_seeds"]:
            for seed in s["seeds"]:
                start = plan["starts"][str(seed)][slot]
                order = CONDITIONS if (slot + seed + velocity) % 2 == 0 else CONDITIONS[::-1]
                for condition in order:
                    directory = root / "runs" / f"seed{seed}" / f"raw{start['raw_dataset_index']}" / f"vseed{velocity}" / condition
                    yield seed, start, velocity, condition, directory


def aggregate(plan, root):
    rows = []
    for seed, start, velocity, condition, directory in tasks(plan, root):
        value = last_attempt(directory)
        if value:
            rows.append({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in value.items()})
    write_csv(root / "per_run_results.csv", rows)
    by_key = {(r["train_seed"], r["raw_dataset_index"], r["velocity_seed"], r["condition"]): r for r in rows}
    pairs = []
    metrics = ("structural_survival", "nve_tail_temperature_k", "nve_flagged_frame_fraction",
               "nve_maximum_energy_deviation_ev", "nvt_tail_temperature_k")
    for seed, raw, velocity in sorted({key[:3] for key in by_key}):
        a, b = (by_key.get((seed, raw, velocity, c)) for c in CONDITIONS)
        row = {"train_seed": seed, "raw_dataset_index": raw, "velocity_seed": velocity,
               "paired_terminal": int(bool(a and b and a["status"] in TERMINAL and b["status"] in TERMINAL)),
               "paired_complete": int(bool(a and b and a["status"] == b["status"] == "complete"))}
        if (a and b and a.get("initial_state_sha256") and b.get("initial_state_sha256")
                and a["initial_state_sha256"] != b["initial_state_sha256"]):
            raise ValueError(f"Paired physical initial states differ: {seed}/{raw}/{velocity}")
        for metric in metrics:
            valid = row["paired_terminal"] if metric == "structural_survival" else row["paired_complete"]
            row[f"{metric}_momentum_minus_absolute"] = (b[metric] - a[metric]
                if valid and a.get(metric) is not None and b.get(metric) is not None else None)
        pairs.append(row)
    write_csv(root / "paired_results.csv", pairs)
    seeds = []
    for seed in plan["settings"]["seeds"]:
        members = [r for r in pairs if r["train_seed"] == seed]
        row = {"train_seed": seed, "expected_pairs": plan["settings"]["configs_per_seed"] * len(plan["settings"]["velocity_seeds"]),
               "paired_terminal": sum(r["paired_terminal"] for r in members),
               "paired_complete": sum(r["paired_complete"] for r in members)}
        for metric in metrics:
            key = f"{metric}_momentum_minus_absolute"
            values = [r[key] for r in members if r[key] is not None]
            row[key] = statistics.mean(values) if values else None
        seeds.append(row)
    write_csv(root / "paired_seed_summary.csv", seeds)
    expected = len(list(tasks(plan, root)))
    state = {"expected_runs": expected, "terminal_runs": sum(r["status"] in TERMINAL for r in rows),
             "complete_runs": sum(r["status"] == "complete" for r in rows),
             "numerically_aborted_runs": sum(r["status"] == "aborted" for r in rows),
             "error_or_interrupted_runs": sum(r["status"] not in TERMINAL for r in rows),
             "note": "Partial summaries are provisional. Only paired completed runs contribute temperature/energy contrasts; numerical aborts count against structural survival. Seed rows are the units for across-model comparison."}
    write_json(root / "study_status.json", state)
    return state


def integer_list(value):
    result = [int(x) for x in value.split(",")]
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("Use distinct nonnegative comma-separated integers")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-root", default="checkpoints/waterbox_study_zbl_bonded_ext70")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out", type=Path, default=Path("results/waterbox_equilibration_study_ext70_v1"))
    parser.add_argument("--seeds", type=integer_list, default=list(range(6)))
    parser.add_argument("--velocity-seeds", type=integer_list, default=[0, 1])
    parser.add_argument("--configs-per-seed", type=int, default=3)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--nvt-ps", type=float, default=1.0)
    parser.add_argument("--nve-ps", type=float, default=5.0)
    parser.add_argument("--tdamp-fs", type=float, default=100.0)
    parser.add_argument("--log-stride", type=int, default=10)
    parser.add_argument("--structure-stride", type=int, default=10)
    parser.add_argument("--trajectory-stride", type=int, default=100)
    parser.add_argument("--maximum-temperature-k", type=float, default=5000.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        if (args.out / ".run.lock").exists():
            parser.error("Study is running/locked. Read study_status.json for live progress; analyze after it stops.")
        print(json.dumps(aggregate(json.loads((args.out / "study_plan.json").read_text()), args.out), indent=2))
        return
    for name in ("configs_per_seed", "temperature_k", "dt_fs", "nvt_ps", "nve_ps", "tdamp_fs",
                 "log_stride", "structure_stride", "trajectory_stride", "maximum_temperature_k"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.structure_stride % args.log_stride or args.trajectory_stride % args.log_stride:
        parser.error("Structural/trajectory strides must be multiples of log stride")
    for duration in (args.nvt_ps, args.nve_ps):
        steps = duration * 1000 / args.dt_fs
        if not math.isclose(steps, round(steps), abs_tol=1e-7) or round(steps) < 1 or round(steps) % args.log_stride:
            parser.error("Phase durations must contain an integer multiple of log-stride steps")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / ".run.lock"
    try:
        handle = lock.open("x")
    except FileExistsError:
        parser.error(f"Study is locked: {lock}. Check its PID/host; remove this lock only after confirming no study process is running.")
    with handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
    try:
        plan, dataset, group_ids, groups = prepare(args)
        print(f"Frozen plan: {args.out / 'study_plan.json'}", flush=True)
        write_csv(args.out / "preflight.csv", preflight(plan, dataset, group_ids))
        if args.prepare_only:
            print("PREPARATION PASSED: all requested checkpoints and starts validated. No MD launched.", flush=True)
            return
        all_tasks = list(tasks(plan, args.out))
        started, newly_finished = time.monotonic(), 0
        for seed, start, velocity, condition, directory in all_tasks:
            previous = last_attempt(directory)
            if previous and previous["status"] in TERMINAL:
                print(f"SKIP {seed}/{start['raw_dataset_index']}/{velocity}/{condition}: {previous['status']}", flush=True)
                continue
            print(f"RUN seed={seed} raw={start['raw_dataset_index']} velocity={velocity} {condition}", flush=True)
            result = run_case(plan, dataset, group_ids, groups, seed, start, velocity, condition, directory)
            state = aggregate(plan, args.out)
            newly_finished += 1
            remaining = state["expected_runs"] - state["terminal_runs"]
            eta_h = (time.monotonic() - started) / newly_finished * remaining / 3600
            print(f"FINISHED {result['status']}: {state['terminal_runs']}/{len(all_tasks)} terminal; rough remaining {eta_h:.1f} h", flush=True)
        state = aggregate(plan, args.out)
        print(json.dumps(state, indent=2), flush=True)
        if state["terminal_runs"] != state["expected_runs"]:
            raise SystemExit(1)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
