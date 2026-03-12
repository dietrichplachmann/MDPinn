#!/usr/bin/env python
"""
Automated experiment runner for standard vs physics-informed TorchMD-NET.

Highlights:
- Config-driven multi-experiment execution.
- Physics model-size sweeps (embedding_dimension / num_layers / num_rbf).
- Named physics checkpoints: <molecule>_<single|bulk>_<embedding>_<layers>.ckpt
- Consolidated CSV + Markdown tables for paper-ready summaries.
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchmdnet.datasets import MD17

from compare_models import compare_models
from rollout_nve import load_lnnp_from_ckpt, get_atomic_masses, velocity_verlet_rollout
from train_physics import train_physics_informed_model
from train_standard import train_standard_model


def _merge(base, override):
    out = dict(base)
    out.update(override or {})
    return out


def _sanitize_token(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))


def _make_physics_grid(e):
    """Return list of model-size variants to run for physics."""
    grid = e.get("physics_model_grid")
    if grid:
        return grid
    return [
        {
            "embedding_dimension": int(e.get("embedding_dimension", 256)),
            "num_layers": int(e.get("num_layers", 6)),
            "num_rbf": int(e.get("num_rbf", 64)),
        }
    ]


def run_rollout_summary(
    ckpt_path,
    dataset,
    molecule,
    data_root,
    steps,
    dt,
    n_rollouts,
    seed,
    device,
    energy_log_stride=20,
):
    if dataset != "MD17":
        raise NotImplementedError("Rollout summary currently supports MD17 only.")

    torch.manual_seed(seed)
    model = load_lnnp_from_ckpt(str(ckpt_path), device=device)

    full = MD17(root=data_root, molecules=molecule)
    train_size = int(0.8 * len(full))
    val_size = int(0.1 * len(full))
    test_size = len(full) - train_size - val_size
    _, _, test_data = random_split(
        full,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_indices = sorted(list(test_data.indices))
    test_start_indices = [i for i in test_indices if (i + 1) < len(full)]
    if not test_start_indices:
        raise RuntimeError("No valid rollout start indices found in test set.")

    stride = max(1, len(test_start_indices) // n_rollouts)
    starts = test_start_indices[::stride][:n_rollouts]
    if len(starts) < n_rollouts:
        extra = torch.randint(0, len(test_start_indices), (n_rollouts - len(starts),)).tolist()
        starts += [test_start_indices[i] for i in extra]

    failed = 0
    max_abs_drifts = []
    final_drifts = []
    for start_idx in starts:
        s0 = full[start_idx]
        s1 = full[start_idx + 1]

        z = s0.z.to(device)
        x0 = s0.pos.to(device).float()
        x1 = s1.pos.to(device).float()
        masses = get_atomic_masses(z).to(device)
        v0 = (x1 - x0) / dt

        out = velocity_verlet_rollout(
            model=model,
            z=z,
            masses_amu=masses,
            x0=x0,
            v0=v0,
            steps=steps,
            dt_fs=dt,
            device=device,
            energy_log_stride=energy_log_stride,
            progress_stride=0,
            rollout_id=None,
        )
        if out["failed"]:
            failed += 1
        else:
            max_abs_drifts.append(float(out["max_abs_drift"]))
            final_drifts.append(float(out["final_drift"]))

    return {
        "failed": int(failed),
        "success": int(n_rollouts - failed),
        "failure_rate": float(failed / n_rollouts),
        "mean_max_abs_drift_eV": float(sum(max_abs_drifts) / len(max_abs_drifts)) if max_abs_drifts else None,
        "mean_final_drift_eV": float(sum(final_drifts) / len(final_drifts)) if final_drifts else None,
    }


def write_tables(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = [
        "experiment",
        "variant",
        "dataset",
        "molecule",
        "system_type",
        "model",
        "epochs",
        "batch_size",
        "lr",
        "delta_learning",
        "embedding_dimension",
        "num_layers",
        "num_rbf",
        "standard_checkpoint",
        "physics_checkpoint",
        "energy_mae_std",
        "energy_mae_phys",
        "force_mae_std",
        "force_mae_phys",
        "drift_cmp_std",
        "drift_cmp_phys",
        "rollout_fail_std",
        "rollout_fail_phys",
        "rollout_drift_std",
        "rollout_drift_phys",
    ]

    csv_path = out_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in headers})

    md_path = out_dir / "summary_table.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            vals = [str(row.get(k, "")) for k in headers]
            f.write("| " + " | ".join(vals) + " |\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run experiment suite with optional physics-size sweeps.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    defaults = cfg.get("defaults", {})
    output_root = Path(cfg.get("output_root", "results/experiments"))
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in cfg.get("experiments", []):
        name = exp["name"]
        e = _merge(defaults, exp)

        exp_dir = output_root / name
        overwrite_existing = bool(e.get("overwrite_existing", False))
        if overwrite_existing and exp_dir.exists():
            print(f"Overwriting existing experiment directory: {exp_dir}")
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        dataset = e.get("dataset", "MD17")
        molecule = e.get("molecule", "aspirin")
        system_type = _sanitize_token(e.get("system_type", "single"))
        model_type = e.get("model", "tensornet")
        batch_size = int(e.get("batch_size", 32))
        epochs = int(e.get("epochs", 10))
        lr = float(e.get("lr", 1e-4))
        delta_learning = bool(e.get("delta_learning", False))
        baseline_eps = float(e.get("baseline_eps", 0.01))
        baseline_sigma = float(e.get("baseline_sigma", 1.0))
        baseline_cutoff = float(e.get("baseline_cutoff", 5.0))
        device = e.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        run_standard = bool(e.get("run_standard", True))
        run_physics = bool(e.get("run_physics", True))
        run_compare = bool(e.get("run_compare", True))
        run_rollout = bool(e.get("run_rollout", False))

        std_over = e.get("standard_overrides", {})
        phys_over = e.get("physics_overrides", {})
        rollout_cfg = e.get("rollout", {})
        physics_grid = _make_physics_grid(e)

        print("\n" + "=" * 70)
        print(f"Experiment: {name}")
        print("=" * 70)

        for variant in physics_grid:
            emb = int(variant.get("embedding_dimension", 256))
            layers = int(variant.get("num_layers", 6))
            num_rbf = int(variant.get("num_rbf", 64))
            variant_tag = f"e{emb}_l{layers}"
            ckpt_base = f"{_sanitize_token(molecule)}_{system_type}_{emb}_{layers}"
            std_ckpt_stem = f"{ckpt_base}_standard"
            phys_ckpt_stem = f"{ckpt_base}_physics"

            ckpt_std_dir = exp_dir / "checkpoints" / "standard" / std_ckpt_stem
            log_std_dir = exp_dir / "logs" / "standard" / variant_tag
            std_ckpt = ckpt_std_dir / f"{std_ckpt_stem}.ckpt"

            ckpt_phys_dir = exp_dir / "checkpoints" / "physics_informed" / phys_ckpt_stem
            log_phys_dir = exp_dir / "logs" / "physics_informed" / variant_tag
            cmp_dir = exp_dir / "comparison" / variant_tag
            phys_ckpt = ckpt_phys_dir / f"{phys_ckpt_stem}.ckpt"

            print(f"\n[Variant] {variant_tag} -> std={std_ckpt.name}, phys={phys_ckpt.name}")

            if run_standard:
                std_kwargs = {
                    "dataset": dataset,
                    "molecule": molecule,
                    "batch_size": batch_size,
                    "num_epochs": epochs,
                    "lr": lr,
                    "model_type": model_type,
                    "save_dir": str(ckpt_std_dir),
                    "log_dir": str(log_std_dir),
                    "delta_learning": delta_learning,
                    "baseline_epsilon_eV": baseline_eps,
                    "baseline_sigma_A": baseline_sigma,
                    "baseline_cutoff_A": baseline_cutoff,
                    "embedding_dimension": emb,
                    "num_layers": layers,
                    "num_rbf": num_rbf,
                    "checkpoint_name": std_ckpt_stem,
                }
                std_kwargs.update(std_over)
                train_standard_model(**std_kwargs)

            if run_physics:
                phys_kwargs = {
                    "dataset": dataset,
                    "molecule": molecule,
                    "batch_size": batch_size,
                    "num_epochs": epochs,
                    "lr": lr,
                    "model_type": model_type,
                    "save_dir": str(ckpt_phys_dir),
                    "log_dir": str(log_phys_dir),
                    "delta_learning": delta_learning,
                    "baseline_epsilon_eV": baseline_eps,
                    "baseline_sigma_A": baseline_sigma,
                    "baseline_cutoff_A": baseline_cutoff,
                    "embedding_dimension": emb,
                    "num_layers": layers,
                    "num_rbf": num_rbf,
                    "checkpoint_name": phys_ckpt_stem,
                }
                phys_kwargs.update(phys_over)
                train_physics_informed_model(**phys_kwargs)

            summary = {
                "experiment": name,
                "variant": variant_tag,
                "dataset": dataset,
                "molecule": molecule,
                "system_type": system_type,
                "model": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "delta_learning": delta_learning,
                "embedding_dimension": emb,
                "num_layers": layers,
                "num_rbf": num_rbf,
                "standard_checkpoint": str(std_ckpt),
                "physics_checkpoint": str(phys_ckpt),
            }

            if run_compare:
                cmp = compare_models(
                    standard_checkpoint=str(std_ckpt),
                    physics_checkpoint=str(phys_ckpt),
                    dataset=dataset,
                    molecule=molecule,
                    output_dir=str(cmp_dir),
                    device=device,
                )
                summary.update(
                    {
                        "energy_mae_std": cmp["standard_metrics"]["energy_mae"],
                        "energy_mae_phys": cmp["physics_metrics"]["energy_mae"],
                        "force_mae_std": cmp["standard_metrics"]["force_mae"],
                        "force_mae_phys": cmp["physics_metrics"]["force_mae"],
                        "drift_cmp_std": cmp["standard_drift"]["energy_drift_mean"],
                        "drift_cmp_phys": cmp["physics_drift"]["energy_drift_mean"],
                    }
                )

            if run_rollout:
                standard_rollout = run_rollout_summary(
                    ckpt_path=std_ckpt,
                    dataset=dataset,
                    molecule=molecule,
                    data_root=rollout_cfg.get("data_root", "./data"),
                    steps=int(rollout_cfg.get("steps", 5000)),
                    dt=float(rollout_cfg.get("dt", 0.1)),
                    n_rollouts=int(rollout_cfg.get("n_rollouts", 10)),
                    seed=int(rollout_cfg.get("seed", 42)),
                    device=device,
                    energy_log_stride=int(rollout_cfg.get("energy_log_stride", 20)),
                )
                phys_roll = run_rollout_summary(
                    ckpt_path=phys_ckpt,
                    dataset=dataset,
                    molecule=molecule,
                    data_root=rollout_cfg.get("data_root", "./data"),
                    steps=int(rollout_cfg.get("steps", 5000)),
                    dt=float(rollout_cfg.get("dt", 0.1)),
                    n_rollouts=int(rollout_cfg.get("n_rollouts", 10)),
                    seed=int(rollout_cfg.get("seed", 42)),
                    device=device,
                    energy_log_stride=int(rollout_cfg.get("energy_log_stride", 20)),
                )
                summary.update(
                    {
                        "rollout_fail_std": standard_rollout["failure_rate"] if standard_rollout else None,
                        "rollout_fail_phys": phys_roll["failure_rate"],
                        "rollout_drift_std": standard_rollout["mean_max_abs_drift_eV"] if standard_rollout else None,
                        "rollout_drift_phys": phys_roll["mean_max_abs_drift_eV"],
                    }
                )
                with open(exp_dir / f"rollout_summary_{variant_tag}.json", "w") as f:
                    json.dump({"standard": standard_rollout, "physics": phys_roll}, f, indent=2)

            with open(exp_dir / f"experiment_summary_{variant_tag}.json", "w") as f:
                json.dump(summary, f, indent=2)
            rows.append(summary)

    write_tables(rows, output_root)
    print("All experiments completed.")


if __name__ == "__main__":
    main()
