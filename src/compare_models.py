#!/usr/bin/env python
"""
Compare standard vs physics-informed checkpoints.

Important delta-learning behavior:
- If a checkpoint was trained in delta mode, this script reconstructs
  absolute energy/forces by adding analytic baseline terms back in.

Chemist view:
- We always compare on absolute observables (E and F), because those are what
  matter for real MD behavior and what your advisor will care about.
- Delta checkpoints are converted back to absolute predictions before metrics.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from torchmdnet.datasets import MD17
from torchmdnet.module import LNNP

from baseline_potential import lj_energy_forces_batched


# PyTorch 2.7 checkpoint compatibility.
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


def load_checkpoint(checkpoint_path, device="cpu"):
    """Load checkpoint and attach delta metadata needed for absolute inference.

    The extra metadata flags tell downstream code whether this checkpoint emits:
    - absolute quantities directly, or
    - residual quantities that must be added to U_ref/F_ref.
    """
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
    model.load_state_dict(checkpoint["state_dict"])

    model._delta_learning = bool(hparams.get("delta_learning", False))
    model._baseline_eps = float(hparams.get("baseline_epsilon_eV", 0.01))
    model._baseline_sigma = float(hparams.get("baseline_sigma_A", 1.0))
    model._baseline_cutoff = float(hparams.get("baseline_cutoff_A", 5.0))

    return model.eval().to(device)


def create_dataset(dataset_name="MD17", molecule="aspirin", data_root="./data"):
    """Create MD17 dataset and fixed random split."""
    full_dataset = MD17(root=data_root, molecules=molecule)

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train_data, val_data, test_data, full_dataset


def predict_absolute_energy_forces(model, batch):
    """Predict absolute energy/forces regardless of training mode.

    Returns:
    - energy_abs: (B,1)
    - force_abs: (N,3)
    """
    # Need grad-enabled forward because TorchMD-Net force prediction uses autograd.
    with torch.enable_grad():
        energy_pred, force_pred = model(batch.z, batch.pos, batch=batch.batch)

    # Absolute checkpoint: already in physical units.
    if not getattr(model, "_delta_learning", False):
        return energy_pred, force_pred

    # Delta checkpoint: convert (DeltaU, DeltaF) -> (U_hyb, F_hyb).
    u_ref, f_ref = lj_energy_forces_batched(
        z=batch.z,
        pos=batch.pos.detach(),
        batch=batch.batch,
        epsilon_eV=getattr(model, "_baseline_eps", 0.01),
        sigma_A=getattr(model, "_baseline_sigma", 1.0),
        r_cut_A=getattr(model, "_baseline_cutoff", 5.0),
    )

    energy_abs = energy_pred.squeeze(-1) + u_ref
    energy_abs = energy_abs.unsqueeze(1)
    force_abs = force_pred + f_ref
    return energy_abs, force_abs


def evaluate_on_dataset(model, dataset, device="cuda", batch_size=32, max_samples=None):
    """Run absolute-energy/force evaluation on dataset subset.

    We deliberately evaluate everything in absolute space so standard and delta
    runs are directly comparable.
    """
    model = model.to(device).eval()
    dataloader = GeometricDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    energies_pred, energies_true = [], []
    forces_pred, forces_true = [], []

    samples_processed = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            batch.pos.requires_grad_(True)

            energy_abs, force_abs = predict_absolute_energy_forces(model, batch)

            energies_pred.append(energy_abs.detach().cpu())
            energies_true.append(batch.y.cpu())
            forces_pred.append(force_abs.detach().cpu())
            forces_true.append(batch.neg_dy.cpu())

            samples_processed += batch.num_graphs
            if max_samples and samples_processed >= max_samples:
                break

    return {
        "energy_pred": torch.cat(energies_pred).numpy(),
        "energy_true": torch.cat(energies_true).numpy(),
        "force_pred": torch.cat(forces_pred).numpy(),
        "force_true": torch.cat(forces_true).numpy(),
    }


def compute_metrics(results):
    """Compute MAE/RMSE/max-error and R2 for energy and forces."""
    energy_pred = results["energy_pred"].flatten()
    energy_true = results["energy_true"].flatten()
    force_pred = results["force_pred"].flatten()
    force_true = results["force_true"].flatten()

    return {
        "energy_mae": np.mean(np.abs(energy_pred - energy_true)),
        "energy_rmse": np.sqrt(np.mean((energy_pred - energy_true) ** 2)),
        "energy_max_error": np.max(np.abs(energy_pred - energy_true)),
        "force_mae": np.mean(np.abs(force_pred - force_true)),
        "force_rmse": np.sqrt(np.mean((force_pred - force_true) ** 2)),
        "force_max_error": np.max(np.abs(force_pred - force_true)),
        "energy_r2": np.corrcoef(energy_pred, energy_true)[0, 1] ** 2,
        "force_r2": np.corrcoef(force_pred, force_true)[0, 1] ** 2,
    }


def evaluate_energy_conservation(model, dataset, device="cuda", traj_length=100, num_trajs=10):
    """Estimate mean/max drift of absolute predicted potential along trajectories.

    Note: this function checks potential drift only (not full kinetic+potential
    Hamiltonian drift), so interpret it as a model smoothness/stability proxy.
    """
    model = model.to(device).eval()

    max_start = len(dataset) - traj_length
    if max_start <= 0:
        return {"energy_drift_mean": 0.0, "energy_drift_std": 0.0, "energy_drift_max": 0.0}

    starts = np.random.choice(max_start, size=min(num_trajs, max_start), replace=False)
    energy_drifts = []

    with torch.no_grad():
        for start_idx in tqdm(starts, desc="Testing trajectories"):
            energies = []
            for t in range(traj_length):
                sample = dataset[start_idx + t].to(device)
                sample.pos.requires_grad_(True)
                sample.batch = torch.zeros(sample.z.size(0), dtype=torch.long, device=device)

                energy_abs, _ = predict_absolute_energy_forces(model, sample)
                energies.append(energy_abs.detach().squeeze().item())

            energies = np.asarray(energies)
            energy_drifts.append(np.abs(energies - energies[0]).mean())

    return {
        "energy_drift_mean": float(np.mean(energy_drifts)),
        "energy_drift_std": float(np.std(energy_drifts)),
        "energy_drift_max": float(np.max(energy_drifts)),
    }


def plot_parity(standard_results, physics_results, output_dir):
    """Create parity plots for both models (energy and forces)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    e_pred = standard_results["energy_pred"].flatten()
    e_true = standard_results["energy_true"].flatten()
    axes[0, 0].scatter(e_true, e_pred, alpha=0.3, s=10)
    axes[0, 0].plot([e_true.min(), e_true.max()], [e_true.min(), e_true.max()], "r--", lw=2)
    axes[0, 0].set_title("Standard - Energy")

    e_pred = physics_results["energy_pred"].flatten()
    e_true = physics_results["energy_true"].flatten()
    axes[0, 1].scatter(e_true, e_pred, alpha=0.3, s=10)
    axes[0, 1].plot([e_true.min(), e_true.max()], [e_true.min(), e_true.max()], "r--", lw=2)
    axes[0, 1].set_title("Physics-Informed - Energy")

    f_pred = standard_results["force_pred"].flatten()
    f_true = standard_results["force_true"].flatten()
    axes[1, 0].scatter(f_true, f_pred, alpha=0.1, s=5)
    axes[1, 0].plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], "r--", lw=2)
    axes[1, 0].set_title("Standard - Force")

    f_pred = physics_results["force_pred"].flatten()
    f_true = physics_results["force_true"].flatten()
    axes[1, 1].scatter(f_true, f_pred, alpha=0.1, s=5)
    axes[1, 1].plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], "r--", lw=2)
    axes[1, 1].set_title("Physics-Informed - Force")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "parity_plots.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_error_distributions(standard_results, physics_results, output_dir):
    """Create histogram plots of prediction errors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    err = (standard_results["energy_pred"] - standard_results["energy_true"]).flatten()
    axes[0, 0].hist(err, bins=50, alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("Standard - Energy Error")

    err = (physics_results["energy_pred"] - physics_results["energy_true"]).flatten()
    axes[0, 1].hist(err, bins=50, alpha=0.7, edgecolor="black", color="orange")
    axes[0, 1].set_title("Physics-Informed - Energy Error")

    err = (standard_results["force_pred"] - standard_results["force_true"]).flatten()
    axes[1, 0].hist(err, bins=50, alpha=0.7, edgecolor="black")
    axes[1, 0].set_title("Standard - Force Error")

    err = (physics_results["force_pred"] - physics_results["force_true"]).flatten()
    axes[1, 1].hist(err, bins=50, alpha=0.7, edgecolor="black", color="orange")
    axes[1, 1].set_title("Physics-Informed - Force Error")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(standard_metrics, physics_metrics, output_dir):
    """Create simple bar chart summary of key metrics."""
    metrics_to_plot = [
        ("energy_mae", "Energy MAE"),
        ("energy_rmse", "Energy RMSE"),
        ("force_mae", "Force MAE"),
        ("force_rmse", "Force RMSE"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (key, title) in enumerate(metrics_to_plot):
        values = [standard_metrics[key], physics_metrics[key]]
        axes[idx].bar(["Standard", "Physics"], values, color=["#3498db", "#e74c3c"], alpha=0.8)
        axes[idx].set_title(title)
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def compare_models(
    standard_checkpoint,
    physics_checkpoint,
    dataset="MD17",
    molecule="aspirin",
    output_dir="results/comparison",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Run end-to-end comparison and write plots + JSON metrics.

    Workflow:
    1) Load both checkpoints.
    2) Reconstruct absolute predictions where needed.
    3) Compute error metrics + drift proxies.
    4) Save visual artifacts for advisor review.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints...")
    standard_model = load_checkpoint(standard_checkpoint, device=device)
    physics_model = load_checkpoint(physics_checkpoint, device=device)

    print("Loading dataset...")
    _, _, test_data, full_dataset = create_dataset(dataset, molecule)

    print("Evaluating test set...")
    standard_results = evaluate_on_dataset(standard_model, test_data, device=device)
    physics_results = evaluate_on_dataset(physics_model, test_data, device=device)

    print("Computing metrics...")
    standard_metrics = compute_metrics(standard_results)
    physics_metrics = compute_metrics(physics_results)

    print("Evaluating energy conservation...")
    standard_drift = evaluate_energy_conservation(standard_model, full_dataset, device=device)
    physics_drift = evaluate_energy_conservation(physics_model, full_dataset, device=device)

    print("Generating plots...")
    plot_parity(standard_results, physics_results, output_dir)
    plot_error_distributions(standard_results, physics_results, output_dir)
    plot_metrics_comparison(standard_metrics, physics_metrics, output_dir)

    results_summary = {
        "standard_metrics": standard_metrics,
        "physics_metrics": physics_metrics,
        "standard_drift": standard_drift,
        "physics_drift": physics_drift,
        "dataset": dataset,
        "molecule": molecule,
    }

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    return results_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare TorchMD-NET models")
    parser.add_argument("--standard", type=str, required=True)
    parser.add_argument("--physics", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="MD17")
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--output-dir", type=str, default="results/comparison")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    compare_models(
        standard_checkpoint=args.standard,
        physics_checkpoint=args.physics,
        dataset=args.dataset,
        molecule=args.molecule,
        output_dir=args.output_dir,
        device=args.device,
    )
