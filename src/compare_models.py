#!/usr/bin/env python
"""
Model Comparison Script - UPDATED for new training framework
Compare standard vs physics-informed TorchMD-NET models
Generate comprehensive comparison plots and metrics
"""

import torch

# Fix PyTorch 2.7 torch.load compatibility
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader

from torchmdnet.datasets import MD17
from torchmdnet.module import LNNP

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load LNNP model from checkpoint

    Args:
        checkpoint_path: Path to .ckpt file
        device: Device to load model on

    Returns:
        Loaded LNNP model
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get hyperparameters
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
        elif 'hparams' in checkpoint:
            hparams = checkpoint['hparams']
        else:
            raise ValueError("No hyperparameters found in checkpoint")

        # Create model
        model = LNNP(hparams)

        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError("No state_dict found in checkpoint")

        model.eval()
        model.to(device)

        print(f"✓ Model loaded from {checkpoint_path}")
        return model

    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        raise


def create_dataset(dataset_name='MD17', molecule='aspirin', data_root='./data'):
    """
    Create dataset and split into train/val/test

    Returns:
        train_data, val_data, test_data, full_dataset
    """
    print(f"Loading {dataset_name} dataset: {molecule}")

    # Load full dataset
    full_dataset = MD17(root=data_root, molecules=molecule)
    print(f"✓ Dataset loaded: {len(full_dataset)} samples")

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Split: {train_size} train, {val_size} val, {test_size} test")

    return train_data, val_data, test_data, full_dataset


def evaluate_on_dataset(model, dataset, device='cuda', batch_size=32, max_samples=None):
    """
    Evaluate model on dataset

    Args:
        model: LNNP model
        dataset: PyTorch dataset
        device: Device to run on
        batch_size: Batch size for evaluation
        max_samples: Optional limit on number of samples

    Returns:
        Dictionary with predictions and targets
    """
    model = model.to(device)
    model.eval()

    # Create dataloader
    dataloader = GeometricDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Use 0 for evaluation to avoid issues
    )

    energies_pred = []
    energies_true = []
    forces_pred = []
    forces_true = []

    samples_processed = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)

            # Forward pass
            energy_pred, force_pred = model(batch.z, batch.pos, batch=batch.batch)

            # Collect predictions and targets
            energies_pred.append(energy_pred.cpu())
            energies_true.append(batch.y.cpu())
            forces_pred.append(force_pred.cpu())
            forces_true.append(batch.neg_dy.cpu())

            samples_processed += batch.num_graphs

            if max_samples and samples_processed >= max_samples:
                break

    # Concatenate all batches
    results = {
        'energy_pred': torch.cat(energies_pred).numpy(),
        'energy_true': torch.cat(energies_true).numpy(),
        'force_pred': torch.cat(forces_pred).numpy(),
        'force_true': torch.cat(forces_true).numpy(),
    }

    return results


def compute_metrics(results):
    """
    Compute evaluation metrics

    Args:
        results: Dict with predictions and targets

    Returns:
        Dict with metrics
    """
    energy_pred = results['energy_pred'].flatten()
    energy_true = results['energy_true'].flatten()
    force_pred = results['force_pred'].flatten()
    force_true = results['force_true'].flatten()

    metrics = {
        # Energy metrics
        'energy_mae': np.mean(np.abs(energy_pred - energy_true)),
        'energy_rmse': np.sqrt(np.mean((energy_pred - energy_true) ** 2)),
        'energy_max_error': np.max(np.abs(energy_pred - energy_true)),

        # Force metrics
        'force_mae': np.mean(np.abs(force_pred - force_true)),
        'force_rmse': np.sqrt(np.mean((force_pred - force_true) ** 2)),
        'force_max_error': np.max(np.abs(force_pred - force_true)),

        # Correlation
        'energy_r2': np.corrcoef(energy_pred, energy_true)[0, 1] ** 2,
        'force_r2': np.corrcoef(force_pred, force_true)[0, 1] ** 2,
    }

    return metrics


def evaluate_energy_conservation(model, dataset, device='cuda', traj_length=100, num_trajs=10):
    """
    Evaluate energy conservation along trajectories

    Args:
        model: LNNP model
        dataset: Full dataset
        device: Device
        traj_length: Length of each trajectory
        num_trajs: Number of trajectories to test

    Returns:
        Dict with energy drift statistics
    """
    model = model.to(device)
    model.eval()

    energy_drifts = []

    # Sample random starting points for trajectories
    max_start = len(dataset) - traj_length
    if max_start <= 0:
        print(f"⚠ Dataset too small for trajectory length {traj_length}")
        return {'energy_drift_mean': 0.0, 'energy_drift_std': 0.0}

    starts = np.random.choice(max_start, size=min(num_trajs, max_start), replace=False)

    with torch.no_grad():
        for start_idx in tqdm(starts, desc="Testing trajectories"):
            energies = []

            for t in range(traj_length):
                sample = dataset[start_idx + t]
                sample = sample.to(device)

                # Create batch tensor
                batch = torch.zeros(sample.z.size(0), dtype=torch.long, device=device)

                # Predict energy
                energy, _ = model(sample.z, sample.pos, batch=batch)
                energies.append(energy.item())

            # Compute drift from initial energy
            energies = np.array(energies)
            drift = np.abs(energies - energies[0])
            energy_drifts.append(drift.mean())

    return {
        'energy_drift_mean': np.mean(energy_drifts),
        'energy_drift_std': np.std(energy_drifts),
        'energy_drift_max': np.max(energy_drifts),
    }


def plot_parity(standard_results, physics_results, output_dir, title_suffix=''):
    """Generate parity plots comparing predictions to targets"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Standard model - Energy
    ax = axes[0, 0]
    e_pred = standard_results['energy_pred'].flatten()
    e_true = standard_results['energy_true'].flatten()
    ax.scatter(e_true, e_pred, alpha=0.3, s=10)
    ax.plot([e_true.min(), e_true.max()], [e_true.min(), e_true.max()], 'r--', lw=2)
    ax.set_xlabel('True Energy (eV)')
    ax.set_ylabel('Predicted Energy (eV)')
    ax.set_title('Standard Model - Energy')
    ax.grid(True, alpha=0.3)

    # Physics model - Energy
    ax = axes[0, 1]
    e_pred = physics_results['energy_pred'].flatten()
    e_true = physics_results['energy_true'].flatten()
    ax.scatter(e_true, e_pred, alpha=0.3, s=10)
    ax.plot([e_true.min(), e_true.max()], [e_true.min(), e_true.max()], 'r--', lw=2)
    ax.set_xlabel('True Energy (eV)')
    ax.set_ylabel('Predicted Energy (eV)')
    ax.set_title('Physics-Informed Model - Energy')
    ax.grid(True, alpha=0.3)

    # Standard model - Forces
    ax = axes[1, 0]
    f_pred = standard_results['force_pred'].flatten()
    f_true = standard_results['force_true'].flatten()
    ax.scatter(f_true, f_pred, alpha=0.1, s=5)
    ax.plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], 'r--', lw=2)
    ax.set_xlabel('True Forces (eV/Å)')
    ax.set_ylabel('Predicted Forces (eV/Å)')
    ax.set_title('Standard Model - Forces')
    ax.grid(True, alpha=0.3)

    # Physics model - Forces
    ax = axes[1, 1]
    f_pred = physics_results['force_pred'].flatten()
    f_true = physics_results['force_true'].flatten()
    ax.scatter(f_true, f_pred, alpha=0.1, s=5)
    ax.plot([f_true.min(), f_true.max()], [f_true.min(), f_true.max()], 'r--', lw=2)
    ax.set_xlabel('True Forces (eV/Å)')
    ax.set_ylabel('Predicted Forces (eV/Å)')
    ax.set_title('Physics-Informed Model - Forces')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'parity_plots{title_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved parity plots")


def plot_error_distributions(standard_results, physics_results, output_dir):
    """Plot error distributions"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Energy errors - Standard
    ax = axes[0, 0]
    errors = standard_results['energy_pred'] - standard_results['energy_true']
    ax.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Energy Error (eV)')
    ax.set_ylabel('Frequency')
    ax.set_title('Standard Model - Energy Errors')
    ax.grid(True, alpha=0.3)

    # Energy errors - Physics
    ax = axes[0, 1]
    errors = physics_results['energy_pred'] - physics_results['energy_true']
    ax.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Energy Error (eV)')
    ax.set_ylabel('Frequency')
    ax.set_title('Physics-Informed Model - Energy Errors')
    ax.grid(True, alpha=0.3)

    # Force errors - Standard
    ax = axes[1, 0]
    errors = standard_results['force_pred'] - standard_results['force_true']
    ax.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Force Error (eV/Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('Standard Model - Force Errors')
    ax.grid(True, alpha=0.3)

    # Force errors - Physics
    ax = axes[1, 1]
    errors = physics_results['force_pred'] - physics_results['force_true']
    ax.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Force Error (eV/Å)')
    ax.set_ylabel('Frequency')
    ax.set_title('Physics-Informed Model - Force Errors')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Saved error distribution plots")


def plot_metrics_comparison(standard_metrics, physics_metrics, output_dir):
    """Bar plot comparing metrics"""

    metrics_to_plot = [
        ('energy_mae', 'Energy MAE (eV)'),
        ('energy_rmse', 'Energy RMSE (eV)'),
        ('force_mae', 'Force MAE (eV/Å)'),
        ('force_rmse', 'Force RMSE (eV/Å)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx]

        values = [standard_metrics[metric_key], physics_metrics[metric_key]]
        labels = ['Standard', 'Physics-Informed']
        colors = ['#3498db', '#e74c3c']

        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')

        # Calculate improvement
        improvement = ((standard_metrics[metric_key] - physics_metrics[metric_key]) /
                       standard_metrics[metric_key] * 100)
        if improvement > 0:
            ax.text(0.5, max(values) * 0.9, f'↓ {improvement:.1f}% improvement',
                    ha='center', transform=ax.transData,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Saved metrics comparison plot")


def compare_models(
        standard_checkpoint,
        physics_checkpoint,
        dataset='MD17',
        molecule='aspirin',
        output_dir='results/comparison',
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Complete model comparison workflow
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Model Comparison: Standard vs Physics-Informed")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset} - {molecule}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # Load models
    print("Step 1: Loading models...")
    standard_model = load_checkpoint(standard_checkpoint, device=device)
    physics_model = load_checkpoint(physics_checkpoint, device=device)

    # Load dataset
    print("\nStep 2: Loading dataset...")
    train_data, val_data, test_data, full_dataset = create_dataset(dataset, molecule)

    # Evaluate on test set
    print("\nStep 3: Evaluating on test set...")
    standard_results = evaluate_on_dataset(standard_model, test_data, device=device)
    physics_results = evaluate_on_dataset(physics_model, test_data, device=device)

    # Compute metrics
    print("\nStep 4: Computing metrics...")
    standard_metrics = compute_metrics(standard_results)
    physics_metrics = compute_metrics(physics_results)

    # Print metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nStandard Model:")
    for key, value in standard_metrics.items():
        print(f"  {key:20s}: {value:.6f}")

    print("\nPhysics-Informed Model:")
    for key, value in physics_metrics.items():
        print(f"  {key:20s}: {value:.6f}")

    print("\nImprovements (Physics over Standard):")
    for key in standard_metrics.keys():
        improvement = ((standard_metrics[key] - physics_metrics[key]) /
                       standard_metrics[key] * 100)
        symbol = "↓" if improvement > 0 else "↑"
        print(f"  {key:20s}: {symbol} {abs(improvement):.2f}%")

    # Energy conservation test
    print("\nStep 5: Testing energy conservation...")
    standard_drift = evaluate_energy_conservation(standard_model, full_dataset, device=device)
    physics_drift = evaluate_energy_conservation(physics_model, full_dataset, device=device)

    print("\nEnergy Drift (mean over trajectories):")
    print(f"  Standard:        {standard_drift['energy_drift_mean']:.6f} eV")
    print(f"  Physics-Informed: {physics_drift['energy_drift_mean']:.6f} eV")

    # Generate plots
    print("\nStep 6: Generating plots...")
    plot_parity(standard_results, physics_results, output_dir)
    plot_error_distributions(standard_results, physics_results, output_dir)
    plot_metrics_comparison(standard_metrics, physics_metrics, output_dir)

    # Save results to JSON
    results_summary = {
        'standard_metrics': standard_metrics,
        'physics_metrics': physics_metrics,
        'standard_drift': standard_drift,
        'physics_drift': physics_drift,
        'dataset': dataset,
        'molecule': molecule,
    }

    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✓ Comparison complete!")
    print(f"  Results saved to: {output_dir}")
    print("=" * 60)

    return results_summary


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare TorchMD-NET models')
    parser.add_argument('--standard', type=str, required=True,
                        help='Path to standard model checkpoint')
    parser.add_argument('--physics', type=str, required=True,
                        help='Path to physics-informed model checkpoint')
    parser.add_argument('--dataset', type=str, default='MD17')
    parser.add_argument('--molecule', type=str, default='aspirin')
    parser.add_argument('--output-dir', type=str, default='results/comparison')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    compare_models(
        standard_checkpoint=args.standard,
        physics_checkpoint=args.physics,
        dataset=args.dataset,
        molecule=args.molecule,
        output_dir=args.output_dir,
        device=args.device
    )