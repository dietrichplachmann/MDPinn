#!/usr/bin/env python
"""
Model Comparison Script
Compare standard vs physics-informed TorchMD-NET models
Generate comprehensive comparison plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm

from torchmdnet.models.model import load_model
from torchmdnet.data import DataModule

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_models(standard_checkpoint, physics_checkpoint):
    """Load both models from checkpoints"""
    print("Loading models...")

    standard_model = load_model(standard_checkpoint, derivative=True)
    standard_model.eval()

    physics_model = load_model(physics_checkpoint, derivative=True)
    physics_model.eval()

    print("✓ Models loaded")
    return standard_model, physics_model


def evaluate_on_dataset(model, dataloader, device='cuda', max_batches=None):
    """
    Evaluate model on dataset

    Returns:
        Dictionary with predictions and targets
    """
    model = model.to(device)
    model.eval()

    energies_pred = []
    energies_true = []
    forces_pred = []
    forces_true = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and i >= max_batches:
                break

            # Move to device
            batch = batch.to(device)

            # Predict
            energy, forces = model(batch.z, batch.pos, batch=batch.batch)

            # Collect results
            energies_pred.append(energy.cpu())
            energies_true.append(batch.y.cpu())
            forces_pred.append(forces.cpu())
            forces_true.append(batch.neg_dy.cpu())

    # Concatenate all batches
    results = {
        'energy_pred': torch.cat(energies_pred),
        'energy_true': torch.cat(energies_true),
        'forces_pred': torch.cat(forces_pred),
        'forces_true': torch.cat(forces_true),
    }

    return results


def compute_metrics(results):
    """Compute evaluation metrics"""

    # Energy metrics
    energy_pred = results['energy_pred'].numpy().flatten()
    energy_true = results['energy_true'].numpy().flatten()

    energy_mae = np.mean(np.abs(energy_pred - energy_true))
    energy_rmse = np.sqrt(np.mean((energy_pred - energy_true) ** 2))
    energy_r2 = 1 - np.sum((energy_true - energy_pred) ** 2) / np.sum((energy_true - energy_true.mean()) ** 2)

    # Force metrics
    forces_pred = results['forces_pred'].numpy().reshape(-1)
    forces_true = results['forces_true'].numpy().reshape(-1)

    force_mae = np.mean(np.abs(forces_pred - forces_true))
    force_rmse = np.sqrt(np.mean((forces_pred - forces_true) ** 2))
    force_r2 = 1 - np.sum((forces_true - forces_pred) ** 2) / np.sum((forces_true - forces_true.mean()) ** 2)

    metrics = {
        'energy': {
            'mae': energy_mae,
            'rmse': energy_rmse,
            'r2': energy_r2,
        },
        'force': {
            'mae': force_mae,
            'rmse': force_rmse,
            'r2': force_r2,
        }
    }

    return metrics


def plot_energy_comparison(standard_results, physics_results, standard_metrics,
                           physics_metrics, output_dir):
    """Plot energy prediction comparison"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Standard model
    ax = axes[0]
    energy_pred = standard_results['energy_pred'].numpy().flatten()
    energy_true = standard_results['energy_true'].numpy().flatten()

    ax.scatter(energy_true, energy_pred, alpha=0.5, s=10)
    ax.plot([energy_true.min(), energy_true.max()],
            [energy_true.min(), energy_true.max()],
            'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('True Energy (eV)')
    ax.set_ylabel('Predicted Energy (eV)')
    ax.set_title(
        f'Standard Model\nMAE: {standard_metrics["energy"]["mae"]:.4f} eV, R²: {standard_metrics["energy"]["r2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Physics-informed model
    ax = axes[1]
    energy_pred = physics_results['energy_pred'].numpy().flatten()
    energy_true = physics_results['energy_true'].numpy().flatten()

    ax.scatter(energy_true, energy_pred, alpha=0.5, s=10, color='orange')
    ax.plot([energy_true.min(), energy_true.max()],
            [energy_true.min(), energy_true.max()],
            'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('True Energy (eV)')
    ax.set_ylabel('Predicted Energy (eV)')
    ax.set_title(
        f'Physics-Informed Model\nMAE: {physics_metrics["energy"]["mae"]:.4f} eV, R²: {physics_metrics["energy"]["r2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'energy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Energy comparison plot saved")


def plot_force_comparison(standard_results, physics_results, standard_metrics,
                          physics_metrics, output_dir):
    """Plot force prediction comparison"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Standard model
    ax = axes[0]
    forces_pred = standard_results['forces_pred'].numpy().reshape(-1)
    forces_true = standard_results['forces_true'].numpy().reshape(-1)

    # Sample for visualization (too many points)
    n_sample = min(10000, len(forces_pred))
    idx = np.random.choice(len(forces_pred), n_sample, replace=False)

    ax.scatter(forces_true[idx], forces_pred[idx], alpha=0.3, s=5)
    ax.plot([forces_true.min(), forces_true.max()],
            [forces_true.min(), forces_true.max()],
            'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('True Force (eV/Å)')
    ax.set_ylabel('Predicted Force (eV/Å)')
    ax.set_title(
        f'Standard Model\nMAE: {standard_metrics["force"]["mae"]:.4f} eV/Å, R²: {standard_metrics["force"]["r2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Physics-informed model
    ax = axes[1]
    forces_pred = physics_results['forces_pred'].numpy().reshape(-1)
    forces_true = physics_results['forces_true'].numpy().reshape(-1)

    ax.scatter(forces_true[idx], forces_pred[idx], alpha=0.3, s=5, color='orange')
    ax.plot([forces_true.min(), forces_true.max()],
            [forces_true.min(), forces_true.max()],
            'r--', linewidth=2, label='Perfect')

    ax.set_xlabel('True Force (eV/Å)')
    ax.set_ylabel('Predicted Force (eV/Å)')
    ax.set_title(
        f'Physics-Informed Model\nMAE: {physics_metrics["force"]["mae"]:.4f} eV/Å, R²: {physics_metrics["force"]["r2"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'force_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Force comparison plot saved")


def plot_error_distributions(standard_results, physics_results, output_dir):
    """Plot error distributions"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Energy errors
    standard_energy_err = (standard_results['energy_pred'] - standard_results['energy_true']).numpy().flatten()
    physics_energy_err = (physics_results['energy_pred'] - physics_results['energy_true']).numpy().flatten()

    # Energy error distribution
    ax = axes[0, 0]
    ax.hist(standard_energy_err, bins=50, alpha=0.5, label='Standard', density=True)
    ax.hist(physics_energy_err, bins=50, alpha=0.5, label='Physics-Informed', density=True)
    ax.set_xlabel('Energy Error (eV)')
    ax.set_ylabel('Density')
    ax.set_title('Energy Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy error box plot
    ax = axes[0, 1]
    ax.boxplot([standard_energy_err, physics_energy_err],
               labels=['Standard', 'Physics-Informed'])
    ax.set_ylabel('Energy Error (eV)')
    ax.set_title('Energy Error Box Plot')
    ax.grid(True, alpha=0.3)

    # Force errors
    standard_force_err = (standard_results['forces_pred'] - standard_results['forces_true']).numpy().reshape(-1)
    physics_force_err = (physics_results['forces_pred'] - physics_results['forces_true']).numpy().reshape(-1)

    # Force error distribution
    ax = axes[1, 0]
    ax.hist(standard_force_err, bins=50, alpha=0.5, label='Standard', density=True)
    ax.hist(physics_force_err, bins=50, alpha=0.5, label='Physics-Informed', density=True)
    ax.set_xlabel('Force Error (eV/Å)')
    ax.set_ylabel('Density')
    ax.set_title('Force Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Force error box plot
    ax = axes[1, 1]
    ax.boxplot([standard_force_err, physics_force_err],
               labels=['Standard', 'Physics-Informed'])
    ax.set_ylabel('Force Error (eV/Å)')
    ax.set_title('Force Error Box Plot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Error distribution plot saved")


def plot_metrics_comparison(standard_metrics, physics_metrics, output_dir):
    """Bar plot comparing metrics"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Energy metrics
    ax = axes[0]
    metrics = ['MAE', 'RMSE']
    standard_vals = [standard_metrics['energy']['mae'], standard_metrics['energy']['rmse']]
    physics_vals = [physics_metrics['energy']['mae'], physics_metrics['energy']['rmse']]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, standard_vals, width, label='Standard', alpha=0.8)
    ax.bar(x + width / 2, physics_vals, width, label='Physics-Informed', alpha=0.8)

    ax.set_ylabel('Error (eV)')
    ax.set_title('Energy Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Force metrics
    ax = axes[1]
    standard_vals = [standard_metrics['force']['mae'], standard_metrics['force']['rmse']]
    physics_vals = [physics_metrics['force']['mae'], physics_metrics['force']['rmse']]

    ax.bar(x - width / 2, standard_vals, width, label='Standard', alpha=0.8)
    ax.bar(x + width / 2, physics_vals, width, label='Physics-Informed', alpha=0.8)

    ax.set_ylabel('Error (eV/Å)')
    ax.set_title('Force Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Metrics comparison plot saved")


def create_summary_report(standard_metrics, physics_metrics, output_dir):
    """Create text summary report"""

    report = f"""
MODEL COMPARISON REPORT
{'=' * 60}

ENERGY METRICS
{'-' * 60}
                    Standard        Physics-Informed    Improvement
MAE (eV):          {standard_metrics['energy']['mae']:10.6f}    {physics_metrics['energy']['mae']:10.6f}    {(1 - physics_metrics['energy']['mae'] / standard_metrics['energy']['mae']) * 100:6.2f}%
RMSE (eV):         {standard_metrics['energy']['rmse']:10.6f}    {physics_metrics['energy']['rmse']:10.6f}    {(1 - physics_metrics['energy']['rmse'] / standard_metrics['energy']['rmse']) * 100:6.2f}%
R²:                {standard_metrics['energy']['r2']:10.6f}    {physics_metrics['energy']['r2']:10.6f}    {(physics_metrics['energy']['r2'] - standard_metrics['energy']['r2']) * 100:6.2f}%

FORCE METRICS
{'-' * 60}
                    Standard        Physics-Informed    Improvement
MAE (eV/Å):        {standard_metrics['force']['mae']:10.6f}    {physics_metrics['force']['mae']:10.6f}    {(1 - physics_metrics['force']['mae'] / standard_metrics['force']['mae']) * 100:6.2f}%
RMSE (eV/Å):       {standard_metrics['force']['rmse']:10.6f}    {physics_metrics['force']['rmse']:10.6f}    {(1 - physics_metrics['force']['rmse'] / standard_metrics['force']['rmse']) * 100:6.2f}%
R²:                {standard_metrics['force']['r2']:10.6f}    {physics_metrics['force']['r2']:10.6f}    {(physics_metrics['force']['r2'] - standard_metrics['force']['r2']) * 100:6.2f}%

{'=' * 60}

CONCLUSION:
"""

    # Add conclusion based on improvements
    energy_improvement = (1 - physics_metrics['energy']['mae'] / standard_metrics['energy']['mae']) * 100
    force_improvement = (1 - physics_metrics['force']['mae'] / standard_metrics['force']['mae']) * 100

    if energy_improvement > 0 and force_improvement > 0:
        report += f"\n✓ Physics-informed model shows improvement in both energy ({energy_improvement:.1f}%) and force ({force_improvement:.1f}%) predictions.\n"
    elif energy_improvement > 0:
        report += f"\n✓ Physics-informed model shows improvement in energy predictions ({energy_improvement:.1f}%).\n"
    elif force_improvement > 0:
        report += f"\n✓ Physics-informed model shows improvement in force predictions ({force_improvement:.1f}%).\n"
    else:
        report += "\n⚠ Physics-informed model did not show improvement. Consider adjusting loss weights.\n"

    report += "\nSee plots in the output directory for detailed visualizations.\n"

    # Save report
    report_path = Path(output_dir) / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"\n✓ Report saved to {report_path}")


def compare_models(standard_checkpoint, physics_checkpoint, dataset='MD17',
                   molecule='aspirin', output_dir='results/plots'):
    """
    Main comparison function

    Args:
        standard_checkpoint: Path to standard model checkpoint
        physics_checkpoint: Path to physics-informed model checkpoint
        dataset: Dataset name
        molecule: Molecule name (for MD17)
        output_dir: Directory to save plots
    """

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # Load models
    standard_model, physics_model = load_models(standard_checkpoint, physics_checkpoint)

    # Load test dataset
    print("\nLoading test dataset...")
    data = DataModule(
        dataset=dataset,
        dataset_root='./data',
        dataset_arg=molecule if dataset in ['MD17', 'rMD17'] else '7',
        batch_size=32,
        num_workers=4,
        splits=[0.8, 0.1, 0.1],
        seed=42
    )
    data.setup('test')
    test_loader = data.test_dataloader()

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Evaluate standard model
    print("\nEvaluating standard model...")
    standard_results = evaluate_on_dataset(standard_model, test_loader, device, max_batches=100)
    standard_metrics = compute_metrics(standard_results)

    # Evaluate physics-informed model
    print("\nEvaluating physics-informed model...")
    physics_results = evaluate_on_dataset(physics_model, test_loader, device, max_batches=100)
    physics_metrics = compute_metrics(physics_results)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_energy_comparison(standard_results, physics_results,
                           standard_metrics, physics_metrics, output_dir)

    plot_force_comparison(standard_results, physics_results,
                          standard_metrics, physics_metrics, output_dir)

    plot_error_distributions(standard_results, physics_results, output_dir)

    plot_metrics_comparison(standard_metrics, physics_metrics, output_dir)

    # Create summary report
    create_summary_report(standard_metrics, physics_metrics, output_dir)

    # Save metrics as JSON
    comparison_data = {
        'standard': standard_metrics,
        'physics_informed': physics_metrics,
    }

    with open(Path(output_dir) / 'metrics.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare TorchMD-NET models')
    parser.add_argument('--standard', type=str, required=True,
                        help='Path to standard model checkpoint')
    parser.add_argument('--physics', type=str, required=True,
                        help='Path to physics-informed model checkpoint')
    parser.add_argument('--dataset', type=str, default='MD17')
    parser.add_argument('--molecule', type=str, default='aspirin')
    parser.add_argument('--output-dir', type=str, default='results/plots')

    args = parser.parse_args()

    compare_models(
        standard_checkpoint=args.standard,
        physics_checkpoint=args.physics,
        dataset=args.dataset,
        molecule=args.molecule,
        output_dir=args.output_dir
    )