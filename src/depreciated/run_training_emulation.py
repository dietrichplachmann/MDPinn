#!/usr/bin/env python
"""
Main Runner Script for TorchMD-NET Training
Supports both standard and physics-informed training modes
"""

import os
import sys
import argparse
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║         TorchMD-NET Training System                       ║
    ║         Physics-Informed Neural Network Potentials        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_user_choice():
    """Interactive menu for choosing training mode"""
    print("\nSelect training mode:")
    print("  [1] Standard TorchMD-NET (baseline)")
    print("  [2] Physics-Informed TorchMD-NET (with NVE, PBC, momentum losses)")
    print("  [3] Compare both models (train both and generate comparison plots)")
    print("  [4] Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return choice
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data',
        'checkpoints',
        'checkpoints/standard',
        'checkpoints/physics_informed',
        'logs',
        'logs/standard',
        'logs/physics_informed',
        'results',
        'results/plots',
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("✓ Directories created")


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")

    required = {
        'torch': 'PyTorch',
        'torchmdnet': 'TorchMD-NET',
        'pytorch_lightning': 'PyTorch Lightning',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy',
    }

    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(package)

    if missing:
        print("\n⚠ Missing dependencies detected!")
        print("Install with:")
        if 'torchmdnet' in missing:
            print("  pip install torchmd-net-cu11 --extra-index-url https://download.pytorch.org/whl/cu118")
            missing.remove('torchmdnet')
        if missing:
            print(f"  pip install {' '.join(missing)}")
        return False

    print("\n✓ All dependencies satisfied")
    return True


def run_standard_training(args):
    """Run standard TorchMD-NET training"""
    print("\n" + "=" * 60)
    print("STANDARD TORCHMD-NET TRAINING")
    print("=" * 60)

    from train_standard import train_standard_model

    train_standard_model(
        dataset=args.dataset,
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        save_dir='checkpoints/standard',
        log_dir='logs/standard'
    )

    print("\n✓ Standard training complete!")
    print(f"  Model saved to: checkpoints/standard/")


def run_physics_informed_training(args):
    """Run physics-informed training"""
    print("\n" + "=" * 60)
    print("PHYSICS-INFORMED TORCHMD-NET TRAINING")
    print("=" * 60)

    from train_physics import train_physics_informed_model

    train_physics_informed_model(
        dataset=args.dataset,
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        save_dir='checkpoints/physics_informed',
        log_dir='logs/physics_informed',
        # Physics loss weights
        force_weight=args.force_weight,
        energy_weight=args.energy_weight,
        nve_weight=args.nve_weight,
        pbc_weight=args.pbc_weight,
        momentum_weight=args.momentum_weight,
        traj_length=args.traj_length,
    )

    print("\n✓ Physics-informed training complete!")
    print(f"  Model saved to: checkpoints/physics_informed/")


def run_comparison(args):
    """Train both models and generate comparison plots"""
    print("\n" + "=" * 60)
    print("TRAINING BOTH MODELS FOR COMPARISON")
    print("=" * 60)

    # Train standard model
    print("\n[1/3] Training standard model...")
    run_standard_training(args)

    # Train physics-informed model
    print("\n[2/3] Training physics-informed model...")
    run_physics_informed_training(args)

    # Generate comparison
    print("\n[3/3] Generating comparison plots...")
    from compare_models import compare_models

    compare_models(
        standard_checkpoint='checkpoints/standard/best_model.ckpt',
        physics_checkpoint='checkpoints/physics_informed/best_model.ckpt',
        dataset=args.dataset,
        molecule=args.molecule,
        output_dir='results/plots'
    )

    print("\n✓ Comparison complete!")
    print(f"  Results saved to: results/plots/")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='TorchMD-NET Training Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset options
    parser.add_argument('--dataset', type=str, default='MD17',
                        choices=['MD17', 'rMD17', 'MD22', 'QM9'],
                        help='Dataset to use')
    parser.add_argument('--molecule', type=str, default='aspirin',
                        help='Molecule name (for MD17/rMD17)')

    # Model options
    parser.add_argument('--model', type=str, default='tensornet',
                        choices=['tensornet', 'equivariant-transformer', 'graph-network'],
                        help='Model architecture')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')

    # Standard loss weights
    parser.add_argument('--force-weight', type=float, default=0.95,
                        help='Weight for force loss')
    parser.add_argument('--energy-weight', type=float, default=0.05,
                        help='Weight for energy loss')

    # Physics-informed loss weights
    parser.add_argument('--nve-weight', type=float, default=1.0,
                        help='Weight for NVE conservation loss')
    parser.add_argument('--pbc-weight', type=float, default=0.1,
                        help='Weight for periodic boundary condition loss')
    parser.add_argument('--momentum-weight', type=float, default=0.01,
                        help='Weight for momentum conservation loss')

    # Trajectory options
    parser.add_argument('--traj-length', type=int, default=100,
                        help='Trajectory length for NVE loss')

    # Mode (can override interactive selection)
    parser.add_argument('--mode', type=str, choices=['standard', 'physics', 'compare'],
                        help='Training mode (skips interactive menu)')

    return parser.parse_args()


def main():
    """Main entry point"""
    print_banner()

    # Parse arguments
    args = parse_args()

    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Please install missing dependencies and try again.")
        sys.exit(1)

    # Setup directories
    setup_directories()

    # Get training mode
    if args.mode:
        mode_map = {'standard': '1', 'physics': '2', 'compare': '3'}
        choice = mode_map[args.mode]
    else:
        choice = get_user_choice()

    if choice == '4':
        print("\nExiting...")
        sys.exit(0)

    # Display configuration
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Dataset:      {args.dataset}")
    if args.dataset in ['MD17', 'rMD17']:
        print(f"  Molecule:     {args.molecule}")
    print(f"  Model:        {args.model}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    if choice in ['2', '3']:
        print(f"\n  Physics Loss Weights:")
        print(f"    Force:      {args.force_weight}")
        print(f"    Energy:     {args.energy_weight}")
        print(f"    NVE:        {args.nve_weight}")
        print(f"    PBC:        {args.pbc_weight}")
        print(f"    Momentum:   {args.momentum_weight}")
        print(f"  Traj Length:  {args.traj_length}")
    print("=" * 60)

    # Confirm
    confirm = input("\nProceed with training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        sys.exit(0)

    # Run selected mode
    try:
        if choice == '1':
            run_standard_training(args)
        elif choice == '2':
            run_physics_informed_training(args)
        elif choice == '3':
            run_comparison(args)

        print("\n" + "=" * 60)
        print("ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()