#!/usr/bin/env python
"""
Main CLI entry point for training and comparison.

This runner now treats delta-learning as the default advanced path:
- `train_standard.py` handles supervised training with optional delta labels.
- `train_physics.py` adds physics regularization on top of that.
"""

import sys
import argparse
from pathlib import Path


def print_banner():
    """Print a simple startup banner."""
    print("\n" + "=" * 70)
    print("TorchMD-NET Training Runner (Delta-Learning Ready)")
    print("=" * 70)


def get_user_choice():
    """Interactive mode selector when --mode is not provided."""
    print("\nSelect training mode:")
    print("  [1] Standard training")
    print("  [2] Physics-informed training")
    print("  [3] Train both + compare")
    print("  [4] Exit")

    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        if choice in ["1", "2", "3", "4"]:
            return choice
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


def setup_directories():
    """Create output directories expected by the training scripts."""
    for d in [
        "data",
        "checkpoints",
        "checkpoints/standard",
        "checkpoints/physics_informed",
        "logs",
        "logs/standard",
        "logs/physics_informed",
        "results",
        "results/plots",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("Directories ready.")


def check_dependencies():
    """Check packages required by the current src entry points."""
    print("\nChecking dependencies...")

    required = {
        "torch": "PyTorch",
        "torchmdnet": "TorchMD-NET",
        "lightning": "Lightning",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
    }

    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  OK {name}")
        except ImportError:
            print(f"  MISSING {name}")
            missing.append(package)

    if not missing:
        print("All required dependencies are installed.")
        return True

    print("\nInstall missing dependencies with pip, then rerun.")
    if "torchmdnet" in missing:
        print("  pip install torchmd-net-cu11 --extra-index-url https://download.pytorch.org/whl/cu118")
    print(f"  pip install {' '.join([m for m in missing if m != 'torchmdnet'])}")
    return False


def run_standard_training(args):
    """Run standard trainer (can still be delta-learning when enabled)."""
    print("\n" + "=" * 70)
    print("STANDARD TRAINING")
    print("=" * 70)

    from train_standard import train_standard_model

    train_standard_model(
        dataset=args.dataset,
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        save_dir="checkpoints/standard",
        log_dir="logs/standard",
        delta_learning=args.delta_learning,
        baseline_epsilon_eV=args.baseline_eps,
        baseline_sigma_A=args.baseline_sigma,
        baseline_cutoff_A=args.baseline_cutoff,
    )


def run_physics_informed_training(args):
    """Run physics-informed trainer (also delta-capable)."""
    print("\n" + "=" * 70)
    print("PHYSICS-INFORMED TRAINING")
    print("=" * 70)

    from train_physics import train_physics_informed_model

    train_physics_informed_model(
        dataset=args.dataset,
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        save_dir="checkpoints/physics_informed",
        log_dir="logs/physics_informed",
        force_weight=args.force_weight,
        energy_weight=args.energy_weight,
        nve_weight=args.nve_weight,
        pbc_weight=args.pbc_weight,
        momentum_weight=args.momentum_weight,
        traj_length=args.traj_length,
        delta_learning=args.delta_learning,
        baseline_epsilon_eV=args.baseline_eps,
        baseline_sigma_A=args.baseline_sigma,
        baseline_cutoff_A=args.baseline_cutoff,
    )


def run_comparison(args):
    """Train both models and then run metric/plot comparison."""
    print("\n" + "=" * 70)
    print("TRAIN + COMPARE")
    print("=" * 70)

    print("\n[1/3] Training standard model...")
    run_standard_training(args)

    print("\n[2/3] Training physics-informed model...")
    run_physics_informed_training(args)

    print("\n[3/3] Comparing checkpoints...")
    from compare_models import compare_models

    compare_models(
        standard_checkpoint="checkpoints/standard/best_model.ckpt",
        physics_checkpoint="checkpoints/physics_informed/best_model.ckpt",
        dataset=args.dataset,
        molecule=args.molecule,
        output_dir="results/plots",
    )


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="TorchMD-NET training runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", type=str, default="MD17", choices=["MD17", "rMD17", "MD22", "QM9"])
    parser.add_argument("--molecule", type=str, default="aspirin")

    parser.add_argument(
        "--model",
        type=str,
        default="tensornet",
        choices=["tensornet", "equivariant-transformer", "graph-network"],
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Delta-learning controls.
    parser.add_argument("--delta-learning", action="store_true")
    parser.add_argument("--baseline-eps", type=float, default=0.01)
    parser.add_argument("--baseline-sigma", type=float, default=1.0)
    parser.add_argument("--baseline-cutoff", type=float, default=5.0)

    parser.add_argument("--force-weight", type=float, default=0.95)
    parser.add_argument("--energy-weight", type=float, default=0.05)

    parser.add_argument("--nve-weight", type=float, default=1.0)
    parser.add_argument("--pbc-weight", type=float, default=0.1)
    parser.add_argument("--momentum-weight", type=float, default=0.01)
    parser.add_argument("--traj-length", type=int, default=100)

    parser.add_argument("--mode", type=str, choices=["standard", "physics", "compare"])

    return parser.parse_args()


def main():
    """Program entry point."""
    print_banner()
    args = parse_args()

    if not check_dependencies():
        sys.exit(1)

    setup_directories()

    if args.mode:
        choice = {"standard": "1", "physics": "2", "compare": "3"}[args.mode]
    else:
        choice = get_user_choice()

    if choice == "4":
        print("Exiting.")
        sys.exit(0)

    print("\nConfiguration summary")
    print(f"  dataset={args.dataset}")
    print(f"  molecule={args.molecule}")
    print(f"  model={args.model}")
    print(f"  batch_size={args.batch_size}")
    print(f"  epochs={args.epochs}")
    print(f"  lr={args.lr}")
    print(f"  delta_learning={args.delta_learning}")
    if args.delta_learning:
        print(
            f"  baseline=(eps={args.baseline_eps}, sigma={args.baseline_sigma}, cutoff={args.baseline_cutoff})"
        )

    confirm = input("\nProceed with training? (y/n): ").strip().lower()
    if confirm != "y":
        print("Training cancelled.")
        sys.exit(0)

    try:
        if choice == "1":
            run_standard_training(args)
        elif choice == "2":
            run_physics_informed_training(args)
        elif choice == "3":
            run_comparison(args)
        print("\nDone.")
    except Exception as exc:
        print(f"\nTraining failed: {exc}")
        raise


if __name__ == "__main__":
    main()
