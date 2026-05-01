#!/usr/bin/env python
"""
Main CLI entry point for training only.

This runner is intentionally limited to model production:
- `train_standard.py` handles supervised training with optional delta labels.
- `train_physics.py` adds physics regularization on top of that.
- Post-training analysis belongs in `run_evaluation.py` so checkpoint
  selection and graph generation can be rerun independently of training.

Physical interpretation:
- Without `--delta-learning`, the neural network learns the FULL potential
  surface directly from reference energies/forces.
- With `--delta-learning`, the network learns only the correction
  DeltaU(R) to an analytic baseline U_ref(R), so the deployed model is
  U_hyb(R) = U_ref(R) + DeltaU(R).
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
    print("  [3] Optuna tuning")
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
        "results/optuna",
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


def check_tuning_dependencies():
    """Check packages required by the Optuna tuner."""
    if not check_dependencies():
        return False

    try:
        __import__("optuna")
        print("  OK Optuna")
        return True
    except ImportError:
        print("  MISSING Optuna")
        print("\nInstall missing dependencies with pip, then rerun.")
        print("  pip install optuna")
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
        embedding_dimension=args.embedding_dimension,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        checkpoint_name=args.checkpoint_name,
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
        nve_freq=args.nve_freq,
        nve_warmup_epochs=args.nve_warmup_epochs,
        nve_ramp_epochs=args.nve_ramp_epochs,
        nve_relative=args.nve_relative,
        nve_relative_eps=args.nve_relative_eps,
        delta_learning=args.delta_learning,
        baseline_epsilon_eV=args.baseline_eps,
        baseline_sigma_A=args.baseline_sigma,
        baseline_cutoff_A=args.baseline_cutoff,
        embedding_dimension=args.embedding_dimension,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        checkpoint_name=args.checkpoint_name,
    )


def run_optuna_tuning(args):
    """Run the Optuna tuner from a JSON config file."""
    if not args.tuning_config:
        raise ValueError("--tuning-config is required when mode=tune.")
    if not check_tuning_dependencies():
        raise RuntimeError("Required tuning dependencies are not installed.")

    print("\n" + "=" * 70)
    print("OPTUNA TUNING")
    print("=" * 70)

    from optuna_tuning import run_study

    run_study(args.tuning_config)


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
    # Think of these as selecting the analytic "first guess" Hamiltonian:
    # the NN then learns what that baseline misses.
    parser.add_argument("--delta-learning", action="store_true")
    parser.add_argument("--baseline-eps", type=float, default=0.01)
    parser.add_argument("--baseline-sigma", type=float, default=1.0)
    parser.add_argument("--baseline-cutoff", type=float, default=5.0)

    parser.add_argument("--force-weight", type=float, default=0.95)
    parser.add_argument("--energy-weight", type=float, default=0.05)

    parser.add_argument("--nve-weight", type=float, default=0.01)
    parser.add_argument("--pbc-weight", type=float, default=0.0)
    parser.add_argument("--momentum-weight", type=float, default=0.01)
    parser.add_argument("--traj-length", type=int, default=100)
    parser.add_argument("--nve-freq", type=int, default=50)
    parser.add_argument("--nve-warmup-epochs", type=int, default=5)
    parser.add_argument("--nve-ramp-epochs", type=int, default=20)
    parser.add_argument("--nve-relative", dest="nve_relative", action="store_true")
    parser.add_argument("--nve-absolute", dest="nve_relative", action="store_false")
    parser.set_defaults(nve_relative=True)
    parser.add_argument("--nve-relative-eps", type=float, default=1e-6)
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-rbf", type=int, default=64)
    parser.add_argument("--checkpoint-name", type=str, default="best_model")

    parser.add_argument("--mode", type=str, choices=["standard", "physics", "tune"])
    parser.add_argument("--tuning-config", type=str, default=None)

    return parser.parse_args()


def main():
    """Program entry point."""
    print_banner()
    args = parse_args()

    if args.mode and not (check_tuning_dependencies() if args.mode == "tune" else check_dependencies()):
        sys.exit(1)

    setup_directories()

    if args.mode:
        choice = {"standard": "1", "physics": "2", "tune": "3"}[args.mode]
    else:
        choice = get_user_choice()

    if not args.mode:
        dependency_ok = check_tuning_dependencies() if choice == "3" else check_dependencies()
        if not dependency_ok:
            sys.exit(1)

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
    if choice in ["1", "2"]:
        print(f"  embedding_dimension={args.embedding_dimension}")
        print(f"  num_layers={args.num_layers}")
        print(f"  num_rbf={args.num_rbf}")
        print(f"  checkpoint_name={args.checkpoint_name}")
    if choice == "2":
        print(f"  momentum_weight={args.momentum_weight}")
        print(f"  nve_weight={args.nve_weight}")
        print(f"  nve_freq={args.nve_freq}")
        print(f"  nve_warmup_epochs={args.nve_warmup_epochs}")
        print(f"  nve_ramp_epochs={args.nve_ramp_epochs}")
        print(f"  nve_relative={args.nve_relative}")
    if choice == "3":
        print(f"  tuning_config={args.tuning_config}")

    print("  evaluation=run separately via src/run_evaluation.py")

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
            run_optuna_tuning(args)
        print("\nDone.")
    except Exception as exc:
        print(f"\nTraining failed: {exc}")
        raise


if __name__ == "__main__":
    main()
