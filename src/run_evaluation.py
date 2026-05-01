#!/usr/bin/env python
"""
Dedicated post-training evaluation runner.

This entry point is intentionally separate from `run_training.py` so the user can:
- train checkpoints once,
- choose which checkpoint(s) are worth inspecting,
- rerun evaluation plots and summaries without retraining.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_models import (
    compare_models,
    compute_metrics,
    create_dataset,
    evaluate_energy_conservation,
    evaluate_on_dataset,
    load_checkpoint,
)
from compare_rollouts import run_rollout_comparison
from experiment_suite import plot_rollout_drift_comparison, run_rollout_summary


def _single_model_plot(results, metrics, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    energy_true = results["energy_true"].flatten()
    energy_pred = results["energy_pred"].flatten()
    energy_pred = energy_pred - np.mean(energy_pred - energy_true)
    force_true = results["force_true"].flatten()
    force_pred = results["force_pred"].flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(energy_true, energy_pred, alpha=0.3, s=10)
    axes[0, 0].plot([energy_true.min(), energy_true.max()], [energy_true.min(), energy_true.max()], "r--", lw=2)
    axes[0, 0].set_title(f"{label} - Energy Parity")
    axes[0, 0].set_xlabel("Reference energy")
    axes[0, 0].set_ylabel("Predicted energy")

    energy_err = energy_pred - energy_true
    axes[0, 1].hist(energy_err, bins=50, alpha=0.75, edgecolor="black")
    axes[0, 1].set_title(f"{label} - Energy Error (bias aligned)")
    axes[0, 1].set_xlabel("Energy error")

    axes[1, 0].scatter(force_true, force_pred, alpha=0.08, s=5)
    axes[1, 0].plot([force_true.min(), force_true.max()], [force_true.min(), force_true.max()], "r--", lw=2)
    axes[1, 0].set_title(f"{label} - Force Parity")
    axes[1, 0].set_xlabel("Reference force")
    axes[1, 0].set_ylabel("Predicted force")

    force_err = force_pred - force_true
    axes[1, 1].hist(force_err, bins=50, alpha=0.75, edgecolor="black")
    axes[1, 1].set_title(f"{label} - Force Error")
    axes[1, 1].set_xlabel("Force error")

    metric_text = "\n".join(
        [
            f"Energy MAE: {metrics['energy_mae']:.6f}",
            f"Energy RMSE: {metrics['energy_rmse']:.6f}",
            f"Force MAE: {metrics['force_mae']:.6f}",
            f"Force RMSE: {metrics['force_rmse']:.6f}",
        ]
    )
    axes[0, 1].text(
        0.98,
        0.98,
        metric_text,
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black"},
    )

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "single_model_plots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_single_checkpoint(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint...")
    model = load_checkpoint(args.checkpoint, device=args.device)

    print("Loading dataset...")
    _, _, test_data, full_dataset = create_dataset(args.dataset, args.molecule, args.data_root)

    print("Evaluating held-out frames...")
    results = evaluate_on_dataset(model, test_data, device=args.device)
    metrics = compute_metrics(results)
    drift_proxy = evaluate_energy_conservation(
        model,
        full_dataset,
        device=args.device,
        traj_length=args.drift_traj_length,
        num_trajs=args.drift_num_trajs,
    )

    print("Running rollouts...")
    rollout = run_rollout_summary(
        ckpt_path=args.checkpoint,
        dataset=args.dataset,
        molecule=args.molecule,
        data_root=args.data_root,
        steps=args.steps,
        dt=args.dt,
        n_rollouts=args.n_rollouts,
        seed=args.seed,
        device=args.device,
        energy_log_stride=args.energy_log_stride,
    )

    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "molecule": args.molecule,
            "data_root": args.data_root,
            "steps": args.steps,
            "dt": args.dt,
            "n_rollouts": args.n_rollouts,
            "energy_log_stride": args.energy_log_stride,
            "seed": args.seed,
            "device": args.device,
        },
        "metrics": metrics,
        "energy_drift_proxy": drift_proxy,
        "rollout": rollout,
    }

    with open(output_dir / "evaluation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    _single_model_plot(results, metrics, output_dir, args.label)
    plot_rollout_drift_comparison(
        rollout,
        {},
        output_dir / "rollout_drift.png",
        title=f"{args.label} Rollout Drift",
    )
    print(f"Wrote: {output_dir}")


def evaluate_checkpoint_pair(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running framewise comparison...")
    comparison_dir = output_dir / "framewise"
    compare_models(
        standard_checkpoint=args.standard,
        physics_checkpoint=args.candidate,
        dataset=args.dataset,
        molecule=args.molecule,
        data_root=args.data_root,
        output_dir=str(comparison_dir),
        device=args.device,
    )

    print("Running rollout comparison...")
    rollout_dir = output_dir / "rollout"
    run_rollout_comparison(
        standard_checkpoint=args.standard,
        candidate_checkpoint=args.candidate,
        dataset=args.dataset,
        molecule=args.molecule,
        data_root=args.data_root,
        steps=args.steps,
        dt=args.dt,
        n_rollouts=args.n_rollouts,
        energy_log_stride=args.energy_log_stride,
        seed=args.seed,
        device=args.device,
        output_dir=str(rollout_dir),
        standard_label=args.standard_label,
        candidate_label=args.candidate_label,
    )
    print(f"Wrote: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run post-training evaluation and graph generation for one or two checkpoints."
    )
    parser.add_argument("--dataset", type=str, default="MD17")
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--energy-log-stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift-traj-length", type=int, default=100)
    parser.add_argument("--drift-num-trajs", type=int, default=10)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=str, help="Evaluate one checkpoint.")
    mode.add_argument("--standard", type=str, help="Reference checkpoint for paired comparison.")

    parser.add_argument("--candidate", type=str, help="Candidate checkpoint for paired comparison.")
    parser.add_argument("--label", type=str, default="Model")
    parser.add_argument("--standard-label", type=str, default="Standard")
    parser.add_argument("--candidate-label", type=str, default="Candidate")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checkpoint:
        evaluate_single_checkpoint(args)
        return

    if not args.candidate:
        raise ValueError("--candidate is required when --standard is used.")
    evaluate_checkpoint_pair(args)


if __name__ == "__main__":
    main()
