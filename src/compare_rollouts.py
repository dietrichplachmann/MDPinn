#!/usr/bin/env python
"""
Run matched rollout comparisons for two trained checkpoints.

Typical use:
    python src/compare_rollouts.py ^
      --standard checkpoints/standard/best_model.ckpt ^
      --candidate checkpoints/physics_informed/best_model.ckpt ^
      --molecule aspirin ^
      --n-rollouts 100
"""

import argparse
import json
from pathlib import Path

import torch

from experiment_suite import plot_rollout_drift_comparison, run_rollout_summary


def _build_comparison_summary(standard_rollout, candidate_rollout, standard_label, candidate_label):
    """Build a compact aggregate comparison record for downstream inspection."""
    std_fail = standard_rollout.get("failure_rate")
    cand_fail = candidate_rollout.get("failure_rate")
    std_drift = standard_rollout.get("mean_max_abs_drift_eV")
    cand_drift = candidate_rollout.get("mean_max_abs_drift_eV")
    std_final = standard_rollout.get("mean_final_drift_eV")
    cand_final = candidate_rollout.get("mean_final_drift_eV")

    return {
        "labels": {
            "standard": standard_label,
            "candidate": candidate_label,
        },
        "failure_rate_delta": None if std_fail is None or cand_fail is None else float(cand_fail - std_fail),
        "mean_max_abs_drift_delta_eV": None
        if std_drift is None or cand_drift is None
        else float(cand_drift - std_drift),
        "mean_final_drift_delta_eV": None
        if std_final is None or cand_final is None
        else float(cand_final - std_final),
        "better_failure_rate": None
        if std_fail is None or cand_fail is None
        else (candidate_label if cand_fail < std_fail else standard_label if std_fail < cand_fail else "tie"),
        "better_mean_max_abs_drift": None
        if std_drift is None or cand_drift is None
        else (candidate_label if cand_drift < std_drift else standard_label if std_drift < cand_drift else "tie"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare rollout stability for two trained checkpoints.")
    parser.add_argument("--standard", type=str, required=True, help="Path to the standard baseline checkpoint.")
    parser.add_argument("--candidate", type=str, required=True, help="Path to the candidate checkpoint to compare.")
    parser.add_argument("--dataset", type=str, default="MD17")
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--energy-log-stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/rollout_comparison")
    parser.add_argument("--standard-label", type=str, default="Standard")
    parser.add_argument("--candidate-label", type=str, default="Candidate")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running standard rollouts...")
    standard_rollout = run_rollout_summary(
        ckpt_path=args.standard,
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

    print("Running candidate rollouts...")
    candidate_rollout = run_rollout_summary(
        ckpt_path=args.candidate,
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

    comparison = {
        "config": {
            "dataset": args.dataset,
            "molecule": args.molecule,
            "data_root": args.data_root,
            "steps": args.steps,
            "dt": args.dt,
            "n_rollouts": args.n_rollouts,
            "energy_log_stride": args.energy_log_stride,
            "seed": args.seed,
            "device": args.device,
            "standard_checkpoint": args.standard,
            "candidate_checkpoint": args.candidate,
        },
        "standard": standard_rollout,
        "candidate": candidate_rollout,
        "comparison": _build_comparison_summary(
            standard_rollout,
            candidate_rollout,
            standard_label=args.standard_label,
            candidate_label=args.candidate_label,
        ),
    }

    summary_path = output_dir / "rollout_comparison.json"
    with open(summary_path, "w") as handle:
        json.dump(comparison, handle, indent=2)
    print(f"Wrote: {summary_path}")

    plot_rollout_drift_comparison(
        standard_rollout,
        candidate_rollout,
        output_dir / "rollout_drift_comparison.png",
        title=f"{args.standard_label} vs {args.candidate_label} Rollout Drift",
    )

    print(json.dumps(comparison["comparison"], indent=2))


if __name__ == "__main__":
    main()
