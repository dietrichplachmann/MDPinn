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
import csv
import json
from statistics import median
from pathlib import Path

import torch

from experiment_suite import (
    plot_rollout_drift_comparison,
    plot_single_rollout_drift_comparison,
    run_rollout_summary,
)


def _mean_or_none(values):
    return None if not values else float(sum(values) / len(values))


def _median_or_none(values):
    return None if not values else float(median(values))


def _build_paired_rollout_summary(standard_rollout, candidate_rollout, standard_label, candidate_label):
    standard_rows = standard_rollout.get("rollouts", [])
    candidate_rows = candidate_rollout.get("rollouts", [])
    pairs = list(zip(standard_rows, candidate_rows))

    std_abs_final = []
    cand_abs_final = []
    std_max = []
    cand_max = []
    candidate_better_abs_final = 0
    candidate_better_max = 0
    paired_successes = 0

    for std_row, cand_row in pairs:
        if std_row.get("failed") or cand_row.get("failed"):
            continue
        paired_successes += 1

        std_final = abs(float(std_row["final_drift"]))
        cand_final = abs(float(cand_row["final_drift"]))
        std_peak = float(std_row["max_abs_drift"])
        cand_peak = float(cand_row["max_abs_drift"])

        std_abs_final.append(std_final)
        cand_abs_final.append(cand_final)
        std_max.append(std_peak)
        cand_max.append(cand_peak)

        if cand_final < std_final:
            candidate_better_abs_final += 1
        if cand_peak < std_peak:
            candidate_better_max += 1

    return {
        "paired_successes": paired_successes,
        "mean_abs_final_drift_eV": {
            "standard": _mean_or_none(std_abs_final),
            "candidate": _mean_or_none(cand_abs_final),
            "delta": None if not std_abs_final or not cand_abs_final else float(_mean_or_none(cand_abs_final) - _mean_or_none(std_abs_final)),
            "better": None
            if not std_abs_final or not cand_abs_final
            else (candidate_label if _mean_or_none(cand_abs_final) < _mean_or_none(std_abs_final) else standard_label if _mean_or_none(std_abs_final) < _mean_or_none(cand_abs_final) else "tie"),
        },
        "median_abs_final_drift_eV": {
            "standard": _median_or_none(std_abs_final),
            "candidate": _median_or_none(cand_abs_final),
        },
        "median_max_abs_drift_eV": {
            "standard": _median_or_none(std_max),
            "candidate": _median_or_none(cand_max),
        },
        "candidate_better_count_abs_final": candidate_better_abs_final,
        "candidate_better_count_max_abs_drift": candidate_better_max,
    }


def _build_single_rollout_comparison(standard_row, candidate_row, standard_label, candidate_label):
    std_final = standard_row.get("final_drift")
    cand_final = candidate_row.get("final_drift")
    std_max = standard_row.get("max_abs_drift")
    cand_max = candidate_row.get("max_abs_drift")

    std_abs_final = None if std_final is None else abs(float(std_final))
    cand_abs_final = None if cand_final is None else abs(float(cand_final))

    return {
        "labels": {
            "standard": standard_label,
            "candidate": candidate_label,
        },
        "start_idx_standard": standard_row.get("start_idx"),
        "start_idx_candidate": candidate_row.get("start_idx"),
        "matched_start_idx": standard_row.get("start_idx") if standard_row.get("start_idx") == candidate_row.get("start_idx") else None,
        "standard_failed": bool(standard_row.get("failed")),
        "candidate_failed": bool(candidate_row.get("failed")),
        "standard_final_drift_eV": std_final,
        "candidate_final_drift_eV": cand_final,
        "standard_abs_final_drift_eV": std_abs_final,
        "candidate_abs_final_drift_eV": cand_abs_final,
        "abs_final_drift_delta_eV": None
        if std_abs_final is None or cand_abs_final is None
        else float(cand_abs_final - std_abs_final),
        "standard_max_abs_drift_eV": std_max,
        "candidate_max_abs_drift_eV": cand_max,
        "max_abs_drift_delta_eV": None if std_max is None or cand_max is None else float(cand_max - std_max),
        "better_abs_final_drift": None
        if std_abs_final is None or cand_abs_final is None
        else (candidate_label if cand_abs_final < std_abs_final else standard_label if std_abs_final < cand_abs_final else "tie"),
        "better_max_abs_drift": None
        if std_max is None or cand_max is None
        else (candidate_label if cand_max < std_max else standard_label if std_max < cand_max else "tie"),
    }


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
        "paired_metrics": _build_paired_rollout_summary(
            standard_rollout,
            candidate_rollout,
            standard_label=standard_label,
            candidate_label=candidate_label,
        ),
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

    per_rollout_dir = output_dir / "per_rollout"
    per_rollout_dir.mkdir(parents=True, exist_ok=True)
    per_rollout_rows = []
    for idx, (standard_row, candidate_row) in enumerate(
        zip(standard_rollout.get("rollouts", []), candidate_rollout.get("rollouts", []))
    ):
        rollout_dir = per_rollout_dir / f"rollout_{idx:03d}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        single_comparison = _build_single_rollout_comparison(
            standard_row,
            candidate_row,
            standard_label=args.standard_label,
            candidate_label=args.candidate_label,
        )
        single_summary = {
            "config": {
                "dataset": args.dataset,
                "molecule": args.molecule,
                "steps": args.steps,
                "dt": args.dt,
                "energy_log_stride": args.energy_log_stride,
                "seed": args.seed,
                "device": args.device,
                "standard_checkpoint": args.standard,
                "candidate_checkpoint": args.candidate,
                "rollout_index": idx,
            },
            "standard": standard_row,
            "candidate": candidate_row,
            "comparison": single_comparison,
        }
        with open(rollout_dir / "rollout_comparison.json", "w") as handle:
            json.dump(single_summary, handle, indent=2)
        print(f"Wrote: {rollout_dir / 'rollout_comparison.json'}")

        per_rollout_rows.append(
            {
                "rollout_index": idx,
                "start_idx_standard": single_comparison["start_idx_standard"],
                "start_idx_candidate": single_comparison["start_idx_candidate"],
                "matched_start_idx": single_comparison["matched_start_idx"],
                "standard_failed": single_comparison["standard_failed"],
                "candidate_failed": single_comparison["candidate_failed"],
                "standard_final_drift_eV": single_comparison["standard_final_drift_eV"],
                "candidate_final_drift_eV": single_comparison["candidate_final_drift_eV"],
                "standard_abs_final_drift_eV": single_comparison["standard_abs_final_drift_eV"],
                "candidate_abs_final_drift_eV": single_comparison["candidate_abs_final_drift_eV"],
                "abs_final_drift_delta_eV": single_comparison["abs_final_drift_delta_eV"],
                "standard_max_abs_drift_eV": single_comparison["standard_max_abs_drift_eV"],
                "candidate_max_abs_drift_eV": single_comparison["candidate_max_abs_drift_eV"],
                "max_abs_drift_delta_eV": single_comparison["max_abs_drift_delta_eV"],
                "better_abs_final_drift": single_comparison["better_abs_final_drift"],
                "better_max_abs_drift": single_comparison["better_max_abs_drift"],
                "json_path": str(rollout_dir / "rollout_comparison.json"),
                "plot_path": str(rollout_dir / "rollout_drift_comparison.png"),
            }
        )

        plot_single_rollout_drift_comparison(
            standard_row,
            candidate_row,
            rollout_dir / "rollout_drift_comparison.png",
            title=f"{args.standard_label} vs {args.candidate_label} Rollout Drift #{idx}",
            standard_label=args.standard_label,
            physics_label=args.candidate_label,
        )

    csv_path = output_dir / "per_rollout_summary.csv"
    csv_fields = [
        "rollout_index",
        "start_idx_standard",
        "start_idx_candidate",
        "matched_start_idx",
        "standard_failed",
        "candidate_failed",
        "standard_final_drift_eV",
        "candidate_final_drift_eV",
        "standard_abs_final_drift_eV",
        "candidate_abs_final_drift_eV",
        "abs_final_drift_delta_eV",
        "standard_max_abs_drift_eV",
        "candidate_max_abs_drift_eV",
        "max_abs_drift_delta_eV",
        "better_abs_final_drift",
        "better_max_abs_drift",
        "json_path",
        "plot_path",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in per_rollout_rows:
            writer.writerow(row)
    print(f"Wrote: {csv_path}")

    plot_rollout_drift_comparison(
        standard_rollout,
        candidate_rollout,
        output_dir / "rollout_drift_comparison.png",
        title=f"{args.standard_label} vs {args.candidate_label} Rollout Drift",
    )

    print(json.dumps(comparison["comparison"], indent=2))


if __name__ == "__main__":
    main()
