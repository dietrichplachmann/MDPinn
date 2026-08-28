#!/usr/bin/env python
"""Direct before/after rollout-stability comparison for one StABlE
fine-tuning run: same training seed and condition, same checkpoint
architecture - only the checkpoint itself differs (an ext70 baseline vs. its
own StABlE-fine-tuned descendant). run_rollout_study.py's own machinery
doesn't fit this directly (it's built to compare water_absolute vs.
water_absolute+momentum at one fixed checkpoint root, not "checkpoint A vs.
checkpoint B" for the same condition) - but its run_matrix/aggregate_results
don't actually care what the "condition" dict keys mean, so this reuses them
directly rather than duplicating the matrix/CSV/markdown-writing logic.

Matches the existing ext70 seed-1 velocity-axis baseline exactly (confirmed
directly from its own energy_history.csv timestamps, not assumed):
DATA_SEED=42 (read from run_rollout_study's own module-level global via the
imported run_matrix - not re-specified here, so it can't drift out of sync),
test_config_index=0, velocity seeds 0-4, dt=0.1, steps=10000 - so this run's
numbers are directly comparable to
results/waterbox_rollout_study_zbl_bonded_ext70_seed1/summary_table.md
without re-deriving anything. --vary config mirrors that baseline's config
axis the same way (config indices 1-5, fixed velocity seed 0).

Usage:
    python src/compare_stable_rollout.py \\
        --baseline-ckpt checkpoints/waterbox_study_zbl_bonded_ext70/water_absolute/seed1/best_model.ckpt \\
        --finetuned-ckpt checkpoints/waterbox_study_zbl_bonded_ext70_stable/water_absolute/seed1/stable_final.ckpt \\
        --out results/waterbox_rollout_stable_compare/water_absolute_seed1
"""

from __future__ import annotations

from pathlib import Path

from run_rollout_study import aggregate_results, run_matrix

VELOCITY_SEEDS = [0, 1, 2, 3, 4]
FIXED_TEST_CONFIG_INDEX = 0
CONFIG_INDICES = [1, 2, 3, 4, 5]
FIXED_VELOCITY_SEED = 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--finetuned-ckpt", type=str, required=True)
    parser.add_argument("--baseline-label", type=str, default="before_finetune")
    parser.add_argument("--finetuned-label", type=str, default="after_finetune")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--vary", choices=["velocity", "config"], default="velocity",
        help="Which replicate axis to sweep - matches run_rollout_study.py's own --vary exactly, "
        "same replicate seeds/indices, so results land on the same axis as the existing ext70 "
        "seed-1 baseline (summary_table.md / summary_table_by_config.md).",
    )
    parser.add_argument(
        "--steps", type=int, default=10000,
        help="Default 10000 matches the existing ext70 seed-1 baseline exactly (confirmed from its "
        "own energy_history.csv: 10000 steps at dt=0.1 -> 1000 fs). Only change this if "
        "deliberately comparing against a differently-scaled baseline.",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Default 0.1, NOT run_rollout_study.py's own default of 0.5 - this checkpoint carries "
        "the bonded-exclusion ZBL prior, which this project already established needs dt=0.1 "
        "(0.5 fs causes a genuine numerical explosion with any ZBL variant, unrelated to real "
        "stability - see CLAUDE.md's Q4 resolution). Matches the ext70 seed-1 baseline exactly.",
    )
    parser.add_argument("--temperature-k", type=float, default=300.0)
    args = parser.parse_args()

    conditions = {args.baseline_label: args.baseline_ckpt, args.finetuned_label: args.finetuned_ckpt}
    out_root = Path(args.out)

    if args.vary == "velocity":
        replicates = [(f"vseed{v}", v, FIXED_TEST_CONFIG_INDEX) for v in VELOCITY_SEEDS]
        raw_csv = out_root / "raw_results.csv"
        summary_csv = out_root / "summary_table.csv"
        summary_md = out_root / "summary_table.md"
    else:
        replicates = [(f"cfg{c}", FIXED_VELOCITY_SEED, c) for c in CONFIG_INDICES]
        raw_csv = out_root / "raw_results_by_config.csv"
        summary_csv = out_root / "summary_table_by_config.csv"
        summary_md = out_root / "summary_table_by_config.md"

    note = (
        f"Same training seed/condition, before ({args.baseline_label}={args.baseline_ckpt}) vs. "
        f"after ({args.finetuned_label}={args.finetuned_ckpt}) StABlE fine-tuning. "
        f"{args.vary}-axis replicates at dt={args.dt}, matching "
        "results/waterbox_rollout_study_zbl_bonded_ext70_seed1's own DATA_SEED/test_config_index "
        "exactly for direct comparability."
    )

    rows, failed = run_matrix(
        conditions, replicates, args.steps, args.dt, args.temperature_k, raw_csv, out_root / "runs",
    )
    aggregate_results(rows, summary_csv, summary_md, note)
    if failed:
        print(f"\n{len(failed)} cell(s) failed and are NOT in the results above: {failed}")


if __name__ == "__main__":
    main()
