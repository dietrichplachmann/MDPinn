#!/usr/bin/env python
"""Run the paired all-seed WaterBox NVT or NPT property matrix.

Every condition uses the same test configuration and velocity seeds within a
training seed.  The data split seed equals the model training seed, so the
starting structure is held out from that checkpoint's training data.  Runs
with a complete manifest are skipped, making remote execution resumable.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from run_waterbox_ensemble import run_ensemble


CONDITIONS = ("water_absolute", "water_absolute+momentum")


def _parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _is_complete(out_dir: Path) -> bool:
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        return json.loads(manifest_path.read_text()).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", choices=("nvt", "npt"), required=True)
    parser.add_argument(
        "--checkpoint-root", default="checkpoints/waterbox_study_zbl_bonded_ext70"
    )
    parser.add_argument(
        "--results-root", default="results/waterbox_property_study_zbl_bonded_ext70"
    )
    parser.add_argument("--seeds", type=_parse_ints, default=_parse_ints("0,1,2,3,4,5"))
    parser.add_argument("--velocity-seeds", type=_parse_ints, default=_parse_ints("0,1"))
    parser.add_argument("--test-config-index", type=int, default=0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--equilibration-ps", type=float, default=20.0)
    parser.add_argument("--production-ps", type=float, default=100.0)
    parser.add_argument("--tdamp-fs", type=float, default=100.0)
    parser.add_argument("--pdamp-fs", type=float, default=1000.0)
    parser.add_argument("--stress-fd-epsilon", type=float, default=0.003)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Rerun cells even when their existing manifest says complete.")
    args = parser.parse_args()

    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    velocity_seeds = args.velocity_seeds[:1] if args.smoke_test else args.velocity_seeds
    conditions = CONDITIONS[:1] if args.smoke_test else CONDITIONS
    equilibration_ps = 0.01 if args.smoke_test else args.equilibration_ps
    production_ps = 0.01 if args.smoke_test else args.production_ps
    results_root = Path(
        f"{args.results_root}_smoke" if args.smoke_test else args.results_root
    )

    failures = []
    completed = 0
    skipped = 0
    for train_seed in seeds:
        for condition in conditions:
            checkpoint = Path(args.checkpoint_root) / condition / f"seed{train_seed}" / "best_model.ckpt"
            if not checkpoint.exists():
                failures.append(f"missing checkpoint: {checkpoint}")
                continue
            for velocity_seed in velocity_seeds:
                out_dir = (
                    results_root / args.ensemble / condition /
                    f"seed{train_seed}" / f"vseed{velocity_seed}"
                )
                if _is_complete(out_dir) and not args.force:
                    print(f"SKIP complete: {out_dir}")
                    skipped += 1
                    continue
                print(f"RUN: {out_dir}")
                try:
                    manifest = run_ensemble(
                        checkpoint=str(checkpoint), ensemble=args.ensemble, out=str(out_dir),
                        data_seed=train_seed, test_config_index=args.test_config_index,
                        velocity_seed=velocity_seed, temperature_k=args.temperature_k,
                        pressure_bar=args.pressure_bar, dt_fs=args.dt_fs,
                        equilibration_ps=equilibration_ps, production_ps=production_ps,
                        tdamp_fs=args.tdamp_fs, pdamp_fs=args.pdamp_fs,
                        stress_fd_epsilon=args.stress_fd_epsilon,
                    )
                    if manifest["status"] == "complete":
                        completed += 1
                    else:
                        failures.append(f"{out_dir}: {manifest['status']} ({manifest['abort_reason']})")
                except Exception as exc:
                    traceback.print_exc()
                    failures.append(f"{out_dir}: {type(exc).__name__}: {exc}")

    print(f"Completed={completed}; skipped={skipped}; failures={len(failures)}")
    for failure in failures:
        print(f"FAILED: {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
