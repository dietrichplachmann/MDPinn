#!/usr/bin/env python
"""Combine per-rollout energy histories into a presentation-ready CSV.

The script discovers the seed-0 results directory and its ``_seed<N>``
siblings, then creates one row for every velocity- or configuration-axis
rollout.  It uses only the Python standard library so it can also be run on
the local analysis-only checkout.

Metric definitions
------------------
``plateau_temperature_k``
    Mean temperature over the last 30% of logged frames, matching
    ``rollout_waterbox_ase.py``.
``potential_drop_100fs_ev``
    E_pot(0) - E_pot(100 fs), so a drop is positive.
``kinetic_rise_100fs_ev`` and ``total_energy_change_100fs_ev``
    E(100 fs) - E(0), retaining the physical sign.
``predicted_temperature_rise_k``
    Potential drop converted completely to kinetic energy, calculated as
    ``potential_drop * initial_temperature / initial_kinetic_energy``.  This
    uses the trajectory's own kinetic-energy/temperature conversion and
    therefore preserves the degrees-of-freedom convention used by ASE.
``time_to_temperature_threshold_fs``
    First threshold crossing, linearly interpolated between logged frames.
    Blank if the threshold is never reached.
``energy_drift_mev_per_atom``
    Signed endpoint total-energy change divided by the atom count.
``completed_1ps``
    1 when the last logged time is at least 1000 fs, otherwise 0.

Example
-------
python src/postprocess_rollouts.py \
  --results-prefix results/waterbox_rollout_study_zbl_bonded_ext70 \
  --expected-seeds 0,1,2,3,4,5 --require-complete
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path


OUTPUT_COLUMNS = [
    "train_seed",
    "condition",
    "replicate_axis",
    "velocity_seed",
    "test_config_index",
    "initial_temperature_k",
    "plateau_temperature_k",
    "temperature_rise_k",
    "potential_drop_100fs_ev",
    "kinetic_rise_100fs_ev",
    "total_energy_change_100fs_ev",
    "predicted_temperature_rise_k",
    "maximum_temperature_k",
    "time_to_temperature_threshold_fs",
    "energy_drift_mev_per_atom",
    "completed_1ps",
]

EXPECTED_CONDITIONS = ("water_absolute", "water_absolute+momentum")
VELOCITY_LABELS = tuple(f"vseed{i}" for i in range(5))
CONFIG_LABELS = tuple(f"cfg{i}" for i in range(1, 6))


def _read_history(path: Path) -> list[dict[str, float]]:
    required = {"time_fs", "epot_ev", "ekin_ev", "etot_ev", "temperature_k"}
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = {key: float(raw[key]) for key in required}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric history value") from exc
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"{path}:{line_number}: non-finite history value")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    rows.sort(key=lambda row: row["time_fs"])
    return rows


def _interpolate(rows: list[dict[str, float]], time_fs: float, column: str) -> float | None:
    if time_fs < rows[0]["time_fs"] or time_fs > rows[-1]["time_fs"]:
        return None
    for left, right in zip(rows, rows[1:]):
        if math.isclose(time_fs, left["time_fs"], rel_tol=0.0, abs_tol=1e-12):
            return left[column]
        if left["time_fs"] <= time_fs <= right["time_fs"]:
            width = right["time_fs"] - left["time_fs"]
            if width == 0:
                return right[column]
            fraction = (time_fs - left["time_fs"]) / width
            return left[column] + fraction * (right[column] - left[column])
    if math.isclose(time_fs, rows[-1]["time_fs"], rel_tol=0.0, abs_tol=1e-12):
        return rows[-1][column]
    return None


def _threshold_crossing(rows: list[dict[str, float]], threshold_k: float) -> float | None:
    if rows[0]["temperature_k"] >= threshold_k:
        return rows[0]["time_fs"]
    for left, right in zip(rows, rows[1:]):
        t0 = left["temperature_k"]
        t1 = right["temperature_k"]
        if t0 < threshold_k <= t1:
            if t1 == t0:
                return right["time_fs"]
            fraction = (threshold_k - t0) / (t1 - t0)
            return left["time_fs"] + fraction * (right["time_fs"] - left["time_fs"])
    return None


def _seed_from_root(root: Path, base_name: str) -> int | None:
    if root.name == base_name:
        return 0
    match = re.fullmatch(re.escape(base_name) + r"_seed(\d+)", root.name)
    return int(match.group(1)) if match else None


def _discover_roots(results_prefix: Path) -> dict[int, Path]:
    roots: dict[int, Path] = {}
    base_name = results_prefix.name
    candidates = [results_prefix]
    if results_prefix.parent.exists():
        candidates.extend(results_prefix.parent.glob(f"{base_name}_seed*"))
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        seed = _seed_from_root(candidate, base_name)
        if seed is None:
            continue
        if seed in roots:
            raise ValueError(f"duplicate results roots for train seed {seed}: {roots[seed]} and {candidate}")
        roots[seed] = candidate
    return dict(sorted(roots.items()))


def _rollout_metadata(label: str) -> tuple[str, int, int]:
    velocity_match = re.fullmatch(r"vseed(\d+)", label)
    if velocity_match:
        return "velocity", int(velocity_match.group(1)), 0
    config_match = re.fullmatch(r"cfg(\d+)", label)
    if config_match:
        return "config", 0, int(config_match.group(1))
    raise ValueError(f"unrecognized rollout label {label!r}; expected vseed<N> or cfg<N>")


def _fmt(value: float | int | None) -> str | int:
    if value is None:
        return ""
    if isinstance(value, int):
        return value
    return f"{value:.12g}"


def _summarize_history(
    rows: list[dict[str, float]], *, train_seed: int, condition: str, label: str,
    evaluation_time_fs: float, plateau_fraction: float, threshold_k: float,
    n_atoms: int, completion_time_fs: float,
) -> dict[str, str | int]:
    replicate_axis, velocity_seed, test_config_index = _rollout_metadata(label)
    initial = rows[0]
    tail_n = max(1, int(len(rows) * plateau_fraction))
    plateau_temperature = sum(row["temperature_k"] for row in rows[-tail_n:]) / tail_n

    epot_at_eval = _interpolate(rows, evaluation_time_fs, "epot_ev")
    ekin_at_eval = _interpolate(rows, evaluation_time_fs, "ekin_ev")
    etot_at_eval = _interpolate(rows, evaluation_time_fs, "etot_ev")
    if epot_at_eval is None or ekin_at_eval is None or etot_at_eval is None:
        potential_drop = kinetic_rise = total_change = predicted_temperature_rise = None
    else:
        potential_drop = initial["epot_ev"] - epot_at_eval
        kinetic_rise = ekin_at_eval - initial["ekin_ev"]
        total_change = etot_at_eval - initial["etot_ev"]
        predicted_temperature_rise = (
            potential_drop * initial["temperature_k"] / initial["ekin_ev"]
            if initial["ekin_ev"] != 0 else None
        )

    return {
        "train_seed": train_seed,
        "condition": condition,
        "replicate_axis": replicate_axis,
        "velocity_seed": velocity_seed,
        "test_config_index": test_config_index,
        "initial_temperature_k": _fmt(initial["temperature_k"]),
        "plateau_temperature_k": _fmt(plateau_temperature),
        "temperature_rise_k": _fmt(plateau_temperature - initial["temperature_k"]),
        "potential_drop_100fs_ev": _fmt(potential_drop),
        "kinetic_rise_100fs_ev": _fmt(kinetic_rise),
        "total_energy_change_100fs_ev": _fmt(total_change),
        "predicted_temperature_rise_k": _fmt(predicted_temperature_rise),
        "maximum_temperature_k": _fmt(max(row["temperature_k"] for row in rows)),
        "time_to_temperature_threshold_fs": _fmt(_threshold_crossing(rows, threshold_k)),
        "energy_drift_mev_per_atom": _fmt(
            (rows[-1]["etot_ev"] - initial["etot_ev"]) * 1000.0 / n_atoms
        ),
        "completed_1ps": int(rows[-1]["time_fs"] + 1e-9 >= completion_time_fs),
    }


def _expected_histories(root: Path) -> list[tuple[str, str, Path]]:
    expected: list[tuple[str, str, Path]] = []
    for condition in EXPECTED_CONDITIONS:
        for label in (*VELOCITY_LABELS, *CONFIG_LABELS):
            expected.append((condition, label, root / "runs" / condition / label / "energy_history.csv"))
    return expected


def _numeric(row: dict[str, str | int], column: str) -> float | None:
    value = row.get(column, "")
    if value in (None, ""):
        return None
    return float(value)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install it in the analysis environment "
            "or rerun with --no-plots to generate only the CSV."
        ) from exc
    return plt


def _save_figure(fig, plot_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    written = []
    for extension in formats:
        path = plot_dir / f"{stem}.{extension}"
        save_args = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            save_args["dpi"] = dpi
        fig.savefig(path, **save_args)
        written.append(path)
    return written


def _plot_rollout_temperature(plt, rows, plot_dir, formats, dpi):
    colors = {"water_absolute": "#376795", "water_absolute+momentum": "#d05a47"}
    labels = {"water_absolute": "Absolute", "water_absolute+momentum": "Absolute + momentum"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for axis_name, ax in zip(("velocity", "config"), axes):
        seeds = sorted({int(row["train_seed"]) for row in rows if row["replicate_axis"] == axis_name})
        summaries = {}
        for seed in seeds:
            for condition in EXPECTED_CONDITIONS:
                values = [
                    float(row["plateau_temperature_k"])
                    for row in rows
                    if int(row["train_seed"]) == seed
                    and row["condition"] == condition
                    and row["replicate_axis"] == axis_name
                ]
                if values:
                    summaries[(seed, condition)] = _mean_sd(values)
        for seed in seeds:
            pair = [summaries.get((seed, condition)) for condition in EXPECTED_CONDITIONS]
            if all(value is not None for value in pair):
                ax.plot([seed - 0.08, seed + 0.08], [pair[0][0], pair[1][0]], color="#a7a7a7", lw=1.0, zorder=1)
        for offset, condition in zip((-0.08, 0.08), EXPECTED_CONDITIONS):
            x_values, means, errors = [], [], []
            for seed in seeds:
                summary = summaries.get((seed, condition))
                if summary is not None:
                    x_values.append(seed + offset)
                    means.append(summary[0])
                    errors.append(summary[1])
            ax.errorbar(
                x_values, means, yerr=errors, fmt="o", ms=6, capsize=3, lw=1.2,
                color=colors[condition], label=labels[condition], zorder=2,
            )
        ax.set_title(f"{axis_name.capitalize()} replicates")
        ax.set_xlabel("Training seed")
        ax.set_xticks(seeds)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Plateau temperature (K)\nmean ± SD across five replicates")
    axes[1].legend(frameon=False)
    fig.suptitle("Rollout heating by trained model seed")
    fig.tight_layout()
    paths = _save_figure(fig, plot_dir, "01_rollout_temperature_by_seed", formats, dpi)
    plt.close(fig)
    return paths


def _select_representative_pair(records):
    pairs = {}
    for record in records:
        row = record["summary"]
        key = (
            int(row["train_seed"]), row["replicate_axis"], int(row["velocity_seed"]),
            int(row["test_config_index"]),
        )
        pairs.setdefault(key, {})[row["condition"]] = record
    complete = []
    for key, pair in pairs.items():
        if all(condition in pair for condition in EXPECTED_CONDITIONS):
            pair_mean = statistics.mean(float(pair[c]["summary"]["plateau_temperature_k"]) for c in EXPECTED_CONDITIONS)
            complete.append((key, pair, pair_mean))
    if not complete:
        return None
    median_value = statistics.median(item[2] for item in complete)
    return min(complete, key=lambda item: abs(item[2] - median_value))


def _plot_energy_conversion(plt, records, plot_dir, formats, dpi, n_atoms):
    selected = _select_representative_pair(records)
    if selected is None:
        return []
    key, pair, _ = selected
    colors = {"water_absolute": "#376795", "water_absolute+momentum": "#d05a47"}
    labels = {"water_absolute": "Absolute", "water_absolute+momentum": "Absolute + momentum"}
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.2), sharex=True)
    for condition in EXPECTED_CONDITIONS:
        history = pair[condition]["history"]
        time = [row["time_fs"] for row in history]
        axes[0].plot(time, [row["temperature_k"] for row in history], color=colors[condition], label=labels[condition])
        epot0, ekin0, etot0 = history[0]["epot_ev"], history[0]["ekin_ev"], history[0]["etot_ev"]
        axes[1].plot(time, [row["epot_ev"] - epot0 for row in history], color=colors[condition], ls="-", label=f"{labels[condition]} ΔEpot")
        axes[1].plot(time, [row["ekin_ev"] - ekin0 for row in history], color=colors[condition], ls="--", label=f"{labels[condition]} ΔEkin")
        axes[2].plot(time, [(row["etot_ev"] - etot0) * 1000.0 / n_atoms for row in history], color=colors[condition], label=labels[condition])
    axes[0].set_ylabel("Temperature (K)")
    axes[1].set_ylabel("Energy change (eV)")
    axes[2].set_ylabel("Δ total energy\n(meV/atom)")
    axes[2].set_xlabel("Time (fs)")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    for ax in axes:
        ax.axhline(0, color="#777777", lw=0.7, alpha=0.6)
        ax.grid(alpha=0.2)
    seed, axis_name, velocity_seed, config_index = key
    fig.suptitle(
        "Representative matched rollout selected by median pair temperature\n"
        f"train seed {seed}; axis={axis_name}; velocity seed={velocity_seed}; config={config_index}"
    )
    fig.tight_layout()
    paths = _save_figure(fig, plot_dir, "02_energy_conversion_representative", formats, dpi)
    plt.close(fig)
    return paths


def _plot_potential_drop(plt, rows, plot_dir, formats, dpi):
    colors = {"water_absolute": "#376795", "water_absolute+momentum": "#d05a47"}
    markers = {"velocity": "o", "config": "s"}
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    for condition in EXPECTED_CONDITIONS:
        for axis_name in ("velocity", "config"):
            selected = [row for row in rows if row["condition"] == condition and row["replicate_axis"] == axis_name]
            x_values = [_numeric(row, "potential_drop_100fs_ev") for row in selected]
            y_values = [_numeric(row, "temperature_rise_k") for row in selected]
            valid = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]
            if valid:
                ax.scatter(
                    [item[0] for item in valid], [item[1] for item in valid],
                    c=colors[condition], marker=markers[axis_name], s=38, alpha=0.75,
                    label=f"{'Absolute + momentum' if condition.endswith('momentum') else 'Absolute'}; {axis_name}",
                )
    ax.set_xlabel("Potential-energy drop during first 100 fs (eV)")
    ax.set_ylabel("Plateau temperature rise (K)")
    ax.set_title("Early potential relaxation predicts subsequent heating")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    paths = _save_figure(fig, plot_dir, "03_potential_drop_vs_temperature_rise", formats, dpi)
    plt.close(fig)
    return paths


def _plot_threshold_time(plt, rows, plot_dir, formats, dpi, threshold_k, completion_time_fs):
    colors = {"water_absolute": "#376795", "water_absolute+momentum": "#d05a47"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for axis_name, ax in zip(("velocity", "config"), axes):
        for condition_index, condition in enumerate(EXPECTED_CONDITIONS):
            selected = [row for row in rows if row["condition"] == condition and row["replicate_axis"] == axis_name]
            offset = -0.10 if condition_index == 0 else 0.10
            crossed_x, crossed_y, censored_x = [], [], []
            for row in selected:
                x_value = int(row["train_seed"]) + offset
                crossing = _numeric(row, "time_to_temperature_threshold_fs")
                if crossing is None:
                    censored_x.append(x_value)
                else:
                    crossed_x.append(x_value)
                    crossed_y.append(crossing)
            ax.scatter(crossed_x, crossed_y, color=colors[condition], s=24, alpha=0.65)
            if censored_x:
                ax.scatter(censored_x, [completion_time_fs] * len(censored_x), facecolors="none", edgecolors=colors[condition], marker="^", s=48)
        seeds = sorted({int(row["train_seed"]) for row in selected}) if selected else []
        ax.set_xticks(seeds)
        ax.set_xlabel("Training seed")
        ax.set_title(f"{axis_name.capitalize()} replicates")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(f"Time to {threshold_k:g} K (fs)")
    fig.suptitle("Heating-onset distribution; open triangles did not cross by 1 ps")
    fig.tight_layout()
    paths = _save_figure(fig, plot_dir, "04_time_to_temperature_threshold", formats, dpi)
    plt.close(fig)
    return paths


def _derive_static_results_path(results_prefix: Path) -> Path:
    name = results_prefix.name.replace("waterbox_rollout_study", "waterbox_study", 1)
    return results_prefix.parent / name / "raw_results.csv"


def _plot_static_vs_rollout(plt, rows, static_path, plot_dir, formats, dpi):
    if not static_path.exists():
        print(f"Plot warning: static results not found at {static_path}; skipping static-vs-rollout plot.", file=sys.stderr)
        return []
    with static_path.open(newline="") as handle:
        static_rows = list(csv.DictReader(handle))
    rollout_means = {}
    for seed in sorted({int(row["train_seed"]) for row in rows}):
        for condition in EXPECTED_CONDITIONS:
            values = [
                float(row["temperature_rise_k"]) for row in rows
                if int(row["train_seed"]) == seed and row["condition"] == condition
            ]
            if values:
                rollout_means[(seed, condition)] = statistics.mean(values)
    colors = {"water_absolute": "#376795", "water_absolute+momentum": "#d05a47"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    metrics = (
        ("force_mae", "Test force MAE (eV/Å)"),
        ("mean_per_molecule_momentum_violation", "Mean per-molecule momentum violation"),
    )
    for ax, (column, xlabel) in zip(axes, metrics):
        for condition in EXPECTED_CONDITIONS:
            x_values, y_values = [], []
            for row in static_rows:
                if row.get("condition") != condition:
                    continue
                key = (int(row["seed"]), condition)
                if key in rollout_means and row.get(column) not in (None, ""):
                    x_values.append(float(row[column]))
                    y_values.append(rollout_means[key])
            ax.scatter(x_values, y_values, color=colors[condition], s=48, label="Absolute + momentum" if condition.endswith("momentum") else "Absolute")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean rollout temperature rise (K)")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Static test metrics versus dynamical heating")
    fig.tight_layout()
    paths = _save_figure(fig, plot_dir, "05_static_metrics_vs_rollout_heating", formats, dpi)
    plt.close(fig)
    return paths


def _generate_plots(rows, records, *, plot_dir, formats, dpi, threshold_k, completion_time_fs, static_path, n_atoms):
    plt = _load_matplotlib()
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []
    written += _plot_rollout_temperature(plt, rows, plot_dir, formats, dpi)
    written += _plot_energy_conversion(plt, records, plot_dir, formats, dpi, n_atoms)
    written += _plot_potential_drop(plt, rows, plot_dir, formats, dpi)
    written += _plot_threshold_time(plt, rows, plot_dir, formats, dpi, threshold_k, completion_time_fs)
    written += _plot_static_vs_rollout(plt, rows, static_path, plot_dir, formats, dpi)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-prefix", type=Path,
        default=Path("results/waterbox_rollout_study_zbl_bonded_ext70"),
        help="Seed-0 results directory; _seed<N> sibling directories are discovered automatically.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path. Defaults to <results-prefix>/all_seed_rollout_metrics.csv.",
    )
    parser.add_argument("--expected-seeds", default="0,1,2,3,4,5")
    parser.add_argument("--evaluation-time-fs", type=float, default=100.0)
    parser.add_argument("--plateau-fraction", type=float, default=0.30)
    parser.add_argument("--temperature-threshold-k", type=float, default=600.0)
    parser.add_argument("--n-atoms", type=int, default=192)
    parser.add_argument("--completion-time-fs", type=float, default=1000.0)
    parser.add_argument("--no-plots", action="store_true", help="Generate the combined CSV without figures.")
    parser.add_argument(
        "--plot-dir", type=Path, default=None,
        help="Figure directory. Defaults to <output-directory>/plots.",
    )
    parser.add_argument(
        "--plot-formats", default="png,pdf",
        help="Comma-separated figure formats understood by matplotlib (default: png,pdf).",
    )
    parser.add_argument("--plot-dpi", type=int, default=220)
    parser.add_argument(
        "--static-results", type=Path, default=None,
        help="Static raw_results.csv used for the static-vs-rollout figure. Derived from the results prefix by default.",
    )
    parser.add_argument(
        "--require-complete", action="store_true",
        help="Return a nonzero status if any expected seed or rollout history is missing/incomplete.",
    )
    args = parser.parse_args()

    if not 0 < args.plateau_fraction <= 1:
        parser.error("--plateau-fraction must be in (0, 1]")
    if args.n_atoms <= 0:
        parser.error("--n-atoms must be positive")

    expected_seeds = {int(item.strip()) for item in args.expected_seeds.split(",") if item.strip()}
    roots = _discover_roots(args.results_prefix)
    output = args.output or args.results_prefix / "all_seed_rollout_metrics.csv"
    problems: list[str] = []
    missing_seeds = sorted(expected_seeds.difference(roots))
    if missing_seeds:
        problems.append(f"missing training-seed result roots: {missing_seeds}")

    output_rows: list[dict[str, str | int]] = []
    history_records = []
    for train_seed, root in roots.items():
        if train_seed not in expected_seeds:
            continue
        for condition, label, history_path in _expected_histories(root):
            if not history_path.exists():
                problems.append(f"missing history: {history_path}")
                continue
            try:
                history = _read_history(history_path)
                row = _summarize_history(
                    history,
                    train_seed=train_seed,
                    condition=condition,
                    label=label,
                    evaluation_time_fs=args.evaluation_time_fs,
                    plateau_fraction=args.plateau_fraction,
                    threshold_k=args.temperature_threshold_k,
                    n_atoms=args.n_atoms,
                    completion_time_fs=args.completion_time_fs,
                )
            except ValueError as exc:
                problems.append(str(exc))
                continue
            if row["completed_1ps"] != 1:
                problems.append(
                    f"incomplete {args.completion_time_fs:g} fs rollout: "
                    f"seed={train_seed}, condition={condition}, label={label}, "
                    f"last_time={history[-1]['time_fs']:g} fs"
                )
            output_rows.append(row)
            history_records.append({"summary": row, "history": history, "path": history_path})

    axis_order = {"velocity": 0, "config": 1}
    condition_order = {name: index for index, name in enumerate(EXPECTED_CONDITIONS)}
    output_rows.sort(key=lambda row: (
        int(row["train_seed"]), condition_order.get(str(row["condition"]), 99),
        axis_order.get(str(row["replicate_axis"]), 99), int(row["velocity_seed"]),
        int(row["test_config_index"]),
    ))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rollout rows to {output}")
    print(
        f"Definitions: plateau=last {args.plateau_fraction:.0%}; "
        f"early energy window={args.evaluation_time_fs:g} fs; "
        f"temperature threshold={args.temperature_threshold_k:g} K; "
        f"completion={args.completion_time_fs:g} fs; atoms={args.n_atoms}."
    )
    if not args.no_plots:
        formats = [item.strip().lower().lstrip(".") for item in args.plot_formats.split(",") if item.strip()]
        if not formats:
            parser.error("--plot-formats must contain at least one format")
        plot_dir = args.plot_dir or output.parent / "plots"
        static_path = args.static_results or _derive_static_results_path(args.results_prefix)
        try:
            plot_paths = _generate_plots(
                output_rows,
                history_records,
                plot_dir=plot_dir,
                formats=formats,
                dpi=args.plot_dpi,
                threshold_k=args.temperature_threshold_k,
                completion_time_fs=args.completion_time_fs,
                static_path=static_path,
                n_atoms=args.n_atoms,
            )
        except RuntimeError as exc:
            print(f"Plot generation failed: {exc}", file=sys.stderr)
            return 3
        print(f"Wrote {len(plot_paths)} plot files to {plot_dir}")
    if problems:
        print("Completeness warnings:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        if args.require_complete:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
