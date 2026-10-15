#!/usr/bin/env python
"""Plot matched absolute/momentum WaterBox NVT -> NVE results, without MD.

Use --watch alongside the simulation process to refresh figures as runs finish.
This only reads study inputs/results and writes figures under OUT/figures. It
does not acquire the MD lock or regenerate the runner's summary CSVs. Failed
or partial traces are retained; scalar physical comparisons require both
paired runs to complete. All means describe repeats within a training seed,
not independently trained models or confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

from run_waterbox_equilibration_study import CONDITIONS, RULES, TERMINAL, tasks


COLORS = {CONDITIONS[0]: "#2563a8", CONDITIONS[1]: "#bc4a24"}
LABELS = {CONDITIONS[0]: "Absolute", CONDITIONS[1]: "Momentum"}


def read_rows(path):
    """Live result files may have one partially written final row."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [r for r in csv.DictReader(handle) if None not in r and all(v is not None for v in r.values())]


def load_runs(plan, root):
    runs, signature = {}, []
    for seed, start, velocity, condition, directory in tasks(plan, root):
        manifests = sorted(directory.glob("attempt_*/run_manifest.json"))
        if not manifests:
            continue
        path = manifests[-1]
        # Atomic manifest replacement may briefly race with reading metadata.
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        signature.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
        run = {"manifest": manifest, "directory": path.parent}
        for name, filename in (("therm", "thermodynamic_history.csv"), ("structure", "structural_history.csv")):
            file = path.parent / filename
            run[name] = read_rows(file)
            if file.exists():
                signature.append((str(file), file.stat().st_mtime_ns, file.stat().st_size))
        runs[(seed, start["raw_dataset_index"], velocity, condition)] = run
    return runs, tuple(signature)


def paired_runs(runs, seed, complete_only=False):
    pairs = []
    keys = sorted({key[:3] for key in runs if key[0] == seed})
    for key in keys:
        a, b = (runs.get((*key, c)) for c in CONDITIONS)
        if not a or not b:
            continue
        ma, mb = a["manifest"], b["manifest"]
        if (ma.get("initial_state_sha256") and mb.get("initial_state_sha256")
                and ma["initial_state_sha256"] != mb["initial_state_sha256"]):
            raise ValueError(f"Mismatched paired initial states: {key}")
        statuses = [ma["status"], mb["status"]]
        if complete_only and statuses != ["complete", "complete"]:
            continue
        if not complete_only and not all(s in TERMINAL for s in statuses):
            continue
        pairs.append((a, b))
    return pairs


def series(run, metric):
    """Reference each energy to its own run/phase; never compare offsets."""
    definitions = {
        "temperature": ("therm", "temperature_k", None, False, 1.),
        "potential": ("therm", "epot_ev", None, True, 1.),
        "nve_drift": ("therm", "etot_ev", "nve", True, 1000. / 192.),
        "angle": ("structure", "hoh_angle_min_deg", None, False, 1.),
        "oh_max": ("structure", "oh_max_a", None, False, 1.),
        "oo_min": ("structure", "intermolecular_oo_min_a", None, False, 1.),
        "hh_min": ("structure", "intramolecular_hh_min_a", None, False, 1.),
        "oh_inter": ("structure", "intermolecular_oh_min_a", None, False, 1.),
        "nve_kinetic": ("therm", "ekin_ev", "nve", True, 1.),
    }
    source, column, phase, relative, factor = definitions[metric]
    x, y = [], []
    for row in run[source]:
        if phase is not None and row.get("phase") != phase:
            continue
        try:
            t, value = float(row["time_fs"]) / 1000., float(row[column])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(t) and math.isfinite(value):
            x.append(t); y.append(value)
    x, y = np.asarray(x), np.asarray(y)
    if len(y) and relative:
        y = y - y[0]
    return x, y * factor


def downsample_extrema(x, y, maximum=1800):
    """Reduce figure size while preserving both extremes of each time bin."""
    if len(x) <= maximum:
        return x, y
    selected = {0, len(x) - 1}
    for indices in np.array_split(np.arange(len(x)), max(1, maximum // 2 - 1)):
        selected.add(int(indices[np.argmin(y[indices])]))
        selected.add(int(indices[np.argmax(y[indices])]))
    indices = sorted(selected)
    return x[indices], y[indices]


def common_mean(runs, metric):
    """Intersect exact time grids; do not extrapolate terminated trajectories."""
    curves = [series(r, metric) for r in runs]
    curves = [(x, y) for x, y in curves if len(x)]
    if not curves:
        return np.array([]), np.array([])
    # Integer femtosecond-like keys preserve the configured sub-fs resolution.
    mapping = [{round(float(t), 10): float(v) for t, v in zip(x, y)} for x, y in curves]
    shared = set(mapping[0]).intersection(*(set(m) for m in mapping[1:]))
    grid = sorted(shared)
    return np.array(grid), np.array([np.mean([m[t] for m in mapping]) for t in grid])


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = directory / f"{name}.{suffix}"
        temp = directory / f".{name}.{suffix}.tmp"
        fig.savefig(temp, format=suffix, dpi=160, facecolor="white")
        temp.replace(path)


def seed_dashboard(plan, runs, seed, directory, plt):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    expected = plan["settings"]["configs_per_seed"] * len(plan["settings"]["velocity_seeds"])
    complete = paired_runs(runs, seed, complete_only=True)
    terminal = paired_runs(runs, seed)
    nvt = plan["settings"]["nvt_ps"]
    end = nvt + plan["settings"]["nve_ps"]
    rules = plan.get("rules", RULES)
    panels = (
        ("temperature", "Temperature (K)", plan["settings"]["temperature_k"]),
        ("potential", "Potential-energy change\nfrom initial frame (eV)", None),
        ("nve_drift", "NVE total-energy change\nfrom handoff (meV/atom)", 0.),
        ("angle", "Minimum H–O–H angle (degrees)", rules["minimum_hoh_angle_deg"]),
        ("oh_max", "Maximum assigned O–H (Å)", rules["maximum_assigned_oh_a"]),
        ("oo_min", "Minimum intermolecular O–O (Å)", rules["minimum_intermolecular_oo_a"]),
        ("hh_min", "Minimum intramolecular H–H (Å)", rules["minimum_intramolecular_hh_a"]),
        ("oh_inter", "Minimum intermolecular O–H (Å)", rules["minimum_intermolecular_oh_a"]),
        ("nve_kinetic", "NVE kinetic-energy change\nfrom handoff (eV)", 0.),
    )
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle(f"Training seed {seed}: absolute vs momentum\n"
                 f"Finished matched pairs: {len(terminal)}/{expected}; both complete: {len(complete)}/{expected}", fontsize=16)
    for ax, (metric, label, threshold) in zip(axes.flat, panels):
        has_data = False
        for condition in CONDITIONS:
            individuals = [r for key, r in runs.items() if key[0] == seed and key[3] == condition]
            for run in individuals:
                x, y = series(run, metric)
                if not len(x):
                    continue
                has_data = True
                xplot, yplot = downsample_extrema(x, y)
                status = run["manifest"]["status"]
                ax.plot(xplot, yplot, color=COLORS[condition], alpha=.28, linewidth=.7,
                        linestyle="-" if status == "complete" else "--", rasterized=True)
                if status != "complete":
                    ax.plot(x[-1], y[-1], "x", color=COLORS[condition], markersize=5)
            member = 0 if condition == CONDITIONS[0] else 1
            x, y = common_mean([p[member] for p in complete], metric)
            if len(x):
                ax.plot(*downsample_extrema(x, y), color=COLORS[condition], linewidth=2.)
        if threshold is not None:
            ax.axhline(threshold, color="#555555", linestyle=":", linewidth=1.)
        if not metric.startswith("nve_"):
            ax.axvspan(0, nvt, color="#e6ebef", alpha=.7, zorder=-2)
        ax.axvline(nvt, color="#777777", linestyle="--", linewidth=.8)
        ax.set(xlabel="Time from start (ps)", ylabel=label, xlim=(0, end))
        ax.grid(alpha=.15)
        if not has_data:
            ax.text(.5, .5, "Awaiting data", transform=ax.transAxes, ha="center", color="#777777")
    handles = [Line2D([], [], color=COLORS[c], lw=2, label=LABELS[c]) for c in CONDITIONS]
    handles.extend([Patch(facecolor="#e6ebef", label="NVT preparation"),
                    Line2D([], [], color="#555555", linestyle=":", label="Target / diagnostic threshold")])
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, .045))
    fig.text(.5, .012, "Thin traces: individual runs; dashed / ×: incomplete or aborted. Thick traces: means of both-complete matched pairs only.\n"
             "Means are within-seed descriptions, not confidence intervals; failed runs remain in the status/outcome figure. Thresholds are operational diagnostics.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .92))
    save_figure(fig, directory, f"seed{seed}_trajectories")
    plt.close(fig)


def metric_values(pairs, metric, duration_ps):
    values = [[], []]
    for pair in pairs:
        for i, run in enumerate(pair):
            m = run["manifest"]
            if metric == "survival":
                value = 100. * m["structural_survival"]
            elif metric == "time_without_flags":
                event = m.get("first_structural_event")
                value = (float(event["time_fs"]) / 1000. if event else
                         duration_ps if m["status"] == "complete" else float(m.get("last_time_fs", 0.)) / 1000.)
                value = min(duration_ps, value)
            else:
                factor = 100. if metric == "nve_flagged_frame_fraction" else 1000. / 192. if metric == "nve_maximum_energy_deviation_ev" else 1.
                raw = m.get(metric)
                value = None if raw is None else float(raw) * factor
            values[i].append(value)
    # Remove whole pairs when either side lacks a metric; never average unpaired cohorts.
    valid = [(a, b) for a, b in zip(*values) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    return ([p[0] for p in valid], [p[1] for p in valid])


def overview(plan, runs, directory, plt):
    from matplotlib.lines import Line2D
    seeds = plan["settings"]["seeds"]
    total_ps = plan["settings"]["nvt_ps"] + plan["settings"]["nve_ps"]
    expected = plan["settings"]["configs_per_seed"] * len(plan["settings"]["velocity_seeds"])
    panels = (
        ("survival", "Finished without\nstructural flags (%)", False),
        ("time_without_flags", f"Time to first flag / abort\n(capped at {total_ps:g} ps)", False),
        ("nve_tail_temperature_k", "NVE final-200-fs\nmean temperature (K)", True),
        ("nve_flagged_frame_fraction", "NVE frames with any\nstructural flag (%)", True),
        ("nve_maximum_energy_deviation_ev", "Maximum NVE |energy change|\n(meV/atom)", True),
        ("nve_intermolecular_oo_min_a", "Mean trajectory minimum\nO–O distance (Å)", True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Matched absolute–momentum outcomes within each training seed", fontsize=16)
    for ax, (metric, label, complete_only) in zip(axes.flat, panels):
        ticklabels = []
        for j, seed in enumerate(seeds):
            pairs = paired_runs(runs, seed, complete_only)
            a, b = metric_values(pairs, metric, total_ps)
            ticklabels.append(f"{seed}\n{len(a)}/{expected}")
            for va, vb in zip(a, b):
                ax.plot([j - .16, j + .16], [va, vb], color="#b5bdc4", alpha=.5, linewidth=.8, zorder=1)
            for offset, values, cond in ((-.16, a, CONDITIONS[0]), (.16, b, CONDITIONS[1])):
                if values:
                    ax.scatter(np.full(len(values), j + offset), values, color=COLORS[cond], s=12, alpha=.4, zorder=2)
                    ax.scatter(j + offset, np.mean(values), color=COLORS[cond], s=55, marker="D", zorder=3)
        ax.set(xticks=range(len(seeds)), xticklabels=ticklabels, ylabel=label,
               xlabel="Training seed / available matched pairs")
        ax.set_title("Both-complete pairs" if complete_only else "Terminal pairs, including numerical aborts", fontsize=10)
        if metric in ("survival", "nve_flagged_frame_fraction"):
            ax.set_ylim(-5, 105)
        elif metric == "time_without_flags":
            ax.set_ylim(-.03 * total_ps, 1.05 * total_ps)
        elif metric == "nve_tail_temperature_k":
            ax.axhline(plan["settings"]["temperature_k"], color="#555", linestyle=":", linewidth=1.)
        ax.grid(axis="y", alpha=.2)
    fig.legend(handles=[Line2D([], [], marker="D", color=COLORS[c], linestyle="", label=LABELS[c]) for c in CONDITIONS],
               loc="lower center", bbox_to_anchor=(.5, .045), ncol=2)
    fig.text(.5, .012, "Small points / connectors: matched configuration–velocity pairs. Diamonds: within-seed means. Missing comparisons stay blank.\n"
             "Plots remain provisional until the planned matrix is finished. Completing a run or passing thresholds does not establish equilibrium or physical accuracy.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .94))
    save_figure(fig, directory, "paired_seed_outcomes")
    plt.close(fig)


def static_errors(path, directory, seeds, plt):
    """Original matched-test static errors, separate from dynamical outcomes."""
    rows = read_rows(path)
    by_key = {(int(r["seed"]), r["condition"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, label in zip(axes, ("energy_mae", "force_mae"),
                                 ("Energy MAE (eV per 192-atom box)", "Force MAE (eV/Å)")):
        for j, seed in enumerate(seeds):
            members = [by_key.get((seed, c)) for c in CONDITIONS]
            if not all(members):
                continue
            values = [float(r[metric]) for r in members]
            if not all(math.isfinite(v) for v in values):
                continue
            ax.plot([j - .13, j + .13], values, color="#b5bdc4", linewidth=1.)
            for offset, value, condition in zip((-.13, .13), values, CONDITIONS):
                ax.scatter(j + offset, value, color=COLORS[condition], s=50,
                           label=LABELS[condition] if j == 0 else None)
        ax.set(xticks=range(len(seeds)), xticklabels=seeds, xlabel="Training seed", ylabel=label)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, .06))
    fig.suptitle("Existing ext70 checkpoints: matched-test static prediction errors", fontsize=14)
    fig.text(.5, .015, "One checkpoint per condition and training seed. These errors are separate from equilibrated trajectory performance.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .92))
    save_figure(fig, directory, "static_test_errors")
    plt.close(fig)


def render(plan, runs, root, static_results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = root / "figures"
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for seed in plan["settings"]["seeds"]:
            seed_dashboard(plan, runs, seed, directory, plt)
        overview(plan, runs, directory, plt)
        if static_results is not None and static_results.exists():
            static_errors(static_results, directory, plan["settings"]["seeds"], plt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/waterbox_equilibration_study_ext70_v1"))
    parser.add_argument("--watch", action="store_true", help="Refresh as run manifests change; exit when every planned run is terminal")
    parser.add_argument("--interval-seconds", type=float, default=60.)
    parser.add_argument("--static-results", type=Path,
                        default=Path("results/waterbox_study_zbl_bonded_ext70/raw_results.csv"))
    parser.add_argument("--static-only", action="store_true", help="Plot existing six-seed static errors before MD results exist")
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds < 5:
        parser.error("--interval-seconds must be finite and at least five")
    if args.static_only:
        if not args.static_results.exists():
            parser.error(f"Missing {args.static_results}")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        static_errors(args.static_results, args.out / "figures", list(range(6)), plt)
        return
    previous = None
    while True:
        path = args.out / "study_plan.json"
        if not path.exists():
            if not args.watch:
                parser.error(f"Missing {path}; prepare the study first")
            print("Waiting for study_plan.json...", flush=True)
        else:
            try:
                plan = json.loads(path.read_text())
                runs, signature = load_runs(plan, args.out)
                signature += ((str(path), path.stat().st_mtime_ns),)
                if args.static_results.exists():
                    signature += ((str(args.static_results), args.static_results.stat().st_mtime_ns),)
                if signature != previous:
                    render(plan, runs, args.out, args.static_results)
                    previous = signature
                    print(f"Updated PNG/PDF figures: {args.out / 'figures'}", flush=True)
                expected = len(list(tasks(plan, args.out)))
                if len(runs) == expected and all(r["manifest"]["status"] in TERMINAL for r in runs.values()):
                    return
            except (OSError, ValueError, KeyError) as exc:
                if not args.watch:
                    raise
                warnings.warn(f"Plot refresh deferred: {exc}")
        if not args.watch:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
