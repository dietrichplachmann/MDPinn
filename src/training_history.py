from __future__ import annotations

import csv
import json
from pathlib import Path

import lightning.pytorch as pl


class MetricHistoryCallback(pl.Callback):
    """Persist per-epoch metrics for later plotting and inspection."""

    def __init__(self, output_dir: str | Path, run_name: str):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.run_name = str(run_name)
        self.rows: list[dict[str, float | int]] = []

    @staticmethod
    def _to_float(value):
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return

        metrics = {"epoch": int(trainer.current_epoch)}
        for key, value in trainer.callback_metrics.items():
            scalar = self._to_float(value)
            if scalar is not None:
                metrics[key] = scalar

        if len(metrics) > 1:
            self.rows.append(metrics)

    def on_fit_end(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{self.run_name}_history.json"
        csv_path = self.output_dir / f"{self.run_name}_history.csv"

        self.rows = self._non_empty_rows(self.rows)

        with open(json_path, "w") as handle:
            json.dump(self.rows, handle, indent=2)

        headers = sorted({key for row in self.rows for key in row.keys()})
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        self._write_plot()
        self._write_plot(exclude_epoch0=True)

    @staticmethod
    def _non_empty_rows(rows):
        return [row for row in rows if any(key != "epoch" for key in row.keys())]

    @staticmethod
    def _series(rows, key):
        xs = []
        ys = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            xs.append(row["epoch"])
            ys.append(value)
        return xs, ys

    def _write_plot(self, exclude_epoch0=False):
        rows = self._non_empty_rows(self.rows)
        if exclude_epoch0:
            rows = [row for row in rows if float(row["epoch"]) != 0.0]
        if not rows:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        total_keys = [
            key
            for key in (
                "train_total_mse_loss",
                "val_total_mse_loss",
                "train_total_with_physics",
                "val_rollout_score",
            )
            if any(key in row for row in rows)
        ]
        energy_keys = [
            key
            for key in (
                "train_y_mse_loss",
                "val_y_mse_loss",
                "val_energy_mae",
            )
            if any(key in row for row in rows)
        ]
        force_keys = [
            key
            for key in (
                "train_neg_dy_mse_loss",
                "val_neg_dy_mse_loss",
                "val_force_mae",
            )
            if any(key in row for row in rows)
        ]
        physics_keys = [
            key
            for key in (
                "train_loss_momentum",
                "train_loss_nve",
                "train_loss_pbc",
                "train_loss_momentum_weighted",
                "train_loss_nve_weighted",
                "train_loss_pbc_weighted",
                "train_physics_to_supervised_ratio",
                "train_nve_to_supervised_ratio",
                "train_momentum_to_supervised_ratio",
                "train_nve_mean_abs_drift_eV",
                "train_nve_max_abs_drift_eV",
                "train_nve_mean_abs_drift_per_atom_eV",
                "train_nve_max_abs_drift_per_atom_eV",
                "val_rollout_median_mean_abs_drift_eV",
                "val_rollout_median_max_abs_drift_eV",
                "val_rollout_failure_rate",
                "train_short_rollout_mean_abs_drift_eV",
                "train_short_rollout_max_abs_drift_eV",
            )
            if any(key in row for row in rows)
        ]
        groups = [
            ("Total Loss Curves", total_keys, "Loss"),
            ("Energy Metrics", energy_keys, "Loss / metric"),
            ("Force Metrics", force_keys, "Loss / metric"),
            ("Physics / Rollout Metrics", physics_keys, "Loss / metric"),
        ]
        groups = [(title, keys, ylabel) for title, keys, ylabel in groups if keys]

        if not groups:
            return

        nrows = len(groups)
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        title_suffix = " (excluding epoch 0)" if exclude_epoch0 else ""
        for ax, (title, keys, ylabel) in zip(axes, groups):
            for key in keys:
                xs, ys = self._series(rows, key)
                if ys:
                    ax.plot(xs, ys, linewidth=1.8, label=key)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title}{title_suffix}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        suffix = "_history_no_epoch0.png" if exclude_epoch0 else "_history.png"
        fig.savefig(self.output_dir / f"{self.run_name}{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)
