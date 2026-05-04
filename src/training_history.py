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
        metrics = {"epoch": int(trainer.current_epoch)}
        for key, value in trainer.callback_metrics.items():
            scalar = self._to_float(value)
            if scalar is not None:
                metrics[key] = scalar
        self.rows.append(metrics)

    def on_fit_end(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"{self.run_name}_history.json"
        csv_path = self.output_dir / f"{self.run_name}_history.csv"

        with open(json_path, "w") as handle:
            json.dump(self.rows, handle, indent=2)

        headers = sorted({key for row in self.rows for key in row.keys()})
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        self._write_plot()

    def _write_plot(self):
        if not self.rows:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        epochs = [row["epoch"] for row in self.rows]
        total_keys = [
            key
            for key in (
                "train_total_mse_loss",
                "val_total_mse_loss",
                "train_total_with_physics",
                "val_rollout_score",
            )
            if any(key in row for row in self.rows)
        ]
        component_keys = [
            key
            for key in (
                "train_y_mse_loss",
                "val_y_mse_loss",
                "train_neg_dy_mse_loss",
                "val_neg_dy_mse_loss",
                "train_loss_momentum",
                "train_loss_nve",
                "train_loss_pbc",
                "train_loss_momentum_weighted",
                "train_loss_nve_weighted",
                "train_loss_pbc_weighted",
                "train_physics_to_supervised_ratio",
                "train_nve_to_supervised_ratio",
                "train_momentum_to_supervised_ratio",
                "val_rollout_median_mean_abs_drift_eV",
                "val_rollout_median_max_abs_drift_eV",
                "val_rollout_failure_rate",
                "val_force_mae",
                "val_energy_mae",
                "train_short_rollout_mean_abs_drift_eV",
                "train_short_rollout_max_abs_drift_eV",
            )
            if any(key in row for row in self.rows)
        ]

        nrows = 2 if component_keys else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        for key in total_keys:
            ys = [row.get(key) for row in self.rows]
            axes[0].plot(epochs, ys, linewidth=2.0, label=key)
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Loss Curves")
        axes[0].grid(True, alpha=0.3)
        if total_keys:
            axes[0].legend(loc="best")

        if component_keys:
            for key in component_keys:
                ys = [row.get(key) for row in self.rows]
                axes[1].plot(epochs, ys, linewidth=1.8, label=key)
            axes[1].set_ylabel("Loss")
            axes[1].set_title("Component Metrics")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="best")

        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{self.run_name}_history.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
