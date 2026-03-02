#!/usr/bin/env python
"""
Standard TorchMD-NET training with optional delta-learning.

Core idea:
- Absolute mode: network learns total energy/forces directly.
- Delta mode: network learns correction terms relative to an analytic baseline.
"""

import json
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader

from torchmdnet.datasets import MD17
from torchmdnet.module import LNNP

from baseline_potential import lj_energy_forces_batched


# PyTorch 2.7 checkpoint compatibility.
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


class DeltaLNNP(LNNP):
    """LNNP subclass that converts absolute labels into residual labels.

    For each training mini-batch in delta mode:
    - y_target = y_true - U_ref
    - F_target = F_true - F_ref

    The network therefore learns DeltaU and DeltaF.
    """

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.delta_learning = bool(hparams.get("delta_learning", False))
        self.baseline_eps = float(hparams.get("baseline_epsilon_eV", 0.01))
        self.baseline_sigma = float(hparams.get("baseline_sigma_A", 1.0))
        self.baseline_cutoff = float(hparams.get("baseline_cutoff_A", 5.0))

    def data_transform(self, batch):
        """Run base transform first, then residualize labels when enabled."""
        batch = super().data_transform(batch)

        if not self.delta_learning:
            return batch

        # Compute baseline on current coordinates.
        U_ref, F_ref = lj_energy_forces_batched(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch,
            epsilon_eV=self.baseline_eps,
            sigma_A=self.baseline_sigma,
            r_cut_A=self.baseline_cutoff,
        )

        # Energies are per-graph.
        if hasattr(batch, "y") and batch.y is not None:
            y = batch.y
            if y.ndim == 1:
                y = y.unsqueeze(1)
            batch.y = (y.squeeze(-1) - U_ref).unsqueeze(1)

        # Forces are per-atom.
        if hasattr(batch, "neg_dy") and batch.neg_dy is not None:
            batch.neg_dy = batch.neg_dy - F_ref

        return batch


def train_standard_model(
    dataset="MD17",
    molecule="aspirin",
    batch_size=32,
    num_epochs=100,
    lr=1e-4,
    model_type="tensornet",
    save_dir="checkpoints/standard",
    log_dir="logs/standard",
    delta_learning=False,
    baseline_epsilon_eV=0.01,
    baseline_sigma_A=1.0,
    baseline_cutoff_A=5.0,
):
    """Train standard model and optionally residualize targets for delta-learning."""

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Standard Training")
    print("=" * 70)
    print(f"dataset={dataset}, molecule={molecule}, model={model_type}")
    print(f"delta_learning={delta_learning}")

    print("Loading dataset...")
    full_dataset = MD17(root="./data", molecules=molecule)
    print(f"Dataset loaded: {len(full_dataset)} samples")

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = GeometricDataLoader(val_data, batch_size=batch_size, num_workers=4)
    test_loader = GeometricDataLoader(test_data, batch_size=batch_size, num_workers=4)

    # Include delta settings in checkpoint hyperparameters for reproducible inference.
    model_args = {
        "delta_learning": bool(delta_learning),
        "baseline_epsilon_eV": float(baseline_epsilon_eV),
        "baseline_sigma_A": float(baseline_sigma_A),
        "baseline_cutoff_A": float(baseline_cutoff_A),
        "model": model_type,
        "prior_model": None,
        "output_model": "Scalar",
        "load_model": None,
        "remove_ref_energy": False,
        "train_loss": "mse_loss",
        "train_loss_arg": None,
        "charge": False,
        "spin": False,
        "precision": 32,
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "embedding_dimension": 256,
        "num_layers": 6,
        "num_rbf": 64,
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "activation": "silu",
        "max_z": 100,
        "max_num_neighbors": 128,
        "derivative": True,
        "lr": lr,
        "lr_patience": 15,
        "lr_min": 1e-7,
        "lr_factor": 0.8,
        "lr_warmup_steps": 0,
        "weight_decay": 0.0,
        "y_weight": 0.05,
        "neg_dy_weight": 0.95,
        "ema_alpha_y": 1.0,
        "ema_alpha_neg_dy": 1.0,
        "atom_filter": -1,
        "reduce_op": "add",
        "equivariance_invariance_group": "O(3)",
        "box_vecs": None,
        "check_errors": True,
        "static_shapes": False,
        "vector_cutoff": False,
        "aggr": "add",
        "neighbor_embedding": True,
        "attn_activation": "silu",
        "num_heads": 8,
        "distance_influence": "both",
    }

    print("Creating model...")
    model = DeltaLNNP(model_args) if delta_learning else LNNP(model_args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_mse_loss",
        dirpath=save_dir,
        filename="best_model",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    early_stop = EarlyStopping(monitor="val_total_mse_loss", patience=30, mode="min")
    logger = TensorBoardLogger(save_dir=log_dir, name="standard")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        log_every_n_steps=10,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("Testing best checkpoint...")
    test_results = trainer.test(model, test_loader, ckpt_path="best")

    config = {
        "model_args": model_args,
        "training": {
            "dataset": dataset,
            "molecule": molecule,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "delta_learning": bool(delta_learning),
        },
        "test_results": test_results[0] if test_results else None,
    }

    with open(Path(save_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. Model: {save_dir}/best_model.ckpt")
    return trainer, model, test_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="aspirin")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="tensornet")
    parser.add_argument("--delta-learning", action="store_true")
    parser.add_argument("--baseline-eps", type=float, default=0.01)
    parser.add_argument("--baseline-sigma", type=float, default=1.0)
    parser.add_argument("--baseline-cutoff", type=float, default=5.0)
    args = parser.parse_args()

    train_standard_model(
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        delta_learning=args.delta_learning,
        baseline_epsilon_eV=args.baseline_eps,
        baseline_sigma_A=args.baseline_sigma,
        baseline_cutoff_A=args.baseline_cutoff,
    )
