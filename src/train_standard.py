#!/usr/bin/env python
"""
Standard TorchMD-NET training with optional delta-learning.

Core idea:
- Absolute mode: network learns total energy/forces directly.
- Delta mode: network learns correction terms relative to an analytic baseline.

Chemist view:
- Absolute mode asks the NN to approximate the whole PES in one shot.
- Delta mode says: "start from a physically motivated baseline U_ref, then learn
  only the residual chemistry DeltaU that baseline misses."
- Forces always come from gradients of the learned scalar energy, so the model
  remains conservative by construction.
"""

import json
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader as GeometricDataLoader

from torchmdnet.datasets import MD17
from torchmdnet.module import LNNP

from baseline_potential import load_reference_energy_offset_eV, reference_energy_forces_batched
from data_splits import contiguous_split


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
        self.baseline_molecule = str(hparams.get("baseline_molecule", hparams.get("molecule", "aspirin")))
        self.baseline_eps = float(hparams.get("baseline_epsilon_eV", 0.01))
        self.baseline_sigma = float(hparams.get("baseline_sigma_A", 1.0))
        self.baseline_cutoff = float(hparams.get("baseline_cutoff_A", 5.0))
        self.baseline_energy_offset = float(
            hparams.get("baseline_energy_offset_eV", load_reference_energy_offset_eV(self.baseline_molecule))
        )

    def data_transform(self, batch):
        """Run base transform first, then residualize labels when enabled.

        Why this hook matters:
        - TorchMD-Net/LNNP computes losses against `batch.y` and `batch.neg_dy`.
        - By rewriting those labels here, we can train the same architecture
          either on absolute targets or residual targets without rewriting
          the downstream Lightning training loop.
        """
        batch = super().data_transform(batch)

        if not self.delta_learning:
            return batch

        # Compute baseline on current coordinates.
        # Physically: this is the part of the force field we choose to keep
        # analytic and interpretable.
        U_ref, F_ref = reference_energy_forces_batched(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch,
            molecule=self.baseline_molecule,
            box_l=batch.box if "box" in batch else None,
            epsilon_eV=self.baseline_eps,
            sigma_A=self.baseline_sigma,
            r_cut_A=self.baseline_cutoff,
            energy_offset_eV=self.baseline_energy_offset,
        )

        # Energies are per-graph.
        # y_true is reference total potential for each molecular graph.
        # In delta mode we train on residual energy:
        #   DeltaE_target = E_true - E_ref
        if hasattr(batch, "y") and batch.y is not None:
            y = batch.y
            if y.ndim == 1:
                y = y.unsqueeze(1)
            batch.y = (y.squeeze(-1) - U_ref).unsqueeze(1)

        # Forces are per-atom.
        # Same idea for forces:
        #   DeltaF_target = F_true - F_ref
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
    embedding_dimension=256,
    num_layers=6,
    num_rbf=64,
    checkpoint_name="best_model",
    train_loss="mse_loss",
    train_loss_arg=None,
    energy_weight=0.05,
    force_weight=0.95,
    weight_decay=0.0,
    lr_patience=15,
    lr_min=1e-7,
    lr_factor=0.8,
    num_workers=4,
    seed=42,
    trainer_callbacks=None,
    trainer_kwargs=None,
):
    """Train standard model and optionally residualize targets for delta-learning.

    Data flow summary:
    1) Load MD17 snapshots (R, Z, E*, F*).
    2) Build train/val/test split.
    3) In delta mode, transform labels to (E* - E_ref, F* - F_ref).
    4) Fit NN on those targets.
    5) Save checkpoint containing the delta/baseline metadata so inference code
       can reconstruct absolute energies/forces later.
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Standard Training")
    print("=" * 70)
    print(f"dataset={dataset}, molecule={molecule}, model={model_type}")
    print(f"delta_learning={delta_learning}")

    if dataset != "MD17":
        raise NotImplementedError(f"Dataset '{dataset}' is not implemented in train_standard.py (supported: MD17).")

    pl.seed_everything(seed, workers=True)

    print("Loading dataset...")
    full_dataset = MD17(root="./data", molecules=molecule)
    print(f"Dataset loaded: {len(full_dataset)} samples")

    train_data, val_data, test_data = contiguous_split(full_dataset)

    train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = GeometricDataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = GeometricDataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    # Include delta settings in checkpoint hyperparameters for reproducible inference.
    # This is essential because evaluation/rollout must know whether to do:
    #   U_abs = U_model                      (absolute mode)
    #   U_abs = U_ref + U_model (DeltaU)     (delta mode)
    model_args = {
        "delta_learning": bool(delta_learning),
        "baseline_molecule": molecule,
        "baseline_epsilon_eV": float(baseline_epsilon_eV),
        "baseline_sigma_A": float(baseline_sigma_A),
        "baseline_cutoff_A": float(baseline_cutoff_A),
        "baseline_energy_offset_eV": float(load_reference_energy_offset_eV(molecule)),
        "model": model_type,
        "prior_model": None,
        "output_model": "Scalar",
        "load_model": None,
        "remove_ref_energy": False,
        "train_loss": train_loss,
        "train_loss_arg": train_loss_arg,
        "charge": False,
        "spin": False,
        "precision": 32,
        "cutoff_lower": 0.0,
        "cutoff_upper": 5.0,
        "embedding_dimension": int(embedding_dimension),
        "num_layers": int(num_layers),
        "num_rbf": int(num_rbf),
        "rbf_type": "expnorm",
        "trainable_rbf": False,
        "activation": "silu",
        "max_z": 100,
        "max_num_neighbors": 128,
        "derivative": True,
        "lr": lr,
        "lr_patience": lr_patience,
        "lr_min": lr_min,
        "lr_factor": lr_factor,
        "lr_warmup_steps": 0,
        "weight_decay": weight_decay,
        "y_weight": energy_weight,
        "neg_dy_weight": force_weight,
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
    # In delta mode we use DeltaLNNP so labels are converted on-the-fly.
    # In absolute mode we use vanilla LNNP.
    model = DeltaLNNP(model_args) if delta_learning else LNNP(model_args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_mse_loss",
        dirpath=save_dir,
        filename=checkpoint_name,
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    early_stop = EarlyStopping(monitor="val_total_mse_loss", patience=30, mode="min")
    logger = TensorBoardLogger(save_dir=log_dir, name="standard")
    trainer_callbacks = list(trainer_callbacks or [])
    trainer_kwargs = dict(trainer_kwargs or {})

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop, *trainer_callbacks],
        logger=logger,
        log_every_n_steps=10,
        inference_mode=False,
        **trainer_kwargs,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    fit_metrics = {key: float(value.item()) for key, value in trainer.callback_metrics.items() if hasattr(value, "item")}

    print("Testing best checkpoint...")
    test_results = trainer.test(model, test_loader, ckpt_path="best")
    test_callback_metrics = {
        key: float(value.item()) for key, value in trainer.callback_metrics.items() if hasattr(value, "item")
    }
    best_model_score = checkpoint_callback.best_model_score
    best_model_score = float(best_model_score.item()) if best_model_score is not None else None
    best_model_path = checkpoint_callback.best_model_path or str(Path(save_dir) / f"{checkpoint_name}.ckpt")
    val_metrics = dict(fit_metrics)
    val_metrics.update({f"post_test.{key}": value for key, value in test_callback_metrics.items()})
    if best_model_score is not None:
        val_metrics.setdefault("val_total_mse_loss", best_model_score)

    config = {
        "model_args": model_args,
        "training": {
            "dataset": dataset,
            "molecule": molecule,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "seed": seed,
            "weight_decay": weight_decay,
            "train_loss": train_loss,
            "train_loss_arg": train_loss_arg,
            "energy_weight": energy_weight,
            "force_weight": force_weight,
            "delta_learning": bool(delta_learning),
        },
        "validation_metrics": val_metrics,
        "best_model_path": best_model_path,
        "best_model_score": best_model_score,
        "test_results": test_results[0] if test_results else None,
    }

    with open(Path(save_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training complete. Model: {save_dir}/{checkpoint_name}.ckpt")
    return {
        "trainer": trainer,
        "model": model,
        "test_results": test_results,
        "best_model_path": best_model_path,
        "best_model_score": best_model_score,
        "validation_metrics": val_metrics,
        "config": config,
    }


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
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-rbf", type=int, default=64)
    parser.add_argument("--checkpoint-name", type=str, default="best_model")
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
        embedding_dimension=args.embedding_dimension,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        checkpoint_name=args.checkpoint_name,
    )
