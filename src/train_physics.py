#!/usr/bin/env python
"""
Physics-informed training with optional delta-learning.

This extends standard TorchMD-NET training by adding:
- momentum conservation regularization,
- optional NVE drift regularization on short trajectories.

Chemist view:
- Base supervised loss fits reference E and F (or residuals in delta mode).
- Physics losses are soft constraints that bias training toward trajectories
  with better dynamical behavior (less spurious drift / symmetry violation).
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
from physics_losses import momentum_symmetry_loss, nve_loss_from_trajectory, build_trajectory_batch


# PyTorch 2.7 checkpoint compatibility.
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, "weights_only": False})


class DeltaLNNP(LNNP):
    """Convert absolute labels to residual labels when delta-learning is enabled.

    This mirrors `train_standard.py` so both training modes use the same
    physical semantics for delta-learning.
    """

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.delta_learning = bool(hparams.get("delta_learning", False))
        self.baseline_eps = float(hparams.get("baseline_epsilon_eV", 0.01))
        self.baseline_sigma = float(hparams.get("baseline_sigma_A", 1.0))
        self.baseline_cutoff = float(hparams.get("baseline_cutoff_A", 5.0))

    def data_transform(self, batch):
        batch = super().data_transform(batch)
        if not self.delta_learning:
            return batch

        U_ref, F_ref = lj_energy_forces_batched(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch,
            epsilon_eV=self.baseline_eps,
            sigma_A=self.baseline_sigma,
            r_cut_A=self.baseline_cutoff,
        )

        if hasattr(batch, "y") and batch.y is not None:
            y = batch.y
            if y.ndim == 1:
                y = y.unsqueeze(1)
            batch.y = (y.squeeze(-1) - U_ref).unsqueeze(1)

        if hasattr(batch, "neg_dy") and batch.neg_dy is not None:
            batch.neg_dy = batch.neg_dy - F_ref

        return batch


class PhysicsInformedLNNP(DeltaLNNP):
    """LNNP with extra physics losses added inside `step` during training.

    Interpretation of added terms:
    - momentum loss: discourages net translational/rotational force imbalance.
    - NVE loss: discourages potential-energy drift along short trajectory segments.
    """

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        self.momentum_weight = float(hparams.get("momentum_weight", 0.01))
        self.nve_weight = float(hparams.get("nve_weight", 0.01))
        self.pbc_weight = float(hparams.get("pbc_weight", 0.0))

        self.traj_length = int(hparams.get("traj_length", 100))
        self.nve_freq = int(hparams.get("nve_freq", 50))
        self.nve_warmup_epochs = int(hparams.get("nve_warmup_epochs", 5))
        self.nve_ramp_epochs = int(hparams.get("nve_ramp_epochs", 20))
        self.nve_relative = bool(hparams.get("nve_relative", True))
        self.nve_relative_eps = float(hparams.get("nve_relative_eps", 1e-6))

        self.full_dataset = None
        self.train_batch_counter = 0

    def _effective_nve_weight(self):
        """Return epoch-scheduled NVE weight (warmup + linear ramp)."""
        base = self.nve_weight
        epoch = int(getattr(self, "current_epoch", 0))

        if epoch < self.nve_warmup_epochs:
            return 0.0

        if self.nve_ramp_epochs <= 0:
            return base

        ramp_pos = epoch - self.nve_warmup_epochs + 1
        scale = min(1.0, max(0.0, ramp_pos / self.nve_ramp_epochs))
        return base * scale

    def _predict_absolute_energy(self, z, pos, batch):
        """Return absolute potential energy for NVE loss.

        In absolute mode: U_abs = U_model
        In delta mode:    U_abs = U_ref + DeltaU_model
        """
        # `self.model` is the underlying TorchMD-Net energy model.
        out = self.model(z, pos, batch=batch)
        if isinstance(out, tuple):
            out = out[0]

        # Absolute-mode checkpoint already predicts total U(R).
        if not self.delta_learning:
            return out

        # Delta-mode checkpoint predicts DeltaU(R); reconstruct total U_hyb(R).
        u_ref, _ = lj_energy_forces_batched(
            z=z,
            pos=pos,
            batch=batch,
            epsilon_eV=self.baseline_eps,
            sigma_A=self.baseline_sigma,
            r_cut_A=self.baseline_cutoff,
        )
        return out.squeeze(-1) + u_ref

    def step(self, batch, loss_fn_list, stage):
        """Compute default LNNP loss, then add physics terms during train stage."""
        total_loss = super().step(batch, loss_fn_list, stage)

        if stage != "train":
            return total_loss

        try:
            # Recompute model force prediction for physics penalties.
            # We need force vectors explicitly because momentum symmetry is a
            # force-level constraint.
            batch.pos = batch.pos.clone().detach().requires_grad_(True)
            _, neg_dy = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
            )

            loss_momentum = torch.tensor(0.0, device=self.device)
            loss_nve = torch.tensor(0.0, device=self.device)

            if self.momentum_weight > 0:
                # Apply momentum symmetry per molecule, then average.
                unique_batches = torch.unique(batch.batch)
                for mol_idx in unique_batches:
                    mask = batch.batch == mol_idx
                    loss_momentum += momentum_symmetry_loss(batch.pos[mask], neg_dy[mask])
                loss_momentum = loss_momentum / len(unique_batches)

            effective_nve_weight = self._effective_nve_weight()
            if effective_nve_weight > 0 and self.train_batch_counter % self.nve_freq == 0:
                # NVE is expensive, so it is sampled every nve_freq steps.
                loss_nve = self._compute_nve_loss(self.train_batch_counter)

            self.train_batch_counter += 1

            self.log("train_loss_momentum", loss_momentum, on_step=False, on_epoch=True)
            self.log("train_loss_nve", loss_nve, on_step=False, on_epoch=True)
            self.log("train_nve_weight_effective", effective_nve_weight, on_step=False, on_epoch=True)

            physics_loss = self.momentum_weight * loss_momentum + effective_nve_weight * loss_nve
            total_loss = total_loss + physics_loss
            self.log("train_total_with_physics", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        except Exception as exc:
            print(f"Warning: physics loss computation failed: {exc}")

        return total_loss

    def _compute_nve_loss(self, batch_idx):
        """Compute trajectory drift penalty on absolute energy.

        Important: this always evaluates absolute energy drift.
        In delta mode that means baseline + learned residual, matching the
        deployed hybrid potential rather than just DeltaU alone.
        """
        if self.full_dataset is None:
            return torch.tensor(0.0, device=self.device)

        dataset_size = len(self.full_dataset)
        max_start = dataset_size - self.traj_length
        if max_start <= 0:
            return torch.tensor(0.0, device=self.device)

        start_idx = (batch_idx * 137) % max_start
        traj_batch = build_trajectory_batch(self.full_dataset, start_idx, self.traj_length, self.device)

        # Wrap absolute-energy predictor to match expected callable signature
        # expected by nve_loss_from_trajectory(...).
        def abs_energy_model(z, pos, batch):
            return self._predict_absolute_energy(z, pos, batch=batch)

        try:
            return nve_loss_from_trajectory(
                abs_energy_model,
                traj_batch,
                self.device,
                relative=self.nve_relative,
                eps=self.nve_relative_eps,
            )
        except Exception as exc:
            print(f"Warning: NVE loss failed: {exc}")
            return torch.tensor(0.0, device=self.device)


def train_physics_informed_model(
    dataset="MD17",
    molecule="aspirin",
    batch_size=32,
    num_epochs=100,
    lr=1e-4,
    model_type="tensornet",
    save_dir="checkpoints/physics_informed",
    log_dir="logs/physics_informed",
    force_weight=0.95,
    energy_weight=0.05,
    momentum_weight=0.01,
    nve_weight=0.01,
    pbc_weight=0.0,
    traj_length=100,
    nve_freq=50,
    nve_warmup_epochs=5,
    nve_ramp_epochs=20,
    nve_relative=True,
    nve_relative_eps=1e-6,
    delta_learning=False,
    baseline_epsilon_eV=0.01,
    baseline_sigma_A=1.0,
    baseline_cutoff_A=5.0,
):
    """Train physics-informed model with optional delta-learning targets.

    Practical meaning:
    - Standard terms fit reference data.
    - Added physics terms reduce unphysical behavior between data points.
    - Delta mode keeps the same constraints, but applied to the hybrid model.
    """

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Physics-Informed Training")
    print("=" * 70)
    print(f"dataset={dataset}, molecule={molecule}, model={model_type}")
    print(f"delta_learning={delta_learning}")

    full_dataset = MD17(root="./data", molecules=molecule)

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
        "y_weight": energy_weight,
        "neg_dy_weight": force_weight,
        "ema_alpha_y": 1.0,
        "ema_alpha_neg_dy": 1.0,
        "momentum_weight": momentum_weight,
        "nve_weight": nve_weight,
        "pbc_weight": pbc_weight,
        "traj_length": traj_length,
        "nve_freq": nve_freq,
        "nve_warmup_epochs": nve_warmup_epochs,
        "nve_ramp_epochs": nve_ramp_epochs,
        "nve_relative": bool(nve_relative),
        "nve_relative_eps": float(nve_relative_eps),
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

    # PhysicsInformedLNNP includes both delta label handling and physics losses.
    model = PhysicsInformedLNNP(model_args)
    model.full_dataset = full_dataset

    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_mse_loss",
        dirpath=save_dir,
        filename="best_model",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_total_mse_loss", patience=30, mode="min")
    logger = TensorBoardLogger(save_dir=log_dir, name="physics_informed")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1000.0,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Kept for interface compatibility with standard trainer.
    _ = test_loader
    test_results = None

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
        "physics_weights": {
            "momentum": momentum_weight,
            "nve": nve_weight,
            "pbc": pbc_weight,
        },
        "physics_schedule": {
            "nve_freq": nve_freq,
            "nve_warmup_epochs": nve_warmup_epochs,
            "nve_ramp_epochs": nve_ramp_epochs,
            "nve_relative": bool(nve_relative),
            "nve_relative_eps": float(nve_relative_eps),
        },
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
    parser.add_argument("--momentum-weight", type=float, default=0.01)
    parser.add_argument("--nve-weight", type=float, default=0.01)
    parser.add_argument("--traj-length", type=int, default=100)
    parser.add_argument("--nve-freq", type=int, default=50)
    parser.add_argument("--nve-warmup-epochs", type=int, default=5)
    parser.add_argument("--nve-ramp-epochs", type=int, default=20)
    parser.add_argument("--nve-relative", dest="nve_relative", action="store_true")
    parser.add_argument("--nve-absolute", dest="nve_relative", action="store_false")
    parser.set_defaults(nve_relative=True)
    parser.add_argument("--nve-relative-eps", type=float, default=1e-6)
    parser.add_argument("--delta-learning", action="store_true")
    parser.add_argument("--baseline-eps", type=float, default=0.01)
    parser.add_argument("--baseline-sigma", type=float, default=1.0)
    parser.add_argument("--baseline-cutoff", type=float, default=5.0)
    args = parser.parse_args()

    train_physics_informed_model(
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        momentum_weight=args.momentum_weight,
        nve_weight=args.nve_weight,
        traj_length=args.traj_length,
        nve_freq=args.nve_freq,
        nve_warmup_epochs=args.nve_warmup_epochs,
        nve_ramp_epochs=args.nve_ramp_epochs,
        nve_relative=args.nve_relative,
        nve_relative_eps=args.nve_relative_eps,
        delta_learning=args.delta_learning,
        baseline_epsilon_eV=args.baseline_eps,
        baseline_sigma_A=args.baseline_sigma,
        baseline_cutoff_A=args.baseline_cutoff,
    )
