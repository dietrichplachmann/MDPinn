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
from physics_losses import (
    momentum_symmetry_loss,
    nve_loss_from_trajectory,
    nve_loss_with_kinetic_energy,
    build_trajectory_batch,
    periodic_bc_loss_improved,
    get_atomic_masses,
)


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
        # `total_energy` is the recommended default for MD because it constrains
        # the physically relevant Hamiltonian drift rather than potential-only drift.
        self.nve_loss_mode = str(hparams.get("nve_loss_mode", "total_energy"))
        # Keep the training-side trajectory timestep aligned with rollout/eval.
        self.nve_dt_fs = float(hparams.get("nve_dt_fs", 0.5))

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
            loss_pbc = torch.tensor(0.0, device=self.device)

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
                # For short development sweeps, avoid setting this so high that
                # the physics term almost never fires.
                loss_nve = self._compute_nve_loss(self.train_batch_counter)

            if self.pbc_weight > 0 and hasattr(batch, "box") and batch.box is not None:
                # Apply periodicity regularization per graph when box vectors are available.
                unique_batches = torch.unique(batch.batch)
                for mol_idx in unique_batches:
                    mask = batch.batch == mol_idx
                    box_l = self._extract_box_lengths(batch.box, int(mol_idx.item()))
                    if box_l is None:
                        continue

                    local_batch = torch.zeros(int(mask.sum().item()), dtype=torch.long, device=self.device)
                    loss_pbc += periodic_bc_loss_improved(
                        self.model,
                        batch.pos[mask],
                        batch.z[mask],
                        box_l,
                        neg_dy[mask],
                        local_batch,
                    )
                loss_pbc = loss_pbc / len(unique_batches)

            self.train_batch_counter += 1

            self.log("train_loss_momentum", loss_momentum, on_step=False, on_epoch=True)
            self.log("train_loss_nve", loss_nve, on_step=False, on_epoch=True)
            self.log("train_loss_pbc", loss_pbc, on_step=False, on_epoch=True)
            self.log("train_nve_weight_effective", effective_nve_weight, on_step=False, on_epoch=True)

            physics_loss = (
                self.momentum_weight * loss_momentum
                + effective_nve_weight * loss_nve
                + self.pbc_weight * loss_pbc
            )
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
            if self.nve_loss_mode == "total_energy":
                masses = get_atomic_masses(traj_batch["Z"]).to(self.device)
                return nve_loss_with_kinetic_energy(
                    abs_energy_model,
                    traj_batch,
                    self.device,
                    masses=masses,
                    dt=self.nve_dt_fs,
                )

            if self.nve_loss_mode != "potential_only":
                raise ValueError(
                    f"Unsupported nve_loss_mode='{self.nve_loss_mode}'. "
                    "Expected 'total_energy' or 'potential_only'."
                )

            return nve_loss_from_trajectory(
                abs_energy_model,
                traj_batch,
                self.device,
                relative=self.nve_relative,
                eps=self.nve_relative_eps,
                dt=self.nve_dt_fs,
            )
        except Exception as exc:
            print(f"Warning: NVE loss failed: {exc}")
            return torch.tensor(0.0, device=self.device)

    def _extract_box_lengths(self, box, graph_idx):
        """Extract orthorhombic box lengths (Lx,Ly,Lz) for one graph if available."""
        if box is None:
            return None

        try:
            # Supported shapes commonly seen in batched data:
            # - (B, 3, 3): full box vectors per graph
            # - (3, 3): single box matrix
            # - (B, 3): lengths per graph
            # - (3,): single set of lengths
            if box.dim() == 3:
                box_g = box[graph_idx]
                return torch.linalg.norm(box_g, dim=1)
            if box.dim() == 2:
                if box.shape[0] == 3 and box.shape[1] == 3:
                    return torch.linalg.norm(box, dim=1)
                if box.shape[1] == 3:
                    return box[graph_idx]
            if box.dim() == 1 and box.numel() == 3:
                return box
        except Exception:
            return None

        return None


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
    nve_loss_mode="total_energy",
    nve_dt_fs=0.5,
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
    weight_decay=0.0,
    lr_patience=15,
    lr_min=1e-7,
    lr_factor=0.8,
    num_workers=4,
    seed=42,
    trainer_callbacks=None,
    trainer_kwargs=None,
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

    if dataset != "MD17":
        raise NotImplementedError(f"Dataset '{dataset}' is not implemented in train_physics.py (supported: MD17).")

    pl.seed_everything(seed, workers=True)

    full_dataset = MD17(root="./data", molecules=molecule)

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = GeometricDataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = GeometricDataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

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
        "momentum_weight": momentum_weight,
        "nve_weight": nve_weight,
        "pbc_weight": pbc_weight,
        "traj_length": traj_length,
        "nve_freq": nve_freq,
        "nve_warmup_epochs": nve_warmup_epochs,
        "nve_ramp_epochs": nve_ramp_epochs,
        "nve_relative": bool(nve_relative),
        "nve_relative_eps": float(nve_relative_eps),
        "nve_loss_mode": str(nve_loss_mode),
        "nve_dt_fs": float(nve_dt_fs),
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
        filename=checkpoint_name,
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_total_mse_loss", patience=30, mode="min")
    logger = TensorBoardLogger(save_dir=log_dir, name="physics_informed")
    trainer_callbacks = list(trainer_callbacks or [])
    trainer_kwargs = dict(trainer_kwargs or {})

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop, *trainer_callbacks],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1000.0,
        inference_mode=False,
        **trainer_kwargs,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("Testing best checkpoint...")
    test_results = trainer.test(model, test_loader, ckpt_path="best")
    val_metrics = {key: float(value) for key, value in trainer.callback_metrics.items() if hasattr(value, "item")}
    best_model_score = checkpoint_callback.best_model_score
    best_model_score = float(best_model_score.item()) if best_model_score is not None else None
    best_model_path = checkpoint_callback.best_model_path or str(Path(save_dir) / f"{checkpoint_name}.ckpt")

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
            "delta_learning": bool(delta_learning),
        },
        "validation_metrics": val_metrics,
        "best_model_path": best_model_path,
        "best_model_score": best_model_score,
        "test_results": test_results[0] if test_results else None,
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
            "nve_loss_mode": str(nve_loss_mode),
            "nve_dt_fs": float(nve_dt_fs),
        },
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
    parser.add_argument("--nve-loss-mode", type=str, default="total_energy")
    parser.add_argument("--nve-dt-fs", type=float, default=0.5)
    parser.add_argument("--delta-learning", action="store_true")
    parser.add_argument("--baseline-eps", type=float, default=0.01)
    parser.add_argument("--baseline-sigma", type=float, default=1.0)
    parser.add_argument("--baseline-cutoff", type=float, default=5.0)
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-rbf", type=int, default=64)
    parser.add_argument("--checkpoint-name", type=str, default="best_model")
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
        nve_loss_mode=args.nve_loss_mode,
        nve_dt_fs=args.nve_dt_fs,
        delta_learning=args.delta_learning,
        baseline_epsilon_eV=args.baseline_eps,
        baseline_sigma_A=args.baseline_sigma,
        baseline_cutoff_A=args.baseline_cutoff,
        embedding_dimension=args.embedding_dimension,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        checkpoint_name=args.checkpoint_name,
    )
