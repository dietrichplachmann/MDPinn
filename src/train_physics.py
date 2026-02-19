#!/usr/bin/env python
"""
Physics-Informed TorchMD-NET Training with NVE Loss - COMPLETE
"""

import torch

# CRITICAL: Fix PyTorch 2.7 torch.load compatibility
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

import lightning.pytorch as pl  # Changed from pytorch_lightning
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
import json
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader

from torchmdnet.datasets import MD17
from torchmdnet.models.model import create_model
from torchmdnet.module import LNNP  # CORRECT LOCATION

# Import physics losses
try:
    from physics_losses_COMPLETE import (
        momentum_symmetry_loss,
        nve_loss_from_trajectory,
        build_trajectory_batch,
    )

    PHYSICS_LOSSES_AVAILABLE = True
    print("✓ Physics losses loaded")
except ImportError:
    print("✗ Warning: physics_losses_COMPLETE.py not found")
    PHYSICS_LOSSES_AVAILABLE = False


class PhysicsInformedLNNP(LNNP):
    """Extended LNNP with physics losses including NVE"""

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Physics loss weights
        self.momentum_weight = hparams.get('momentum_weight', 0.01)
        self.nve_weight = hparams.get('nve_weight', 1.0)
        self.pbc_weight = hparams.get('pbc_weight', 0.0)  # Disabled for gas phase

        # NVE parameters
        self.traj_length = hparams.get('traj_length', 100)
        self.nve_freq = hparams.get('nve_freq', 10)  # Compute every N batches

        # Dataset reference (will be set externally)
        self.full_dataset = None

        print(f"Physics loss weights:")
        print(f"  Momentum: {self.momentum_weight}")
        print(f"  NVE:      {self.nve_weight} (every {self.nve_freq} batches)")
        print(f"  PBC:      {self.pbc_weight} (disabled for gas phase)")

    def training_step(self, batch, batch_idx):
        """Training with physics losses"""

        # Standard forward pass
        pred, deriv_pred = self(batch.z, batch.pos, batch=batch.batch)
        loss_y, loss_dy = self.loss_fn(pred, batch.y, deriv_pred, batch.neg_dy)

        # Initialize physics losses
        loss_momentum = torch.tensor(0.0, device=self.device)
        loss_nve = torch.tensor(0.0, device=self.device)

        if PHYSICS_LOSSES_AVAILABLE:
            # 1. Momentum conservation loss (computed every batch - cheap)
            if self.momentum_weight > 0:
                unique_batches = torch.unique(batch.batch)
                for mol_idx in unique_batches:
                    mask = batch.batch == mol_idx
                    pos_mol = batch.pos[mask]
                    forces_mol = deriv_pred[mask]
                    loss_momentum += momentum_symmetry_loss(pos_mol, forces_mol)
                loss_momentum = loss_momentum / len(unique_batches)

            # 2. NVE loss (computed periodically - expensive)
            if self.nve_weight > 0 and batch_idx % self.nve_freq == 0:
                loss_nve = self._compute_nve_loss(batch_idx)

        # Total loss
        total_loss = (
                loss_y +
                loss_dy +
                self.momentum_weight * loss_momentum +
                self.nve_weight * loss_nve
        )

        # Logging
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_loss_y', loss_y, on_step=False, on_epoch=True)
        self.log('train_loss_dy', loss_dy, on_step=False, on_epoch=True)
        self.log('train_loss_momentum', loss_momentum, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_loss_nve', loss_nve, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def _compute_nve_loss(self, batch_idx):
        """
        Compute NVE loss by building trajectory from dataset
        """
        if self.full_dataset is None:
            return torch.tensor(0.0, device=self.device)

        try:
            # Calculate starting index for trajectory
            # Use batch_idx to vary which trajectory we sample
            dataset_size = len(self.full_dataset)
            max_start = dataset_size - self.traj_length

            if max_start <= 0:
                # Dataset too small for trajectories
                return torch.tensor(0.0, device=self.device)

            # Cycle through dataset based on batch_idx
            start_idx = (batch_idx * 137) % max_start  # 137 is prime for better coverage

            # Build trajectory batch
            traj_batch = build_trajectory_batch(
                self.full_dataset,
                start_idx,
                self.traj_length,
                self.device
            )

            # Compute NVE loss using representation model
            # Note: self.representation_model is the actual energy predictor
            loss_nve = nve_loss_from_trajectory(
                self.representation_model,
                traj_batch,
                self.device
            )

            return loss_nve

        except Exception as e:
            # If anything goes wrong, return zero loss
            # This prevents training from crashing
            print(f"Warning: NVE loss computation failed: {e}")
            return torch.tensor(0.0, device=self.device)

    def validation_step(self, batch, batch_idx):
        """Validation with physics losses"""
        pred, deriv_pred = self(batch.z, batch.pos, batch=batch.batch)
        loss_y, loss_dy = self.loss_fn(pred, batch.y, deriv_pred, batch.neg_dy)

        # Only compute momentum loss in validation (NVE too expensive)
        loss_momentum = torch.tensor(0.0, device=self.device)
        if PHYSICS_LOSSES_AVAILABLE and self.momentum_weight > 0:
            unique_batches = torch.unique(batch.batch)
            for mol_idx in unique_batches:
                mask = batch.batch == mol_idx
                pos_mol = batch.pos[mask]
                forces_mol = deriv_pred[mask]
                loss_momentum += momentum_symmetry_loss(pos_mol, forces_mol)
            loss_momentum = loss_momentum / len(unique_batches)

        total_loss = loss_y + loss_dy + self.momentum_weight * loss_momentum

        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_loss_y', loss_y, on_step=False, on_epoch=True)
        self.log('val_loss_dy', loss_dy, on_step=False, on_epoch=True)
        self.log('val_loss_momentum', loss_momentum, on_step=False, on_epoch=True)

        return total_loss


def train_physics_informed_model(
        dataset='MD17',
        molecule='aspirin',
        batch_size=32,
        num_epochs=100,
        lr=0.0001,
        model_type='tensornet',
        save_dir='checkpoints/physics_informed',
        log_dir='logs/physics_informed',
        force_weight=0.95,
        energy_weight=0.05,
        momentum_weight=0.01,
        nve_weight=1.0,
        pbc_weight=0.0,
        traj_length=100,
        nve_freq=10,
):
    """Train physics-informed model with NVE loss"""

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Physics-Informed TorchMD-NET with NVE Loss")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset}")
    print(f"Molecule: {molecule}")
    print(f"Model: {model_type}")
    print(f"\nPhysics Weights:")
    print(f"  Force:    {force_weight}")
    print(f"  Energy:   {energy_weight}")
    print(f"  Momentum: {momentum_weight}")
    print(f"  NVE:      {nve_weight} (traj_length={traj_length}, freq={nve_freq})")
    print(f"  PBC:      {pbc_weight} (disabled)")
    print(f"{'=' * 60}\n")

    # Load dataset
    print("Loading dataset...")
    full_dataset = MD17(root='./data', molecules=molecule)
    print(f"✓ Dataset loaded: {len(full_dataset)} samples")

    # Check if dataset is large enough for trajectories
    if len(full_dataset) < traj_length:
        print(f"⚠ Warning: Dataset ({len(full_dataset)}) smaller than trajectory length ({traj_length})")
        print(f"  NVE loss may not work properly. Consider reducing traj_length.")

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Split: {train_size} train, {val_size} val, {test_size} test")

    # Create dataloaders - use PyTorch Geometric DataLoader for graph data
    train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = GeometricDataLoader(val_data, batch_size=batch_size, num_workers=4)
    test_loader = GeometricDataLoader(test_data, batch_size=batch_size, num_workers=4)

    # Model configuration - ALL required parameters
    model_args = {
        # Model type
        'model': model_type,
        'prior_model': None,
        'output_model': 'Scalar',

        # LNNP required parameters
        'load_model': None,
        'remove_ref_energy': False,
        'train_loss': 'mse_loss',
        'train_loss_arg': None,
        'charge': False,
        'spin': False,

        # Precision and dtype
        'precision': 32,

        # Cutoffs
        'cutoff_lower': 0.0,
        'cutoff_upper': 5.0,

        # Architecture
        'embedding_dimension': 256,
        'num_layers': 6,
        'num_rbf': 64,
        'rbf_type': 'expnorm',
        'trainable_rbf': False,
        'activation': 'silu',
        'max_z': 100,
        'max_num_neighbors': 128,

        # Training
        'derivative': True,
        'lr': lr,
        'lr_patience': 15,
        'lr_min': 1e-7,
        'lr_factor': 0.8,
        'lr_warmup_steps': 0,
        'weight_decay': 0.0,
        'y_weight': energy_weight,
        'neg_dy_weight': force_weight,

        # EMA parameters
        'ema_alpha_y': 1.0,
        'ema_alpha_neg_dy': 1.0,

        # Physics loss weights (custom)
        'momentum_weight': momentum_weight,
        'nve_weight': nve_weight,
        'pbc_weight': pbc_weight,
        'traj_length': traj_length,
        'nve_freq': nve_freq,

        # Required by create_model
        'atom_filter': -1,
        'reduce_op': 'add',
        'equivariance_invariance_group': 'O(3)',
        'box_vecs': None,
        'check_errors': True,
        'static_shapes': False,
        'vector_cutoff': False,

        # For graph-network (if used)
        'aggr': 'add',
        'neighbor_embedding': True,

        # For transformer models (if used)
        'attn_activation': 'silu',
        'num_heads': 8,
        'distance_influence': 'both',
    }

    # Create model
    print("Creating physics-informed model...")
    try:
        if model_type == 'tensornet':
            from torchmdnet.models.torchmd_t import TorchMD_T
            representation = TorchMD_T(**model_args)
        else:
            raise ValueError(f"Only tensornet supported for now")

        model = PhysicsInformedLNNP(model_args, representation_model=representation)

        # CRITICAL: Give model access to full dataset for NVE loss
        model.full_dataset = full_dataset

        print("✓ Physics-informed model created with NVE loss")
    except Exception as e:
        print(f"✗ Physics-informed model creation failed: {e}")
        print("Falling back to standard model (no physics losses)...")

        # Fallback: LNNP creates model internally
        model = LNNP(model_args)
        print("✓ Standard model created")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename='best_model',
        save_top_k=1,
        mode='min',
        save_last=True,  # Also save last epoch
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=30, mode='min')

    # Logger
    logger = TensorBoardLogger(save_dir=log_dir, name='physics_informed')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1000.0,
    )

    # Train
    print("\nStarting training...")
    print(f"Monitor with: tensorboard --logdir={log_dir}\n")
    trainer.fit(model, train_loader, val_loader)

    # Test
    print("\nTesting...")
    test_results = trainer.test(model, test_loader, ckpt_path='best')

    # Save config
    config = {
        'model_args': model_args,
        'training': {
            'dataset': dataset,
            'molecule': molecule,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr,
        },
        'physics_weights': {
            'momentum': momentum_weight,
            'nve': nve_weight,
            'pbc': pbc_weight,
            'traj_length': traj_length,
            'nve_freq': nve_freq,
        },
        'test_results': test_results[0] if test_results else None,
    }

    with open(Path(save_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Model: {save_dir}/best_model.ckpt")
    print(f"  Logs: {log_dir}")

    return trainer, model, test_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--molecule', type=str, default='aspirin')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum-weight', type=float, default=0.01)
    parser.add_argument('--nve-weight', type=float, default=1.0)
    parser.add_argument('--traj-length', type=int, default=100)
    parser.add_argument('--nve-freq', type=int, default=10)

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
    )