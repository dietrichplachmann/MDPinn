#!/usr/bin/env python
"""
CORRECTED Physics-Informed TorchMD-NET Training Script
Fixed DataModule arguments and model_args defaults
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import json

from torchmdnet.models.model import create_model, LNNP
from torchmdnet.data import DataModule

# Import physics losses (make sure physics_losses.py is in the same directory or PYTHONPATH)
try:
    from physics_losses import (
        nve_loss_from_md17_trajectory,
        momentum_symmetry_loss,
        periodic_bc_loss_improved,
    )

    PHYSICS_LOSSES_AVAILABLE = True
except ImportError:
    print("Warning: physics_losses.py not found. Physics-informed losses will be disabled.")
    PHYSICS_LOSSES_AVAILABLE = False


class PhysicsInformedLNNP(LNNP):
    """
    Extended LNNP with physics-informed losses
    """

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        # Physics loss weights
        self.nve_weight = hparams.get('nve_weight', 1.0)
        self.pbc_weight = hparams.get('pbc_weight', 0.1)
        self.momentum_weight = hparams.get('momentum_weight', 0.01)
        self.traj_length = hparams.get('traj_length', 100)

        print(f"Physics loss weights:")
        print(f"  NVE:      {self.nve_weight}")
        print(f"  PBC:      {self.pbc_weight}")
        print(f"  Momentum: {self.momentum_weight}")

    def training_step(self, batch, batch_idx):
        """
        Override training step to add physics losses
        """
        # Standard forward pass
        pred, deriv_pred = self(batch.z, batch.pos, batch=batch.batch)

        # Standard energy and force losses
        loss_y, loss_dy = self.loss_fn(pred, batch.y, deriv_pred, batch.neg_dy)

        # Initialize physics losses
        loss_momentum = torch.tensor(0.0, device=self.device)
        loss_pbc = torch.tensor(0.0, device=self.device)
        loss_nve = torch.tensor(0.0, device=self.device)

        if PHYSICS_LOSSES_AVAILABLE:
            # 1. Momentum conservation loss
            if self.momentum_weight > 0:
                unique_batches = torch.unique(batch.batch)
                for mol_idx in unique_batches:
                    mask = batch.batch == mol_idx
                    pos_mol = batch.pos[mask]
                    forces_mol = deriv_pred[mask]
                    loss_momentum += momentum_symmetry_loss(pos_mol, forces_mol)
                loss_momentum = loss_momentum / len(unique_batches)

        # Total loss
        total_loss = (
                loss_y +
                loss_dy +
                self.momentum_weight * loss_momentum +
                self.pbc_weight * loss_pbc +
                self.nve_weight * loss_nve
        )

        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_loss_y', loss_y, prog_bar=False)
        self.log('train_loss_dy', loss_dy, prog_bar=False)
        self.log('train_loss_momentum', loss_momentum, prog_bar=True)
        self.log('train_loss_pbc', loss_pbc, prog_bar=False)
        self.log('train_loss_nve', loss_nve, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation with physics losses
        """
        pred, deriv_pred = self(batch.z, batch.pos, batch=batch.batch)
        loss_y, loss_dy = self.loss_fn(pred, batch.y, deriv_pred, batch.neg_dy)

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

        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_loss_y', loss_y)
        self.log('val_loss_dy', loss_dy)
        self.log('val_loss_momentum', loss_momentum)

        return total_loss


def create_physics_informed_model(hparams):
    """
    Create a physics-informed model
    """
    # Create base representation model first
    from torchmdnet.models import model as model_module

    # Get the representation model class
    if hparams['model'] == 'tensornet':
        from torchmdnet.models.torchmd_t import TorchMD_T
        representation_model = TorchMD_T(**hparams)
    elif hparams['model'] == 'equivariant-transformer':
        from torchmdnet.models.torchmd_et import TorchMD_ET
        representation_model = TorchMD_ET(**hparams)
    elif hparams['model'] == 'graph-network':
        from torchmdnet.models.torchmd_gn import TorchMD_GN
        representation_model = TorchMD_GN(**hparams)
    else:
        raise ValueError(f"Unknown model type: {hparams['model']}")

    # Wrap in physics-informed Lightning module
    model = PhysicsInformedLNNP(hparams, representation_model=representation_model)

    return model


def train_physics_informed_model(
        dataset='MD17',
        molecule='aspirin',
        batch_size=32,
        num_epochs=100,
        lr=0.0001,
        model_type='tensornet',
        save_dir='checkpoints/physics_informed',
        log_dir='logs/physics_informed',
        # Physics loss weights
        force_weight=0.95,
        energy_weight=0.05,
        nve_weight=1.0,
        pbc_weight=0.1,
        momentum_weight=0.01,
        traj_length=100,
):
    """
    Train a physics-informed TorchMD-NET model
    CORRECTED VERSION
    """

    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training Physics-Informed TorchMD-NET (CORRECTED)")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset}")
    if dataset in ['MD17', 'rMD17']:
        print(f"Molecule: {molecule}")
    print(f"Model: {model_type}")
    print(f"\nPhysics Loss Weights:")
    print(f"  Force:    {force_weight}")
    print(f"  Energy:   {energy_weight}")
    print(f"  NVE:      {nve_weight}")
    print(f"  PBC:      {pbc_weight}")
    print(f"  Momentum: {momentum_weight}")
    print(f"{'=' * 60}\n")

    # CORRECTED Model configuration
    model_args = {
        # Model type
        'model': model_type,

        # CRITICAL required arguments
        'prior_model': None,
        'output_model': 'Scalar',

        # Architecture
        'embedding_dimension': 256,
        'num_layers': 6,
        'num_rbf': 64,
        'rbf_type': 'expnorm',
        'trainable_rbf': False,
        'activation': 'silu',
        'max_z': 100,
        'max_num_neighbors': 128,
        'aggr': 'add',
        'neighbor_embedding': True,

        # Training settings
        'derivative': True,
        'lr': lr,
        'lr_patience': 15,
        'lr_min': 1e-7,
        'lr_factor': 0.8,
        'lr_warmup_steps': 0,
        'weight_decay': 0.0,

        # EMA
        'ema_alpha_y': 1.0,
        'ema_alpha_dy': 1.0,

        # Standard loss weights
        'energy_weight': energy_weight,
        'force_weight': force_weight,

        # Physics loss weights (custom)
        'nve_weight': nve_weight,
        'pbc_weight': pbc_weight,
        'momentum_weight': momentum_weight,
        'traj_length': traj_length,

        # Gradient clipping
        'gradient_clipping': 1000.0,

        # Dataset
        'dataset': dataset,
        'dataset_arg': molecule if dataset in ['MD17', 'rMD17'] else 'U0',
    }

    # Model-specific args
    if model_type == 'equivariant-transformer':
        model_args.update({
            'attn_activation': 'silu',
            'num_heads': 8,
            'distance_influence': 'both',
        })

    # Create physics-informed model
    print("Creating physics-informed model...")
    try:
        model = create_physics_informed_model(model_args)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        print("\nTrying standard model creation...")
        try:
            model = create_model(model_args)
            print("✓ Standard model created (physics losses may not work)")
        except Exception as e2:
            print(f"✗ Both creation methods failed: {e2}")
            raise

    # CORRECTED Setup data
    print("Loading dataset...")
    try:
        data = DataModule(
            dataset=dataset,
            dataset_root='./data',
            dataset_arg=molecule if dataset in ['MD17', 'rMD17'] else 'U0',  # CORRECT
            batch_size=batch_size,
            num_workers=4,
            splits=[0.8, 0.1, 0.1],
            seed=42,
        )
        print("✓ DataModule created successfully")
    except Exception as e:
        print(f"✗ Error creating DataModule: {e}")
        raise

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename='model-{epoch:02d}-{val_loss:.6f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        save_on_train_epoch_end=False,
    )

    best_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename='best_model',
        save_top_k=1,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        mode='min',
        verbose=True
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='physics_informed_training',
        version=molecule if dataset in ['MD17', 'rMD17'] else 'qm9'
    )

    # Trainer
    print("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, best_checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=1000.0,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    print("\nStarting physics-informed training...")
    print(f"Monitor logs with: tensorboard --logdir={log_dir}\n")

    try:
        trainer.fit(model, data)
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        raise

    # Test
    print("\nTesting best model...")
    try:
        test_results = trainer.test(model, data, ckpt_path='best')
    except Exception as e:
        print(f"Warning: Testing failed: {e}")
        test_results = None

    # Save configuration
    config = {
        'model_args': model_args,
        'training': {
            'dataset': dataset,
            'molecule': molecule if dataset in ['MD17', 'rMD17'] else 'N/A',
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr,
        },
        'physics_weights': {
            'nve': nve_weight,
            'pbc': pbc_weight,
            'momentum': momentum_weight,
        },
        'test_results': test_results[0] if test_results else None,
    }

    config_path = Path(save_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Physics-informed training complete!")
    print(f"  Best model: {save_dir}/best_model.ckpt")
    print(f"  Config: {config_path}")
    print(f"  Logs: {log_dir}")

    return trainer, model, test_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train physics-informed TorchMD-NET (CORRECTED)')
    parser.add_argument('--dataset', type=str, default='MD17')
    parser.add_argument('--molecule', type=str, default='aspirin')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model', type=str, default='tensornet')
    parser.add_argument('--nve-weight', type=float, default=1.0)
    parser.add_argument('--pbc-weight', type=float, default=0.1)
    parser.add_argument('--momentum-weight', type=float, default=0.01)
    parser.add_argument('--traj-length', type=int, default=100)

    args = parser.parse_args()

    train_physics_informed_model(
        dataset=args.dataset,
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        nve_weight=args.nve_weight,
        pbc_weight=args.pbc_weight,
        momentum_weight=args.momentum_weight,
        traj_length=args.traj_length,
    )