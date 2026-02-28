#!/usr/bin/env python
"""
Standard TorchMD-NET Training Script - FIXED for your environment
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

from baseline_potential import lj_energy_forces_batched



class DeltaLNNP(LNNP):
    """LNNP wrapper that converts absolute labels to Δ-labels on the fly.

    When hparams['delta_learning'] is True:
      y_target := y_true - U_ref
      neg_dy_target := F_true - F_ref
    and the network learns ΔU, ΔF. (Hybrid = ref + model at inference.)
    """

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.delta_learning = bool(getattr(self.hparams, "delta_learning", False)) if hasattr(self, "hparams") else bool(hparams.get("delta_learning", False))
        self.baseline_eps = float(hparams.get("baseline_epsilon_eV", 0.01))
        self.baseline_sigma = float(hparams.get("baseline_sigma_A", 1.0))
        self.baseline_cutoff = float(hparams.get("baseline_cutoff_A", 5.0))

    def data_transform(self, batch):
        batch = super().data_transform(batch)

        if not self.delta_learning:
            return batch

        # Compute analytic baseline on the *current* positions.
        # Note: baseline ignores Z for now; that's OK for a first Δ-learning pivot.
        U_ref, F_ref = lj_energy_forces_batched(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch,
            epsilon_eV=self.baseline_eps,
            sigma_A=self.baseline_sigma,
            r_cut_A=self.baseline_cutoff,
        )

        # Energy labels in MD17 are per-graph. Subtract per-graph baseline.
        if hasattr(batch, "y") and batch.y is not None:
            y = batch.y
            if y.ndim == 1:
                y = y.unsqueeze(1)
            # Map per-atom batch indices -> per-graph baseline energy
            # batch.y is per-graph (B,1), but random_split+GeometricDataLoader uses batch.batch to indicate graph.
            # The order of graphs in the mini-batch is the sorted unique(batch.batch) used in lj_energy_forces_batched.
            # That matches PyG's internal ordering for a standard collate.
            batch.y = (y.squeeze(-1) - U_ref).unsqueeze(1)

        # Force labels are per-atom (N,3)
        if hasattr(batch, "neg_dy") and batch.neg_dy is not None:
            batch.neg_dy = batch.neg_dy - F_ref

        return batch


def train_standard_model(
        dataset='MD17',
        molecule='aspirin',
        batch_size=32,
        num_epochs=100,
        lr=0.0001,
        model_type='tensornet',
        save_dir='checkpoints/standard',
        log_dir='logs/standard',
        delta_learning: bool = False,
        baseline_epsilon_eV: float = 0.01,
        baseline_sigma_A: float = 1.0,
        baseline_cutoff_A: float = 5.0,
):
    """Train a standard TorchMD-NET model"""

    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training Standard TorchMD-NET")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset}")
    print(f"Molecule: {molecule}")
    print(f"Model: {model_type}")
    print(f"{'=' * 60}\n")

    # Load dataset using MD17 directly (NOT DataModule)
    print("Loading dataset...")
    full_dataset = MD17(root='./data', molecules=molecule)
    print(f"✓ Dataset loaded: {len(full_dataset)} samples")

    # Split dataset manually
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
        'y_weight': 0.05,  # Same as energy_weight
        'neg_dy_weight': 0.95,  # Same as force_weight

        # EMA parameters
        'ema_alpha_y': 1.0,
        'ema_alpha_neg_dy': 1.0,

        # Required by create_model
        'atom_filter': -1,  # No atom filtering
        'reduce_op': 'add',  # Reduction operation for output
        'equivariance_invariance_group': 'O(3)',  # For TensorNet
        'box_vecs': None,  # No periodic box
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
    print("Creating model...")
    try:
        # LNNP creates the model internally from hparams
        model = DeltaLNNP(model_args) if delta_learning else LNNP(model_args)
        print("✓ Model created")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        raise

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_mse_loss',  # LNNP uses this name
        dirpath=save_dir,
        filename='best_model',
        save_top_k=1,
        mode='min',
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor='val_total_mse_loss',  # LNNP uses this name
        patience=30,
        mode='min',
    )

    # Logger
    logger = TensorBoardLogger(save_dir=log_dir, name='standard')

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        log_every_n_steps=10,
    )

    # Train
    print("\nStarting training...")
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
        'test_results': test_results[0] if test_results else None,
    }

    with open(Path(save_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Model: {save_dir}/best_model.ckpt")

    return trainer, model, test_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--molecule', type=str, default='aspirin')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model', type=str, default='tensornet')

    args = parser.parse_args()

    train_standard_model(
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
    )