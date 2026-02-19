#!/usr/bin/env python
"""
Standard TorchMD-NET Training Script - FIXED for your environment
"""

import torch

# CRITICAL: Fix PyTorch 2.7 torch.load compatibility
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import json
from torch.utils.data import DataLoader, random_split

from torchmdnet.datasets import MD17
from torchmdnet.models.model import create_model


def train_standard_model(
        dataset='MD17',
        molecule='aspirin',
        batch_size=32,
        num_epochs=100,
        lr=0.0001,
        model_type='tensornet',
        save_dir='checkpoints/standard',
        log_dir='logs/standard',
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

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    # Model configuration - ALL required parameters
    model_args = {
        # Model type
        'model': model_type,
        'prior_model': None,
        'output_model': 'Scalar',

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
        'energy_weight': 0.05,
        'force_weight': 0.95,

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
        model = create_model(model_args)
        print("✓ Model created")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        raise

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename='model-{epoch:02d}-{val_loss:.6f}',
        save_top_k=3,
        mode='min',
    )

    best_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename='best_model',
        save_top_k=1,
        mode='min',
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
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
        callbacks=[checkpoint_callback, best_checkpoint, early_stop],
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