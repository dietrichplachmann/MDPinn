#!/usr/bin/env python
"""
CORRECTED Standard TorchMD-NET Training Script
Fixed DataModule arguments and model_args defaults
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import json

from torchmdnet.models.model import create_model
from torchmdnet.data import DataModule


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
    """
    Train a standard TorchMD-NET model
    CORRECTED VERSION with proper arguments
    """

    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training Standard TorchMD-NET (CORRECTED)")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset}")
    if dataset in ['MD17', 'rMD17']:
        print(f"Molecule: {molecule}")
    print(f"Model: {model_type}")
    print(f"{'=' * 60}\n")

    # CORRECTED Model configuration with ALL required arguments
    model_args = {
        # Model type
        'model': model_type,

        # CRITICAL - These are often missing!
        'prior_model': None,  # No atomic reference energies
        'output_model': 'Scalar',  # Scalar energy output

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
        'derivative': True,  # Train on forces
        'lr': lr,
        'lr_patience': 15,
        'lr_min': 1e-7,
        'lr_factor': 0.8,
        'lr_warmup_steps': 0,
        'weight_decay': 0.0,

        # EMA (Exponential Moving Average) - required by some versions
        'ema_alpha_y': 1.0,  # 1.0 = no EMA for energy
        'ema_alpha_dy': 1.0,  # 1.0 = no EMA for forces

        # Loss weights
        'energy_weight': 0.05,
        'force_weight': 0.95,

        # Gradient clipping
        'gradient_clipping': 1000.0,

        # Dataset info (sometimes required)
        'dataset': dataset,
        'dataset_arg': molecule if dataset in ['MD17', 'rMD17'] else 'U0',
    }

    # Model-specific additional arguments
    if model_type == 'equivariant-transformer':
        model_args.update({
            'attn_activation': 'silu',
            'num_heads': 8,
            'distance_influence': 'both',
        })
    elif model_type == 'graph-network':
        # Graph network specific args (if any)
        pass

    # Create model
    print("Creating model...")
    try:
        model = create_model(model_args)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        print("\nModel args that were used:")
        for k, v in model_args.items():
            print(f"  {k}: {v}")
        raise

    # CORRECTED Setup data with proper argument names
    print("Loading dataset...")
    try:
        data = DataModule(
            dataset=dataset,
            dataset_root='./data',
            dataset_arg=molecule if dataset in ['MD17', 'rMD17'] else 'U0',  # CORRECT: dataset_arg, not molecule
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
        name='standard_training',
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
    print("\nStarting training...")
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
        'test_results': test_results[0] if test_results else None,
    }

    config_path = Path(save_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Best model: {save_dir}/best_model.ckpt")
    print(f"  Config: {config_path}")
    print(f"  Logs: {log_dir}")

    return trainer, model, test_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train standard TorchMD-NET model (CORRECTED)')
    parser.add_argument('--dataset', type=str, default='MD17')
    parser.add_argument('--molecule', type=str, default='aspirin')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model', type=str, default='tensornet',
                        choices=['tensornet', 'equivariant-transformer', 'graph-network'])
    parser.add_argument('--save-dir', type=str, default='checkpoints/standard')
    parser.add_argument('--log-dir', type=str, default='logs/standard')

    args = parser.parse_args()

    train_standard_model(
        dataset=args.dataset,
        molecule=args.molecule,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        model_type=args.model,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )