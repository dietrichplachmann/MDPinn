#!/usr/bin/env python
"""
Standard TorchMD-NET Training (Baseline)
No physics-informed losses, just standard energy and force training
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

    Args:
        dataset: Dataset name (MD17, rMD17, QM9)
        molecule: Molecule name (for MD17/rMD17)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        model_type: Model architecture (tensornet, equivariant-transformer, graph-network)
        save_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """

    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training Standard TorchMD-NET")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset}")
    if dataset in ['MD17', 'rMD17']:
        print(f"Molecule: {molecule}")
    print(f"Model: {model_type}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"{'=' * 60}\n")

    # Model configuration
    model_args = {
        'model': model_type,
        'embedding_dimension': 256,
        'num_layers': 6,
        'num_rbf': 64,
        'rbf_type': 'expnorm',
        'trainable_rbf': False,
        'activation': 'silu',
        'max_z': 100,
        'max_num_neighbors': 128,

        # Training settings
        'derivative': True,  # Train on forces
        'lr': lr,
        'lr_patience': 15,
        'lr_min': 1e-7,
        'lr_factor': 0.8,
        'weight_decay': 0.0,

        # Loss weights - standard
        'force_weight': 0.95,
        'energy_weight': 0.05,

        'gradient_clipping': 1000.0,

        # Dataset
        'dataset': dataset,
        'dataset_arg': molecule if dataset in ['MD17', 'rMD17'] else '7',
    }

    # Additional settings for specific models
    if model_type == 'equivariant-transformer':
        model_args.update({
            'attn_activation': 'silu',
            'num_heads': 8,
            'distance_influence': 'both',
        })

    # Create model
    print("Creating model...")
    if 'precision' not in model_args:
        model_args['precision'] = 32
    # using defaults from documentation, this can be changed later
    if 'cutoff_lower' not in model_args:
        model_args['cutoff_lower'] = 0.0
    if 'cutoff_upper' not in model_args:
        model_args['cutoff_upper'] = 5.0
    if 'max_z' not in model_args:
        model_args['max_z'] = 128
    if 'max_num_neighbors' not in model_args:
        model_args['max_num_neighbors'] = 64
    model = create_model(model_args)

    # Setup data
    print("Loading dataset...")
    data = DataModule(
        dataset=dataset,
        dataset_root='./data',
        dataset_arg=molecule if dataset in ['MD17', 'rMD17'] else '7',
        batch_size=batch_size,
        num_workers=4,
        splits=[0.8, 0.1, 0.1],
        seed=42
    )

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

    # Save best model separately for easy access
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

    trainer.fit(model, data)

    # Test
    print("\nTesting best model...")
    test_results = trainer.test(model, data, ckpt_path='best')

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

    parser = argparse.ArgumentParser(description='Train standard TorchMD-NET model')
    parser.add_argument('--dataset', type=str, default='MD17')
    parser.add_argument('--molecule', type=str, default='aspirin')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--model', type=str, default='tensornet')
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