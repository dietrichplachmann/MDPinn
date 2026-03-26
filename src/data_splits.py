"""Shared dataset split helpers for trajectory-aware experiments."""

from __future__ import annotations

from torch.utils.data import Subset


def contiguous_split(dataset, train_frac=0.8, val_frac=0.1):
    """Split a trajectory-like dataset into contiguous train/val/test blocks.

    This preserves temporal locality within each split, which is important for
    trajectory-based regularizers and rollout-style evaluation.
    """
    n_samples = len(dataset)
    train_size = int(train_frac * n_samples)
    val_size = int(val_frac * n_samples)
    test_size = n_samples - train_size - val_size

    train_end = train_size
    val_end = train_size + val_size

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, val_end + test_size))

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )
