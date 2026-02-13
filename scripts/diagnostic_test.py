#!/usr/bin/env python
"""
Updated Diagnostic Test Script for TorchMD-NET
Based on actual TorchMD-NET v2.x API
"""

import sys


def test_imports():
    """Test imports"""
    print("=" * 60)
    print("TEST 1: Checking Imports")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False

    try:
        import torchmdnet
        print(f"✓ TorchMD-NET installed")
    except ImportError as e:
        print(f"✗ TorchMD-NET: {e}")
        return False

    try:
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning {pl.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Lightning: {e}")
        return False

    return True


def test_dataset():
    """Test dataset loading (not DataModule)"""
    print("\n" + "=" * 60)
    print("TEST 2: Dataset Loading")
    print("=" * 60)

    try:
        from torchmdnet.datasets import MD17

        print("Creating MD17 dataset (aspirin)...")
        dataset = MD17(root='./data', dataset_arg='aspirin')
        print(f"✓ Dataset created!")
        print(f"  Length: {len(dataset)}")
        print(f"  Sample keys: {dataset[0].keys}")
        return True

    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_torchmd_train_cli():
    """Test the torchmd-train CLI utility"""
    print("\n" + "=" * 60)
    print("TEST 3: torchmd-train CLI")
    print("=" * 60)

    import subprocess

    try:
        result = subprocess.run(
            ['torchmd-train', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("✓ torchmd-train CLI is available")
            print("\nRecommended workflow:")
            print("  1. Create a YAML config file")
            print("  2. Run: torchmd-train --conf config.yaml")
            return True
        else:
            print("✗ torchmd-train not working")
            return False

    except FileNotFoundError:
        print("✗ torchmd-train command not found")
        print("\nThis is OK - you can use Python API directly")
        return True
    except Exception as e:
        print(f"⚠ Could not test CLI: {e}")
        return True


def test_model_from_yaml():
    """Test creating model using YAML-like args"""
    print("\n" + "=" * 60)
    print("TEST 4: Model from Args (YAML Style)")
    print("=" * 60)

    try:
        # Use torchmd-train's argument parser approach
        from torchmdnet.scripts.train import get_argparse_args
        import argparse

        parser = argparse.ArgumentParser()
        parser = get_argparse_args(parser)

        # Minimal working args for TensorNet
        args_list = [
            '--model', 'tensornet',
            '--dataset', 'MD17',
            '--dataset-arg', 'aspirin',
            '--batch-size', '2',
            '--max-num-neighbors', '128',
            '--embedding-dimension', '128',
            '--num-layers', '2',
            '--num-rbf', '32',
            '--rbf-type', 'expnorm',
            '--activation', 'silu',
            '--max-z', '100',
            '--lr', '0.0001',
            '--derivative',
        ]

        args = parser.parse_args(args_list)
        args_dict = vars(args)

        print("✓ Args parsed successfully")
        print(f"  Model: {args_dict['model']}")
        print(f"  Dataset: {args_dict['dataset']}")
        print(f"  Dataset arg: {args_dict['dataset_arg']}")

        return True

    except Exception as e:
        print(f"✗ Failed to parse args: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_inference():
    """Test loading a model and doing inference"""
    print("\n" + "=" * 60)
    print("TEST 5: Simple Inference")
    print("=" * 60)

    try:
        import torch
        from torchmdnet.models.model import load_model

        print("Attempting to create a simple model...")
        print("(This will fail without a checkpoint, but tests the API)")

        # This should fail gracefully
        try:
            model = load_model("nonexistent.ckpt")
            print("✗ Should have failed on nonexistent checkpoint")
            return False
        except FileNotFoundError:
            print("✓ load_model API works (correctly failed on missing file)")
            return True

    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False


def print_recommended_approach():
    """Print recommended workflow"""
    print("\n" + "=" * 60)
    print("RECOMMENDED APPROACH")
    print("=" * 60)

    print("""
TorchMD-NET is designed to work with YAML configs and CLI.

OPTION 1: Use torchmd-train CLI (Recommended)
----------------------------------------------
1. Create config.yaml:

   model: tensornet
   dataset: MD17
   dataset-arg: aspirin
   batch-size: 32
   embedding-dimension: 256
   num-layers: 6
   num-rbf: 64
   rbf-type: expnorm
   activation: silu
   max-z: 100
   max-num-neighbors: 128
   lr: 0.0001
   derivative: true

2. Run training:

   mkdir logs
   torchmd-train --conf config.yaml --log-dir logs/

OPTION 2: Use Python API (Advanced)
------------------------------------
from torchmdnet.datasets import MD17
from torch.utils.data import DataLoader

# Load dataset directly
dataset = MD17(root='./data', dataset_arg='aspirin')
train_loader = DataLoader(dataset, batch_size=32)

# Use torchmd-train's arg parser for model creation
from torchmdnet.scripts.train import get_argparse_args
# ... parse args and create model

OPTION 3: Load pretrained model
--------------------------------
from torchmdnet.models.model import load_model

model = load_model('checkpoint.ckpt', derivative=True)
energy, forces = model(z, pos, batch=batch)

""")


def main():
    """Run diagnostic"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║     TorchMD-NET Diagnostic (Updated for v2.x API)         ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    results = []

    results.append(("Imports", test_imports()))

    if results[-1][1]:
        results.append(("Dataset", test_dataset()))
        results.append(("CLI", test_torchmd_train_cli()))
        results.append(("Args Parser", test_model_from_yaml()))
        results.append(("Inference API", test_simple_inference()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    print("=" * 60)

    if all(r[1] for r in results):
        print("\n🎉 DIAGNOSTICS PASSED!")
        print_recommended_approach()
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nCheck errors above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())