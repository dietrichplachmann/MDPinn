#!/usr/bin/env python
"""
Diagnostic Test Script
Run this to identify exactly what's wrong with your setup
"""

import sys


def test_imports():
    """Test all required imports"""
    print("=" * 60)
    print("TEST 1: Checking Imports")
    print("=" * 60)

    all_good = True

    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch FAILED: {e}")
        all_good = False
        return False

    # Test TorchMD-NET
    try:
        import torchmdnet
        try:
            version = torchmdnet.__version__
        except AttributeError:
            version = "installed (version unknown)"
        print(f"✓ TorchMD-NET {version}")
    except ImportError as e:
        print(f"✗ TorchMD-NET FAILED: {e}")
        all_good = False
        return False

    # Test PyTorch Lightning
    try:
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning {pl.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch Lightning FAILED: {e}")
        all_good = False

    return all_good


def test_datamodule():
    """Test DataModule creation"""
    print("\n" + "=" * 60)
    print("TEST 2: DataModule Creation")
    print("=" * 60)

    try:
        from torchmdnet.data import DataModule

        print("Testing with MD17 aspirin...")
        data = DataModule(
            dataset='MD17',
            dataset_root='./data',
            dataset_arg='aspirin',  # CORRECT parameter name
            batch_size=2,
            num_workers=0,  # Use 0 for debugging
            splits=[0.8, 0.1, 0.1],
        )
        print("✓ DataModule created successfully!")
        print(f"  Dataset: {data.dataset}")
        print(f"  Batch size: {data.batch_size}")
        return True

    except TypeError as e:
        print(f"✗ TypeError: {e}")
        print("\nLikely issue: Wrong parameter name")
        print("Check if you're using:")
        print("  ✓ dataset_arg='aspirin'  (CORRECT)")
        print("  ✗ molecule='aspirin'     (WRONG)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Creation")
    print("=" * 60)

    try:
        from torchmdnet.models.model import create_model

        print("Testing minimal model args...")
        model_args = {
            # Required
            'model': 'tensornet',
            'prior_model': None,
            'output_model': 'Scalar',

            # Architecture
            'embedding_dimension': 128,
            'num_layers': 3,
            'num_rbf': 32,
            'rbf_type': 'expnorm',
            'trainable_rbf': False,
            'activation': 'silu',
            'max_z': 100,
            'max_num_neighbors': 128,

            # Training
            'derivative': True,
            'lr': 0.0001,

            # Loss weights
            'energy_weight': 0.05,
            'force_weight': 0.95,

            # EMA (might be required)
            'ema_alpha_y': 1.0,
            'ema_alpha_dy': 1.0,
        }

        print("Creating model...")
        model = create_model(model_args)
        print("✓ Model created successfully!")

        # Test forward pass
        import torch
        print("\nTesting forward pass...")
        z = torch.tensor([1, 6, 8], dtype=torch.long)
        pos = torch.randn(3, 3)
        batch = torch.zeros(3, dtype=torch.long)

        energy, forces = model(z, pos, batch=batch)
        print(f"✓ Forward pass successful!")
        print(f"  Energy shape: {energy.shape}")
        print(f"  Forces shape: {forces.shape}")

        return True

    except KeyError as e:
        print(f"✗ KeyError: {e}")
        print("\nMissing required argument!")
        print("Common missing args:")
        print("  - prior_model")
        print("  - output_model")
        print("  - ema_alpha_y")
        print("  - ema_alpha_dy")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test complete training setup"""
    print("\n" + "=" * 60)
    print("TEST 4: Full Integration Test")
    print("=" * 60)

    try:
        import torch
        import pytorch_lightning as pl
        from torchmdnet.models.model import create_model
        from torchmdnet.data import DataModule

        print("Creating model...")
        model_args = {
            'model': 'tensornet',
            'prior_model': None,
            'output_model': 'Scalar',
            'embedding_dimension': 128,
            'num_layers': 2,
            'num_rbf': 32,
            'rbf_type': 'expnorm',
            'trainable_rbf': False,
            'activation': 'silu',
            'max_z': 100,
            'derivative': True,
            'lr': 0.0001,
            'energy_weight': 0.05,
            'force_weight': 0.95,
            'ema_alpha_y': 1.0,
            'ema_alpha_dy': 1.0,
        }
        model = create_model(model_args)

        print("Creating DataModule...")
        data = DataModule(
            dataset='MD17',
            dataset_root='./data',
            dataset_arg='aspirin',
            batch_size=2,
            num_workers=0,
            splits=[0.8, 0.1, 0.1],
        )

        print("Creating Trainer...")
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )

        print("Running 1 training step...")
        trainer.fit(model, data)

        print("✓ Full integration test passed!")
        print("\nYour setup is working correctly!")
        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_error(error_msg):
    """Diagnose common errors"""
    print("\n" + "=" * 60)
    print("ERROR DIAGNOSIS")
    print("=" * 60)

    if "unexpected keyword argument" in error_msg:
        print("DIAGNOSIS: Wrong parameter name")
        print("\nCommon fixes:")
        print("1. DataModule:")
        print("   ✗ molecule='aspirin'")
        print("   ✓ dataset_arg='aspirin'")
        print("")
        print("2. Model:")
        print("   Add: 'prior_model': None")
        print("   Add: 'output_model': 'Scalar'")

    elif "missing" in error_msg and "required" in error_msg:
        print("DIAGNOSIS: Missing required argument")
        print("\nAdd these to model_args:")
        print("  'prior_model': None,")
        print("  'output_model': 'Scalar',")
        print("  'ema_alpha_y': 1.0,")
        print("  'ema_alpha_dy': 1.0,")

    elif "KeyError" in error_msg:
        print("DIAGNOSIS: Dictionary key not found")
        print("\nMake sure model_args includes:")
        print("  All required keys listed in DEBUGGING_GUIDE.md")


def main():
    """Run all diagnostic tests"""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║         TorchMD-NET Diagnostic Test Suite                 ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))

    if results[-1][1]:  # Only continue if imports work
        results.append(("DataModule", test_datamodule()))
        results.append(("Model Creation", test_model_creation()))

        if all(r[1] for r in results):
            results.append(("Full Integration", test_full_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    print("=" * 60)

    if all(r[1] for r in results):
        print("\n🎉 ALL TESTS PASSED!")
        print("Your environment is correctly set up.")
        print("\nYou can now run:")
        print("  python train_standard_CORRECTED.py")
        print("  python train_physics_informed_CORRECTED.py")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())