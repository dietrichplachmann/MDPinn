#!/bin/bash

################################################################################
# Quick Setup - TorchMD-NET Physics-Informed Training
#
# Minimal setup script for quick installation
# Usage: bash quick_setup.sh
################################################################################

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     TorchMD-NET Physics-Informed Training - Quick Setup       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected"
    CUDA_VERSION="11.8"  # Default
else
    echo "⚠ No GPU detected - will install CPU version"
    CUDA_VERSION="cpu"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment: torchmd-physics"
python3 -m venv torchmd-physics
source torchmd-physics/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install PyTorch
echo "Installing PyTorch..."
if [ "$CUDA_VERSION" = "11.8" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
elif [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

# Install TorchMD-NET
echo "Installing TorchMD-NET..."
if [ "$CUDA_VERSION" = "11.8" ]; then
    pip install torchmd-net-cu11 --extra-index-url https://download.pytorch.org/whl/cu118 -q
else
    pip install torchmd-net-cpu --extra-index-url https://download.pytorch.org/whl/cpu -q
fi

# Install other dependencies
echo "Installing dependencies..."
pip install pytorch-lightning matplotlib seaborn tqdm numpy -q tensorboardX tensorboard

# Create directories
echo "Creating project directories..."
mkdir -p data checkpoints/{standard,physics_informed} logs/{standard,physics_informed} results/plots

# Test installation
echo ""
echo "Testing installation..."
python -c "import torch; import torchmdnet; print('✓ Installation successful!')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE!                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "To activate environment:"
echo "  source torchmd-physics/bin/activate"
echo ""
echo "To start training:"
echo "  python run_training.py"
echo ""