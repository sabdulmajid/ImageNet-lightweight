#!/bin/bash
# ImageNet-lightweight Setup Script for Linux/Mac with NVIDIA GPU
# Automatically installs all dependencies with CUDA 11.8 support

set -e

echo "========================================"
echo "ImageNet-lightweight Setup (Unix)"
echo "========================================"
echo ""

# Check Python
echo "[1/5] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.8+ first."
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "      $PYTHON_VERSION"

# Check NVIDIA GPU
echo "[2/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "      GPU: $GPU_NAME"
else
    echo "      WARNING: NVIDIA GPU not detected. Training will be slow!"
fi

# Install PyTorch with CUDA
echo "[3/5] Installing PyTorch 2.7.1+cu118..."
python3 -m pip install --upgrade pip > /dev/null
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "[4/5] Installing dependencies..."
python3 -m pip install timm pyyaml tqdm tensorboard matplotlib seaborn pandas fvcore pynvml

# Verify installation
echo "[5/5] Verifying setup..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'      PyTorch {torch.__version__} with GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Test: python3 test_setup.py"
echo "2. Prepare data: python3 scripts/prepare_data.py --imagenet-path /path/to/imagenet --output data/imagenet100 --num-classes 100"
echo "3. Train: python3 train.py --config configs/vit_small.yaml"
