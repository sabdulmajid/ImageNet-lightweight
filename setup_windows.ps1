# ImageNet-lightweight Setup Script for Windows with NVIDIA GPU
# Automatically installs all dependencies with CUDA 11.8 support

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ImageNet-lightweight Setup (Windows)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}
Write-Host "      $pythonVersion" -ForegroundColor Green

# Check NVIDIA GPU
Write-Host "[2/5] Checking GPU..." -ForegroundColor Yellow
$gpuCheck = python -c "import subprocess; print(subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0)" 2>$null
if ($gpuCheck -eq "True") {
    $gpuName = python -c "import subprocess; r=subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True); print(r.stdout.strip())"
    Write-Host "      GPU: $gpuName" -ForegroundColor Green
} else {
    Write-Host "      WARNING: NVIDIA GPU not detected. Training will be slow!" -ForegroundColor Yellow
}

# Install PyTorch with CUDA
Write-Host "[3/5] Installing PyTorch 2.7.1+cu118..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: PyTorch installation failed" -ForegroundColor Red
    exit 1
}

# Install other dependencies
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Yellow
python -m pip install timm pyyaml tqdm tensorboard matplotlib seaborn pandas fvcore pynvml
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Dependencies installation failed" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "[5/5] Verifying setup..." -ForegroundColor Yellow
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'      PyTorch {torch.__version__} with GPU: {torch.cuda.get_device_name(0)}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: GPU verification failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Test: python test_setup.py" -ForegroundColor White
Write-Host "2. Prepare data: python scripts/prepare_data.py --imagenet-path /path/to/imagenet --output data/imagenet100 --num-classes 100" -ForegroundColor White
Write-Host "3. Train: python train.py --config configs/vit_small.yaml" -ForegroundColor White
