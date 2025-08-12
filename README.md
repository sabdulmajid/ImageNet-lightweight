# ImageNet-lightweight

Production-ready training and benchmarking for lightweight vision models on ImageNet subsets. Optimized for single GPU (RTX 3060 12GB).

## Quick Start

**Windows (NVIDIA GPU):**
```powershell
.\setup_windows.ps1
python test_setup.py
```

**Linux/Mac (NVIDIA GPU):**
```bash
chmod +x setup_unix.sh
./setup_unix.sh
python3 test_setup.py
```

**Manual training:**
```bash
# Prepare ImageNet-100 dataset
python scripts/prepare_data.py --imagenet-path /path/to/ILSVRC2012 --output data/imagenet100 --num-classes 100

# Train
python train.py --config configs/vit_small.yaml

# Evaluate
python eval.py --config configs/vit_small.yaml --checkpoint experiments/vit_small/checkpoints/best_model.pth

# Benchmark
python benchmark.py --configs-dir configs --checkpoints-dir experiments --data-path data/imagenet100 --output results/benchmark.json
```

## Models

| Model | Params | GFLOPs | Batch | Train Time |
|-------|--------|--------|-------|------------|
| MobileNetV3-Large | 5.5M | 0.22 | 128 | 2-3h |
| MobileNetV3-Small | 2.5M | 0.06 | 192 | 1.5-2h |
| EfficientNetV2-S | 21.5M | 2.9 | 64 | 4-5h |
| MobileViT-S | 5.6M | 2.0 | 96 | 3-4h |
| MobileViT-XS | 2.3M | 1.0 | 128 | 2-3h |
| ResNet50 | 25.6M | 4.1 | 96 | 3-4h |
| **ViT-Small** | **21.7M** | **4.25** | **64** | **4-5h** |

## Configuration

All configs in `configs/*.yaml` are pre-tuned for RTX 3060 (12GB). Key settings:

- **Mixed precision**: `use_amp: true` (2x speedup)
- **Gradient accumulation**: Effective batch = `batch_size × gradient_accumulation_steps`
- **AutoAugment**: Enabled by default
- **Label smoothing**: 0.1
- **Cosine LR schedule**: 90 epochs default

Adjust `batch_size` if you have different VRAM:
- 8GB: Reduce by 50%, increase `gradient_accumulation_steps: 2`
- 16GB: Increase by 30%
- 24GB: Increase by 100%

## Expected Results (ImageNet-100)

| Model | Top-1 Acc | Top-5 Acc | Inference | Throughput |
|-------|-----------|-----------|-----------|------------|
| MobileNetV3-Large | 75-80% | 92-95% | 4.5ms | 220 FPS |
| MobileNetV3-Small | 70-75% | 88-92% | 3.0ms | 330 FPS |
| EfficientNetV2-S | 78-83% | 94-96% | 13ms | 77 FPS |
| MobileViT-S | 76-81% | 93-95% | 11ms | 91 FPS |
| MobileViT-XS | 72-77% | 90-93% | 7ms | 143 FPS |
| ResNet50 | 77-82% | 93-96% | 8ms | 125 FPS |
| **ViT-Small** | **79-84%** | **94-97%** | **15ms** | **67 FPS** |

## Case Study: Vision Transformer (ViT-Small)

**Hardware:** NVIDIA GeForce RTX 3060 (12GB)  
**Test:** Mixed precision training with gradient accumulation

### Memory & Speed Analysis (Batch Size Scaling)
| Batch Size | Memory Usage | Time/Step | Status |
|------------|--------------|-----------|--------|
| 32 | 1,402 MB | 88 ms | ✓ |
| 48 | 1,940 MB | 118 ms | ✓ |
| **64** | **2,466 MB** | **150 ms** | **✓ Optimal** |
| 80 | 2,999 MB | 181 ms | ✓ |
| 96 | 3,536 MB | 213 ms | ✓ |

**Findings:**
- **Optimal configuration**: Batch size 64 with gradient accumulation × 2 = effective batch 128
- **Linear scaling**: Memory and time scale predictably with batch size
- **Headroom**: Only 3.5GB at batch 96, leaving 8.5GB for larger batches if needed
- **Throughput**: ~426 images/sec at batch 64 (150ms/step)
- **Efficiency**: ViT benefits from attention optimization in PyTorch 2.7.1+cu118

**Why ViT for ImageNet-100?**
1. **Strong accuracy**: Transformers excel at learning visual representations (79-84% expected)
2. **Reasonable compute**: 4.25 GFLOPs is manageable on consumer GPUs
3. **Research relevance**: ViT is the foundation for modern vision models (CLIP, DINO, etc.)
4. **Scalability**: Same architecture scales from tiny to huge models

**Recommended for researchers who want:**
- State-of-art architectures on limited compute
- Baseline for comparing novel transformer designs
- Understanding attention-based vision models

## Training Features

- Mixed precision (AMP)
- Gradient accumulation
- Automatic checkpointing (best + periodic)
- Resume training support
- TensorBoard logging
- GPU power monitoring
- Progress bars
- Comprehensive metrics (accuracy, loss, speed, power)

## Project Structure

```
ImageNet-lightweight/
├── configs/           # Model configs (YAML)
├── scripts/
│   ├── dataset.py    # Data loading & augmentation
│   ├── models.py     # Model factory & checkpointing
│   ├── utils.py      # Metrics, logging, power monitoring
│   └── prepare_data.py
├── train.py          # Training script
├── eval.py           # Evaluation script
├── benchmark.py      # Batch benchmarking
├── plot.py           # Visualization
└── requirements.txt
```

## Monitor Training

```powershell
# TensorBoard
tensorboard --logdir experiments/mobilenetv3_large/logs/tensorboard

# Watch GPU
nvidia-smi -l 1

# View logs
Get-Content experiments/mobilenetv3_large/logs/training_log.jsonl -Tail 10 -Wait
```

## Resume Training

```powershell
python train.py --config configs/mobilenetv3_large.yaml --resume experiments/mobilenetv3_large/checkpoints/checkpoint_epoch_50.pth
```

## Train All Models

```powershell
.\train_all.ps1
```

## Troubleshooting

**Out of Memory**: Reduce `batch_size` in config, increase `gradient_accumulation_steps`

**Slow training**: Enable `use_amp: true`, increase `num_workers`

**Poor accuracy**: Use `pretrained: true`, train longer, verify dataset

**GPU not detected**: Check CUDA installation with `nvidia-smi`

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA 11.8+
- NVIDIA GPU (12GB+ recommended)
- 16GB+ RAM
- 50GB+ storage for ImageNet-100

## License

MIT
