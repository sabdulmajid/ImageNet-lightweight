"""ViT Case Study - Quick benchmark test"""
import torch
from scripts.models import create_model_with_info
from torch.cuda.amp import autocast, GradScaler
import time

print("="*60)
print("Vision Transformer Small - Case Study")
print("="*60)

# Setup
device = torch.device('cuda')
model, info = create_model_with_info('vit_small_patch16_224', num_classes=100, pretrained=False)
model = model.to(device)
model.train()

print(f"\nModel: vit_small_patch16_224")
print(f"Parameters: {info['params']/1e6:.2f}M")
print(f"GFLOPs: {info['gflops']:.2f}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Benchmark training step with different batch sizes
print(f"\n{'Batch Size':<12} {'Memory (MB)':<15} {'Time/Step (ms)':<18} {'Status'}")
print("-" * 60)

for batch_size in [32, 48, 64, 80, 96]:
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = GradScaler()
        
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        targets = torch.randint(0, 100, (batch_size,)).to(device)
        
        # Warmup
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        # Timed run
        start = time.time()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"{batch_size:<12} {memory_mb:<15.0f} {elapsed:<18.1f} ✓")
        torch.cuda.reset_peak_memory_stats()
        
        del images, targets, outputs, loss
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"{batch_size:<12} {'OOM':<15} {'-':<18} ✗")
            torch.cuda.empty_cache()
            break
        else:
            raise

print("\n" + "="*60)
print("Recommendation: batch_size=64 with gradient_accumulation=2")
print("Effective batch size: 128 (optimal for ViT training)")
print("="*60)
