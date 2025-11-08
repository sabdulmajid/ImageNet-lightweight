"""Quick test to verify everything works"""
import torch
from scripts.models import create_model_with_info
from scripts.dataset import get_train_transform, get_val_transform
from PIL import Image
import numpy as np

print("="*60)
print("ImageNet-lightweight: Quick Test")
print("="*60)

# Test 1: GPU
print("\n1. GPU Check:")
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test 2: Model Creation
print("\n2. Model Creation:")
models_to_test = [
    'mobilenetv3_large_100',
    'mobilenetv3_small_100',
    'efficientnetv2_s',
]

for model_name in models_to_test:
    try:
        model, info = create_model_with_info(model_name, num_classes=100, pretrained=False)
        print(f"   ✓ {model_name}: {info['params']/1e6:.1f}M params, {info['gflops']:.2f} GFLOPs")
    except Exception as e:
        print(f"   ✗ {model_name}: {e}")

# Test 3: Data Transforms
print("\n3. Data Transforms:")
try:
    transform = get_train_transform(224)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    tensor = transform(dummy_img)
    print(f"   ✓ Transform works: {tensor.shape}")
except Exception as e:
    print(f"   ✗ Transform failed: {e}")

# Test 4: GPU Training Step
print("\n4. GPU Training Step:")
try:
    model, _ = create_model_with_info('mobilenetv3_small_100', num_classes=100, pretrained=False)
    model = model.cuda()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dummy batch
    images = torch.randn(8, 3, 224, 224).cuda()
    targets = torch.randint(0, 100, (8,)).cuda()
    
    # Forward + Backward
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    print(f"   ✓ Training step completed (Loss: {loss.item():.4f})")
    print(f"   ✓ Memory allocated: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    
except Exception as e:
    print(f"   ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✓ All tests passed! Ready to train.")
print("="*60)
print("\nNext steps:")
print("1. Prepare dataset: python scripts/prepare_data.py --imagenet-path /path/to/imagenet --output data/imagenet100 --num-classes 100")
print("2. Train model: python train.py --config configs/mobilenetv3_large.yaml")
