import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple
from fvcore.nn import FlopCountAnalysis, parameter_count


class ModelFactory:
    """Factory for creating lightweight vision models."""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
        """Create model from timm library or custom implementation."""
        
        model_configs = {
            'mobilenetv3_large_100': 'mobilenetv3_large_100',
            'mobilenetv3_small_100': 'mobilenetv3_small_100',
            'efficientnetv2_s': 'tf_efficientnetv2_s',
            'mobilevit_s': 'mobilevit_s',
            'mobilevit_xs': 'mobilevit_xs',
            'resnet50': 'resnet50',
            'vit_small_patch16_224': 'vit_small_patch16_224',
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(model_configs.keys())}")
        
        timm_name = model_configs[model_name]
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        
        return model
    
    @staticmethod
    def get_model_info(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> dict:
        """Calculate model statistics: params, FLOPs, memory."""
        model.eval()
        device = next(model.parameters()).device
        
        dummy_input = torch.randn(1, *input_size).to(device)
        
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        
        params = parameter_count(model)
        total_params = params['']
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
        mem_bufs = sum(b.numel() * b.element_size() for b in model.buffers())
        total_memory = (mem_params + mem_bufs) / (1024 ** 2)
        
        return {
            'params': total_params,
            'trainable_params': trainable_params,
            'flops': total_flops,
            'gflops': total_flops / 1e9,
            'memory_mb': total_memory,
        }


def create_model_with_info(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    input_size: Tuple[int, int, int] = (3, 224, 224),
) -> Tuple[nn.Module, dict]:
    """Create model and return it with statistics."""
    model = ModelFactory.create_model(model_name, num_classes, pretrained)
    info = ModelFactory.get_model_info(model, input_size)
    return model, info


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[int, float]:
    """Load model checkpoint and optionally optimizer/scheduler state."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc1', 0.0)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return start_epoch, best_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_acc1: float,
    save_path: str,
    is_best: bool = False,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc1': best_acc1,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('checkpoint', 'best_model')
        torch.save(checkpoint, best_path)
