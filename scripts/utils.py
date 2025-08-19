import torch
import torch.nn as nn
import time
import json
from pathlib import Path
from typing import Dict, Optional, List
from collections import defaultdict
import numpy as np

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Energy monitoring disabled.")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """Track multiple metrics over training."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get(self, key: str) -> List:
        return self.metrics[key]
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            self.metrics = defaultdict(list, json.load(f))


class PowerMonitor:
    """Monitor GPU power consumption using nvidia-smi."""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.power_readings = []
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self.enabled = True
            except Exception as e:
                print(f"Warning: Failed to initialize NVML: {e}")
                self.enabled = False
        else:
            self.enabled = False
    
    def sample(self):
        """Sample current power draw in watts."""
        if not self.enabled:
            return None
        
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_w = power_mw / 1000.0
            self.power_readings.append(power_w)
            return power_w
        except Exception as e:
            print(f"Warning: Failed to read power: {e}")
            return None
    
    def get_stats(self) -> Dict[str, float]:
        """Get power consumption statistics."""
        if not self.power_readings:
            return {'avg_power_w': 0, 'max_power_w': 0, 'min_power_w': 0}
        
        return {
            'avg_power_w': np.mean(self.power_readings),
            'max_power_w': np.max(self.power_readings),
            'min_power_w': np.min(self.power_readings),
        }
    
    def reset(self):
        """Reset power readings."""
        self.power_readings = []
    
    def __del__(self):
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.Tensor]:
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def measure_inference_time(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Measure model inference time with warmup."""
    model.eval()
    model = model.to(device)
    
    dummy_input = torch.randn(*input_size, device=device)
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = 1.0 / avg_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'throughput_fps': throughput,
    }


def save_training_log(
    log_dir: str,
    epoch: int,
    train_loss: float,
    train_acc1: float,
    val_loss: float,
    val_acc1: float,
    val_acc5: float,
    lr: float,
    epoch_time: float,
    power_stats: Optional[Dict[str, float]] = None,
):
    """Save training log to JSON file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / 'training_log.jsonl'
    
    log_entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'train_acc1': train_acc1,
        'val_loss': val_loss,
        'val_acc1': val_acc1,
        'val_acc5': val_acc5,
        'lr': lr,
        'epoch_time': epoch_time,
    }
    
    if power_stats:
        log_entry.update(power_stats)
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def load_training_log(log_dir: str) -> List[Dict]:
    """Load training log from JSON file."""
    log_file = Path(log_dir) / 'training_log.jsonl'
    
    if not log_file.exists():
        return []
    
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    
    return logs


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0
