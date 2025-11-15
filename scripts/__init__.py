"""
ImageNet-lightweight: One-GPU Benchmarking Suite

A production-ready toolkit for training and benchmarking lightweight vision models
on ImageNet subsets using consumer hardware.
"""

__version__ = "1.0.0"
__author__ = ""
__license__ = "MIT"

from scripts.dataset import build_dataset, build_dataloader
from scripts.models import create_model_with_info, ModelFactory
from scripts.utils import (
    AverageMeter,
    MetricTracker,
    PowerMonitor,
    accuracy,
    measure_inference_time,
    set_seed,
)

__all__ = [
    'build_dataset',
    'build_dataloader',
    'create_model_with_info',
    'ModelFactory',
    'AverageMeter',
    'MetricTracker',
    'PowerMonitor',
    'accuracy',
    'measure_inference_time',
    'set_seed',
]
