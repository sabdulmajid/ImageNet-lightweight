import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

from scripts.dataset import build_dataset, build_dataloader
from scripts.models import create_model_with_info, load_checkpoint
from scripts.utils import AverageMeter, accuracy, measure_inference_time


def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
    
    return top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, default=None, help='Override data path from config')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.data_path:
        config['data_path'] = args.data_path
    
    batch_size = args.batch_size if args.batch_size else config['batch_size'] * 2
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Building validation dataset...")
    _, val_dataset, num_classes = build_dataset(
        data_path=config['data_path'],
        subset_classes=config.get('num_classes', None),
        input_size=config['input_size'],
        seed=config.get('seed', 42),
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config['num_workers'],
        shuffle=False,
    )
    
    print(f"Validation set: {len(val_dataset)} images, {num_classes} classes")
    
    print(f"Creating model: {config['model_name']}")
    model, model_info = create_model_with_info(
        config['model_name'],
        num_classes=num_classes,
        pretrained=False,
        input_size=(3, config['input_size'], config['input_size']),
    )
    
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint)
    
    model = model.to(device)
    
    print("\nModel Statistics:")
    print(f"  Parameters: {model_info['params']:,}")
    print(f"  GFLOPs: {model_info['gflops']:.2f}")
    print(f"  Memory: {model_info['memory_mb']:.2f} MB")
    
    print("\nEvaluating accuracy...")
    acc1, acc5 = evaluate_model(model, val_loader, device)
    
    print("\nValidation Results:")
    print(f"  Top-1 Accuracy: {acc1:.2f}%")
    print(f"  Top-5 Accuracy: {acc5:.2f}%")
    
    print("\nMeasuring inference speed...")
    inference_stats = measure_inference_time(
        model,
        input_size=(1, 3, config['input_size'], config['input_size']),
        device=str(device),
    )
    
    print(f"  Avg inference time: {inference_stats['avg_time_ms']:.2f} ms")
    print(f"  Throughput: {inference_stats['throughput_fps']:.1f} FPS")


if __name__ == '__main__':
    main()
