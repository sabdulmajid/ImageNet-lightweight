import torch
import argparse
import yaml
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scripts.dataset import build_dataset, build_dataloader
from scripts.models import create_model_with_info, load_checkpoint
from scripts.utils import measure_inference_time, PowerMonitor, AverageMeter, accuracy


def benchmark_model(config_path, checkpoint_path, data_path, device, output_dir):
    """Benchmark a single model."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if data_path:
        config['data_path'] = data_path
    
    _, val_dataset, num_classes = build_dataset(
        data_path=config['data_path'],
        subset_classes=config.get('num_classes', None),
        input_size=config['input_size'],
        seed=config.get('seed', 42),
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        num_workers=config['num_workers'],
        shuffle=False,
    )
    
    model, model_info = create_model_with_info(
        config['model_name'],
        num_classes=num_classes,
        pretrained=False,
        input_size=(3, config['input_size'], config['input_size']),
    )
    
    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model = model.to(device)
    model.eval()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    power_monitor = PowerMonitor(gpu_id=device.index if device.type == 'cuda' else 0)
    power_monitor.reset()
    
    print(f"Evaluating {config['model_name']}...")
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            power_monitor.sample()
    
    inference_stats = measure_inference_time(
        model,
        input_size=(1, 3, config['input_size'], config['input_size']),
        device=str(device),
    )
    
    power_stats = power_monitor.get_stats()
    
    results = {
        'model_name': config['model_name'],
        'params': model_info['params'],
        'gflops': model_info['gflops'],
        'memory_mb': model_info['memory_mb'],
        'top1_acc': top1.avg,
        'top5_acc': top5.avg,
        'inference_ms': inference_stats['avg_time_ms'],
        'throughput_fps': inference_stats['throughput_fps'],
        'avg_power_w': power_stats['avg_power_w'],
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark all models')
    parser.add_argument('--configs-dir', type=str, default='configs', help='Directory with config files')
    parser.add_argument('--checkpoints-dir', type=str, default='experiments', help='Directory with checkpoints')
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet data')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json', help='Output file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    configs_dir = Path(args.configs_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_files = sorted(configs_dir.glob('*.yaml'))
    
    all_results = []
    
    for config_file in config_files:
        model_name = config_file.stem
        checkpoint_path = checkpoints_dir / model_name / 'checkpoints' / 'best_model.pth'
        
        try:
            results = benchmark_model(
                str(config_file),
                str(checkpoint_path),
                args.data_path,
                device,
                output_path.parent,
            )
            all_results.append(results)
            
            print(f"\n{results['model_name']} Results:")
            print(f"  Params: {results['params']:,}")
            print(f"  GFLOPs: {results['gflops']:.2f}")
            print(f"  Top-1: {results['top1_acc']:.2f}%")
            print(f"  Top-5: {results['top5_acc']:.2f}%")
            print(f"  Inference: {results['inference_ms']:.2f} ms")
            print(f"  Throughput: {results['throughput_fps']:.1f} FPS")
            if results['avg_power_w'] > 0:
                print(f"  Avg Power: {results['avg_power_w']:.1f} W")
        
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    df = pd.DataFrame(all_results)
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved to {output_path} and {csv_path}")
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
