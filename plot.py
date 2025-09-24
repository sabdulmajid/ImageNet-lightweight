import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import argparse


sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_training_curves(log_file, output_dir):
    """Plot training and validation curves from log file."""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    
    df = pd.DataFrame(logs)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df['epoch'], df['train_acc1'], label='Train Top-1', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_acc1'], label='Val Top-1', linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_acc5'], label='Val Top-5', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['epoch'], df['lr'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    if 'avg_power_w' in df.columns and df['avg_power_w'].max() > 0:
        axes[1, 1].plot(df['epoch'], df['avg_power_w'], linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Power (W)')
        axes[1, 1].set_title('Average GPU Power Consumption')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].plot(df['epoch'], df['epoch_time'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].set_title('Time per Epoch')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_file, output_dir):
    """Plot comparison charts for multiple models."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    df = df.sort_values('top1_acc', ascending=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    
    axes[0, 0].barh(df['model_name'], df['top1_acc'])
    axes[0, 0].set_xlabel('Top-1 Accuracy (%)')
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    axes[0, 1].barh(df['model_name'], df['params'] / 1e6)
    axes[0, 1].set_xlabel('Parameters (M)')
    axes[0, 1].set_title('Model Size')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    axes[0, 2].barh(df['model_name'], df['gflops'])
    axes[0, 2].set_xlabel('GFLOPs')
    axes[0, 2].set_title('Computational Complexity')
    axes[0, 2].grid(True, alpha=0.3, axis='x')
    
    axes[1, 0].barh(df['model_name'], df['inference_ms'])
    axes[1, 0].set_xlabel('Inference Time (ms)')
    axes[1, 0].set_title('Inference Speed')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    axes[1, 1].barh(df['model_name'], df['throughput_fps'])
    axes[1, 1].set_xlabel('Throughput (FPS)')
    axes[1, 1].set_title('Processing Throughput')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    if 'avg_power_w' in df.columns and df['avg_power_w'].max() > 0:
        axes[1, 2].barh(df['model_name'], df['avg_power_w'])
        axes[1, 2].set_xlabel('Average Power (W)')
        axes[1, 2].set_title('Power Consumption')
        axes[1, 2].grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 2].barh(df['model_name'], df['memory_mb'])
        axes[1, 2].set_xlabel('Memory (MB)')
        axes[1, 2].set_title('Memory Usage')
        axes[1, 2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_efficiency_scatter(results_file, output_dir):
    """Plot efficiency scatter plots."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Efficiency Analysis', fontsize=16, fontweight='bold')
    
    scatter1 = axes[0].scatter(df['gflops'], df['top1_acc'], 
                               s=df['params']/1e5, alpha=0.6)
    axes[0].set_xlabel('GFLOPs')
    axes[0].set_ylabel('Top-1 Accuracy (%)')
    axes[0].set_title('Accuracy vs Complexity (bubble size = params)')
    axes[0].grid(True, alpha=0.3)
    
    for idx, row in df.iterrows():
        axes[0].annotate(row['model_name'], 
                        (row['gflops'], row['top1_acc']),
                        fontsize=8, alpha=0.7)
    
    scatter2 = axes[1].scatter(df['inference_ms'], df['top1_acc'],
                               s=df['params']/1e5, alpha=0.6)
    axes[1].set_xlabel('Inference Time (ms)')
    axes[1].set_ylabel('Top-1 Accuracy (%)')
    axes[1].set_title('Accuracy vs Speed (bubble size = params)')
    axes[1].grid(True, alpha=0.3)
    
    for idx, row in df.iterrows():
        axes[1].annotate(row['model_name'],
                        (row['inference_ms'], row['top1_acc']),
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate plots from results')
    parser.add_argument('--results', type=str, help='Path to benchmark results JSON')
    parser.add_argument('--logs', type=str, help='Path to training log directory')
    parser.add_argument('--output', type=str, default='plots', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.logs:
        log_file = Path(args.logs) / 'training_log.jsonl'
        if log_file.exists():
            print(f"Plotting training curves from {log_file}...")
            plot_training_curves(log_file, output_dir)
            print(f"  Saved to {output_dir / 'training_curves.png'}")
    
    if args.results:
        results_file = Path(args.results)
        if results_file.exists():
            print(f"Plotting model comparison from {results_file}...")
            plot_model_comparison(results_file, output_dir)
            print(f"  Saved to {output_dir / 'model_comparison.png'}")
            
            print("Plotting efficiency analysis...")
            plot_efficiency_scatter(results_file, output_dir)
            print(f"  Saved to {output_dir / 'efficiency_scatter.png'}")
    
    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()
