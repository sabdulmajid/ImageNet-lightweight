import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import time
from pathlib import Path
from tqdm import tqdm

from scripts.dataset import build_dataset, build_dataloader
from scripts.models import create_model_with_info, save_checkpoint, load_checkpoint
from scripts.utils import (
    AverageMeter, PowerMonitor, accuracy, save_training_log,
    set_seed, get_lr
)


def train_epoch(
    model, train_loader, criterion, optimizer, scaler, device,
    epoch, config, power_monitor=None
):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast(enabled=config['use_amp']):
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / config['gradient_accumulation_steps']
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
            if config.get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item() * config['gradient_accumulation_steps'], images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        if power_monitor:
            power_monitor.sample()
        
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'acc1': f"{top1.avg:.2f}%",
        })
    
    return losses.avg, top1.avg


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
    
    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description='Train lightweight models on ImageNet')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--data-path', type=str, default=None, help='Override data path from config')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.data_path:
        config['data_path'] = args.data_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    set_seed(config.get('seed', 42))
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Building datasets...")
    train_dataset, val_dataset, num_classes = build_dataset(
        data_path=config['data_path'],
        subset_classes=config.get('num_classes', None),
        input_size=config['input_size'],
        auto_augment=config.get('auto_augment', True),
        seed=config.get('seed', 42),
    )
    
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
    )
    
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config['batch_size'] * 2,
        num_workers=config['num_workers'],
        shuffle=False,
    )
    
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val, {num_classes} classes")
    
    print(f"Creating model: {config['model_name']}")
    model, model_info = create_model_with_info(
        config['model_name'],
        num_classes=num_classes,
        pretrained=config.get('pretrained', True),
        input_size=(3, config['input_size'], config['input_size']),
    )
    model = model.to(device)
    
    print(f"Model stats: {model_info['params']:,} params, {model_info['gflops']:.2f} GFLOPs")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config.get('min_lr', 1e-6),
    )
    
    scaler = GradScaler(enabled=config['use_amp'])
    
    start_epoch = 0
    best_acc1 = 0.0
    
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        start_epoch, best_acc1 = load_checkpoint(model, args.resume, optimizer, scheduler)
        start_epoch += 1
    
    writer = SummaryWriter(log_dir=log_dir / 'tensorboard')
    
    power_monitor = PowerMonitor(gpu_id=args.gpu)
    
    print("Starting training...")
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        power_monitor.reset()
        
        train_loss, train_acc1 = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, config, power_monitor
        )
        
        val_loss, val_acc1, val_acc5 = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        current_lr = get_lr(optimizer)
        power_stats = power_monitor.get_stats()
        
        print(f"\nEpoch {epoch} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc1:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc@1: {val_acc1:.2f}%, Val Acc@5: {val_acc5:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        if power_stats['avg_power_w'] > 0:
            print(f"  Avg Power: {power_stats['avg_power_w']:.1f}W")
        
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Acc1', train_acc1, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc1', val_acc1, epoch)
        writer.add_scalar('Val/Acc5', val_acc5, epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)
        if power_stats['avg_power_w'] > 0:
            writer.add_scalar('Power/Avg_W', power_stats['avg_power_w'], epoch)
        
        save_training_log(
            log_dir, epoch, train_loss, train_acc1, val_loss, val_acc1, val_acc5,
            current_lr, epoch_time, power_stats
        )
        
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_acc1,
            str(checkpoint_path), is_best
        )
        
        if (epoch + 1) % 10 == 0 or is_best:
            pass
        else:
            if epoch > 0:
                prev_checkpoint = checkpoint_dir / f'checkpoint_epoch_{epoch-1}.pth'
                if prev_checkpoint.exists():
                    prev_checkpoint.unlink()
    
    writer.close()
    print(f"\nTraining complete! Best Val Acc@1: {best_acc1:.2f}%")


if __name__ == '__main__':
    main()
