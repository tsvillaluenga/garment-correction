#!/usr/bin/env python3
"""
Training script for Model 3: Segmentation for on-model images.
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE = None

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data import GarmentSegDataset, create_dataloader
from models.seg_unet import create_seg_model, EnhancedSegmentationUNet
from losses_metrics import CombinedSegLoss, AdvancedSegLoss, compute_segmentation_metrics
from utils import (
    set_seed, setup_logging, CheckpointManager, AverageMeter,
    create_optimizer, create_scheduler, ProgressTracker, get_device, print_system_info,
    TrainingHistory, create_timestamped_dir, save_config_file, update_config_with_results,
    EarlyStopping
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Model 3: Segmentation for on-model images")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--save_dir", type=str, help="Override save directory from config")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict):
    """Create training and validation datasets."""
    # Training dataset
    train_dataset = GarmentSegDataset(
        data_root=config['data_root'],
        split='train',
        img_size=config['img_size'],
        target_type='on_model',  # Different from model 2
        augment_params=config.get('augment', {})
    )
    
    # Validation dataset
    val_dataset = GarmentSegDataset(
        data_root=config['data_root'],
        split='val',
        img_size=config['img_size'],
        target_type='on_model',  # Different from model 2
        augment_params=None  # No augmentation for validation
    )
    
    return train_dataset, val_dataset


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    logger,
    progress_tracker: ProgressTracker,
    use_amp: bool = True
):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    bce_losses = AverageMeter()
    dice_losses = AverageMeter()
    
    progress_tracker.start_batch(len(dataloader), epoch)
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(AMP_DEVICE) if AMP_DEVICE else autocast():
                logits = model(images)
                loss_dict = criterion(logits, masks)
                loss = loss_dict['total']
        else:
            logits = model(images)
            loss_dict = criterion(logits, masks)
            loss = loss_dict['total']
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), images.size(0))
        bce_losses.update(loss_dict['bce'].item(), images.size(0))
        dice_losses.update(loss_dict['dice'].item(), images.size(0))
        
        # Update progress
        progress_tracker.update_batch(batch_idx + 1)
        
        # Log batch progress using the new method
        if batch_idx % 50 == 0:
            loss_info = f"Loss: {losses.avg:.4f} BCE: {bce_losses.avg:.4f} Dice: {dice_losses.avg:.4f}"
            progress_tracker.print_batch_update(batch_idx, len(dataloader), loss_info)
    
    progress_tracker.finish_epoch()
    
    return {
        'train_loss': losses.avg,
        'train_bce': bce_losses.avg,
        'train_dice': dice_losses.avg
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
):
    """Validate for one epoch."""
    model.eval()
    
    losses = AverageMeter()
    bce_losses = AverageMeter()
    dice_losses = AverageMeter()
    
    all_metrics = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            logits = model(images)
            loss_dict = criterion(logits, masks)
            
            # Update loss metrics
            losses.update(loss_dict['total'].item(), images.size(0))
            bce_losses.update(loss_dict['bce'].item(), images.size(0))
            dice_losses.update(loss_dict['dice'].item(), images.size(0))
            
            # Compute segmentation metrics
            seg_metrics = compute_segmentation_metrics(logits, masks, threshold)
            all_metrics.append(seg_metrics)
    
    # Average metrics across all batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[f'val_{key}'] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    avg_metrics.update({
        'val_loss': losses.avg,
        'val_bce': bce_losses.avg,
        'val_dice': dice_losses.avg
    })
    
    return avg_metrics


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override save directory if specified
    if args.save_dir:
        config['save_dir'] = args.save_dir
    
    # Create timestamped save directory
    base_save_dir = Path(config['save_dir'])
    save_dir = create_timestamped_dir(base_save_dir, "train")
    
    # Save configuration file
    config_save_path = save_dir / "training_config.txt"
    save_config_file(config, config_save_path, "Model 3 - On-Model Segmentation")
    
    # Setup logging
    logger = setup_logging(
        log_file=save_dir / "train.log",
        level=logging.DEBUG if args.debug else logging.INFO
    )
    
    # Print system info
    print_system_info()
    
    # Set seed
    set_seed(config['train']['seed'])
    logger.info(f"Set random seed to {config['train']['seed']}")
    
    # Get device
    device = get_device() if args.device is None else torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers']
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers']
    )
    
    # Create model
    logger.info("Creating model...")
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'basic')
    
    if model_type == 'enhanced':
        model = EnhancedSegmentationUNet(
            in_channels=3,
            out_channels=1,
            base_channels=model_config.get('base_channels', 96),
            depth=4,
            use_attention=model_config.get('use_attention', True),
            dropout=model_config.get('dropout', 0.2)
        )
    else:
        model = create_seg_model(
            model_type="basic",
            in_channels=3,
            base_channels=model_config.get('base_channels', 64),
            depth=4,
            dropout=model_config.get('dropout', 0.1)
        )
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    
    # Create loss function
    loss_weights = config.get('loss_weights', {})
    if model_type == 'enhanced':
        criterion = AdvancedSegLoss(
            bce_weight=loss_weights.get('bce_weight', 0.3),
            dice_weight=loss_weights.get('dice_weight', 0.3),
            focal_weight=loss_weights.get('focal_weight', 0.2),
            tversky_weight=loss_weights.get('tversky_weight', 0.1),
            boundary_weight=loss_weights.get('boundary_weight', 0.1)
        )
    else:
        criterion = CombinedSegLoss(bce_weight=1.0, dice_weight=1.0)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_type="adam",
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    scheduler_config = config.get('scheduler', {'type': 'cosine_annealing'})
    scheduler = create_scheduler(
        optimizer,
        scheduler_config,
        epochs=config['train']['epochs']
    )
    
    # AMP scaler
    if config['train']['amp']:
        scaler = GradScaler(AMP_DEVICE) if AMP_DEVICE else GradScaler()
    else:
        scaler = None
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(save_dir)
    
    # Progress tracker
    progress_tracker = ProgressTracker()
    
    # Training history tracker
    history = TrainingHistory()
    
    # Initialize early stopping
    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        early_stop_config = config['early_stopping']
        early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.0),
            mode=early_stop_config.get('mode', 'max'),
            restore_best_weights=early_stop_config.get('restore_best_weights', True)
        )
        progress_tracker.print_info(f"🛑 Early stopping enabled: patience={early_stopping.patience}, mode={early_stopping.mode}", "yellow")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_iou = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = checkpoint_manager.load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume
        )
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('score', 0.0)
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, best IoU: {best_iou:.4f}")
    
    # Training loop
    logger.info("Starting training...")
    progress_tracker.start_epoch(config['train']['epochs'], start_epoch)
    
    for epoch in range(start_epoch, config['train']['epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, progress_tracker, config['train']['amp']
        )
        
        # Validate
        if epoch % config['val']['every_n_epochs'] == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, config['seg']['threshold']
            )
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Add to training history
            history.add_train_metrics(train_metrics)
            history.add_val_metrics(val_metrics)
            
            # Print compact epoch results
            progress_tracker.print_epoch_results(epoch, all_metrics)
            
            # Save checkpoint
            current_iou = val_metrics['val_iou']
            is_best = current_iou > best_iou
            
            if is_best:
                best_iou = current_iou
                progress_tracker.print_best_score(best_iou, "IoU")
            
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, all_metrics,
                is_best=is_best, score=current_iou
            )
            
            progress_tracker.print_checkpoint_saved(checkpoint_path.split('/')[-1])
            
            # Check early stopping
            if early_stopping is not None:
                if early_stopping(current_iou, epoch):
                    early_info = early_stopping.get_info()
                    progress_tracker.print_info(
                        f"🛑 Early stopping triggered! Best IoU: {early_info['best_score']:.4f} at epoch {early_info['best_epoch']}", 
                        "bold red"
                    )
                    break
        
        # Update learning rate
        scheduler_type = scheduler_config.get('type', 'cosine_annealing').lower()
        if scheduler_type == "reduce_on_plateau":
            scheduler.step(1.0 - current_iou)  # Pass (1 - IoU) as loss metric for plateau scheduler
        else:
            scheduler.step()  # Regular step for other schedulers
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        progress_tracker.print_info(f"📈 Learning Rate: {current_lr:.2e}", "dim white")
        
        # Update epoch progress
        progress_tracker.finish_epoch()
    
    # Finish training
    progress_tracker.finish()
    
    # Print training completion info
    if early_stopping is not None and early_stopping.early_stop:
        early_info = early_stopping.get_info()
        progress_tracker.print_info("🛑 Training stopped early!", "bold red")
        progress_tracker.print_info(f"🏆 Best validation IoU: {early_info['best_score']:.4f} at epoch {early_info['best_epoch']}", "bold cyan")
    else:
        progress_tracker.print_info("🎉 Training completed!", "bold green")
        progress_tracker.print_info(f"🏆 Best validation IoU: {best_iou:.4f}", "bold cyan")
    
    # Save final checkpoint
    final_metrics = validate_epoch(
        model, val_loader, criterion, device, config['seg']['threshold']
    )
    checkpoint_manager.save_checkpoint(
        model, optimizer, scheduler, scaler, config['train']['epochs'] - 1,
        final_metrics, is_best=False
    )
    
    # Generate training plots
    plot_path = save_dir / "training_curves.png"
    history_path = save_dir / "training_history.json"
    
    try:
        history.plot_curves(plot_path, "Model 3 - On-Model Segmentation Training Progress")
        history.save_history(history_path)
        progress_tracker.print_info(f"📊 Training plots saved: {plot_path}", "bold blue")
    except Exception as e:
        logger.warning(f"Failed to generate training plots: {e}")
    
    # Update config file with best results
    try:
        best_metrics = {
            "best_iou": best_iou,
            "best_dice": best_dice,
            "total_epochs_trained": config['train']['epochs'],
            "training_completed": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        update_config_with_results(config_save_path, best_metrics)
        progress_tracker.print_info(f"📄 Config updated with results: {config_save_path}", "bold blue")
    except Exception as e:
        logger.warning(f"Failed to update config with results: {e}")


if __name__ == "__main__":
    main()
