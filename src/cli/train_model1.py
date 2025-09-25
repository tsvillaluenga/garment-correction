#!/usr/bin/env python3
"""
Training script for Model 1: Recoloring with cross-attention.
"""
import argparse
import logging
import signal
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

from data import GarmentPairDataset, create_dataloader
from models.recolor_unet import create_recolor_model
from losses_metrics import CombinedRecolorLoss, compute_color_metrics
from utils import (
    set_seed, setup_logging, CheckpointManager, AverageMeter,
    create_optimizer, create_scheduler, ProgressTracker, get_device, print_system_info,
    TrainingHistory, create_timestamped_dir, save_config_file, update_config_with_results
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Model 1: Recoloring with cross-attention")
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
    # Training dataset with degradation
    train_dataset = GarmentPairDataset(
        data_root=config['data_root'],
        split='train',
        img_size=config['img_size'],
        degrade_params=config.get('degrade', {}),
        augment=True
    )
    
    # Validation dataset without degradation
    val_degrade_params = config.get('degrade', {}).copy()
    val_degrade_params['enable'] = True  # Still apply degradation for validation
    
    val_dataset = GarmentPairDataset(
        data_root=config['data_root'],
        split='val',
        img_size=config['img_size'],
        degrade_params=val_degrade_params,
        augment=False
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
    use_amp: bool = True,
    use_gan: bool = False
):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    l1_losses = AverageMeter()
    de_losses = AverageMeter()
    perc_losses = AverageMeter()
    gan_losses = AverageMeter()
    
    progress_tracker.start_batch(len(dataloader), epoch)
    
    for batch_idx, batch in enumerate(dataloader):
        # Unpack batch
        still_ref = batch['still_ref'].to(device, non_blocking=True)
        on_model_input = batch['on_model_input'].to(device, non_blocking=True)
        on_model_target = batch['on_model_target'].to(device, non_blocking=True)
        mask_still = batch['mask_still'].to(device, non_blocking=True)
        mask_on = batch['mask_on'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast(AMP_DEVICE) if AMP_DEVICE else autocast():
                # Forward pass
                pred = model.forward_train(on_model_input, still_ref, mask_on, mask_still)
                
                # Compute losses
                fake_logits = None
                if use_gan and hasattr(model, 'discriminate'):
                    fake_logits = model.discriminate(pred)
                
                loss_dict = criterion(pred, on_model_target, mask_on, fake_logits)
                loss = loss_dict['total']
        else:
            # Forward pass
            pred = model.forward_train(on_model_input, still_ref, mask_on, mask_still)
            
            # Compute losses
            fake_logits = None
            if use_gan and hasattr(model, 'discriminate'):
                fake_logits = model.discriminate(pred)
            
            loss_dict = criterion(pred, on_model_target, mask_on, fake_logits)
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
        losses.update(loss.item(), still_ref.size(0))
        if 'l1' in loss_dict:
            l1_losses.update(loss_dict['l1'].item(), still_ref.size(0))
        if 'delta_e' in loss_dict:
            de_losses.update(loss_dict['delta_e'].item(), still_ref.size(0))
        if 'perceptual' in loss_dict:
            perc_losses.update(loss_dict['perceptual'].item(), still_ref.size(0))
        if 'gan' in loss_dict:
            gan_losses.update(loss_dict['gan'].item(), still_ref.size(0))
        
        # Update progress
        progress_tracker.update_batch(batch_idx + 1)
        
        # Log batch progress using the new method
        if batch_idx % 50 == 0:
            loss_info = (
                f"Loss: {losses.avg:.4f} L1: {l1_losses.avg:.4f} "
                f"ΔE: {de_losses.avg:.4f} Perc: {perc_losses.avg:.4f}"
            )
            if use_gan:
                loss_info += f" GAN: {gan_losses.avg:.4f}"
            progress_tracker.print_batch_update(batch_idx, len(dataloader), loss_info)
    
    progress_tracker.finish_epoch()
    
    metrics = {
        'train_loss': losses.avg,
        'train_l1': l1_losses.avg,
        'train_delta_e': de_losses.avg,
        'train_perceptual': perc_losses.avg
    }
    
    if use_gan:
        metrics['train_gan'] = gan_losses.avg
    
    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_gan: bool = False
):
    """Validate for one epoch."""
    model.eval()
    
    losses = AverageMeter()
    l1_losses = AverageMeter()
    de_losses = AverageMeter()
    perc_losses = AverageMeter()
    
    all_color_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch
            still_ref = batch['still_ref'].to(device, non_blocking=True)
            on_model_input = batch['on_model_input'].to(device, non_blocking=True)
            on_model_target = batch['on_model_target'].to(device, non_blocking=True)
            mask_still = batch['mask_still'].to(device, non_blocking=True)
            mask_on = batch['mask_on'].to(device, non_blocking=True)
            
            # Forward pass
            pred = model.forward_infer(on_model_input, still_ref, mask_on, mask_still)
            
            # Compute losses
            fake_logits = None
            if use_gan and hasattr(model, 'discriminate'):
                fake_logits = model.discriminate(pred)
            
            loss_dict = criterion(pred, on_model_target, mask_on, fake_logits)
            
            # Update loss metrics
            losses.update(loss_dict['total'].item(), still_ref.size(0))
            if 'l1' in loss_dict:
                l1_losses.update(loss_dict['l1'].item(), still_ref.size(0))
            if 'delta_e' in loss_dict:
                de_losses.update(loss_dict['delta_e'].item(), still_ref.size(0))
            if 'perceptual' in loss_dict:
                perc_losses.update(loss_dict['perceptual'].item(), still_ref.size(0))
            
            # Compute color metrics
            color_metrics = compute_color_metrics(pred, on_model_target, mask_on)
            all_color_metrics.append(color_metrics)
    
    # Average color metrics across all batches
    avg_color_metrics = {}
    if all_color_metrics:
        for key in all_color_metrics[0].keys():
            values = [m[key] for m in all_color_metrics if key in m]
            if values:
                avg_color_metrics[f'val_{key}'] = sum(values) / len(values)
    
    # Combine all metrics
    val_metrics = {
        'val_loss': losses.avg,
        'val_l1': l1_losses.avg,
        'val_delta_e': de_losses.avg,
        'val_perceptual': perc_losses.avg
    }
    
    val_metrics.update(avg_color_metrics)
    
    return val_metrics


def signal_handler(signum, frame):
    """Handle interruption signals gracefully."""
    print("\nTraining interrupted by user. Saving current state...")
    sys.exit(0)


def validate_config(config: dict) -> None:
    """Validate configuration parameters."""
    required_keys = ['data_root', 'img_size', 'train', 'model', 'loss_weights', 'save_dir']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate train parameters
    train_config = config['train']
    required_train_keys = ['batch_size', 'epochs', 'lr', 'seed']
    for key in required_train_keys:
        if key not in train_config:
            raise ValueError(f"Missing required train config key: {key}")
    
    # Validate ranges
    if config['img_size'] <= 0 or config['img_size'] > 2048:
        raise ValueError(f"Invalid img_size: {config['img_size']}")
    
    if train_config['batch_size'] <= 0:
        raise ValueError(f"Invalid batch_size: {train_config['batch_size']}")
    
    if train_config['epochs'] <= 0:
        raise ValueError(f"Invalid epochs: {train_config['epochs']}")
    
    if train_config['lr'] <= 0 or train_config['lr'] > 1:
        raise ValueError(f"Invalid learning rate: {train_config['lr']}")


def main():
    # Setup signal handlers for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    args = parse_args()
    
    # Load and validate configuration
    try:
        config = load_config(args.config)
        validate_config(config)
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Override save directory if specified
    if args.save_dir:
        config['save_dir'] = args.save_dir
    
    # Create timestamped save directory
    base_save_dir = Path(config['save_dir'])
    save_dir = create_timestamped_dir(base_save_dir, "train")
    
    # Save configuration file
    config_save_path = save_dir / "training_config.txt"
    save_config_file(config, config_save_path, "Model 1 - Recoloring")
    
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
    use_gan = config['model'].get('use_gan', False)
    
    model = create_recolor_model(
        use_gan=use_gan,
        in_channels=3,
        out_channels=3,
        base_channels=config['model']['base_channels'],
        depth=4,
        num_attn_blocks=config['model']['num_attn_blocks'],
        num_heads=config['model']['num_heads'],
        dropout=0.1
    )
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    
    # Create loss function
    loss_weights = config['loss_weights']
    criterion = CombinedRecolorLoss(
        w_l1=loss_weights['w_l1'],
        w_de=loss_weights['w_de'],
        w_perc=loss_weights['w_perc'],
        w_gan=loss_weights['w_gan']
    )
    criterion.to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_type="adam",
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type="cosine",
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_delta_e = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = checkpoint_manager.load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume
        )
        start_epoch = checkpoint['epoch'] + 1
        best_delta_e = checkpoint.get('score', float('inf'))
        logger.info(f"Resumed from epoch {checkpoint['epoch']}, best ΔE: {best_delta_e:.4f}")
    
    # Training loop
    logger.info("Starting training...")
    progress_tracker.start_epoch(config['train']['epochs'], start_epoch)
    
    for epoch in range(start_epoch, config['train']['epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, logger, progress_tracker,
            config['train']['amp'], use_gan
        )
        
        # Validate
        if epoch % config['val']['every_n_epochs'] == 0:
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, use_gan
            )
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Add to training history
            history.add_train_metrics(train_metrics)
            history.add_val_metrics(val_metrics)
            
            # Print compact epoch results
            progress_tracker.print_epoch_results(epoch, all_metrics)
            
            # Save checkpoint (lower Delta E is better)
            current_delta_e = val_metrics.get('val_delta_e76_mean', val_metrics['val_delta_e'])
            is_best = current_delta_e < best_delta_e
            
            if is_best:
                best_delta_e = current_delta_e
                progress_tracker.print_best_score(best_delta_e, "Delta E")
            
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, all_metrics,
                is_best=is_best, score=current_delta_e
            )
            
            progress_tracker.print_checkpoint_saved(checkpoint_path.split('/')[-1])
        
        # Update learning rate
        scheduler.step()
        
        # Update epoch progress
        progress_tracker.finish_epoch()
    
    # Finish training
    progress_tracker.finish()
    progress_tracker.print_info("🎉 Training completed!", "bold green")
    progress_tracker.print_info(f"🏆 Best validation Delta E: {best_delta_e:.4f}", "bold cyan")
    
    # Save final checkpoint
    final_metrics = validate_epoch(
        model, val_loader, criterion, device, use_gan
    )
    checkpoint_manager.save_checkpoint(
        model, optimizer, scheduler, scaler, config['train']['epochs'] - 1,
        final_metrics, is_best=False
    )
    
    # Generate training plots
    plot_path = save_dir / "training_curves.png"
    history_path = save_dir / "training_history.json"
    
    try:
        history.plot_curves(plot_path, "Model 1 - Recoloring Training Progress")
        history.save_history(history_path)
        progress_tracker.print_info(f"📊 Training plots saved: {plot_path}", "bold blue")
    except Exception as e:
        logger.warning(f"Failed to generate training plots: {e}")
    
    # Update config file with best results
    try:
        best_metrics = {
            "best_delta_e76": best_delta_e,
            "total_epochs_trained": config['train']['epochs'],
            "training_completed": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        update_config_with_results(config_save_path, best_metrics)
        progress_tracker.print_info(f"📄 Config updated with results: {config_save_path}", "bold blue")
    except Exception as e:
        logger.warning(f"Failed to update config with results: {e}")


if __name__ == "__main__":
    main()
