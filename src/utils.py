"""
Utility functions for seeding, AMP, checkpoints, logging, and visualization.
"""
import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import cv2
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, ProgressColumn
from rich.logging import RichHandler
from rich.text import Text
import time
from datetime import datetime


class PercentageColumn(TextColumn):
    """Custom percentage column for rich progress."""
    
    def __init__(self):
        super().__init__("{task.percentage:>5.1f}%")
    
    def render(self, task):
        """Render percentage as text."""
        if task.total:
            percent = (task.completed / task.total) * 100
            return f"{percent:>5.1f}%"
        return "  0.0%"


class CustomTimeRemainingColumn(ProgressColumn):
    """Custom time remaining column that actually works."""
    
    def render(self, task):
        """Render time remaining."""
        if not task.total or task.completed == 0:
            return Text("--:--:--", style="dim")
        
        # Calculate time remaining
        elapsed = task.finished_time - task.start_time if task.finished_time else time.time() - task.start_time
        rate = task.completed / elapsed if elapsed > 0 else 0
        
        if rate > 0:
            remaining = (task.total - task.completed) / rate
            hours = int(remaining // 3600)
            minutes = int((remaining % 3600) // 60)
            seconds = int(remaining % 60)
            return Text(f"{hours:02d}:{minutes:02d}:{seconds:02d}", style="dim")
        
        return Text("--:--:--", style="dim")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging with rich formatting."""
    # Create logger
    logger = logging.getLogger("garment_correction")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Rich console handler
    console = Console()
    rich_handler = RichHandler(console=console, show_time=True, show_path=False)
    rich_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    )
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(self, save_dir: Union[str, Path], keep_best: int = 3, keep_last: int = 2):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_last = keep_last
        self.best_scores = []  # List of (score, path) tuples
        self.last_paths = []   # List of last checkpoint paths
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[GradScaler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        score: Optional[float] = None
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            scaler: AMP scaler
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
            score: Score for ranking (lower is better)
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'metrics': metrics,
            'score': score
        }
        
        # Save only best.pth and latest.pth (no individual epoch files)
        latest_path = self.save_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.save_dir / "best.pth"
            torch.save(checkpoint, best_path)
            checkpoint_path = str(best_path)
        else:
            checkpoint_path = str(latest_path)
        
        return str(checkpoint_path)
    
    def _manage_best_checkpoints(self, new_path: Path, score: float):
        """Manage best checkpoint files."""
        self.best_scores.append((score, new_path))
        self.best_scores.sort(key=lambda x: x[0])  # Sort by score (lower is better)
        
        # Remove excess checkpoints
        while len(self.best_scores) > self.keep_best:
            _, old_path = self.best_scores.pop()
            if old_path.exists():
                old_path.unlink()
    
    def _manage_last_checkpoints(self, new_path: Path):
        """Manage last checkpoint files."""
        self.last_paths.append(new_path)
        
        # Remove excess checkpoints
        while len(self.last_paths) > self.keep_last:
            old_path = self.last_paths.pop(0)
            if old_path.exists():
                old_path.unlink()
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            scheduler: Learning rate scheduler (optional)
            scaler: AMP scaler (optional)
            checkpoint_path: Specific checkpoint path (optional)
            load_best: Whether to load best.pth if checkpoint_path not specified
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.save_dir / ("best.pth" if load_best else "last.pth")
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer."""
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 100,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    if scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, **kwargs)
    elif scheduler_type.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type.lower() == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_type.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def safe_clamp(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """Safely clamp tensor values."""
    return torch.clamp(tensor, min=min_val, max=max_val)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image for visualization."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch element
    
    if tensor.dim() == 3:
        image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        image = tensor.detach().cpu().numpy()
    
    return np.clip(image, 0, 1)


def create_visualization_grid(
    images: List[np.ndarray],
    titles: List[str],
    figsize: tuple = (15, 5),
    save_path: Optional[str] = None
) -> None:
    """Create a grid visualization of images."""
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3:
            axes[i].imshow(img)
        else:
            axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


class ProgressTracker:
    """Track training progress with rich progress bars."""
    
    def __init__(self):
        self.console = Console()
        self.progress = None
        self.main_task = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.start_time = None
        self._last_update_time = 0
        self._batch_updates = []  # Store batch updates to print separately
    
    def start_epoch(self, total_epochs: int, current_epoch: int):
        """Start epoch progress tracking."""
        self.total_epochs = total_epochs
        self.current_epoch = current_epoch
        
        if self.progress is None:
            # Create custom progress bar with thicker bar and percentage
            self.progress = Progress(
                TextColumn("[bold white]Training Progress", justify="left"),
                BarColumn(bar_width=60, style="white", complete_style="bright_green"),
                PercentageColumn(),
                TextColumn("•"),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                CustomTimeRemainingColumn(),
                console=self.console,
                refresh_per_second=1
            )
            self.progress.start()
            self.start_time = time.time()
            
            # Create main task
            self.main_task = self.progress.add_task(
                "Training", 
                total=total_epochs,
                completed=current_epoch
            )
        else:
            # Update existing task
            self.progress.update(self.main_task, completed=current_epoch)
    
    def start_batch(self, total_batches: int, epoch: int):
        """Start batch progress tracking."""
        self.current_epoch = epoch
        # Clear previous batch updates
        self._batch_updates = []
    
    def update_batch(self, completed_batches: int):
        """Update batch progress."""
        # Store batch update for later printing
        current_time = time.time()
        if current_time - self._last_update_time > 0.5:  # Update every 0.5 seconds max
            self._last_update_time = current_time
            self._batch_updates.append((completed_batches, current_time))
    
    def print_batch_update(self, batch_idx: int, total_batches: int, loss_info: str):
        """Print batch update on a separate line below the progress bar."""
        # Clear the line and print batch update
        self.console.print(f"[dim]Epoch {self.current_epoch:03d} [{batch_idx:04d}/{total_batches:04d}] {loss_info}[/dim]")
    
    def finish_epoch(self):
        """Finish epoch progress."""
        # Update main progress
        if self.main_task is not None:
            self.progress.update(self.main_task, completed=self.current_epoch + 1)
    
    def finish(self):
        """Finish all progress tracking."""
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.main_task = None
    
    def print_info(self, message: str, style: str = ""):
        """Print info message below the progress bar."""
        if style:
            self.console.print(f"[{style}]{message}[/{style}]")
        else:
            self.console.print(message)
    
    def print_best_score(self, score: float, score_type: str = "IoU"):
        """Print new best score with highlighting."""
        if score_type.lower() in ["iou", "accuracy"]:
            self.console.print(f"[bold green]🎉 New best {score_type}: {score:.4f}[/bold green]")
        else:  # For losses (lower is better)
            self.console.print(f"[bold green]🎉 New best {score_type}: {score:.4f}[/bold green]")
    
    def print_checkpoint_saved(self, path: str):
        """Print checkpoint saved message."""
        self.console.print(f"[dim]💾 Saved checkpoint: {path}[/dim]")
    
    def print_epoch_results(self, epoch: int, metrics: Dict[str, float]):
        """Print epoch results in a compact format."""
        # Create a compact metrics display
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        
        # Different metrics for different models
        if 'val_iou' in metrics:  # Segmentation models
            main_metric = f"IoU: {metrics['val_iou']:.4f}"
            color = "bright_cyan"
        elif 'val_delta_e' in metrics:  # Recoloring model
            main_metric = f"ΔE: {metrics['val_delta_e']:.4f}"
            color = "bright_magenta"
        else:
            main_metric = f"Loss: {val_loss:.4f}"
            color = "bright_yellow"
        
        # Print compact epoch summary
        self.console.print(
            f"[bold white]Epoch {epoch:03d}[/bold white] "
            f"[dim]│[/dim] "
            f"[red]Loss: {train_loss:.4f}[/red] "
            f"[dim]│[/dim] "
            f"[{color}]{main_metric}[/{color}]"
        )
    
    def print_metrics_table(self, metrics: Dict[str, float], title: str = "Metrics"):
        """Print detailed metrics table (used less frequently)."""
        # Only show this for final results or important milestones
        table = Table(title=f"[bold]{title}[/bold]", show_header=True, header_style="bold blue")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="magenta", justify="right", width=12)
        
        # Sort metrics for better display
        sorted_metrics = sorted(metrics.items())
        
        for key, value in sorted_metrics:
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        self.console.print(table)
        self.console.print()  # Add spacing


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(train_history: Dict[str, List[float]], 
                         val_history: Dict[str, List[float]], 
                         save_path: Path,
                         title: str = "Training Progress"):
    """
    Plot training and validation curves.
    
    Args:
        train_history: Dictionary with training metrics history
        val_history: Dictionary with validation metrics history
        save_path: Path to save the plot
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if hasattr(plt.style, 'available') and 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    sns.set_palette("husl")
    
    # Get all unique metrics
    all_metrics = set(train_history.keys()) | set(val_history.keys())
    
    # Filter out non-plottable metrics
    plottable_metrics = [m for m in all_metrics if m not in ['epoch']]
    
    # Determine subplot layout
    n_metrics = len(plottable_metrics)
    if n_metrics == 0:
        return
    
    # Calculate optimal subplot layout
    if n_metrics <= 2:
        rows, cols = 1, n_metrics
        figsize = (6 * cols, 5)
    elif n_metrics <= 4:
        rows, cols = 2, 2
        figsize = (12, 10)
    elif n_metrics <= 6:
        rows, cols = 2, 3
        figsize = (18, 10)
    else:
        rows, cols = 3, 3
        figsize = (18, 15)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single subplot case
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(plottable_metrics):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get epochs (assuming they're the same for train and val)
        epochs = list(range(1, len(train_history.get(metric, [])) + 1))
        
        # Plot training curve
        if metric in train_history and train_history[metric]:
            ax.plot(epochs, train_history[metric], 
                   label=f'Training {metric}', linewidth=2, marker='o', markersize=3)
        
        # Plot validation curve
        if metric in val_history and val_history[metric]:
            val_epochs = list(range(1, len(val_history[metric]) + 1))
            ax.plot(val_epochs, val_history[metric], 
                   label=f'Validation {metric}', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 for loss metrics
        if 'loss' in metric.lower():
            ax.set_ylim(bottom=0)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Training curves saved: {save_path}")


class TrainingHistory:
    """Class to track training history and generate plots."""
    
    def __init__(self):
        self.train_history = {}
        self.val_history = {}
    
    def add_train_metrics(self, metrics: Dict[str, float]):
        """Add training metrics for current epoch."""
        for key, value in metrics.items():
            if key not in self.train_history:
                self.train_history[key] = []
            self.train_history[key].append(value)
    
    def add_val_metrics(self, metrics: Dict[str, float]):
        """Add validation metrics for current epoch."""
        for key, value in metrics.items():
            if key not in self.val_history:
                self.val_history[key] = []
            self.val_history[key].append(value)
    
    def plot_curves(self, save_path: Path, title: str = "Training Progress"):
        """Generate and save training curves plot."""
        plot_training_curves(self.train_history, self.val_history, save_path, title)
    
    def save_history(self, save_path: Path):
        """Save training history as JSON."""
        import json
        
        history_data = {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Training history saved: {save_path}")


def create_timestamped_dir(base_dir: Path, prefix: str = "") -> Path:
    """
    Create a timestamped directory for saving training results.
    
    Args:
        base_dir: Base directory (e.g., runs/model1_recolor)
        prefix: Optional prefix for the timestamp folder
        
    Returns:
        Path to the timestamped directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        folder_name = f"{prefix}_{timestamp}"
    else:
        folder_name = timestamp
    
    timestamped_dir = base_dir / folder_name
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    
    return timestamped_dir


def save_config_file(config: dict, save_path: Path, model_name: str = "", 
                     best_metrics: dict = None):
    """
    Save training configuration to a text file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the config file
        model_name: Name of the model for the header
        best_metrics: Best metrics achieved during training
    """
    import yaml
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"TRAINING CONFIGURATION - {model_name.upper()}\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        # Write configuration in YAML format
        yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Add best metrics if provided
        if best_metrics:
            f.write("\n" + "="*60 + "\n")
            f.write("BEST TRAINING RESULTS\n")
            f.write("="*60 + "\n")
            for key, value in best_metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("="*60 + "\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("SYSTEM INFORMATION\n")
        f.write("="*60 + "\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU Count: {torch.cuda.device_count()}\n")
            if torch.cuda.device_count() > 0:
                f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
        f.write("="*60 + "\n")


def update_config_with_results(config_path: Path, best_metrics: dict):
    """
    Update existing config file with final training results.
    
    Args:
        config_path: Path to the config file
        best_metrics: Best metrics achieved during training
    """
    if not config_path.exists():
        return
    
    # Read existing content
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Check if results section already exists
    if "BEST TRAINING RESULTS" in content:
        # Replace existing results section
        lines = content.split('\n')
        new_lines = []
        skip = False
        
        for line in lines:
            if "BEST TRAINING RESULTS" in line:
                skip = True
                # Add new results section
                new_lines.append("="*60)
                new_lines.append("BEST TRAINING RESULTS")
                new_lines.append("="*60)
                for key, value in best_metrics.items():
                    if isinstance(value, float):
                        new_lines.append(f"{key}: {value:.6f}")
                    else:
                        new_lines.append(f"{key}: {value}")
                new_lines.append("="*60)
                continue
            elif skip and line.startswith("="*60) and len(line) == 60:
                skip = False
                continue
            elif not skip:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
    else:
        # Add results section before system information
        system_section = "\n" + "="*60 + "\nSYSTEM INFORMATION\n"
        if system_section in content:
            results_section = (
                "\n" + "="*60 + "\n"
                "BEST TRAINING RESULTS\n" +
                "="*60 + "\n"
            )
            for key, value in best_metrics.items():
                if isinstance(value, float):
                    results_section += f"{key}: {value:.6f}\n"
                else:
                    results_section += f"{key}: {value}\n"
            results_section += "="*60 + "\n"
            
            content = content.replace(system_section, results_section + system_section)
    
    # Write updated content
    with open(config_path, 'w') as f:
        f.write(content)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_system_info():
    """Print system information."""
    console = Console()
    
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Information", style="magenta")
    
    # PyTorch info
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("GPU Count", str(torch.cuda.device_count()))
        table.add_row("GPU Name", torch.cuda.get_device_name(0))
        table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Device info
    device = get_device()
    table.add_row("Selected Device", str(device))
    
    console.print(table)
