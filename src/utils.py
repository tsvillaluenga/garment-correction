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


class ThickBarColumn(ProgressColumn):
    """Custom thick progress bar column."""
    
    def __init__(self, bar_width: int = 40, bar_height: int = 3, style: str = "bar.back", complete_style: str = "bar.complete"):
        self.bar_width = bar_width
        self.bar_height = bar_height
        self.style = style
        self.complete_style = complete_style
    
    def render(self, task):
        """Render the thick progress bar."""
        if not task.total:
            return Text("━" * self.bar_width, style=self.style)
        
        completed = int(task.completed / task.total * self.bar_width)
        remaining = self.bar_width - completed
        
        # Create thick bar using block characters
        bar_text = ""
        
        # Completed portion
        if completed > 0:
            bar_text += "█" * completed
        
        # Remaining portion
        if remaining > 0:
            bar_text += "░" * remaining
        
        return Text(bar_text, style=self.complete_style if completed > 0 else self.style)


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
        
        # Save checkpoint
        if is_best and score is not None:
            # Best checkpoint
            checkpoint_path = self.save_dir / f"best_epoch_{epoch:03d}_score_{score:.4f}.pth"
            self._manage_best_checkpoints(checkpoint_path, score)
        else:
            # Regular checkpoint
            checkpoint_path = self.save_dir / f"epoch_{epoch:03d}.pth"
            self._manage_last_checkpoints(checkpoint_path)
        
        torch.save(checkpoint, checkpoint_path)
        
        # Always save a "best.pth" and "last.pth" for easy loading
        if is_best:
            torch.save(checkpoint, self.save_dir / "best.pth")
        torch.save(checkpoint, self.save_dir / "last.pth")
        
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
                ThickBarColumn(bar_width=40, bar_height=3, style="white", complete_style="bright_green"),
                PercentageColumn(),
                TextColumn("•"),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
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
