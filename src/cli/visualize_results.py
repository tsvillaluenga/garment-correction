#!/usr/bin/env python3
"""
Script to create visual comparisons of garment correction results.
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data import load_image, load_mask
from utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Create visual comparisons of results")
    parser.add_argument("--data_root", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory for comparisons")
    parser.add_argument("--img_size", type=int, default=512, help="Image size for processing")
    parser.add_argument("--grid_size", type=int, default=256, help="Size of each image in the grid")
    parser.add_argument("--font_size", type=int, default=14, help="Font size for labels")
    parser.add_argument("--max_items", type=int, help="Maximum number of items to process")
    parser.add_argument("--create_summary", action="store_true", help="Create a summary grid with multiple items")
    parser.add_argument("--summary_rows", type=int, default=4, help="Number of rows in summary grid")
    parser.add_argument("--summary_cols", type=int, default=4, help="Number of columns in summary grid")
    return parser.parse_args()


def load_and_resize_image(image_path: Path, target_size: int) -> np.ndarray:
    """Load and resize image to target size."""
    if not image_path.exists():
        # Create a placeholder image if file doesn't exist
        placeholder = np.ones((target_size, target_size, 3), dtype=np.uint8) * 128
        # Add text indicating missing file
        cv2.putText(placeholder, "Missing", (target_size//4, target_size//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder
    
    try:
        image = load_image(image_path, (target_size, target_size))
        # Convert from [0,1] to [0,255]
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return image_uint8
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        # Return placeholder
        placeholder = np.ones((target_size, target_size, 3), dtype=np.uint8) * 128
        return placeholder


def create_comparison_grid(item_dir: Path, grid_size: int = 256) -> np.ndarray:
    """Create a 2x2 comparison grid for a single item."""
    # Define image paths
    still_path = item_dir / "still.jpg"
    on_model_path = item_dir / "on_model.jpg"
    degraded_path = item_dir / "degraded_on_model.jpg"
    corrected_path = item_dir / "corrected-on-model.jpg"
    
    # Load images
    still_img = load_and_resize_image(still_path, grid_size)
    on_model_img = load_and_resize_image(on_model_path, grid_size)
    degraded_img = load_and_resize_image(degraded_path, grid_size)
    corrected_img = load_and_resize_image(corrected_path, grid_size)
    
    # Create 2x2 grid
    top_row = np.hstack([still_img, on_model_img])
    bottom_row = np.hstack([degraded_img, corrected_img])
    grid = np.vstack([top_row, bottom_row])
    
    return grid


def create_labeled_comparison(item_dir: Path, grid_size: int = 256, font_size: int = 14) -> np.ndarray:
    """Create a labeled comparison image using matplotlib."""
    # Create the grid
    grid = create_comparison_grid(item_dir, grid_size)
    
    # Create matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Garment Correction Results - {item_dir.name}', fontsize=font_size+2, fontweight='bold')
    
    # Define image paths and titles
    images_info = [
        (item_dir / "still.jpg", "Still Product", axes[0, 0]),
        (item_dir / "on_model.jpg", "On-Model (Ground Truth)", axes[0, 1]),
        (item_dir / "degraded_on_model.jpg", "Degraded On-Model", axes[1, 0]),
        (item_dir / "corrected-on-model.jpg", "Corrected Result", axes[1, 1])
    ]
    
    # Load and display images
    for img_path, title, ax in images_info:
        img = load_and_resize_image(img_path, grid_size)
        ax.imshow(img)
        ax.set_title(title, fontsize=font_size, fontweight='bold')
        ax.axis('off')
        
        # Add border color based on image type
        if "still" in title.lower():
            border_color = 'blue'
        elif "ground truth" in title.lower():
            border_color = 'green'
        elif "degraded" in title.lower():
            border_color = 'orange'
        else:  # corrected
            border_color = 'red'
        
        # Add colored border
        rect = Rectangle((0, 0), grid_size-1, grid_size-1, linewidth=3, 
                        edgecolor=border_color, facecolor='none')
        ax.add_patch(rect)
    
    plt.tight_layout()
    
    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return buf


def create_summary_grid(item_dirs: list, output_path: Path, rows: int = 4, cols: int = 4, grid_size: int = 128):
    """Create a summary grid with multiple items."""
    max_items = rows * cols
    selected_items = item_dirs[:max_items]
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    fig.suptitle('Garment Correction Results Summary', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, item_dir in enumerate(selected_items):
        row, col = divmod(i, cols)
        ax = axes[row, col]
        
        # Create mini comparison for this item
        try:
            # Load corrected image (main focus for summary)
            corrected_path = item_dir / "corrected-on-model.jpg"
            img = load_and_resize_image(corrected_path, grid_size)
            ax.imshow(img)
            ax.set_title(f'{item_dir.name}', fontsize=8)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{item_dir.name}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=8)
            ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(selected_items), rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Summary grid saved: {output_path}")


def process_item(item_dir: Path, output_dir: Path, grid_size: int, font_size: int) -> bool:
    """Process a single item and create comparison image."""
    try:
        # Create labeled comparison
        comparison_img = create_labeled_comparison(item_dir, grid_size, font_size)
        
        # Save comparison image
        output_path = output_dir / f"{item_dir.name}_comparison.jpg"
        
        # Convert RGB to BGR for OpenCV
        comparison_bgr = cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), comparison_bgr)
        
        return True
        
    except Exception as e:
        print(f"Error processing {item_dir.name}: {e}")
        return False


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Find test items
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        return
    
    # Look for item directories
    item_dirs = []
    if (data_root / "test").exists():
        test_dir = data_root / "test"
        item_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    else:
        # Assume data_root itself contains item directories
        item_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    
    if not item_dirs:
        logger.error(f"No item directories found in {data_root}")
        return
    
    # Limit items if specified
    if args.max_items:
        item_dirs = item_dirs[:args.max_items]
    
    logger.info(f"Found {len(item_dirs)} items to visualize")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process items
    successful = 0
    failed = 0
    
    for item_dir in tqdm(item_dirs, desc="Creating visualizations"):
        success = process_item(item_dir, output_dir, args.grid_size, args.font_size)
        if success:
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Successfully created {successful} visualizations")
    if failed > 0:
        logger.warning(f"Failed to create {failed} visualizations")
    
    # Create summary grid if requested
    if args.create_summary and successful > 0:
        summary_path = output_dir / "summary_grid.jpg"
        create_summary_grid(item_dirs, summary_path, args.summary_rows, args.summary_cols, args.grid_size//2)
    
    logger.info(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
