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
import io
from PIL import Image

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
    # Define image paths with fallback naming
    still_path = item_dir / "still.jpg" if (item_dir / "still.jpg").exists() else item_dir / "still-life.jpg"
    on_model_path = item_dir / "on_model.jpg" if (item_dir / "on_model.jpg").exists() else item_dir / "on-model.jpg"
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


def compute_masked_similarity(img1_path: Path, img2_path: Path, mask_path: Path, grid_size: int) -> float:
    """Compute comprehensive pixel-by-pixel similarity between two images only in masked regions."""
    try:
        # Load images
        img1 = load_and_resize_image(img1_path, grid_size)
        img2 = load_and_resize_image(img2_path, grid_size)
        
        # Load mask
        mask = load_mask(mask_path, (grid_size, grid_size))
        
        # Convert to float32 and normalize to [0, 1]
        img1_float = img1.astype(np.float32) / 255.0
        img2_float = img2.astype(np.float32) / 255.0
        
        # Apply mask to get only garment regions
        masked_img1 = img1_float * mask[:, :, np.newaxis]  # Broadcast mask to 3 channels
        masked_img2 = img2_float * mask[:, :, np.newaxis]
        
        # Get mask statistics
        mask_pixels = np.sum(mask)
        if mask_pixels == 0:
            return 1.0  # No mask region to compare
        
        # 1. PIXEL-BY-PIXEL COLOR DIFFERENCE (RGB)
        # Compute absolute difference for each pixel in RGB space
        rgb_diff = np.abs(masked_img1 - masked_img2)
        rgb_diff_per_pixel = np.sum(rgb_diff, axis=2)  # Sum across RGB channels
        
        # Average RGB difference per pixel (normalized by number of channels)
        avg_rgb_diff = np.sum(rgb_diff_per_pixel) / (mask_pixels * 3)
        
        # 2. STRUCTURAL SIMILARITY (SSIM-like)
        # Convert to grayscale for structural comparison
        gray1 = np.mean(masked_img1, axis=2)
        gray2 = np.mean(masked_img2, axis=2)
        
        # Compute means and variances in masked regions
        mean1 = np.sum(gray1) / mask_pixels
        mean2 = np.sum(gray2) / mask_pixels
        
        # Variance and covariance
        var1 = np.sum((gray1 - mean1) ** 2) / mask_pixels
        var2 = np.sum((gray2 - mean2) ** 2) / mask_pixels
        covar = np.sum((gray1 - mean1) * (gray2 - mean2)) / mask_pixels
        
        # SSIM components
        c1, c2 = 0.01, 0.03  # Constants for numerical stability
        ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
               ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
        
        # 3. COLOR DISTRIBUTION SIMILARITY (Histogram correlation)
        # Compute histograms for each channel in masked regions
        hist_corr = 0.0
        for channel in range(3):
            hist1, _ = np.histogram(masked_img1[:, :, channel].flatten(), bins=32, range=(0, 1), weights=mask.flatten())
            hist2, _ = np.histogram(masked_img2[:, :, channel].flatten(), bins=32, range=(0, 1), weights=mask.flatten())
            
            # Normalize histograms
            hist1 = hist1 / (np.sum(hist1) + 1e-8)
            hist2 = hist2 / (np.sum(hist2) + 1e-8)
            
            # Correlation coefficient between histograms
            hist_corr += np.corrcoef(hist1, hist2)[0, 1]
        
        hist_corr /= 3  # Average across RGB channels
        
        # 4. EDGE PRESERVATION (Gradient similarity)
        # Compute gradients using Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        grad1_x = cv2.filter2D(gray1, -1, sobel_x)
        grad1_y = cv2.filter2D(gray1, -1, sobel_y)
        grad2_x = cv2.filter2D(gray2, -1, sobel_x)
        grad2_y = cv2.filter2D(gray2, -1, sobel_y)
        
        # Gradient magnitude
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Gradient similarity (correlation)
        grad_corr = np.corrcoef(grad1_mag.flatten(), grad2_mag.flatten())[0, 1]
        if np.isnan(grad_corr):
            grad_corr = 0.0
        
        # 5. COMBINE ALL METRICS WITH WEIGHTS
        # Convert differences to similarities (higher = more similar)
        rgb_similarity = 1.0 - avg_rgb_diff  # RGB difference -> similarity
        ssim_similarity = max(0.0, ssim)     # SSIM (already similarity)
        hist_similarity = max(0.0, hist_corr)  # Histogram correlation
        grad_similarity = max(0.0, grad_corr)  # Gradient correlation
        
        # Weighted combination (emphasize pixel-level differences)
        weights = {
            'rgb': 0.4,      # Pixel-by-pixel color difference (most important)
            'ssim': 0.3,     # Structural similarity
            'hist': 0.2,     # Color distribution
            'grad': 0.1      # Edge preservation
        }
        
        final_similarity = (weights['rgb'] * rgb_similarity + 
                           weights['ssim'] * ssim_similarity + 
                           weights['hist'] * hist_similarity + 
                           weights['grad'] * grad_similarity)
        
        # Ensure result is in [0, 1] range
        final_similarity = np.clip(final_similarity, 0.0, 1.0)
        
        return float(final_similarity)
        
    except Exception as e:
        print(f"Error computing masked similarity for {img1_path} vs {img2_path}: {e}")
        return 0.0


def create_labeled_comparison(item_dir: Path, grid_size: int = 256, font_size: int = 14) -> np.ndarray:
    """Create a labeled comparison image using matplotlib with IoU calculations."""
    # Create the grid
    grid = create_comparison_grid(item_dir, grid_size)
    
    # Compute similarity between on_model and corrected/degraded images using mask
    on_model_path = item_dir / "on_model.jpg" if (item_dir / "on_model.jpg").exists() else item_dir / "on-model.jpg"
    corrected_path = item_dir / "corrected-on-model.jpg"
    degraded_path = item_dir / "degraded_on_model.jpg"
    mask_path = item_dir / "mask_on_model.png"
    
    sim_corrected = compute_masked_similarity(on_model_path, corrected_path, mask_path, grid_size)
    sim_degraded = compute_masked_similarity(on_model_path, degraded_path, mask_path, grid_size)
    
    sim_corrected_pct = sim_corrected * 100
    sim_degraded_pct = sim_degraded * 100
    
    # Create matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Garment Correction Results - {item_dir.name}', fontsize=font_size+2, fontweight='bold')
    
    # Define image paths and titles with fallback naming
    still_path = item_dir / "still.jpg" if (item_dir / "still.jpg").exists() else item_dir / "still-life.jpg"
    on_model_path = item_dir / "on_model.jpg" if (item_dir / "on_model.jpg").exists() else item_dir / "on-model.jpg"
    
    images_info = [
        (still_path, "Still Product", axes[0, 0]),
        (on_model_path, "On-Model (Ground Truth)", axes[0, 1]),
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
    
    # Add similarity information at the bottom
    fig.text(0.5, 0.05, f'Degraded vs Original: {sim_degraded_pct:.1f}% | Corrected vs Original: {sim_corrected_pct:.1f}%', 
             ha='center', va='bottom', fontsize=font_size+1, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Add improvement information
    improvement = sim_corrected_pct - sim_degraded_pct
    improvement_color = "lightgreen" if improvement > 0 else "lightcoral"
    fig.text(0.5, 0.01, f'Improvement: {improvement:+.1f}%', 
             ha='center', va='bottom', fontsize=font_size, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor=improvement_color, alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for IoU text
    
    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    
    # Use buffer_rgba() instead of tostring_rgb() for compatibility
    try:
        # Try the newer method first
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
    except AttributeError:
        # Fallback to older method
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Last resort: save to temporary buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            buf = np.array(img)[:, :, :3]  # Remove alpha channel if present
    
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


def process_item(item_dir: Path, output_dir: Path, grid_size: int, font_size: int) -> tuple[bool, float, float]:
    """Process a single item and create comparison image with IoU."""
    try:
        # Create labeled comparison
        comparison_img = create_labeled_comparison(item_dir, grid_size, font_size)
        
        # Save comparison image
        output_path = output_dir / f"{item_dir.name}_comparison.jpg"
        
        # Convert RGB to BGR for OpenCV
        comparison_bgr = cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), comparison_bgr)
        
        # Compute similarity for return values
        on_model_path = item_dir / "on_model.jpg" if (item_dir / "on_model.jpg").exists() else item_dir / "on-model.jpg"
        corrected_path = item_dir / "corrected-on-model.jpg"
        degraded_path = item_dir / "degraded_on_model.jpg"
        mask_path = item_dir / "mask_on_model.png"
        
        sim_corrected = compute_masked_similarity(on_model_path, corrected_path, mask_path, grid_size)
        sim_degraded = compute_masked_similarity(on_model_path, degraded_path, mask_path, grid_size)
        
        return True, sim_corrected, sim_degraded
        
    except Exception as e:
        print(f"Error processing {item_dir.name}: {e}")
        return False, 0.0, 0.0


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
    sim_corrected_values = []
    sim_degraded_values = []
    improvement_values = []
    
    for item_dir in tqdm(item_dirs, desc="Creating visualizations"):
        success, sim_corrected, sim_degraded = process_item(item_dir, output_dir, args.grid_size, args.font_size)
        if success:
            successful += 1
            sim_corrected_values.append(sim_corrected)
            sim_degraded_values.append(sim_degraded)
            improvement_values.append(sim_corrected - sim_degraded)
        else:
            failed += 1
    
    logger.info(f"Successfully created {successful} visualizations")
    if failed > 0:
        logger.warning(f"Failed to create {failed} visualizations")
    
    # Print similarity statistics
    if sim_corrected_values:
        # Corrected vs Original statistics
        mean_corrected = np.mean(sim_corrected_values)
        std_corrected = np.std(sim_corrected_values)
        min_corrected = np.min(sim_corrected_values)
        max_corrected = np.max(sim_corrected_values)
        
        # Degraded vs Original statistics
        mean_degraded = np.mean(sim_degraded_values)
        std_degraded = np.std(sim_degraded_values)
        min_degraded = np.min(sim_degraded_values)
        max_degraded = np.max(sim_degraded_values)
        
        # Improvement statistics
        mean_improvement = np.mean(improvement_values)
        std_improvement = np.std(improvement_values)
        min_improvement = np.min(improvement_values)
        max_improvement = np.max(improvement_values)
        
        logger.info(f"Masked Similarity Statistics (Corrected vs Original):")
        logger.info(f"  Mean: {mean_corrected:.3f} ({mean_corrected*100:.1f}%)")
        logger.info(f"  Std:  {std_corrected:.3f} ({std_corrected*100:.1f}%)")
        logger.info(f"  Min:  {min_corrected:.3f} ({min_corrected*100:.1f}%)")
        logger.info(f"  Max:  {max_corrected:.3f} ({max_corrected*100:.1f}%)")
        
        logger.info(f"Masked Similarity Statistics (Degraded vs Original):")
        logger.info(f"  Mean: {mean_degraded:.3f} ({mean_degraded*100:.1f}%)")
        logger.info(f"  Std:  {std_degraded:.3f} ({std_degraded*100:.1f}%)")
        logger.info(f"  Min:  {min_degraded:.3f} ({min_degraded*100:.1f}%)")
        logger.info(f"  Max:  {max_degraded:.3f} ({max_degraded*100:.1f}%)")
        
        logger.info(f"Improvement Statistics (Corrected - Degraded):")
        logger.info(f"  Mean: {mean_improvement:+.3f} ({mean_improvement*100:+.1f}%)")
        logger.info(f"  Std:  {std_improvement:.3f} ({std_improvement*100:.1f}%)")
        logger.info(f"  Min:  {min_improvement:+.3f} ({min_improvement*100:+.1f}%)")
        logger.info(f"  Max:  {max_improvement:+.3f} ({max_improvement*100:+.1f}%)")
        
        # Save similarity statistics to file
        stats_path = output_dir / "masked_similarity_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Masked Similarity Statistics for {len(sim_corrected_values)} items:\n\n")
            
            f.write(f"Corrected vs Original (masked regions only):\n")
            f.write(f"  Mean: {mean_corrected:.3f} ({mean_corrected*100:.1f}%)\n")
            f.write(f"  Std:  {std_corrected:.3f} ({std_corrected*100:.1f}%)\n")
            f.write(f"  Min:  {min_corrected:.3f} ({min_corrected*100:.1f}%)\n")
            f.write(f"  Max:  {max_corrected:.3f} ({max_corrected*100:.1f}%)\n\n")
            
            f.write(f"Degraded vs Original (masked regions only):\n")
            f.write(f"  Mean: {mean_degraded:.3f} ({mean_degraded*100:.1f}%)\n")
            f.write(f"  Std:  {std_degraded:.3f} ({std_degraded*100:.1f}%)\n")
            f.write(f"  Min:  {min_degraded:.3f} ({min_degraded*100:.1f}%)\n")
            f.write(f"  Max:  {max_degraded:.3f} ({max_degraded*100:.1f}%)\n\n")
            
            f.write(f"Improvement (Corrected - Degraded):\n")
            f.write(f"  Mean: {mean_improvement:+.3f} ({mean_improvement*100:+.1f}%)\n")
            f.write(f"  Std:  {std_improvement:.3f} ({std_improvement*100:.1f}%)\n")
            f.write(f"  Min:  {min_improvement:+.3f} ({min_improvement*100:+.1f}%)\n")
            f.write(f"  Max:  {max_improvement:+.3f} ({max_improvement*100:+.1f}%)\n\n")
            
            f.write(f"Individual values (masked regions only):\n")
            f.write(f"Item\t\tCorrected\tDegraded\tImprovement\n")
            f.write(f"{'='*60}\n")
            for i, (item_dir, sim_c, sim_d, imp) in enumerate(zip(item_dirs[:len(sim_corrected_values)], sim_corrected_values, sim_degraded_values, improvement_values)):
                f.write(f"{item_dir.name}\t{sim_c:.3f} ({sim_c*100:.1f}%)\t{sim_d:.3f} ({sim_d*100:.1f}%)\t{imp:+.3f} ({imp*100:+.1f}%)\n")
        
        logger.info(f"Masked similarity statistics saved to: {stats_path}")
    
    # Create summary grid if requested
    if args.create_summary and successful > 0:
        summary_path = output_dir / "summary_grid.jpg"
        create_summary_grid(item_dirs, summary_path, args.summary_rows, args.summary_cols, args.grid_size//2)
    
    logger.info(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
