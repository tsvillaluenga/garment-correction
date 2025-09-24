#!/usr/bin/env python3
"""
Evaluation script to compute metrics on corrected images.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data import load_image, load_mask
from losses_metrics import compute_color_metrics
from utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate corrected images")
    parser.add_argument("--data_root", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--results_dir", type=str, help="Directory with corrected results (default: data_root)")
    parser.add_argument("--save_csv", type=str, required=True, help="Path to save metrics CSV")
    parser.add_argument("--img_size", type=int, default=512, help="Image size for processing")
    return parser.parse_args()


def evaluate_item(
    item_dir: Path,
    results_dir: Path,
    img_size: int
) -> dict:
    """Evaluate a single item."""
    # Check required files
    gt_path = item_dir / "on_model.jpg"  # Ground truth
    corrected_path = results_dir / item_dir.name / "corrected-on-model.jpg"
    mask_path = item_dir / "mask_on_model.png"
    
    if not gt_path.exists():
        return {"error": "Ground truth image not found"}
    
    if not corrected_path.exists():
        return {"error": "Corrected image not found"}
    
    if not mask_path.exists():
        return {"error": "Mask not found"}
    
    try:
        # Load images and mask
        gt_img = load_image(gt_path, (img_size, img_size))
        corrected_img = load_image(corrected_path, (img_size, img_size))
        mask = load_mask(mask_path, (img_size, img_size))
        
        # Convert to tensors
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)
        corrected_tensor = torch.from_numpy(corrected_img).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        
        # Compute color metrics
        metrics = compute_color_metrics(corrected_tensor, gt_tensor, mask_tensor)
        
        # Add item name
        metrics['item_name'] = item_dir.name
        
        return metrics
        
    except Exception as e:
        return {"error": str(e)}


def compute_global_metrics(all_metrics: list) -> dict:
    """Compute global statistics across all items."""
    # Filter out error entries
    valid_metrics = [m for m in all_metrics if 'error' not in m]
    
    if not valid_metrics:
        return {}
    
    global_stats = {}
    
    # Get all numeric keys
    numeric_keys = []
    for key in valid_metrics[0].keys():
        if key != 'item_name' and isinstance(valid_metrics[0][key], (int, float)):
            numeric_keys.append(key)
    
    # Compute statistics for each metric
    for key in numeric_keys:
        values = [m[key] for m in valid_metrics if key in m]
        if values:
            global_stats[f'global_{key}_mean'] = np.mean(values)
            global_stats[f'global_{key}_std'] = np.std(values)
            global_stats[f'global_{key}_median'] = np.median(values)
            global_stats[f'global_{key}_min'] = np.min(values)
            global_stats[f'global_{key}_max'] = np.max(values)
    
    # Add count
    global_stats['total_items'] = len(valid_metrics)
    global_stats['failed_items'] = len(all_metrics) - len(valid_metrics)
    
    return global_stats


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Find test items
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        return
    
    # Set results directory
    results_dir = Path(args.results_dir) if args.results_dir else data_root
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
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
    
    logger.info(f"Found {len(item_dirs)} items to evaluate")
    logger.info(f"Results directory: {results_dir}")
    
    # Evaluate items
    all_metrics = []
    
    for item_dir in tqdm(item_dirs, desc="Evaluating items"):
        metrics = evaluate_item(item_dir, results_dir, args.img_size)
        all_metrics.append(metrics)
        
        if 'error' in metrics:
            logger.warning(f"Error evaluating {item_dir.name}: {metrics['error']}")
    
    # Filter successful evaluations
    valid_metrics = [m for m in all_metrics if 'error' not in m]
    error_count = len(all_metrics) - len(valid_metrics)
    
    logger.info(f"Successfully evaluated {len(valid_metrics)} items")
    if error_count > 0:
        logger.warning(f"Failed to evaluate {error_count} items")
    
    if not valid_metrics:
        logger.error("No valid metrics to save")
        return
    
    # Compute global statistics
    global_stats = compute_global_metrics(all_metrics)
    
    # Create DataFrames
    metrics_df = pd.DataFrame(valid_metrics)
    
    # Add global statistics as a summary row
    if global_stats:
        global_row = global_stats.copy()
        global_row['item_name'] = 'GLOBAL_SUMMARY'
        global_df = pd.DataFrame([global_row])
        
        # Combine DataFrames
        final_df = pd.concat([metrics_df, global_df], ignore_index=True)
    else:
        final_df = metrics_df
    
    # Save to CSV
    csv_path = Path(args.save_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(csv_path, index=False)
    
    logger.info(f"Metrics saved to: {csv_path}")
    
    # Print summary statistics
    if global_stats:
        logger.info("\n=== EVALUATION SUMMARY ===")
        logger.info(f"Total items: {global_stats.get('total_items', 0)}")
        logger.info(f"Failed items: {global_stats.get('failed_items', 0)}")
        
        # Print key metrics
        key_metrics = [
            'delta_e76_mean', 'delta_e2000_mean', 
            'psnr_luma', 'psnr_luma_masked'
        ]
        
        for metric in key_metrics:
            mean_key = f'global_{metric}_mean'
            std_key = f'global_{metric}_std'
            if mean_key in global_stats:
                mean_val = global_stats[mean_key]
                std_val = global_stats.get(std_key, 0)
                logger.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()
