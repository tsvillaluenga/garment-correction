#!/usr/bin/env python3
"""
Script to create degraded on-model images for inference.
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data import load_image, load_mask, apply_light_degradation
from utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Create degraded on-model images")
    parser.add_argument("--data_root", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--mode", type=str, default="lab", choices=["hsv", "hsl", "lab", "rgb"], 
                       help="Degradation mode")
    parser.add_argument("--magnitude", type=float, default=0.30, help="Degradation magnitude (default: 0.30 for visible effect)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip_if_exists", action="store_true", 
                       help="Skip items that already have degraded images")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: same as input)")
    parser.add_argument("--output_size", type=int, default=1024, help="Output image size (default: 1024)")
    return parser.parse_args()


def save_image(image: np.ndarray, output_path: Path, output_size: int = None):
    """Save image as JPEG."""
    # Convert to 0-255 range
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Resize if output size is specified
    if output_size is not None:
        h, w = image_uint8.shape[:2]
        if h != output_size or w != output_size:
            image_uint8 = cv2.resize(image_uint8, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def process_item(
    item_dir: Path,
    mode: str,
    magnitude: float,
    seed: int,
    skip_if_exists: bool,
    output_size: int,
    output_dir: Path = None
):
    """Process a single item directory."""
    if output_dir is None:
        output_dir = item_dir
    
    # Check paths
    onmodel_path = item_dir / "on_model.jpg"
    mask_path = item_dir / "mask_on_model.png"
    
    if not onmodel_path.exists():
        print(f"Skipping {item_dir.name}: on_model.jpg not found")
        return False
    
    if not mask_path.exists():
        print(f"Skipping {item_dir.name}: mask_on_model.png not found")
        return False
    
    # Output path (directly in the item directory)
    degraded_path = output_dir / "degraded_on_model.jpg"
    
    # Skip if exists and requested
    if skip_if_exists and degraded_path.exists():
        print(f"Skipping {item_dir.name}: degraded image already exists")
        return True
    
    try:
        # Load image and mask
        onmodel_img = load_image(onmodel_path)
        mask = load_mask(mask_path, (onmodel_img.shape[1], onmodel_img.shape[0]))
        
        # Apply degradation
        degraded_img = apply_light_degradation(
            onmodel_img, mask, mode=mode, magnitude=magnitude, seed=seed
        )
        
        # Save degraded image directly in the item directory
        save_image(degraded_img, degraded_path, output_size)
        
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
    
    logger.info(f"Found {len(item_dirs)} items to process")
    logger.info(f"Degradation mode: {args.mode}, magnitude: {args.magnitude}")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process items
    success_count = 0
    for item_dir in tqdm(item_dirs, desc="Processing items"):
        if process_item(
            item_dir, args.mode, args.magnitude, args.seed,
            args.skip_if_exists, args.output_size, output_dir
        ):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(item_dirs)} items")


if __name__ == "__main__":
    main()
