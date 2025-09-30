#!/usr/bin/env python3
"""
Inference script for recoloring using Model 1.
"""
import argparse
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.recolor_unet import create_recolor_model
from data import load_image, load_mask
from utils import get_device, setup_logging, tensor_to_image


def parse_args():
    parser = argparse.ArgumentParser(description="Recolor garments using trained model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint for recoloring model")
    parser.add_argument("--img_size", type=int, default=512, help="Image size for processing (model input)")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--use_degraded", action="store_true", 
                       help="Apply training-consistent degradation to on_model.jpg (recommended)")
    parser.add_argument("--min_output_size", type=int, default=1024, 
                       help="Minimum output image size (default: 1024)")
    parser.add_argument("--config", type=str, help="Path to model config file (optional)")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite existing output files")
    parser.add_argument("--output_size", type=int, default=1024, help="Output image size (default: 1024)")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device, config_path: str = None):
    """Load recoloring model from checkpoint."""
    # Load checkpoint first to get model parameters
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get model config from checkpoint, fallback to default
    model_config = checkpoint.get('model_config', {})
    
    # If config file provided, load it
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
        print(f"Using model config from: {config_path}")
    else:
        print("Using default model parameters (base_channels=96, num_attn_blocks=3, num_heads=8)")
    
    # Create model with parameters from config or defaults
    model = create_recolor_model(
        use_gan=model_config.get('use_gan', False),
        in_channels=3,
        out_channels=3,
        base_channels=model_config.get('base_channels', 96),  # Match config
        depth=4,
        num_attn_blocks=model_config.get('num_attn_blocks', 3),  # Match config
        num_heads=model_config.get('num_heads', 8),  # Match config
        dropout=0.1
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def save_image(image: np.ndarray, output_path: Path, min_size: int = 512, output_size: int = None):
    """Save image as JPEG, ensuring minimum size and optional resizing."""
    # Convert to 0-255 range
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Resize to output size if specified
    if output_size is not None:
        h, w = image_uint8.shape[:2]
        if h != output_size or w != output_size:
            image_uint8 = cv2.resize(image_uint8, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            print(f"Resized image from {h}x{w} to {output_size}x{output_size}")
    
    # Ensure minimum size (only if output_size not specified or smaller than min_size)
    h, w = image_uint8.shape[:2]
    if h < min_size or w < min_size:
        # Calculate scale factor to reach minimum size
        scale = max(min_size / h, min_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_uint8 = cv2.resize(image_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"Resized image from {h}x{w} to {new_h}x{new_w} to meet minimum size {min_size}x{min_size}")
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def process_item(
    item_dir: Path,
    model,
    device: torch.device,
    img_size: int,
    save_dir: Path,
    use_degraded: bool = False,
    min_output_size: int = 512,
    overwrite: bool = False,
    output_size: int = None
):
    """Process a single item directory."""
    # Check required files
    still_path = item_dir / "still.jpg"
    onmodel_path = item_dir / ("degraded_on_model.jpg" if use_degraded else "on_model.jpg")
    mask_still_path = item_dir / "mask_still.png"
    mask_onmodel_path = item_dir / "mask_on_model.png"
    
    required_files = [still_path, onmodel_path, mask_still_path, mask_onmodel_path]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_files:
        print(f"Skipping {item_dir.name}: missing files {[f.name for f in missing_files]}")
        return False
    
    # Output path (directly in the item directory)
    output_path = item_dir / "corrected-on-model.jpg"
    
    # Skip if already exists (unless overwrite is enabled)
    if output_path.exists() and not overwrite:
        print(f"Skipping {item_dir.name}: output already exists")
        return True
    
    try:
        # Load images and masks
        still_img = load_image(still_path, (img_size, img_size))
        onmodel_img = load_image(onmodel_path, (img_size, img_size))
        mask_still = load_mask(mask_still_path, (img_size, img_size))
        mask_onmodel = load_mask(mask_onmodel_path, (img_size, img_size))
        
        # Debug: Print file info
        print(f"📁 Processing {item_dir.name}:")
        print(f"   Still: {still_path.name} ({still_img.shape})")
        print(f"   On-model: {onmodel_path.name} ({onmodel_img.shape})")
        print(f"   Mask still: {mask_still_path.name} (coverage: {np.mean(mask_still)*100:.1f}%)")
        print(f"   Mask on-model: {mask_onmodel_path.name} (coverage: {np.mean(mask_onmodel)*100:.1f}%)")
        
        # CRITICAL FIX: Apply same degradation as training for consistency
        if use_degraded:
            print(f"🔧 Applying training-consistent degradation...")
            from src.data import apply_light_degradation
            
            # Use same parameters as training (from config)
            degrade_mode = "mixed"  # From model1.yaml
            degrade_magnitude = 0.15  # From model1.yaml (updated)
            degrade_seed = 42  # Fixed seed for reproducibility
            
            # Apply degradation to onmodel_img
            onmodel_input = apply_light_degradation(
                onmodel_img, mask_onmodel,
                mode=degrade_mode,
                magnitude=degrade_magnitude,
                seed=degrade_seed
            )
            
            # Debug: Check degradation
            degrade_diff = np.mean(np.abs(onmodel_input - onmodel_img))
            print(f"   Degradation applied: {degrade_mode}, magnitude={degrade_magnitude}")
            print(f"   Degradation difference: {degrade_diff:.6f}")
        else:
            # Use original image as input
            onmodel_input = onmodel_img
            print(f"⚠️  Using original image (no degradation) - may cause poor results!")
        
        # Convert to tensors
        still_tensor = torch.from_numpy(still_img).permute(2, 0, 1).unsqueeze(0).to(device)
        onmodel_tensor = torch.from_numpy(onmodel_input).permute(2, 0, 1).unsqueeze(0).to(device)  # Use degraded input
        mask_still_tensor = torch.from_numpy(mask_still).unsqueeze(0).unsqueeze(0).to(device)
        mask_onmodel_tensor = torch.from_numpy(mask_onmodel).unsqueeze(0).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            corrected = model.forward_infer(
                onmodel_tensor, still_tensor, mask_onmodel_tensor, mask_still_tensor
            )
            
            # Debug: Check if model is actually making changes
            input_diff = torch.mean(torch.abs(corrected - onmodel_tensor)).item()
            print(f"🔍 Model prediction difference: {input_diff:.6f} (0=no change, >0.01=visible change)")
            
            # Check mask coverage
            mask_coverage = torch.mean(mask_onmodel_tensor).item()
            print(f"🎭 Mask coverage: {mask_coverage:.3f} ({mask_coverage*100:.1f}% of image)")
            
            # Check if prediction is different from input in masked regions
            masked_diff = torch.mean(torch.abs(corrected - onmodel_tensor) * mask_onmodel_tensor).item()
            print(f"🎯 Masked region difference: {masked_diff:.6f}")
        
        # Convert back to numpy
        corrected_img = tensor_to_image(corrected)
        
        # Save result
        save_image(corrected_img, output_path, min_output_size, output_size)
        
        # Debug: Final comparison
        original_diff = np.mean(np.abs(corrected_img - onmodel_img))  # Compare with original (target)
        input_diff = np.mean(np.abs(corrected_img - onmodel_input))   # Compare with degraded input
        print(f"✅ Final results:")
        print(f"   Corrected vs Original: {original_diff:.6f}")
        print(f"   Corrected vs Degraded Input: {input_diff:.6f}")
        print(f"💾 Saved: {output_path.name}")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error processing {item_dir.name}: {e}")
        return False


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Get device
    device = get_device() if args.device is None else torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading recoloring model...")
    model = load_model(args.ckpt, device, args.config)
    logger.info("Model loaded successfully")
    
    # Debug: Print model architecture info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Check if model is in eval mode
    logger.info(f"Model in eval mode: {not model.training}")
    
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
    
    # Process items
    success_count = 0
    for item_dir in tqdm(item_dirs, desc="Processing items"):
        if process_item(
            item_dir, model, device, args.img_size, item_dir, args.use_degraded, args.min_output_size, args.overwrite, args.output_size
        ):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(item_dirs)} items")
    logger.info(f"Results saved to individual item directories in: {data_root}")


if __name__ == "__main__":
    main()
