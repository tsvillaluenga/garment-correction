#!/usr/bin/env python3
"""
Inference script for generating masks using Models 2 and 3.
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

from models.seg_unet import create_seg_model, EnhancedSegmentationUNet, DualInputSegmentationUNet
from data import load_image
from utils import get_device, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Generate masks using segmentation models")
    parser.add_argument("--data_root", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--ckpt_still", type=str, required=True, help="Checkpoint for still segmentation model")
    parser.add_argument("--ckpt_onmodel", type=str, required=True, help="Checkpoint for on-model segmentation model")
    parser.add_argument("--img_size", type=int, default=512, help="Image size for processing (model input)")
    parser.add_argument("--output_size", type=int, default=1024, help="Output image size (default: 1024)")
    parser.add_argument("--thresh", type=float, default=0.5, help="Threshold for binary segmentation")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, help="Output directory (default: same as input)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mask files")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load segmentation model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Detect model architecture from state dict
    # Check if it has dual encoders (DualInputSegmentationUNet)
    has_dual_encoders = any('still_encoder' in key for key in state_dict.keys())
    
    # Check if it has attention layers
    has_attention = any('attention' in key for key in state_dict.keys())
    
    # Detect base_channels from first encoder layer
    if has_dual_encoders:
        # Dual-input model
        first_conv_key = 'onmodel_encoder.init_conv.conv1.conv.weight'
        if first_conv_key in state_dict:
            base_channels = state_dict[first_conv_key].shape[0]
        else:
            base_channels = 96  # Default for dual-input
    else:
        # Single-input model
        first_conv_key = 'init_conv.conv1.conv.weight'
        if first_conv_key in state_dict:
            base_channels = state_dict[first_conv_key].shape[0]
        else:
            # Fallback: check encoder first layer
            encoder_key = 'encoder.0.conv.conv1.conv.weight'
            if encoder_key in state_dict:
                base_channels = state_dict[encoder_key].shape[1]  # Input channels to first encoder
            else:
                base_channels = 64  # Default fallback
    
    print(f"Detected model: base_channels={base_channels}, has_attention={has_attention}, dual_input={has_dual_encoders}")
    
    # Create model based on detected architecture
    if has_dual_encoders:
        model = DualInputSegmentationUNet(
            in_channels=3,
            base_channels=base_channels,
            depth=4,
            use_attention=True,
            dropout=0.2
        )
    elif has_attention or base_channels > 64:
        model = EnhancedSegmentationUNet(
            in_channels=3,
            base_channels=base_channels,
            depth=4,
            use_attention=True,
            dropout=0.2
        )
    else:
        model = create_seg_model(
            model_type="basic",
            in_channels=3,
            base_channels=base_channels,
            depth=4,
            dropout=0.1
        )
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def predict_mask(model, image: np.ndarray, device: torch.device, threshold: float = 0.5, 
                 still_image: np.ndarray = None, still_mask: np.ndarray = None):
    """Predict mask for a single image."""
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Check if model is dual-input
        if hasattr(model, 'still_encoder') and still_image is not None:
            # Dual-input model
            still_tensor = torch.from_numpy(still_image).permute(2, 0, 1).unsqueeze(0).to(device)
            
            if still_mask is not None:
                still_mask_tensor = torch.from_numpy(still_mask).unsqueeze(0).unsqueeze(0).to(device)
            else:
                still_mask_tensor = None
            
            predictions = model(still_tensor, image_tensor, still_mask_tensor)
            logits = predictions['main']  # Use main prediction (on-model segmentation)
        else:
            # Single-input model
            logits = model(image_tensor)
        
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float()
    
    # Convert back to numpy
    mask_np = mask.squeeze().cpu().numpy()
    return mask_np


def save_mask(mask: np.ndarray, output_path: Path):
    """Save mask as PNG image."""
    # Convert to 0-255 range
    mask_uint8 = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), mask_uint8)


def process_item(
    item_dir: Path,
    model_still,
    model_onmodel,
    device: torch.device,
    img_size: int,
    output_size: int,
    threshold: float,
    output_dir: Path = None,
    overwrite: bool = False
):
    """Process a single item directory."""
    if output_dir is None:
        output_dir = item_dir
    
    # Load images
    still_path = item_dir / "still.jpg"
    onmodel_path = item_dir / "on_model.jpg"
    
    if not still_path.exists() or not onmodel_path.exists():
        print(f"Skipping {item_dir.name}: missing images")
        return False
    
    # Check if masks already exist
    mask_still_path = output_dir / "mask_still.png"
    mask_onmodel_path = output_dir / "mask_on_model.png"
    
    if mask_still_path.exists() and mask_onmodel_path.exists() and not overwrite:
        print(f"Skipping {item_dir.name}: masks already exist")
        return True
    
    try:
        # Step 1: Load and process still image (always first)
        still_img = load_image(still_path, (img_size, img_size))
        mask_still = predict_mask(model_still, still_img, device, threshold)
        
        # Step 2: Load on-model image
        onmodel_img = load_image(onmodel_path, (img_size, img_size))
        
        # Step 3: Process on-model image (may use still image and mask as reference)
        if hasattr(model_onmodel, 'still_encoder'):
            # Dual-input model: use still image and mask as reference
            mask_onmodel = predict_mask(
                model_onmodel, onmodel_img, device, threshold,
                still_image=still_img, still_mask=mask_still
            )
            print(f"Using dual-input approach for {item_dir.name}")
        else:
            # Single-input model: traditional approach
            mask_onmodel = predict_mask(model_onmodel, onmodel_img, device, threshold)
            print(f"Using single-input approach for {item_dir.name}")
        
        # Resize masks to output size if different from processing size
        if output_size != img_size:
            mask_still = cv2.resize(mask_still, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
            mask_onmodel = cv2.resize(mask_onmodel, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        
        # Save masks directly in the item directory
        save_mask(mask_still, mask_still_path)
        save_mask(mask_onmodel, mask_onmodel_path)
        
        print(f"✅ Processed {item_dir.name}: masks saved")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {item_dir.name}: {e}")
        return False


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Get device
    device = get_device() if args.device is None else torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load models
    logger.info("Loading segmentation models...")
    model_still = load_model(args.ckpt_still, device)
    model_onmodel = load_model(args.ckpt_onmodel, device)
    logger.info("Models loaded successfully")
    
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
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process items
    success_count = 0
    for item_dir in tqdm(item_dirs, desc="Processing items"):
        if process_item(
            item_dir, model_still, model_onmodel,
            device, args.img_size, args.output_size, args.thresh, output_dir, args.overwrite
        ):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(item_dirs)} items")


if __name__ == "__main__":
    main()
