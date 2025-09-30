"""
Data loading, augmentation, and light degradation utilities.
"""
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_segmentation_transforms(augment_config: Dict, img_size: int = 512) -> A.Compose:
    """
    Get advanced augmentation transforms for segmentation.
    
    Args:
        augment_config: Augmentation configuration dictionary
        img_size: Target image size
        
    Returns:
        Albumentations compose transform
    """
    transforms_list = []
    
    # Geometric transformations (high impact)
    if augment_config.get('hflip', False):
        transforms_list.append(A.HorizontalFlip(p=0.5))
    
    if augment_config.get('rotate_deg', 0) > 0:
        transforms_list.append(A.Rotate(
            limit=augment_config['rotate_deg'], 
            p=0.7,
            border_mode=cv2.BORDER_CONSTANT
        ))
    
    if augment_config.get('scale'):
        scale_range = augment_config['scale']
        transforms_list.append(A.RandomScale(
            scale_limit=(scale_range[0] - 1.0, scale_range[1] - 1.0), 
            p=0.7
        ))
    
    if augment_config.get('elastic_transform', False):
        transforms_list.append(A.ElasticTransform(
            alpha=1, sigma=50, p=0.3,
            border_mode=cv2.BORDER_CONSTANT
        ))
    
    if augment_config.get('grid_distortion', False):
        transforms_list.append(A.GridDistortion(
            num_steps=5, distort_limit=0.3, p=0.3,
            border_mode=cv2.BORDER_CONSTANT
        ))
    
    if augment_config.get('perspective_transform', False):
        transforms_list.append(A.Perspective(
            scale=(0.05, 0.1), p=0.3,
            border_mode=cv2.BORDER_CONSTANT
        ))
    
    # Color augmentations (medium impact)
    if augment_config.get('brightness') and augment_config.get('contrast'):
        brightness_range = augment_config['brightness']
        contrast_range = augment_config['contrast']
        transforms_list.append(A.RandomBrightnessContrast(
            brightness_limit=(brightness_range[0] - 1.0, brightness_range[1] - 1.0),
            contrast_limit=(contrast_range[0] - 1.0, contrast_range[1] - 1.0),
            p=0.5
        ))
    
    if augment_config.get('hue_shift', False):
        transforms_list.append(A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
        ))
    
    if augment_config.get('gaussian_noise', False):
        transforms_list.append(A.GaussNoise(
            noise_scale_factor=0.1, p=0.2
        ))
    
    # Content augmentations (medium-high impact)
    if augment_config.get('cutout', False):
        transforms_list.append(A.CoarseDropout(
            holes=8, height=32, width=32, p=0.3
        ))
    
    if augment_config.get('random_erasing', False):
        # RandomErasing is not available in albumentations, use CoarseDropout instead
        transforms_list.append(A.CoarseDropout(
            holes=4, height=16, width=16, p=0.2
        ))
    
    # Always resize to target size
    transforms_list.append(A.Resize(img_size, img_size))
    
    # Convert to tensor
    transforms_list.append(ToTensorV2())
    
    return A.Compose(transforms_list)


def load_image(path: Union[str, Path], size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image as RGB float32 [0, 1] with comprehensive validation."""
    path = Path(path)
    
    # Validate path exists
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"OpenCV failed to load image: {path}")
        
        # Validate loaded image
        if img.size == 0:
            raise ValueError(f"Empty image loaded: {path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if requested with validation
        if size is not None:
            if not isinstance(size, (tuple, list)) or len(size) != 2:
                raise ValueError("Size must be (width, height) tuple")
            
            width, height = size
            if width <= 0 or height <= 0:
                raise ValueError("Size dimensions must be positive")
            
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize
        img_float = img.astype(np.float32) / 255.0
        
        # Validate result
        if not np.isfinite(img_float).all():
            raise ValueError(f"Image contains non-finite values: {path}")
        
        return img_float
        
    except cv2.error as e:
        raise ValueError(f"OpenCV error loading {path}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading {path}: {e}")


def load_mask(path: Union[str, Path], size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load mask as binary float32 [0, 1] with comprehensive validation and transparency support.
    
    Handles multiple mask formats:
    - PNG with alpha channel (transparency)
    - PNG without alpha channel
    - Grayscale images
    - RGB images
    
    Args:
        path: Path to mask file
        size: Optional (width, height) to resize to
        
    Returns:
        Binary mask as float32 [0, 1] where 1 = foreground, 0 = background
    """
    path = Path(path)

    # Validate path exists
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        # Load mask with full channel support (including alpha)
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to load mask: {path}")
        
        if mask.size == 0:
            raise ValueError(f"Empty mask loaded: {path}")
        
        # Handle different mask formats intelligently
        if len(mask.shape) == 3:
            # Multi-channel image (RGB or RGBA)
            if mask.shape[2] == 4:  # RGBA - use alpha channel
                # Alpha channel: 0 = transparent (background), 255 = opaque (foreground)
                mask_gray = mask[:, :, 3]
                # Convert to binary: any non-transparent pixel = foreground
                binary_mask = (mask_gray > 0).astype(np.float32)
                
            elif mask.shape[2] == 3:  # RGB - convert to grayscale
                mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # Use threshold: >127 = foreground, <=127 = background
                binary_mask = (mask_gray > 127).astype(np.float32)
                
            else:
                raise ValueError(f"Unsupported multi-channel format: {mask.shape}")
                
        elif len(mask.shape) == 2:
            # Grayscale image
            # Use threshold: >127 = foreground, <=127 = background
            binary_mask = (mask > 127).astype(np.float32)
            
        else:
            raise ValueError(f"Unsupported mask format: {mask.shape}")
        
        # Resize if requested with validation
        if size is not None:
            if not isinstance(size, (tuple, list)) or len(size) != 2:
                raise ValueError("Size must be (width, height) tuple")

            width, height = size
            if width <= 0 or height <= 0:
                raise ValueError("Size dimensions must be positive")

            # Use nearest neighbor for binary masks to preserve sharp edges
            binary_mask = cv2.resize(binary_mask, size, interpolation=cv2.INTER_NEAREST)
        
        # Log warning for unusual masks (but only occasionally to avoid spam)
        mask_ratio = np.mean(binary_mask)
        if mask_ratio < 0.001 and random.random() < 0.01:  # 1% chance to log
            import logging
            logging.warning(f"Very small mask region ({mask_ratio:.3%}) in {path}")
        elif mask_ratio > 0.999 and random.random() < 0.01:  # 1% chance to log
            import logging
            logging.warning(f"Nearly full mask ({mask_ratio:.3%}) in {path}")
        
        return binary_mask
        
    except cv2.error as e:
        raise ValueError(f"OpenCV error loading mask {path}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading mask {path}: {e}")


def apply_light_degradation(
    image: np.ndarray,
    mask: np.ndarray,
    mode: str = "lab",
    magnitude: float = 0.04,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply light degradation to masked pixels only.
    
    Args:
        image: RGB image [0, 1] shape (H, W, 3)
        mask: Binary mask [0, 1] shape (H, W)
        mode: 'hsv', 'hsl', 'lab', 'rgb', or enhanced modes like 'mixed'
        magnitude: Degradation strength
        seed: Random seed for reproducibility
        
    Returns:
        Degraded image with same shape as input
    """
    # For enhanced modes, use the enhanced degradation function
    if mode in ["mixed"]:
        return apply_enhanced_degradation(image, mask, mode, magnitude, seed)
    
    if seed is not None:
        np.random.seed(seed)
    
    degraded = image.copy()
    mask_bool = mask > 0.5
    
    if not np.any(mask_bool):
        return degraded
    
    if mode == "hsv":
        # Convert to HSV
        hsv = color.rgb2hsv(image)
        
        # Apply random shifts within bounds
        h_shift = np.random.uniform(-2/360, 2/360)  # ±2 degrees
        s_shift = np.random.uniform(-0.03, 0.03)    # ±3%
        v_shift = np.random.uniform(-0.03, 0.03)    # ±3%
        
        hsv_degraded = hsv.copy()
        hsv_degraded[mask_bool, 0] = np.clip(hsv_degraded[mask_bool, 0] + h_shift, 0, 1)
        hsv_degraded[mask_bool, 1] = np.clip(hsv_degraded[mask_bool, 1] + s_shift, 0, 1)
        hsv_degraded[mask_bool, 2] = np.clip(hsv_degraded[mask_bool, 2] + v_shift, 0, 1)
        
        degraded = color.hsv2rgb(hsv_degraded)
        
    elif mode == "hsl":
        # Convert to HSL (Hue, Saturation, Lightness)
        # Note: scikit-image doesn't have HSL, so we'll use HSV and modify V to behave like L
        hsv = color.rgb2hsv(image)
        
        # Apply shifts with H fixed or minimal variation, S and L with MORE variation
        h_shift = np.random.uniform(-10.0/360, 10.0/360) * magnitude  # ±10 degrees (more visible)
        s_shift = np.random.uniform(-0.5, 0.5) * magnitude          # ±80% (very strong variation)
        l_shift = np.random.uniform(-0.5, 0.5) * magnitude          # ±60% (very strong variation)
        
        # Debug: Print mask info and degradation values
        mask_pixels = np.sum(mask_bool)
        total_pixels = mask_bool.size
        mask_percentage = (mask_pixels / total_pixels) * 100
        
        print(f"🎨 HSL degradation - Mask: {mask_percentage:.1f}% ({mask_pixels}/{total_pixels} pixels)")
        print(f"   H shift: {h_shift*360:.2f}° | S shift: {s_shift*100:.1f}% | L shift: {l_shift*100:.1f}%")
        
        if mask_pixels == 0:
            print("⚠️  WARNING: Mask is empty! No degradation will be applied.")
            return image.copy()
        
        hsv_degraded = hsv.copy()
        
        # Store original values for comparison
        orig_h = hsv[mask_bool, 0].copy()
        orig_s = hsv[mask_bool, 1].copy() 
        orig_v = hsv[mask_bool, 2].copy()
        
        # H: minimal variation (keep hue mostly fixed)
        hsv_degraded[mask_bool, 0] = np.clip(hsv_degraded[mask_bool, 0] + h_shift, 0, 1)
        # S: strong variation (saturation changes) - less aggressive clipping
        new_s = hsv_degraded[mask_bool, 1] + s_shift
        hsv_degraded[mask_bool, 1] = np.clip(new_s, 0.1, 0.9)  # Keep some saturation
        # V: strong variation (lightness changes) - less aggressive clipping  
        new_v = hsv_degraded[mask_bool, 2] + l_shift
        hsv_degraded[mask_bool, 2] = np.clip(new_v, 0.1, 0.9)  # Avoid pure black/white
        
        # Show actual changes applied
        actual_h_change = np.mean(np.abs(hsv_degraded[mask_bool, 0] - orig_h)) * 360
        actual_s_change = np.mean(np.abs(hsv_degraded[mask_bool, 1] - orig_s)) * 100
        actual_v_change = np.mean(np.abs(hsv_degraded[mask_bool, 2] - orig_v)) * 100
        
        print(f"   Actual changes: H={actual_h_change:.2f}°, S={actual_s_change:.1f}%, V={actual_v_change:.1f}%")
        
        degraded = color.hsv2rgb(hsv_degraded)
        
        # Final verification - check if image actually changed
        diff = np.mean(np.abs(degraded - image))
        print(f"   Image difference: {diff:.6f} (0=no change, >0.01=visible)")
        
        if diff < 0.001:
            print("⚠️  WARNING: Very little change detected! Degradation may not be visible.")
        
    elif mode == "lab":
        # Convert to LAB
        lab = color.rgb2lab(image)
        
        # Apply random shifts within bounds
        l_shift = np.random.uniform(-2.0, 2.0)      # ±2 L*
        a_shift = np.random.uniform(-1.5, 1.5)     # ±1.5 a*
        b_shift = np.random.uniform(-1.5, 1.5)     # ±1.5 b*
        
        lab_degraded = lab.copy()
        lab_degraded[mask_bool, 0] = np.clip(lab_degraded[mask_bool, 0] + l_shift, 0, 100)
        lab_degraded[mask_bool, 1] = np.clip(lab_degraded[mask_bool, 1] + a_shift, -127, 128)
        lab_degraded[mask_bool, 2] = np.clip(lab_degraded[mask_bool, 2] + b_shift, -127, 128)
        
        degraded = color.lab2rgb(lab_degraded)
        
    elif mode == "rgb":
        # Apply per-channel offset
        for c in range(3):
            offset = np.random.uniform(-4/255, 4/255)
            degraded[mask_bool, c] = np.clip(degraded[mask_bool, c] + offset, 0, 1)
    
    else:
        raise ValueError(f"Unknown degradation mode: {mode}. Available: hsv, hsl, lab, rgb, mixed")
    
    return np.clip(degraded, 0, 1)


def apply_enhanced_degradation(
    image: np.ndarray, 
    mask: np.ndarray, 
    mode: str = "mixed", 
    magnitude: float = 0.06,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply enhanced degradation with spatial variation and mixed modes.
    
    Args:
        image: RGB image [0, 1] as float32
        mask: Binary mask [0, 1] as float32
        mode: Degradation mode ("mixed", "hsv", "lab", "rgb")
        magnitude: Degradation magnitude multiplier
        seed: Random seed for reproducibility
        
    Returns:
        Degraded image [0, 1] as float32
    """
    if seed is not None:
        np.random.seed(seed)
    
    degraded = image.copy()
    mask_bool = mask > 0.5
    
    if not np.any(mask_bool):
        return degraded
    
    # Enhanced degradation with spatial variation
    if mode == "mixed":
        # Combine multiple degradation types for more realism
        degraded = _apply_mixed_degradation(image, mask_bool, magnitude)
    elif mode == "hsv":
        degraded = _apply_enhanced_hsv_degradation(image, mask_bool, magnitude)
    elif mode == "lab":
        degraded = _apply_enhanced_lab_degradation(image, mask_bool, magnitude)
    elif mode == "rgb":
        degraded = _apply_enhanced_rgb_degradation(image, mask_bool, magnitude)
    else:
        # Fallback to original method
        return apply_light_degradation(image, mask, mode, magnitude, seed)
    
    return np.clip(degraded, 0, 1)


def _apply_mixed_degradation(image: np.ndarray, mask_bool: np.ndarray, magnitude: float) -> np.ndarray:
    """Apply mixed degradation combining LAB and HSV."""
    degraded = image.copy()
    
    # 70% LAB degradation (better for color shifts)
    if np.random.random() < 0.7:
        degraded = _apply_enhanced_lab_degradation(degraded, mask_bool, magnitude * 0.8)
    
    # 50% HSV degradation (better for saturation/brightness)
    if np.random.random() < 0.5:
        degraded = _apply_enhanced_hsv_degradation(degraded, mask_bool, magnitude * 0.6)
    
    # 30% subtle RGB noise
    if np.random.random() < 0.3:
        degraded = _apply_enhanced_rgb_degradation(degraded, mask_bool, magnitude * 0.4)
    
    return degraded


def _apply_enhanced_hsv_degradation(image: np.ndarray, mask_bool: np.ndarray, magnitude: float) -> np.ndarray:
    """Apply HSV degradation with spatial variation."""
    try:
        from scipy.ndimage import zoom
    except ImportError:
        # Fallback to uniform degradation if scipy not available
        return _apply_uniform_hsv_degradation(image, mask_bool, magnitude)
    
    hsv = color.rgb2hsv(image)
    h, w = mask_bool.shape[:2]
    
    # Generate smooth random fields for spatial variation
    h_field = np.random.uniform(-1, 1, (max(1, h//4), max(1, w//4)))
    s_field = np.random.uniform(-1, 1, (max(1, h//4), max(1, w//4)))
    v_field = np.random.uniform(-1, 1, (max(1, h//4), max(1, w//4)))
    
    # Upsample to full resolution
    h_field = zoom(h_field, (h/h_field.shape[0], w/h_field.shape[1]), order=1)[:h, :w]
    s_field = zoom(s_field, (h/s_field.shape[0], w/s_field.shape[1]), order=1)[:h, :w]
    v_field = zoom(v_field, (h/v_field.shape[0], w/v_field.shape[1]), order=1)[:h, :w]
    
    # Apply spatially-varying degradation
    h_shift = h_field * magnitude * (2/360)
    s_shift = s_field * magnitude * 0.03
    v_shift = v_field * magnitude * 0.03
    
    hsv_degraded = hsv.copy()
    hsv_degraded[mask_bool, 0] = np.clip(hsv_degraded[mask_bool, 0] + h_shift[mask_bool], 0, 1)
    hsv_degraded[mask_bool, 1] = np.clip(hsv_degraded[mask_bool, 1] + s_shift[mask_bool], 0, 1)
    hsv_degraded[mask_bool, 2] = np.clip(hsv_degraded[mask_bool, 2] + v_shift[mask_bool], 0, 1)
    
    return color.hsv2rgb(hsv_degraded)


def _apply_enhanced_lab_degradation(image: np.ndarray, mask_bool: np.ndarray, magnitude: float) -> np.ndarray:
    """Apply LAB degradation with spatial variation."""
    try:
        from scipy.ndimage import zoom
    except ImportError:
        # Fallback to uniform degradation if scipy not available
        return _apply_uniform_lab_degradation(image, mask_bool, magnitude)
    
    lab = color.rgb2lab(image)
    h, w = mask_bool.shape[:2]
    
    # Generate smooth random fields
    l_field = np.random.uniform(-1, 1, (max(1, h//4), max(1, w//4)))
    a_field = np.random.uniform(-1, 1, (max(1, h//4), max(1, w//4)))
    b_field = np.random.uniform(-1, 1, (max(1, h//4), max(1, w//4)))
    
    # Upsample to full resolution
    l_field = zoom(l_field, (h/l_field.shape[0], w/l_field.shape[1]), order=1)[:h, :w]
    a_field = zoom(a_field, (h/a_field.shape[0], w/a_field.shape[1]), order=1)[:h, :w]
    b_field = zoom(b_field, (h/b_field.shape[0], w/b_field.shape[1]), order=1)[:h, :w]
    
    # Apply spatially-varying degradation
    l_shift = l_field * magnitude * 2.0
    a_shift = a_field * magnitude * 1.5
    b_shift = b_field * magnitude * 1.5
    
    lab_degraded = lab.copy()
    lab_degraded[mask_bool, 0] = np.clip(lab_degraded[mask_bool, 0] + l_shift[mask_bool], 0, 100)
    lab_degraded[mask_bool, 1] = np.clip(lab_degraded[mask_bool, 1] + a_shift[mask_bool], -127, 128)
    lab_degraded[mask_bool, 2] = np.clip(lab_degraded[mask_bool, 2] + b_shift[mask_bool], -127, 128)
    
    return color.lab2rgb(lab_degraded)


def _apply_enhanced_rgb_degradation(image: np.ndarray, mask_bool: np.ndarray, magnitude: float) -> np.ndarray:
    """Apply RGB degradation with per-channel variation."""
    degraded = image.copy()
    
    # Apply per-channel offset with spatial variation
    for c in range(3):
        offset = np.random.uniform(-4/255, 4/255) * magnitude
        # Add some noise for realism
        noise = np.random.normal(0, 0.5/255, mask_bool.shape) * magnitude
        degraded[mask_bool, c] = np.clip(degraded[mask_bool, c] + offset + noise[mask_bool], 0, 1)
    
    return degraded


def _apply_uniform_hsv_degradation(image: np.ndarray, mask_bool: np.ndarray, magnitude: float) -> np.ndarray:
    """Fallback uniform HSV degradation."""
    hsv = color.rgb2hsv(image)
    
    h_shift = np.random.uniform(-2/360, 2/360) * magnitude
    s_shift = np.random.uniform(-0.03, 0.03) * magnitude
    v_shift = np.random.uniform(-0.03, 0.03) * magnitude
    
    hsv_degraded = hsv.copy()
    hsv_degraded[mask_bool, 0] = np.clip(hsv_degraded[mask_bool, 0] + h_shift, 0, 1)
    hsv_degraded[mask_bool, 1] = np.clip(hsv_degraded[mask_bool, 1] + s_shift, 0, 1)
    hsv_degraded[mask_bool, 2] = np.clip(hsv_degraded[mask_bool, 2] + v_shift, 0, 1)
    
    return color.hsv2rgb(hsv_degraded)


def _apply_uniform_lab_degradation(image: np.ndarray, mask_bool: np.ndarray, magnitude: float) -> np.ndarray:
    """Fallback uniform LAB degradation."""
    lab = color.rgb2lab(image)
    
    l_shift = np.random.uniform(-2.0, 2.0) * magnitude
    a_shift = np.random.uniform(-1.5, 1.5) * magnitude
    b_shift = np.random.uniform(-1.5, 1.5) * magnitude
    
    lab_degraded = lab.copy()
    lab_degraded[mask_bool, 0] = np.clip(lab_degraded[mask_bool, 0] + l_shift, 0, 100)
    lab_degraded[mask_bool, 1] = np.clip(lab_degraded[mask_bool, 1] + a_shift, -127, 128)
    lab_degraded[mask_bool, 2] = np.clip(lab_degraded[mask_bool, 2] + b_shift, -127, 128)
    
    return color.lab2rgb(lab_degraded)


class GarmentPairDataset(Dataset):
    """Dataset for Model 1 - recoloring with degraded inputs."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 512,
        degrade_params: Optional[Dict] = None,
        augment: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.degrade_params = degrade_params or {}
        self.augment = augment and split == "train"
        
        # Find all item directories
        split_dir = self.data_root / split
        self.items = []
        if split_dir.exists():
            for item_dir in sorted(split_dir.iterdir()):
                if item_dir.is_dir():
                    required_files = ["still.jpg", "on_model.jpg", "mask_still.png", "mask_on_model.png"]
                    if all((item_dir / f).exists() for f in required_files):
                        self.items.append(item_dir)
        
        if not self.items:
            raise ValueError(f"No valid items found in {split_dir}")
        
        # Augmentation transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5) if self.augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item_dir = self.items[idx]
        
        # Load images and masks
        still = load_image(item_dir / "still.jpg", (self.img_size, self.img_size))
        on_model = load_image(item_dir / "on_model.jpg", (self.img_size, self.img_size))
        mask_still = load_mask(item_dir / "mask_still.png", (self.img_size, self.img_size))
        mask_on = load_mask(item_dir / "mask_on_model.png", (self.img_size, self.img_size))
        
        # Apply degradation to on_model if enabled
        on_model_input = on_model.copy()
        if self.degrade_params.get("enable", False):
            on_model_input = apply_light_degradation(
                on_model_input,
                mask_on,
                mode=self.degrade_params.get("mode", "lab"),
                magnitude=self.degrade_params.get("magnitude", 0.04),
                seed=None  # Random for training
            )
        
        # Convert to tensors
        still_tensor = torch.from_numpy(still).permute(2, 0, 1)
        on_model_input_tensor = torch.from_numpy(on_model_input).permute(2, 0, 1)
        on_model_target_tensor = torch.from_numpy(on_model).permute(2, 0, 1)
        mask_still_tensor = torch.from_numpy(mask_still).unsqueeze(0)
        mask_on_tensor = torch.from_numpy(mask_on).unsqueeze(0)
        
        # Apply same augmentation to all tensors
        if self.augment:
            # Stack all tensors for consistent augmentation
            stacked = torch.cat([
                still_tensor,
                on_model_input_tensor,
                on_model_target_tensor,
                mask_still_tensor,
                mask_on_tensor
            ], dim=0)
            
            # Apply transform (only horizontal flip for simplicity)
            if random.random() < 0.5:
                stacked = torch.flip(stacked, dims=[2])
            
            # Unstack with .clone() to avoid shared memory/views
            still_tensor = stacked[0:3].clone()
            on_model_input_tensor = stacked[3:6].clone()
            on_model_target_tensor = stacked[6:9].clone()
            mask_still_tensor = stacked[9:10].clone()
            mask_on_tensor = stacked[10:11].clone()
        
        return {
            "still_ref": still_tensor,
            "on_model_input": on_model_input_tensor,
            "on_model_target": on_model_target_tensor,
            "mask_still": mask_still_tensor,
            "mask_on": mask_on_tensor,
            "meta": {
                "item_name": item_dir.name,
                "item_path": str(item_dir)
            }
        }


class GarmentSegDataset(Dataset):
    """Dataset for Models 2 and 3 - segmentation."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 512,
        target_type: str = "still",  # "still" or "on_model"
        augment_params: Optional[Dict] = None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.target_type = target_type
        self.augment_params = augment_params or {}
        
        # Determine file names based on target type
        if target_type == "still":
            self.img_name = "still.jpg"
            self.mask_name = "mask_still.png"
        elif target_type == "on_model":
            self.img_name = "on_model.jpg"
            self.mask_name = "mask_on_model.png"
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        # Find all item directories
        split_dir = self.data_root / split
        self.items = []
        if split_dir.exists():
            for item_dir in sorted(split_dir.iterdir()):
                if item_dir.is_dir():
                    if (item_dir / self.img_name).exists() and (item_dir / self.mask_name).exists():
                        self.items.append(item_dir)
        
        if not self.items:
            raise ValueError(f"No valid items found in {split_dir}")
        
        self.augment = self.augment_params and split == "train"
        
        # Create albumentations transform if augmentations are enabled
        if self.augment:
            self.transform = get_segmentation_transforms(self.augment_params, self.img_size)
        else:
            # Simple transform for validation/test
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item_dir = self.items[idx]
        
        # Load image and mask
        image = load_image(item_dir / self.img_name, (self.img_size, self.img_size))
        mask = load_mask(item_dir / self.mask_name, (self.img_size, self.img_size))
        
        # Convert to uint8 for albumentations (0-255 range)
        image_uint8 = (image * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Apply albumentations transform
        transformed = self.transform(image=image_uint8, mask=mask_uint8)
        
        # Extract transformed image and mask
        image_tensor = transformed['image']  # Already a tensor from ToTensorV2
        mask_tensor = transformed['mask'].unsqueeze(0)  # Add channel dimension
        
        # Convert to float32 and normalize to [0, 1]
        image_tensor = image_tensor.float() / 255.0
        mask_tensor = mask_tensor.float() / 255.0
        
        # Ensure tensors are contiguous and have correct size
        image_tensor = image_tensor.contiguous()
        mask_tensor = mask_tensor.contiguous()
        
        # Double-check tensor shapes
        assert image_tensor.shape == (3, self.img_size, self.img_size), f"Image tensor shape mismatch: {image_tensor.shape}"
        assert mask_tensor.shape == (1, self.img_size, self.img_size), f"Mask tensor shape mismatch: {mask_tensor.shape}"
        
        return image_tensor, mask_tensor


class DualInputSegDataset(Dataset):
    """Dataset for Model 3 with dual inputs: still + on_model for better segmentation."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 512,
        augment_params: Optional[Dict] = None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.augment_params = augment_params or {}
        
        # Find all item directories
        split_dir = self.data_root / split
        self.items = []
        if split_dir.exists():
            for item_dir in sorted(split_dir.iterdir()):
                if item_dir.is_dir():
                    required_files = ["still.jpg", "on_model.jpg", "mask_still.png", "mask_on_model.png"]
                    if all((item_dir / f).exists() for f in required_files):
                        self.items.append(item_dir)
        
        if not self.items:
            raise ValueError(f"No valid items found in {split_dir}")
        
        self.augment = self.augment_params and split == "train"
        
        # Create albumentations transform if augmentations are enabled
        if self.augment:
            self.transform = get_segmentation_transforms(self.augment_params, self.img_size)
        else:
            # Simple transform for validation/test
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item_dir = self.items[idx]
        
        # Load all images and masks
        still = load_image(item_dir / "still.jpg", (self.img_size, self.img_size))
        on_model = load_image(item_dir / "on_model.jpg", (self.img_size, self.img_size))
        mask_still = load_mask(item_dir / "mask_still.png", (self.img_size, self.img_size))
        mask_on_model = load_mask(item_dir / "mask_on_model.png", (self.img_size, self.img_size))
        
        if self.augment:
            # Convert to uint8 for albumentations (0-255 range)
            still_uint8 = (still * 255).astype(np.uint8)
            on_model_uint8 = (on_model * 255).astype(np.uint8)
            mask_still_uint8 = (mask_still * 255).astype(np.uint8)
            mask_on_model_uint8 = (mask_on_model * 255).astype(np.uint8)
            
            # Apply same augmentation to still and its mask
            still_transformed = self.transform(image=still_uint8, mask=mask_still_uint8)
            still_tensor = still_transformed['image'].float() / 255.0
            mask_still_tensor = still_transformed['mask'].unsqueeze(0).float() / 255.0
            
            # Apply same augmentation to on_model and its mask
            on_model_transformed = self.transform(image=on_model_uint8, mask=mask_on_model_uint8)
            on_model_tensor = on_model_transformed['image'].float() / 255.0
            mask_on_model_tensor = on_model_transformed['mask'].unsqueeze(0).float() / 255.0
            
        else:
            # Simple transform for validation/test
            still_transformed = self.transform(image=(still * 255).astype(np.uint8))
            on_model_transformed = self.transform(image=(on_model * 255).astype(np.uint8))
            
            still_tensor = still_transformed['image'].float() / 255.0
            on_model_tensor = on_model_transformed['image'].float() / 255.0
            mask_still_tensor = torch.from_numpy(mask_still).unsqueeze(0).float()
            mask_on_model_tensor = torch.from_numpy(mask_on_model).unsqueeze(0).float()
        
        return {
            "still": still_tensor,                    # Reference garment image
            "on_model": on_model_tensor,             # Image to segment
            "mask_still": mask_still_tensor,         # Reference mask (for auxiliary loss)
            "mask_on_model": mask_on_model_tensor,   # Target mask
            "meta": {
                "item_path": str(item_dir)
            }
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with proper settings."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(num_workers, os.cpu_count() or 1),
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=shuffle  # Drop last batch only for training
    )
