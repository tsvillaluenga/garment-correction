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
    """Load mask as binary float32 [0, 1] with validation."""
    path = Path(path)
    
    # Validate path exists
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {path}")
        
        if mask.size == 0:
            raise ValueError(f"Empty mask loaded: {path}")
        
        # Resize if requested with validation
        if size is not None:
            if not isinstance(size, (tuple, list)) or len(size) != 2:
                raise ValueError("Size must be (width, height) tuple")
            
            width, height = size
            if width <= 0 or height <= 0:
                raise ValueError("Size dimensions must be positive")
            
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary float32
        binary_mask = (mask > 127).astype(np.float32)
        
        # Log warning for unusual masks
        mask_ratio = np.mean(binary_mask)
        if mask_ratio < 0.001:
            import logging
            logging.warning(f"Very small mask region ({mask_ratio:.3%}) in {path}")
        elif mask_ratio > 0.999:
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
        mode: 'hsv', 'lab', or 'rgb'
        magnitude: Degradation strength
        seed: Random seed for reproducibility
        
    Returns:
        Degraded image with same shape as input
    """
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
        raise ValueError(f"Unknown degradation mode: {mode}")
    
    return np.clip(degraded, 0, 1)


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
            
            # Unstack
            still_tensor = stacked[0:3]
            on_model_input_tensor = stacked[3:6]
            on_model_target_tensor = stacked[6:9]
            mask_still_tensor = stacked[9:10]
            mask_on_tensor = stacked[10:11]
        
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
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item_dir = self.items[idx]
        
        # Load image and mask
        image = load_image(item_dir / self.img_name, (self.img_size, self.img_size))
        mask = load_mask(item_dir / self.mask_name, (self.img_size, self.img_size))
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        # Apply augmentations if enabled
        if self.augment:
            # Horizontal flip
            if self.augment_params.get("hflip", False) and random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[2])
            
            # Rotation (small angles only)
            if self.augment_params.get("rotate_deg", 0) > 0:
                angle = random.uniform(
                    -self.augment_params["rotate_deg"],
                    self.augment_params["rotate_deg"]
                )
                # Simple rotation using torch transforms would require PIL conversion
                # For now, skip rotation to keep it simple
            
            # Scale (resize and crop)
            if "scale" in self.augment_params:
                scale_range = self.augment_params["scale"]
                scale = random.uniform(scale_range[0], scale_range[1])
                if scale != 1.0:
                    new_size = int(self.img_size * scale)
                    image_tensor = F.interpolate(
                        image_tensor.unsqueeze(0),
                        size=(new_size, new_size),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0),
                        size=(new_size, new_size),
                        mode="nearest"
                    ).squeeze(0)
                    
                    # Center crop back to original size
                    if new_size > self.img_size:
                        start = (new_size - self.img_size) // 2
                        image_tensor = image_tensor[:, start:start+self.img_size, start:start+self.img_size]
                        mask_tensor = mask_tensor[:, start:start+self.img_size, start:start+self.img_size]
                    elif new_size < self.img_size:
                        # Pad to original size
                        pad = (self.img_size - new_size) // 2
                        image_tensor = F.pad(image_tensor, (pad, pad, pad, pad))
                        mask_tensor = F.pad(mask_tensor, (pad, pad, pad, pad))
        
        return image_tensor, mask_tensor


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
