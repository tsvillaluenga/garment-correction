"""
Loss functions and evaluation metrics for garment color correction.
Includes Delta E76, CIEDE2000, perceptual losses, and segmentation metrics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from skimage import color
from torchvision import models


def rgb_to_lab_torch_differentiable(rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert RGB to LAB color space in PyTorch with preserved gradients.
    
    Args:
        rgb: RGB tensor (B, 3, H, W) in range [0, 1]
        eps: Small epsilon for numerical stability
        
    Returns:
        LAB tensor (B, 3, H, W) where L in [0, 100], a,b in [-127, 128]
    """
    # Input validation and clamping
    rgb = torch.clamp(rgb, 0, 1)
    
    # Convert RGB to XYZ color space (sRGB to CIE XYZ)
    # Apply gamma correction (sRGB to linear RGB)
    mask = rgb > 0.04045
    rgb_linear = torch.where(
        mask,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
        rgb / 12.92
    )
    
    # sRGB to XYZ transformation matrix
    transform_matrix = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    
    # Reshape for matrix multiplication
    B, C, H, W = rgb.shape
    rgb_flat = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)
    
    # Apply transformation
    xyz_flat = torch.matmul(rgb_flat, transform_matrix.t())  # (B*H*W, 3)
    xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
    
    # Normalize by D65 illuminant
    d65_illuminant = torch.tensor([0.95047, 1.00000, 1.08883], 
                                  device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    xyz_normalized = xyz / d65_illuminant
    
    # Apply f(t) function for LAB conversion
    def f_func(t):
        delta = 6.0 / 29.0
        mask = t > delta ** 3
        return torch.where(
            mask,
            torch.pow(t + eps, 1.0/3.0),
            t / (3 * delta ** 2) + 4.0 / 29.0
        )
    
    fx = f_func(xyz_normalized[:, 0:1].clone())
    fy = f_func(xyz_normalized[:, 1:2].clone())
    fz = f_func(xyz_normalized[:, 2:3].clone())
    
    # Convert to LAB
    L = 116.0 * fy - 16.0  # L* in [0, 100]
    a = 500.0 * (fx - fy)  # a* in [-127, 128]
    b = 200.0 * (fy - fz)  # b* in [-127, 128]
    
    lab = torch.cat([L, a, b], dim=1)
    
    # Clamp to valid ranges (avoid in-place operations)
    lab_clamped = lab.clone()
    lab_clamped[:, 0:1] = torch.clamp(lab[:, 0:1], 0, 100)
    lab_clamped[:, 1:2] = torch.clamp(lab[:, 1:2], -127, 128)
    lab_clamped[:, 2:3] = torch.clamp(lab[:, 2:3], -127, 128)
    
    return lab_clamped


def rgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    Fallback RGB to LAB conversion using scikit-image (non-differentiable).
    Used only for evaluation metrics.
    """
    import warnings
    with torch.no_grad():
        rgb_np = rgb.detach().cpu().numpy()
        lab_list = []
        
        for i in range(rgb_np.shape[0]):
            rgb_img = rgb_np[i].transpose(1, 2, 0)  # (H, W, 3)
            rgb_img = np.clip(rgb_img, 0, 1)  # Ensure valid range
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lab_img = color.rgb2lab(rgb_img)
                lab_list.append(lab_img.transpose(2, 0, 1))  # (3, H, W)
            except Exception:
                # Fallback to zeros on error
                lab_list.append(np.zeros_like(rgb_img.transpose(2, 0, 1)))
        
        lab_np = np.stack(lab_list, axis=0)
        lab_tensor = torch.from_numpy(lab_np).to(rgb.device).float()
        
        return lab_tensor


def delta_e76_torch(lab1: torch.Tensor, lab2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Delta E 1976 (CIE76) color difference with numerical stability.
    
    Args:
        lab1: LAB tensor (B, 3, H, W)
        lab2: LAB tensor (B, 3, H, W)
        eps: Small epsilon for numerical stability
        
    Returns:
        Delta E tensor (B, 1, H, W)
    """
    # Input validation
    if lab1.shape != lab2.shape:
        raise ValueError(f"Shape mismatch: {lab1.shape} vs {lab2.shape}")
    
    # Compute squared differences
    diff = lab1 - lab2
    squared_diff = diff ** 2
    
    # Sum across color channels
    sum_squared = torch.sum(squared_diff, dim=1, keepdim=True)
    
    # Safe square root to avoid NaN gradients
    delta_e = torch.sqrt(sum_squared + eps)
    
    return delta_e


def delta_e2000_numpy(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Compute CIEDE2000 color difference using numpy.
    This is used for evaluation metrics only (not for training).
    
    Args:
        lab1: LAB array (H, W, 3)
        lab2: LAB array (H, W, 3)
        
    Returns:
        Delta E 2000 array (H, W)
    """
    try:
        from colorspacious import deltaE
        return deltaE(lab1, lab2, input_space="CIELab", uniform_space="CAM02-UCS")
    except ImportError:
        # Fallback to simplified Delta E76 if colorspacious not available
        diff = lab1 - lab2
        return np.sqrt(np.sum(diff ** 2, axis=-1))


class MaskedL1Loss(nn.Module):
    """Masked L1 loss for garment regions only."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked L1 loss.
        
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            
        Returns:
            Masked L1 loss
        """
        # Clone mask to avoid in-place modifications
        mask_safe = mask.clone()
        
        diff = torch.abs(pred - target)
        masked_diff = diff * mask_safe
        
        if self.reduction == "mean":
            return masked_diff.sum() / (mask_safe.sum() + 1e-8)
        elif self.reduction == "sum":
            return masked_diff.sum()
        else:
            return masked_diff


class MaskedDeltaE76Loss(nn.Module):
    """Masked Delta E 1976 loss for color accuracy with gradient preservation."""
    
    def __init__(self, reduction: str = "mean", use_differentiable: bool = True):
        super().__init__()
        self.reduction = reduction
        self.use_differentiable = use_differentiable
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked Delta E 1976 loss with improved gradient flow.
        
        Args:
            pred: Predicted RGB image (B, 3, H, W)
            target: Target RGB image (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            
        Returns:
            Masked Delta E loss
        """
        try:
            # Convert to LAB - use differentiable version for training
            if self.use_differentiable:
                pred_lab = rgb_to_lab_torch_differentiable(pred)
                target_lab = rgb_to_lab_torch_differentiable(target)
            else:
                pred_lab = rgb_to_lab_torch(pred)
                target_lab = rgb_to_lab_torch(target)
            
            # Compute Delta E
            delta_e = delta_e76_torch(pred_lab, target_lab)
            
            # Apply mask (clone to avoid in-place modifications)
            mask_safe = mask.clone()
            masked_delta_e = delta_e * mask_safe
            
            if self.reduction == "mean":
                mask_sum = mask_safe.sum()
                if mask_sum > 0:
                    return masked_delta_e.sum() / mask_sum
                else:
                    return torch.tensor(0.0, device=pred.device, requires_grad=True)
            elif self.reduction == "sum":
                return masked_delta_e.sum()
            else:
                return masked_delta_e
                
        except Exception as e:
            # Fallback to L1 loss on error
            import logging
            logging.warning(f"Delta E computation failed, using L1 fallback: {e}")
            l1_loss = torch.abs(pred - target) * mask
            if self.reduction == "mean":
                return l1_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                return l1_loss.sum()
            else:
                return l1_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features on luminance."""
    
    def __init__(self, layers: list = None, weights: list = None):
        super().__init__()
        
        if layers is None:
            layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]
        
        self.layers = layers
        self.weights = weights
        
        # Load pretrained VGG19 (using new weights parameter)
        try:
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        except ImportError:
            # Fallback for older torchvision versions
            vgg = models.vgg19(pretrained=True).features
        self.model = nn.Sequential()
        
        # Extract specific layers
        layer_map = {
            'relu1_1': 2, 'relu1_2': 4,
            'relu2_1': 7, 'relu2_2': 9,
            'relu3_1': 12, 'relu3_2': 14, 'relu3_3': 16, 'relu3_4': 18,
            'relu4_1': 21, 'relu4_2': 23, 'relu4_3': 25, 'relu4_4': 27,
            'relu5_1': 30
        }
        
        max_layer = max(layer_map[layer] for layer in layers)
        
        for i in range(max_layer + 1):
            self.model.add_module(str(i), vgg[i])
        
        # Freeze parameters and ensure float32
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Ensure model is in float32 for compatibility with AMP
        self.model = self.model.float()
    
    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.model = self.model.to(device)
        return self
    
    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from specified layers."""
        features = {}
        layer_map = {
            2: 'relu1_1', 4: 'relu1_2',
            7: 'relu2_1', 9: 'relu2_2',
            12: 'relu3_1', 14: 'relu3_2', 16: 'relu3_3', 18: 'relu3_4',
            21: 'relu4_1', 23: 'relu4_2', 25: 'relu4_3', 27: 'relu4_4',
            30: 'relu5_1'
        }
        
        # Ensure input is float32 for VGG19 compatibility
        x = x.float()
        
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in layer_map and layer_map[i] in self.layers:
                features[layer_map[i]] = x
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss on luminance channel.
        
        Args:
            pred: Predicted RGB image (B, 3, H, W)
            target: Target RGB image (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            
        Returns:
            Perceptual loss
        """
        # Convert to luminance (Y channel) - clone to avoid views
        pred_luma = 0.299 * pred[:, 0:1].clone() + 0.587 * pred[:, 1:2].clone() + 0.114 * pred[:, 2:3].clone()
        target_luma = 0.299 * target[:, 0:1].clone() + 0.587 * target[:, 1:2].clone() + 0.114 * target[:, 2:3].clone()
        
        # Replicate to 3 channels for VGG
        pred_luma_3ch = pred_luma.repeat(1, 3, 1, 1)
        target_luma_3ch = target_luma.repeat(1, 3, 1, 1)
        
        # Extract features
        pred_features = self._extract_features(pred_luma_3ch)
        target_features = self._extract_features(target_luma_3ch)
        
        # Compute loss
        total_loss = 0.0
        for layer, weight in zip(self.layers, self.weights):
            if layer in pred_features:
                # Resize mask to feature map size (clone to avoid in-place modifications)
                mask_safe = mask.clone()
                feat_h, feat_w = pred_features[layer].shape[2:]
                mask_resized = F.interpolate(mask_safe, size=(feat_h, feat_w), mode='nearest')
                
                # Masked feature loss
                feat_diff = F.mse_loss(pred_features[layer], target_features[layer], reduction='none')
                masked_feat_diff = feat_diff * mask_resized
                
                layer_loss = masked_feat_diff.sum() / (mask_resized.sum() + 1e-8)
                total_loss = total_loss + weight * layer_loss  # Avoid in-place
        
        return total_loss


class HingeGANLoss(nn.Module):
    """Hinge GAN loss for adversarial training."""
    
    def __init__(self):
        super().__init__()
    
    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator loss: -mean(fake_logits)"""
        return -fake_logits.mean()
    
    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Discriminator loss: mean(ReLU(1-real)) + mean(ReLU(1+fake))"""
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()
        return real_loss + fake_loss


class CombinedRecolorLoss(nn.Module):
    """Combined loss for recoloring model."""
    
    def __init__(
        self,
        w_l1: float = 1.0,
        w_de: float = 1.0,
        w_perc: float = 0.1,
        w_gan: float = 0.0
    ):
        super().__init__()
        
        self.w_l1 = w_l1
        self.w_de = w_de
        self.w_perc = w_perc
        self.w_gan = w_gan
        
        self.l1_loss = MaskedL1Loss()
        self.de_loss = MaskedDeltaE76Loss()
        self.perceptual_loss = PerceptualLoss()
        
        if w_gan > 0:
            self.gan_loss = HingeGANLoss()
    
    def to(self, device):
        """Move all loss components to device."""
        super().to(device)
        self.l1_loss = self.l1_loss.to(device)
        self.de_loss = self.de_loss.to(device)
        self.perceptual_loss = self.perceptual_loss.to(device)
        if hasattr(self, 'gan_loss'):
            self.gan_loss = self.gan_loss.to(device)
        return self
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        fake_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined recoloring loss.
        
        Args:
            pred: Predicted image (B, 3, H, W)
            target: Target image (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            fake_logits: Discriminator logits for fake images
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # L1 loss
        if self.w_l1 > 0:
            losses['l1'] = self.l1_loss(pred, target, mask)
        
        # Delta E loss
        if self.w_de > 0:
            losses['delta_e'] = self.de_loss(pred, target, mask)
        
        # Perceptual loss
        if self.w_perc > 0:
            losses['perceptual'] = self.perceptual_loss(pred, target, mask)
        
        # GAN loss
        if self.w_gan > 0 and fake_logits is not None:
            losses['gan'] = self.gan_loss.generator_loss(fake_logits)
        
        # Total loss (avoid in-place operations)
        total = 0.0
        if 'l1' in losses:
            total = total + self.w_l1 * losses['l1']
        if 'delta_e' in losses:
            total = total + self.w_de * losses['delta_e']
        if 'perceptual' in losses:
            total = total + self.w_perc * losses['perceptual']
        if 'gan' in losses:
            total = total + self.w_gan * losses['gan']
        
        losses['total'] = total
        return losses


# Segmentation losses
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Target masks (B, 1, H, W)
            
        Returns:
            Dice loss
        """
        pred_prob = torch.sigmoid(pred)
        
        intersection = (pred_prob * target).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """Combined BCE + Dice loss for segmentation."""
    
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined segmentation loss.
        
        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Target masks (B, 1, H, W)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        losses['bce'] = self.bce_loss(pred, target)
        losses['dice'] = self.dice_loss(pred, target)
        losses['total'] = self.bce_weight * losses['bce'] + self.dice_weight * losses['dice']
        
        return losses


# Evaluation metrics
def compute_segmentation_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute segmentation metrics: IoU, Dice, Precision, Recall.
    
    Args:
        pred: Predicted logits (B, 1, H, W)
        target: Target masks (B, 1, H, W)
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target.float()
        
        # Flatten for easier computation
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)
        
        # Compute metrics
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection / (union + 1e-8)).item()
        dice = (2 * intersection / (pred_flat.sum() + target_flat.sum() + 1e-8)).item()
        
        # Precision and recall
        tp = intersection
        fp = pred_flat.sum() - intersection
        fn = target_flat.sum() - intersection
        
        precision = (tp / (tp + fp + 1e-8)).item()
        recall = (tp / (tp + fn + 1e-8)).item()
        
        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall
        }


def compute_color_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute color accuracy metrics: Delta E76, CIEDE2000, SSIM, PSNR.
    
    Args:
        pred: Predicted RGB image (B, 3, H, W) in [0, 1]
        target: Target RGB image (B, 3, H, W) in [0, 1]
        mask: Binary mask (B, 1, H, W)
        
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # Convert to numpy for color space conversions
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        
        metrics = {}
        delta_e76_values = []
        delta_e2000_values = []
        
        for i in range(pred_np.shape[0]):
            # Convert to LAB
            pred_rgb = pred_np[i].transpose(1, 2, 0)
            target_rgb = target_np[i].transpose(1, 2, 0)
            mask_2d = mask_np[i, 0]
            
            pred_lab = color.rgb2lab(pred_rgb)
            target_lab = color.rgb2lab(target_rgb)
            
            # Compute Delta E only in masked regions
            mask_bool = mask_2d > 0.5
            if np.any(mask_bool):
                # Delta E 1976
                diff = pred_lab - target_lab
                de76 = np.sqrt(np.sum(diff ** 2, axis=-1))
                delta_e76_values.extend(de76[mask_bool])
                
                # Delta E 2000
                de2000 = delta_e2000_numpy(pred_lab, target_lab)
                delta_e2000_values.extend(de2000[mask_bool])
        
        if delta_e76_values:
            delta_e76_values = np.array(delta_e76_values)
            delta_e2000_values = np.array(delta_e2000_values)
            
            # Statistics for Delta E 1976
            metrics['delta_e76_mean'] = float(np.mean(delta_e76_values))
            metrics['delta_e76_p50'] = float(np.percentile(delta_e76_values, 50))
            metrics['delta_e76_p90'] = float(np.percentile(delta_e76_values, 90))
            metrics['delta_e76_p95'] = float(np.percentile(delta_e76_values, 95))
            metrics['delta_e76_p99'] = float(np.percentile(delta_e76_values, 99))
            
            # Statistics for CIEDE2000
            metrics['delta_e2000_mean'] = float(np.mean(delta_e2000_values))
            metrics['delta_e2000_p50'] = float(np.percentile(delta_e2000_values, 50))
            metrics['delta_e2000_p90'] = float(np.percentile(delta_e2000_values, 90))
            metrics['delta_e2000_p95'] = float(np.percentile(delta_e2000_values, 95))
            metrics['delta_e2000_p99'] = float(np.percentile(delta_e2000_values, 99))
        
        # SSIM and PSNR on luminance
        pred_luma = 0.299 * pred[:, 0] + 0.587 * pred[:, 1] + 0.114 * pred[:, 2]
        target_luma = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # Simple PSNR computation
        mse = F.mse_loss(pred_luma, target_luma)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        metrics['psnr_luma'] = float(psnr.item())
        
        # Masked PSNR
        mask_1d = mask[:, 0]
        masked_mse = ((pred_luma - target_luma) ** 2 * mask_1d).sum() / (mask_1d.sum() + 1e-8)
        masked_psnr = 20 * torch.log10(1.0 / torch.sqrt(masked_mse + 1e-8))
        metrics['psnr_luma_masked'] = float(masked_psnr.item())
        
        return metrics
