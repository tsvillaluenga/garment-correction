"""
Test color conversion and Delta E computations.
"""
import pytest
import numpy as np
import torch
from skimage import color

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from losses_metrics import rgb_to_lab_torch, delta_e76_torch


class TestColorConversions:
    """Test color space conversions and metrics."""
    
    def test_rgb_to_lab_basic(self):
        """Test basic RGB to LAB conversion."""
        # Create a simple RGB image
        rgb = np.array([[[1.0, 0.0, 0.0],   # Pure red
                        [0.0, 1.0, 0.0],   # Pure green
                        [0.0, 0.0, 1.0],   # Pure blue
                        [1.0, 1.0, 1.0]]]) # White
        
        # Convert using our function
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        lab_tensor = rgb_to_lab_torch(rgb_tensor)
        lab_result = lab_tensor.squeeze().permute(1, 2, 0).numpy()
        
        # Convert using scikit-image for comparison
        lab_expected = color.rgb2lab(rgb)
        
        # Check if results are close (allowing for small numerical differences)
        np.testing.assert_allclose(lab_result, lab_expected, rtol=1e-3, atol=1e-3)
    
    def test_delta_e76_identical(self):
        """Test Delta E 1976 for identical colors."""
        # Create identical LAB colors
        lab1 = torch.tensor([[[[50.0, 0.0, 0.0],
                              [75.0, 10.0, -5.0]]]])
        lab1 = lab1.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        lab2 = lab1.clone()
        
        delta_e = delta_e76_torch(lab1, lab2)
        
        # Delta E should be zero for identical colors
        assert torch.allclose(delta_e, torch.zeros_like(delta_e), atol=1e-6)
    
    def test_delta_e76_known_values(self):
        """Test Delta E 1976 for known color differences."""
        # Create two slightly different colors
        lab1 = torch.tensor([[[[50.0, 0.0, 0.0]]]])  # L*=50, a*=0, b*=0
        lab2 = torch.tensor([[[[53.0, 4.0, -3.0]]]])  # L*=53, a*=4, b*=-3
        
        lab1 = lab1.permute(0, 3, 1, 2)
        lab2 = lab2.permute(0, 3, 1, 2)
        
        delta_e = delta_e76_torch(lab1, lab2)
        
        # Expected Delta E = sqrt((53-50)^2 + (4-0)^2 + (-3-0)^2) = sqrt(9+16+9) = sqrt(34) ≈ 5.83
        expected = np.sqrt(3**2 + 4**2 + 3**2)
        
        assert torch.allclose(delta_e, torch.tensor(expected), atol=1e-3)
    
    def test_delta_e76_batch(self):
        """Test Delta E computation on batches."""
        batch_size = 4
        height, width = 32, 32
        
        # Create random LAB tensors
        lab1 = torch.randn(batch_size, 3, height, width)
        lab2 = torch.randn(batch_size, 3, height, width)
        
        # Ensure LAB values are in reasonable ranges
        lab1[:, 0] = torch.clamp(lab1[:, 0] * 20 + 50, 0, 100)  # L*
        lab1[:, 1:] = torch.clamp(lab1[:, 1:] * 30, -127, 128)  # a*, b*
        lab2[:, 0] = torch.clamp(lab2[:, 0] * 20 + 50, 0, 100)  # L*
        lab2[:, 1:] = torch.clamp(lab2[:, 1:] * 30, -127, 128)  # a*, b*
        
        delta_e = delta_e76_torch(lab1, lab2)
        
        # Check output shape
        assert delta_e.shape == (batch_size, 1, height, width)
        
        # Check that all values are non-negative
        assert torch.all(delta_e >= 0)
    
    def test_rgb_lab_roundtrip(self):
        """Test RGB -> LAB -> RGB roundtrip (approximately)."""
        # Create test RGB image
        rgb_original = np.random.rand(1, 64, 64, 3).astype(np.float32)
        
        # Convert to tensor and back
        rgb_tensor = torch.from_numpy(rgb_original).permute(0, 3, 1, 2)
        lab_tensor = rgb_to_lab_torch(rgb_tensor)
        
        # Convert LAB back to RGB using scikit-image
        lab_np = lab_tensor.squeeze().permute(1, 2, 0).numpy()
        rgb_recovered = color.lab2rgb(lab_np)
        
        # Check if roundtrip is approximately correct
        # (Some precision loss is expected due to color space conversion)
        np.testing.assert_allclose(
            rgb_original.squeeze(), rgb_recovered, 
            rtol=1e-2, atol=1e-2
        )
    
    def test_delta_e76_perceptual_ordering(self):
        """Test that Delta E respects perceptual color differences."""
        # White reference
        white_lab = torch.tensor([[[[95.0, 0.0, 0.0]]]])  # Approximately white
        white_lab = white_lab.permute(0, 3, 1, 2)
        
        # Light gray (should be close to white)
        light_gray_lab = torch.tensor([[[[85.0, 0.0, 0.0]]]])
        light_gray_lab = light_gray_lab.permute(0, 3, 1, 2)
        
        # Dark gray (should be far from white)
        dark_gray_lab = torch.tensor([[[[30.0, 0.0, 0.0]]]])
        dark_gray_lab = dark_gray_lab.permute(0, 3, 1, 2)
        
        # Compute Delta E values
        de_light = delta_e76_torch(white_lab, light_gray_lab)
        de_dark = delta_e76_torch(white_lab, dark_gray_lab)
        
        # Dark gray should be farther from white than light gray
        assert torch.all(de_dark > de_light)


if __name__ == "__main__":
    pytest.main([__file__])
