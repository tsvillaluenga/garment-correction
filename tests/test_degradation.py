"""
Test light degradation functions.
"""
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import apply_light_degradation


class TestDegradation:
    """Test degradation functions."""
    
    def test_degradation_preserves_unmasked(self):
        """Test that degradation only affects masked pixels."""
        # Create test image and mask
        image = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0  # Central square is masked
        
        # Apply degradation
        degraded = apply_light_degradation(image, mask, mode="rgb", magnitude=0.1, seed=42)
        
        # Check that unmasked regions are unchanged
        unmasked_region = mask == 0
        np.testing.assert_allclose(
            image[unmasked_region], 
            degraded[unmasked_region], 
            rtol=1e-6, atol=1e-6
        )
    
    def test_degradation_affects_masked(self):
        """Test that degradation changes masked pixels."""
        # Create test image and mask
        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        mask = np.ones((32, 32), dtype=np.float32)
        
        # Apply degradation
        degraded = apply_light_degradation(image, mask, mode="rgb", magnitude=0.1, seed=42)
        
        # Check that masked region is changed
        assert not np.allclose(image, degraded, rtol=1e-6, atol=1e-6)
    
    def test_degradation_reproducible(self):
        """Test that degradation is reproducible with same seed."""
        # Create test image and mask
        image = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.random.rand(32, 32).astype(np.float32) > 0.5
        
        # Apply degradation with same seed
        degraded1 = apply_light_degradation(image, mask, mode="lab", magnitude=0.05, seed=123)
        degraded2 = apply_light_degradation(image, mask, mode="lab", magnitude=0.05, seed=123)
        
        # Results should be identical
        np.testing.assert_allclose(degraded1, degraded2, rtol=1e-6, atol=1e-6)
    
    def test_degradation_different_seeds(self):
        """Test that different seeds produce different results."""
        # Create test image and mask
        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        mask = np.ones((32, 32), dtype=np.float32)
        
        # Apply degradation with different seeds
        degraded1 = apply_light_degradation(image, mask, mode="hsv", magnitude=0.05, seed=42)
        degraded2 = apply_light_degradation(image, mask, mode="hsv", magnitude=0.05, seed=123)
        
        # Results should be different
        assert not np.allclose(degraded1, degraded2, rtol=1e-3, atol=1e-3)
    
    def test_degradation_modes(self):
        """Test all degradation modes work."""
        # Create test image and mask
        image = np.random.rand(16, 16, 3).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        
        modes = ["rgb", "hsv", "lab"]
        
        for mode in modes:
            degraded = apply_light_degradation(image, mask, mode=mode, magnitude=0.05, seed=42)
            
            # Check output shape and type
            assert degraded.shape == image.shape
            assert degraded.dtype == image.dtype
            
            # Check values are in valid range
            assert np.all(degraded >= 0.0)
            assert np.all(degraded <= 1.0)
    
    def test_degradation_magnitude_effect(self):
        """Test that larger magnitude produces larger changes."""
        # Create test image and mask
        image = np.ones((16, 16, 3), dtype=np.float32) * 0.5
        mask = np.ones((16, 16), dtype=np.float32)
        
        # Apply degradation with different magnitudes
        degraded_small = apply_light_degradation(image, mask, mode="rgb", magnitude=0.01, seed=42)
        degraded_large = apply_light_degradation(image, mask, mode="rgb", magnitude=0.1, seed=42)
        
        # Larger magnitude should produce larger changes
        diff_small = np.abs(image - degraded_small).mean()
        diff_large = np.abs(image - degraded_large).mean()
        
        assert diff_large > diff_small
    
    def test_degradation_empty_mask(self):
        """Test degradation with empty mask."""
        # Create test image and empty mask
        image = np.random.rand(16, 16, 3).astype(np.float32)
        mask = np.zeros((16, 16), dtype=np.float32)
        
        # Apply degradation
        degraded = apply_light_degradation(image, mask, mode="rgb", magnitude=0.1, seed=42)
        
        # Image should be unchanged
        np.testing.assert_allclose(image, degraded, rtol=1e-6, atol=1e-6)
    
    def test_degradation_full_mask(self):
        """Test degradation with full mask."""
        # Create test image and full mask
        image = np.random.rand(16, 16, 3).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        
        # Apply degradation
        degraded = apply_light_degradation(image, mask, mode="rgb", magnitude=0.1, seed=42)
        
        # All pixels should be changed
        assert not np.allclose(image, degraded, rtol=1e-6, atol=1e-6)
        
        # But values should still be in valid range
        assert np.all(degraded >= 0.0)
        assert np.all(degraded <= 1.0)
    
    def test_degradation_bounds_hsv(self):
        """Test HSV degradation stays within expected bounds."""
        # Create test image and mask
        image = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.float32)
        
        # Apply HSV degradation multiple times
        for _ in range(10):
            degraded = apply_light_degradation(image, mask, mode="hsv", magnitude=0.05)
            
            # Check bounds
            assert np.all(degraded >= 0.0)
            assert np.all(degraded <= 1.0)
    
    def test_degradation_bounds_lab(self):
        """Test LAB degradation stays within expected bounds."""
        # Create test image and mask
        image = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.float32)
        
        # Apply LAB degradation multiple times
        for _ in range(10):
            degraded = apply_light_degradation(image, mask, mode="lab", magnitude=0.05)
            
            # Check bounds
            assert np.all(degraded >= 0.0)
            assert np.all(degraded <= 1.0)
    
    def test_degradation_invalid_mode(self):
        """Test that invalid mode raises error."""
        image = np.random.rand(16, 16, 3).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        
        with pytest.raises(ValueError):
            apply_light_degradation(image, mask, mode="invalid", magnitude=0.05)


if __name__ == "__main__":
    pytest.main([__file__])
