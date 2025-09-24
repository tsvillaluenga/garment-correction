"""
Test Model 1 (recoloring) forward pass and shapes.
"""
import pytest
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.recolor_unet import RecoloringUNet, create_recolor_model


class TestRecoloringModel:
    """Test recoloring model shapes and forward pass."""
    
    def test_model_creation(self):
        """Test model can be created with different configurations."""
        # Basic model
        model = RecoloringUNet(
            base_channels=32,
            depth=3,
            num_attn_blocks=1,
            num_heads=2
        )
        assert isinstance(model, RecoloringUNet)
        
        # Model with GAN
        model_gan = create_recolor_model(
            use_gan=True,
            base_channels=32,
            depth=3,
            num_attn_blocks=1,
            num_heads=2
        )
        assert hasattr(model_gan, 'discriminator')
    
    def test_forward_shapes(self):
        """Test forward pass shapes are correct."""
        model = RecoloringUNet(
            base_channels=32,
            depth=3,
            num_attn_blocks=1,
            num_heads=2
        )
        model.eval()
        
        batch_size = 2
        height, width = 128, 128
        
        # Create input tensors
        on_model = torch.randn(batch_size, 3, height, width)
        still = torch.randn(batch_size, 3, height, width)
        mask_on = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        mask_still = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        
        with torch.no_grad():
            output = model.forward_train(on_model, still, mask_on, mask_still)
        
        # Check output shape
        assert output.shape == (batch_size, 3, height, width)
        
        # Check output range (should be [0, 1] due to sigmoid)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)
    
    def test_forward_infer_vs_train(self):
        """Test that inference and training forward passes are consistent."""
        model = RecoloringUNet(
            base_channels=16,
            depth=2,
            num_attn_blocks=1,
            num_heads=2
        )
        model.eval()
        
        batch_size = 1
        height, width = 64, 64
        
        # Create input tensors
        on_model = torch.randn(batch_size, 3, height, width)
        still = torch.randn(batch_size, 3, height, width)
        mask_on = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        mask_still = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        
        # Forward passes
        output_train = model.forward_train(on_model, still, mask_on, mask_still)
        output_infer = model.forward_infer(on_model, still, mask_on, mask_still)
        
        # Should be identical in eval mode
        assert torch.allclose(output_train, output_infer, atol=1e-6)
    
    def test_attention_mechanism(self):
        """Test that cross-attention works with different spatial sizes."""
        model = RecoloringUNet(
            base_channels=32,
            depth=2,
            num_attn_blocks=1,
            num_heads=4
        )
        model.eval()
        
        batch_size = 1
        
        # Test with square images
        on_model = torch.randn(batch_size, 3, 64, 64)
        still = torch.randn(batch_size, 3, 64, 64)
        mask_on = torch.ones(batch_size, 1, 64, 64)
        mask_still = torch.ones(batch_size, 1, 64, 64)
        
        with torch.no_grad():
            output = model.forward_train(on_model, still, mask_on, mask_still)
        
        assert output.shape == (batch_size, 3, 64, 64)
    
    def test_mask_compositing(self):
        """Test that mask compositing works correctly."""
        model = RecoloringUNet(
            base_channels=16,
            depth=2,
            num_attn_blocks=1,
            num_heads=2
        )
        model.eval()
        
        batch_size = 1
        height, width = 32, 32
        
        # Create specific inputs to test compositing
        on_model = torch.zeros(batch_size, 3, height, width)  # Black image
        still = torch.ones(batch_size, 3, height, width)      # White image
        
        # Mask that covers half the image
        mask_on = torch.zeros(batch_size, 1, height, width)
        mask_on[:, :, :, width//2:] = 1.0  # Right half is masked
        
        mask_still = torch.ones(batch_size, 1, height, width)
        
        with torch.no_grad():
            output = model.forward_train(on_model, still, mask_on, mask_still)
        
        # Left half (unmasked) should be close to original (black)
        left_half = output[:, :, :, :width//2]
        assert torch.all(left_half < 0.1)  # Should be close to black
        
        # Right half (masked) should be modified by the model
        right_half = output[:, :, :, width//2:]
        # Can't predict exact values, but should be different from input
        assert not torch.allclose(right_half, torch.zeros_like(right_half), atol=0.1)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = RecoloringUNet(
            base_channels=16,
            depth=2,
            num_attn_blocks=1,
            num_heads=2
        )
        model.train()
        
        batch_size = 1
        height, width = 32, 32
        
        # Create input tensors with gradients
        on_model = torch.randn(batch_size, 3, height, width, requires_grad=True)
        still = torch.randn(batch_size, 3, height, width, requires_grad=True)
        mask_on = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        mask_still = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        
        # Forward pass
        output = model.forward_train(on_model, still, mask_on, mask_still)
        
        # Compute loss and backward
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert on_model.grad is not None
        assert still.grad is not None
        assert torch.any(on_model.grad != 0)
        assert torch.any(still.grad != 0)
    
    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = RecoloringUNet(
            base_channels=16,
            depth=2,
            num_attn_blocks=1,
            num_heads=2
        )
        model.eval()
        
        height, width = 64, 64
        
        for batch_size in [1, 2, 4]:
            on_model = torch.randn(batch_size, 3, height, width)
            still = torch.randn(batch_size, 3, height, width)
            mask_on = torch.randint(0, 2, (batch_size, 1, height, width)).float()
            mask_still = torch.randint(0, 2, (batch_size, 1, height, width)).float()
            
            with torch.no_grad():
                output = model.forward_train(on_model, still, mask_on, mask_still)
            
            assert output.shape == (batch_size, 3, height, width)
    
    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        model = RecoloringUNet(
            base_channels=64,
            depth=4,
            num_attn_blocks=2,
            num_heads=4
        )
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have a reasonable number of parameters (not too small, not too large)
        assert 1_000_000 < param_count < 100_000_000
    
    def test_gan_discriminator(self):
        """Test GAN discriminator if enabled."""
        model = create_recolor_model(
            use_gan=True,
            base_channels=32,
            depth=3
        )
        model.eval()
        
        batch_size = 2
        height, width = 64, 64
        
        # Test discriminator
        fake_image = torch.randn(batch_size, 3, height, width)
        
        with torch.no_grad():
            disc_output = model.discriminate(fake_image)
        
        # Discriminator should output patch-level predictions
        assert disc_output.shape[0] == batch_size
        assert disc_output.shape[1] == 1
        assert disc_output.shape[2] > 0  # Height
        assert disc_output.shape[3] > 0  # Width


if __name__ == "__main__":
    pytest.main([__file__])
