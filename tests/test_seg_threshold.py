"""
Test segmentation thresholding and simple IoU computation.
"""
import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.seg_unet import SegmentationUNet
from losses_metrics import compute_segmentation_metrics


class TestSegmentationThresholding:
    """Test segmentation model thresholding and metrics."""
    
    def test_model_predict_threshold(self):
        """Test model prediction with thresholding."""
        model = SegmentationUNet(base_channels=16, depth=2)
        model.eval()
        
        batch_size = 2
        height, width = 64, 64
        
        # Create input
        x = torch.randn(batch_size, 3, height, width)
        
        # Test different thresholds
        thresholds = [0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            with torch.no_grad():
                masks = model.predict(x, threshold=threshold)
            
            # Check output shape and values
            assert masks.shape == (batch_size, 1, height, width)
            assert torch.all((masks == 0) | (masks == 1))  # Binary values only
    
    def test_threshold_effect(self):
        """Test that different thresholds produce different results."""
        model = SegmentationUNet(base_channels=16, depth=2)
        model.eval()
        
        # Create input that will produce probabilities around 0.5
        x = torch.zeros(1, 3, 32, 32)
        
        with torch.no_grad():
            # Get logits and convert to probabilities
            logits = model(x)
            probs = torch.sigmoid(logits)
            
            # Apply different thresholds
            mask_low = model.predict(x, threshold=0.3)
            mask_high = model.predict(x, threshold=0.7)
        
        # Lower threshold should generally produce more positive pixels
        assert torch.sum(mask_low) >= torch.sum(mask_high)
    
    def test_iou_perfect_match(self):
        """Test IoU computation for perfect match."""
        # Create identical predictions and targets
        pred = torch.tensor([[[[1.0, 1.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 1.0],
                              [0.0, 0.0, 1.0, 1.0]]]])
        
        target = pred.clone()
        
        # Convert pred to logits (inverse sigmoid)
        pred_logits = torch.log(pred + 1e-8) - torch.log(1 - pred + 1e-8)
        
        metrics = compute_segmentation_metrics(pred_logits, target, threshold=0.5)
        
        # Perfect match should give IoU = 1, Dice = 1
        assert abs(metrics['iou'] - 1.0) < 1e-6
        assert abs(metrics['dice'] - 1.0) < 1e-6
        assert abs(metrics['precision'] - 1.0) < 1e-6
        assert abs(metrics['recall'] - 1.0) < 1e-6
    
    def test_iou_no_overlap(self):
        """Test IoU computation for no overlap."""
        # Create non-overlapping predictions and targets
        pred = torch.tensor([[[[1.0, 1.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]]]])
        
        target = torch.tensor([[[[0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 1.0],
                                [0.0, 0.0, 1.0, 1.0]]]])
        
        # Convert pred to logits
        pred_logits = torch.log(pred + 1e-8) - torch.log(1 - pred + 1e-8)
        
        metrics = compute_segmentation_metrics(pred_logits, target, threshold=0.5)
        
        # No overlap should give IoU = 0, Dice = 0
        assert abs(metrics['iou'] - 0.0) < 1e-6
        assert abs(metrics['dice'] - 0.0) < 1e-6
        assert abs(metrics['precision'] - 0.0) < 1e-6
        assert abs(metrics['recall'] - 0.0) < 1e-6
    
    def test_iou_partial_overlap(self):
        """Test IoU computation for partial overlap."""
        # Create partially overlapping predictions and targets
        pred = torch.tensor([[[[1.0, 1.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]]]])
        
        target = torch.tensor([[[[0.0, 1.0, 1.0, 0.0],
                                [0.0, 1.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0]]]])
        
        # Convert pred to logits
        pred_logits = torch.log(pred + 1e-8) - torch.log(1 - pred + 1e-8)
        
        metrics = compute_segmentation_metrics(pred_logits, target, threshold=0.5)
        
        # Partial overlap: intersection = 2, union = 6, IoU = 2/6 = 1/3
        expected_iou = 2.0 / 6.0
        expected_dice = 2 * 2.0 / (4.0 + 4.0)  # 2 * intersection / (pred + target)
        
        assert abs(metrics['iou'] - expected_iou) < 1e-6
        assert abs(metrics['dice'] - expected_dice) < 1e-6
    
    def test_metrics_batch(self):
        """Test metrics computation on batches."""
        batch_size = 4
        height, width = 16, 16
        
        # Create random predictions and targets
        pred_logits = torch.randn(batch_size, 1, height, width)
        target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        
        metrics = compute_segmentation_metrics(pred_logits, target, threshold=0.5)
        
        # Check that metrics are in valid ranges
        assert 0.0 <= metrics['iou'] <= 1.0
        assert 0.0 <= metrics['dice'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
    
    def test_threshold_binary_output(self):
        """Test that thresholding produces binary output."""
        # Create probabilities around threshold
        probs = torch.tensor([[[[0.2, 0.4, 0.6, 0.8],
                               [0.1, 0.45, 0.55, 0.9],
                               [0.3, 0.49, 0.51, 0.7],
                               [0.0, 0.5, 1.0, 0.35]]]])
        
        # Convert to logits
        logits = torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8)
        
        # Apply threshold
        threshold = 0.5
        binary_mask = (torch.sigmoid(logits) > threshold).float()
        
        # Check that output is binary
        unique_values = torch.unique(binary_mask)
        assert len(unique_values) <= 2
        assert torch.all((unique_values == 0) | (unique_values == 1))
        
        # Check specific values
        expected = torch.tensor([[[[0, 0, 1, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 1, 1],
                                  [0, 0, 1, 0]]]])
        
        assert torch.equal(binary_mask, expected.float())
    
    def test_empty_masks(self):
        """Test metrics with empty masks."""
        batch_size = 2
        height, width = 8, 8
        
        # All zeros prediction and target
        pred_logits = torch.full((batch_size, 1, height, width), -10.0)  # Very negative = ~0 prob
        target = torch.zeros(batch_size, 1, height, width)
        
        metrics = compute_segmentation_metrics(pred_logits, target, threshold=0.5)
        
        # When both pred and target are empty, IoU and Dice are undefined
        # Our implementation should handle this gracefully
        assert not np.isnan(metrics['iou'])
        assert not np.isnan(metrics['dice'])
    
    def test_model_integration(self):
        """Test integration with actual segmentation model."""
        model = SegmentationUNet(base_channels=8, depth=2)
        model.eval()
        
        batch_size = 2
        height, width = 32, 32
        
        # Create input and target
        x = torch.randn(batch_size, 3, height, width)
        target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
        
        with torch.no_grad():
            # Forward pass
            logits = model(x)
            
            # Compute metrics
            metrics = compute_segmentation_metrics(logits, target, threshold=0.5)
            
            # Test prediction method
            pred_masks = model.predict(x, threshold=0.5)
        
        # Check shapes and values
        assert logits.shape == (batch_size, 1, height, width)
        assert pred_masks.shape == (batch_size, 1, height, width)
        assert torch.all((pred_masks == 0) | (pred_masks == 1))
        
        # Check metrics are valid
        for key in ['iou', 'dice', 'precision', 'recall']:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
