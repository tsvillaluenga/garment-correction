"""
Lightweight U-Net for binary segmentation (Models 2 and 3).
"""
import torch
import torch.nn as nn
from typing import List, Optional

from .blocks import ConvBlock, DoubleConv, DownBlock, UpBlock


class SegmentationUNet(nn.Module):
    """Lightweight U-Net for binary garment segmentation."""
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        use_bn: bool = True,
        activation: str = "relu",
        upsample_mode: str = "transpose",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolution
        self.init_conv = DoubleConv(in_channels, channels[0], use_bn=use_bn, activation=activation)
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(
                DownBlock(channels[i], channels[i + 1], use_bn=use_bn, activation=activation)
            )
        
        # Bottleneck
        self.bottleneck = DoubleConv(
            channels[depth], channels[depth], use_bn=use_bn, activation=activation
        )
        
        # Add dropout to bottleneck
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(
                UpBlock(
                    channels[depth - i],
                    channels[depth - i - 1],
                    channels[depth - i - 1],  # Skip connection channels
                    use_bn=use_bn,
                    activation=activation,
                    upsample_mode=upsample_mode
                )
            )
        
        # Final classification layer
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits tensor (B, 1, H, W)
        """
        # Store skip connections
        skip_connections = []
        
        # Initial convolution
        x = self.init_conv(x)
        skip_connections.append(x)
        
        # Encoder path
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        # Remove the last skip connection (it's the bottleneck input)
        skip_connections = skip_connections[:-1]
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[-(i + 1)]
            x = decoder_block(x, skip)
        
        # Final classification
        logits = self.final_conv(x)
        
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary masks with thresholding.
        
        Args:
            x: Input tensor (B, C, H, W)
            threshold: Threshold for binary classification
            
        Returns:
            Binary mask tensor (B, 1, H, W)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()
        return masks


class EnhancedSegmentationUNet(nn.Module):
    """Enhanced U-Net with attention and residual connections."""
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        use_bn: bool = True,
        activation: str = "relu",
        upsample_mode: str = "transpose",
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolution
        self.init_conv = DoubleConv(in_channels, channels[0], use_bn=use_bn, activation=activation)
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(
                DownBlock(channels[i], channels[i + 1], use_bn=use_bn, activation=activation)
            )
        
        # Bottleneck with attention
        self.bottleneck = DoubleConv(
            channels[depth], channels[depth], use_bn=use_bn, activation=activation
        )
        
        if self.use_attention:
            from .blocks import SpatialAttention
            self.bottleneck_attention = SpatialAttention(channels[depth])
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(
                UpBlock(
                    channels[depth - i],
                    channels[depth - i - 1],
                    channels[depth - i - 1],  # Skip connection channels
                    use_bn=use_bn,
                    activation=activation,
                    upsample_mode=upsample_mode
                )
            )
        
        # Skip connection attention modules
        if self.use_attention:
            from .blocks import SpatialAttention
            self.skip_attentions = nn.ModuleList([
                SpatialAttention(channels[i]) for i in range(depth)
            ])
        
        # Final classification layer
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the enhanced U-Net.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits tensor (B, 1, H, W)
        """
        # Store skip connections
        skip_connections = []
        
        # Initial convolution
        x = self.init_conv(x)
        skip_connections.append(x)
        
        # Encoder path
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        if self.use_attention:
            x = self.bottleneck_attention(x)
        x = self.dropout(x)
        
        # Remove the last skip connection (it's the bottleneck input)
        skip_connections = skip_connections[:-1]
        
        # Decoder path
        for i, decoder_block in enumerate(self.decoder):
            skip = skip_connections[-(i + 1)]
            
            # Apply attention to skip connection
            if self.use_attention:
                skip = self.skip_attentions[self.depth - i - 1](skip)
            
            x = decoder_block(x, skip)
        
        # Final classification
        logits = self.final_conv(x)
        
        return logits
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary masks with thresholding.
        
        Args:
            x: Input tensor (B, C, H, W)
            threshold: Threshold for binary classification
            
        Returns:
            Binary mask tensor (B, 1, H, W)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()
        return masks


def create_seg_model(model_type: str = "basic", **kwargs) -> nn.Module:
    """
    Factory function to create segmentation models.
    
    Args:
        model_type: Type of model ('basic' or 'enhanced')
        **kwargs: Model parameters
        
    Returns:
        Segmentation model
    """
    if model_type == "basic":
        return SegmentationUNet(**kwargs)
    elif model_type == "enhanced":
        return EnhancedSegmentationUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test basic model
    model = SegmentationUNet(base_channels=32, depth=3)
    model.to(device)
    
    x = torch.randn(2, 3, 256, 256).to(device)
    logits = model(x)
    masks = model.predict(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test enhanced model
    enhanced_model = EnhancedSegmentationUNet(base_channels=32, depth=3)
    enhanced_model.to(device)
    
    logits_enhanced = enhanced_model(x)
    print(f"Enhanced model parameters: {sum(p.numel() for p in enhanced_model.parameters())}")
