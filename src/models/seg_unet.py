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


class DualInputSegmentationUNet(nn.Module):
    """Dual-input U-Net with cross-attention for better on-model segmentation."""
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,
        depth: int = 4,
        use_bn: bool = True,
        activation: str = "relu",
        upsample_mode: str = "transpose",
        dropout: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.depth = depth
        self.use_attention = use_attention
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Separate encoders for still and on_model
        self.still_encoder = self._create_encoder(in_channels, channels, use_bn, activation)
        self.onmodel_encoder = self._create_encoder(in_channels, channels, use_bn, activation)
        
        # Cross-attention in bottleneck
        if use_attention:
            from .attention import CrossAttentionBlock
            self.cross_attention = CrossAttentionBlock(
                embed_dim=channels[depth], 
                num_heads=8, 
                dropout=dropout
            )
        
        # Bottleneck processing
        self.bottleneck = DoubleConv(
            channels[depth], channels[depth], use_bn=use_bn, activation=activation
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # Decoder (uses on_model encoder features for skip connections)
        self.decoder = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[depth - i]
            out_ch = channels[depth - i - 1]
            skip_ch = channels[depth - i - 1]  # Skip connection channels
            
            self.decoder.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    skip_channels=skip_ch,
                    use_bn=use_bn,
                    activation=activation,
                    upsample_mode=upsample_mode
                )
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], 1, kernel_size=1)
        
        # Auxiliary decoder for still segmentation (simpler version)
        self.aux_decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[depth], channels[depth//2], 4, 2, 1),  # 32x32 -> 64x64
            nn.BatchNorm2d(channels[depth//2]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[depth//2], channels[depth//4], 4, 2, 1),  # 64x64 -> 128x128
            nn.BatchNorm2d(channels[depth//4]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[depth//4], channels[depth//8], 4, 2, 1),  # 128x128 -> 256x256
            nn.BatchNorm2d(channels[depth//8]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channels[depth//8], channels[0], 4, 2, 1),  # 256x256 -> 512x512
            nn.BatchNorm2d(channels[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], 1, kernel_size=1)  # Final prediction
        )
    
    def _create_encoder(self, in_channels, channels, use_bn, activation):
        """Create encoder layers."""
        encoder = nn.ModuleList()
        
        # Initial convolution
        init_conv = DoubleConv(in_channels, channels[0], use_bn=use_bn, activation=activation)
        
        # Downsampling blocks
        down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            down_blocks.append(
                DownBlock(channels[i], channels[i + 1], use_bn=use_bn, activation=activation)
            )
        
        return nn.ModuleDict({
            'init_conv': init_conv,
            'down_blocks': down_blocks
        })
    
    def _encode(self, x, encoder):
        """Encode input through encoder."""
        features = []
        
        # Initial convolution
        x = encoder['init_conv'](x)
        features.append(x)
        
        # Downsampling
        for down_block in encoder['down_blocks']:
            x = down_block(x)
            features.append(x)
        
        return features
    
    def forward(self, still, on_model, mask_still=None):
        """
        Forward pass with dual inputs.
        
        Args:
            still: Reference garment image (B, 3, H, W)
            on_model: Image to segment (B, 3, H, W)
            mask_still: Optional mask for still image (B, 1, H, W)
            
        Returns:
            dict with 'main' (on_model mask) and 'aux' (still mask) predictions
        """
        # Apply mask to still image if provided (focus on garment region)
        if mask_still is not None:
            # Expand mask to 3 channels and apply
            mask_expanded = mask_still.expand(-1, 3, -1, -1)
            still_masked = still * mask_expanded
            # Add small background to avoid zero features
            still_masked = still_masked + (1 - mask_expanded) * 0.1
        else:
            still_masked = still
        
        # Encode both images
        still_features = self._encode(still_masked, self.still_encoder)
        onmodel_features = self._encode(on_model, self.onmodel_encoder)
        
        # Get bottleneck features
        still_bottleneck = still_features[-1]
        onmodel_bottleneck = onmodel_features[-1]
        
        # Cross-attention: let on_model attend to still features
        if self.use_attention:
            attended_features = self.cross_attention(
                onmodel_bottleneck,  # Query: what to segment
                still_bottleneck,    # Key/Value: garment reference
                still_bottleneck
            )
        else:
            # Simple concatenation fallback
            attended_features = torch.cat([onmodel_bottleneck, still_bottleneck], dim=1)
            attended_features = nn.Conv2d(
                attended_features.shape[1], onmodel_bottleneck.shape[1], 1
            ).to(attended_features.device)(attended_features)
        
        # Process attended features
        x = self.bottleneck(attended_features)
        x = self.dropout(x)
        
        # Auxiliary prediction for still image (for multi-task learning)
        aux_pred = self.aux_decoder(still_bottleneck)
        
        # Decoder with skip connections from on_model encoder
        for i, up_block in enumerate(self.decoder):
            skip_features = onmodel_features[-(i + 2)]  # Skip connection
            x = up_block(x, skip_features)
        
        # Final prediction
        main_pred = self.final_conv(x)
        
        return {
            'main': main_pred,      # On-model segmentation
            'aux': aux_pred         # Still segmentation (auxiliary)
        }


def create_seg_model(model_type: str = "basic", **kwargs) -> nn.Module:
    """
    Factory function to create segmentation models.
    
    Args:
        model_type: Type of model ('basic', 'enhanced', or 'dual_input')
        **kwargs: Model parameters
        
    Returns:
        Segmentation model
    """
    if model_type == "basic":
        return SegmentationUNet(**kwargs)
    elif model_type == "enhanced":
        return EnhancedSegmentationUNet(**kwargs)
    elif model_type == "dual_input":
        return DualInputSegmentationUNet(**kwargs)
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
