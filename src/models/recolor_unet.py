"""
Recoloring U-Net with cross-attention (Model 1).
Two encoders for on-model and still images with cross-attention in bottleneck.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .blocks import DoubleConv, DownBlock, UpBlock, ConvBlock
from .attention import CrossAttentionBlock


class DualEncoder(nn.Module):
    """Dual encoder for processing two input streams."""
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        use_bn: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Initial convolutions for both streams
        self.init_conv_on = DoubleConv(in_channels, channels[0], use_bn=use_bn, activation=activation)
        self.init_conv_still = DoubleConv(in_channels, channels[0], use_bn=use_bn, activation=activation)
        
        # Encoder blocks for both streams
        self.encoder_on = nn.ModuleList()
        self.encoder_still = nn.ModuleList()
        
        for i in range(depth):
            self.encoder_on.append(
                DownBlock(channels[i], channels[i + 1], use_bn=use_bn, activation=activation)
            )
            self.encoder_still.append(
                DownBlock(channels[i], channels[i + 1], use_bn=use_bn, activation=activation)
            )
    
    def forward(self, on_model: torch.Tensor, still: torch.Tensor) -> Tuple[Dict, Dict]:
        """
        Forward pass through both encoders.
        
        Args:
            on_model: On-model image (B, C, H, W)
            still: Still image (B, C, H, W)
            
        Returns:
            Tuple of (on_model_features, still_features) dictionaries
        """
        # Store features at each level
        on_features = {}
        still_features = {}
        
        # Initial convolutions
        x_on = self.init_conv_on(on_model)
        x_still = self.init_conv_still(still)
        
        on_features[0] = x_on
        still_features[0] = x_still
        
        # Encoder paths
        for i, (enc_on, enc_still) in enumerate(zip(self.encoder_on, self.encoder_still)):
            x_on = enc_on(x_on)
            x_still = enc_still(x_still)
            
            on_features[i + 1] = x_on
            still_features[i + 1] = x_still
        
        return on_features, still_features


class CrossAttentionBottleneck(nn.Module):
    """Bottleneck with cross-attention between on-model and still features."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_blocks: int = 2,
        dropout: float = 0.1,
        use_pos_embed: bool = True
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        # Cross-attention blocks
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_pos_embed=use_pos_embed
            ) for _ in range(num_blocks)
        ])
        
        # Additional processing
        self.process_conv = DoubleConv(embed_dim, embed_dim)
        
    def forward(self, on_features: torch.Tensor, still_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between on-model and still features.
        
        Args:
            on_features: On-model features (B, C, H, W)
            still_features: Still features (B, C, H, W)
            
        Returns:
            Enhanced on-model features (B, C, H, W)
        """
        x = on_features
        
        # Apply multiple cross-attention blocks
        for cross_attn in self.cross_attention_blocks:
            x = cross_attn(
                query=x,
                key=still_features,
                value=still_features
            )
        
        # Additional processing
        x = self.process_conv(x)
        
        return x


class RecoloringUNet(nn.Module):
    """
    Recoloring U-Net with dual encoders and cross-attention.
    
    Architecture:
    - Two separate encoders for on-model and still images
    - Cross-attention in bottleneck where Q from on-model, K,V from still
    - Single decoder with skip connections from on-model encoder
    - Output compositing outside the network
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 4,
        num_attn_blocks: int = 2,
        num_heads: int = 8,
        use_bn: bool = True,
        activation: str = "relu",
        upsample_mode: str = "transpose",
        dropout: float = 0.1,
        use_pos_embed: bool = True
    ):
        super().__init__()
        
        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Dual encoder
        self.encoder = DualEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            use_bn=use_bn,
            activation=activation
        )
        
        # Cross-attention bottleneck
        self.bottleneck = CrossAttentionBottleneck(
            embed_dim=channels[depth],
            num_heads=num_heads,
            num_blocks=num_attn_blocks,
            dropout=dropout,
            use_pos_embed=use_pos_embed
        )
        
        # Decoder (single path with skip connections from on-model encoder)
        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(
                UpBlock(
                    channels[depth - i],
                    channels[depth - i - 1],
                    channels[depth - i - 1],  # Skip connection channels from on-model
                    use_bn=use_bn,
                    activation=activation,
                    upsample_mode=upsample_mode
                )
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        
        # Optional output activation
        self.output_activation = nn.Sigmoid()
    
    def forward_train(
        self,
        on_model_input: torch.Tensor,
        still_ref: torch.Tensor,
        mask_on: torch.Tensor,
        mask_still: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            on_model_input: Degraded on-model image (B, 3, H, W)
            still_ref: Reference still image (B, 3, H, W)
            mask_on: On-model mask (B, 1, H, W)
            mask_still: Still mask (B, 1, H, W) - optional
            
        Returns:
            Corrected on-model image (B, 3, H, W)
        """
        # Encode both inputs
        on_features, still_features = self.encoder(on_model_input, still_ref)
        
        # Cross-attention in bottleneck
        bottleneck_features = self.bottleneck(
            on_features[self.depth],
            still_features[self.depth]
        )
        
        # Decode with skip connections from on-model encoder
        x = bottleneck_features
        for i, decoder_block in enumerate(self.decoder):
            skip = on_features[self.depth - i - 1]
            x = decoder_block(x, skip)
        
        # Final prediction
        y_pred = self.final_conv(x)
        y_pred = self.output_activation(y_pred)
        
        # Composite with original input (mask-aware)
        # Use .clone() to avoid in-place operations that could break gradients
        mask_on_safe = mask_on.clone()
        output = mask_on_safe * y_pred + (1 - mask_on_safe) * on_model_input
        
        return output
    
    def forward_infer(
        self,
        on_model_input: torch.Tensor,
        still_ref: torch.Tensor,
        mask_on: torch.Tensor,
        mask_still: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for inference (same as training but with no_grad).
        
        Args:
            on_model_input: Input on-model image (B, 3, H, W)
            still_ref: Reference still image (B, 3, H, W)
            mask_on: On-model mask (B, 1, H, W)
            mask_still: Still mask (B, 1, H, W) - optional
            
        Returns:
            Corrected on-model image (B, 3, H, W)
        """
        with torch.no_grad():
            return self.forward_train(on_model_input, still_ref, mask_on, mask_still)
    
    def forward(
        self,
        on_model_input: torch.Tensor,
        still_ref: torch.Tensor,
        mask_on: torch.Tensor,
        mask_still: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Default forward pass (training mode)."""
        return self.forward_train(on_model_input, still_ref, mask_on, mask_still)


class RecoloringUNetWithGAN(RecoloringUNet):
    """Recoloring U-Net with optional GAN discriminator support."""
    
    def __init__(self, *args, use_gan: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gan = use_gan
        
        if self.use_gan:
            # Simple discriminator for adversarial training
            self.discriminator = self._build_discriminator()
    
    def _build_discriminator(self) -> nn.Module:
        """Build a simple patch discriminator."""
        return nn.Sequential(
            # Input: 3 channels (RGB image)
            ConvBlock(3, 64, kernel_size=4, stride=2, padding=1, use_bn=False, activation="leaky_relu"),
            ConvBlock(64, 128, kernel_size=4, stride=2, padding=1, activation="leaky_relu"),
            ConvBlock(128, 256, kernel_size=4, stride=2, padding=1, activation="leaky_relu"),
            ConvBlock(256, 512, kernel_size=4, stride=1, padding=1, activation="leaky_relu"),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # Output: (B, 1, H/8, W/8) - patch-level predictions
        )
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Run discriminator on input image."""
        if not self.use_gan:
            raise RuntimeError("GAN mode is disabled")
        return self.discriminator(x)


def create_recolor_model(use_gan: bool = False, **kwargs) -> RecoloringUNet:
    """
    Factory function to create recoloring models.
    
    Args:
        use_gan: Whether to use GAN discriminator
        **kwargs: Model parameters
        
    Returns:
        Recoloring model
    """
    if use_gan:
        return RecoloringUNetWithGAN(use_gan=True, **kwargs)
    else:
        return RecoloringUNet(**kwargs)


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = RecoloringUNet(
        base_channels=64,
        depth=4,
        num_attn_blocks=2,
        num_heads=4
    )
    model.to(device)
    
    # Test inputs
    B, H, W = 2, 256, 256
    on_model = torch.randn(B, 3, H, W).to(device)
    still = torch.randn(B, 3, H, W).to(device)
    mask_on = torch.randint(0, 2, (B, 1, H, W)).float().to(device)
    mask_still = torch.randint(0, 2, (B, 1, H, W)).float().to(device)
    
    # Forward pass
    output = model(on_model, still, mask_on, mask_still)
    
    print(f"On-model input shape: {on_model.shape}")
    print(f"Still input shape: {still.shape}")
    print(f"Mask on shape: {mask_on.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test GAN version
    gan_model = RecoloringUNetWithGAN(
        base_channels=32,
        depth=3,
        use_gan=True
    )
    gan_model.to(device)
    
    small_img = torch.randn(B, 3, 128, 128).to(device)
    disc_output = gan_model.discriminate(small_img)
    print(f"Discriminator output shape: {disc_output.shape}")
    print(f"GAN model parameters: {sum(p.numel() for p in gan_model.parameters())}")
