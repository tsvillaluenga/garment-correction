"""
Basic building blocks for U-Net architectures.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with optional normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bn: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn
        )
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two consecutive convolution blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        use_bn: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.conv1 = ConvBlock(in_channels, mid_channels, use_bn=use_bn, activation=activation)
        self.conv2 = ConvBlock(mid_channels, out_channels, use_bn=use_bn, activation=activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class DownBlock(nn.Module):
    """Downsampling block with max pooling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bn: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, use_bn=use_bn, activation=activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.maxpool(x))


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        use_bn: bool = True,
        activation: str = "relu",
        upsample_mode: str = "transpose"
    ):
        super().__init__()
        
        if upsample_mode == "transpose":
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
        elif upsample_mode == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvBlock(in_channels, in_channels // 2, kernel_size=1, padding=0,
                         use_bn=use_bn, activation=activation)
            )
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")
        
        self.conv = DoubleConv(
            in_channels // 2 + skip_channels, out_channels,
            use_bn=use_bn, activation=activation
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        
        if diff_h > 0 or diff_w > 0:
            x = F.pad(x, (diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2))
        
        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_bn: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(
            in_channels, out_channels, stride=stride, use_bn=use_bn, activation=activation
        )
        self.conv2 = ConvBlock(
            out_channels, out_channels, use_bn=use_bn, activation="none"
        )
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.skip = nn.Identity()
        
        if activation == "relu":
            self.final_activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.final_activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.final_activation = nn.GELU()
        else:
            self.final_activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        return self.final_activation(out + residual)


class SpatialAttention(nn.Module):
    """Simple spatial attention mechanism."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention using global average pooling."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
