# models/segmentation/unet.py

"""
2D U-Net for Brain Tumor Segmentation.

Architecture:
- Encoder: 4 downsampling blocks
- Bottleneck: 2 conv blocks
- Decoder: 4 upsampling blocks with skip connections
- Output: 4-class segmentation (background, NCR/NET, ED, ET)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class UNetConfig:
    """Configuration for segmentation U-Net."""
    in_channels: int = 4           # T1, T1ce, T2, FLAIR
    num_classes: int = 4           # Background + 3 tumor regions
    base_channels: int = 32
    depth: int = 4                 # Number of encoder/decoder levels
    dropout_rate: float = 0.1
    use_attention: bool = True
    use_deep_supervision: bool = False


class ConvBlock(nn.Module):
    """Double convolution block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections."""
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        
        # Upsample gate to match skip size
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode='bilinear', align_corners=False)
        
        attention = self.relu(g + s)
        attention = self.psi(attention)
        
        return skip * attention


class EncoderBlock(nn.Module):
    """Encoder block with convolution and downsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout_rate: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )
        
        if use_attention:
            self.attention = AttentionGate(
                in_channels // 2, skip_channels, skip_channels // 2
            )
        else:
            self.attention = None
        
        self.conv = ConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            dropout_rate
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Pad if necessary
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                      diff_h // 2, diff_h - diff_h // 2])
        
        if self.attention is not None:
            skip = self.attention(x, skip)
        
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationUNet(nn.Module):
    """
    2D U-Net for brain tumor segmentation.
    
    Input: (B, 4, H, W) - 4 MRI modalities
    Output: (B, num_classes, H, W) - segmentation logits
    """
    
    def __init__(self, config: Optional[UNetConfig] = None):
        super().__init__()
        
        if config is None:
            config = UNetConfig()
        
        self.config = config
        
        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = config.in_channels
        
        channels = [config.base_channels * (2 ** i) for i in range(config.depth)]
        
        for out_ch in channels:
            self.encoders.append(EncoderBlock(in_ch, out_ch, config.dropout_rate))
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = ConvBlock(
            channels[-1], channels[-1] * 2, config.dropout_rate
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        in_ch = channels[-1] * 2
        
        for i in range(config.depth - 1, -1, -1):
            skip_ch = channels[i]
            out_ch = channels[i]
            self.decoders.append(DecoderBlock(
                in_ch, skip_ch, out_ch,
                config.dropout_rate, config.use_attention
            ))
            in_ch = out_ch
        
        # Output
        self.output = nn.Conv2d(channels[0], config.num_classes, kernel_size=1)
        
        # Deep supervision (optional)
        if config.use_deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv2d(channels[i], config.num_classes, kernel_size=1)
                for i in range(config.depth - 1)
            ])
        else:
            self.deep_outputs = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 4, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Encoder
        skip_connections = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        decoder_features = []
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skip_connections) - 1 - i
            x = decoder(x, skip_connections[skip_idx])
            decoder_features.append(x)
        
        # Output
        logits = self.output(x)
        
        if return_features:
            return logits, decoder_features
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get segmentation prediction."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class SegmentationUNetSmall(SegmentationUNet):
    """Smaller U-Net variant for faster training."""
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 16
    ):
        config = UNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=4,
            dropout_rate=0.1,
            use_attention=False,
            use_deep_supervision=False
        )
        super().__init__(config)


class SegmentationUNetLarge(SegmentationUNet):
    """Larger U-Net variant for better performance."""
    
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 64
    ):
        config = UNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=5,
            dropout_rate=0.15,
            use_attention=True,
            use_deep_supervision=True
        )
        super().__init__(config)


def create_segmentation_model(
    model_type: str = 'standard',
    in_channels: int = 4,
    num_classes: int = 4,
    **kwargs
) -> SegmentationUNet:
    """
    Factory function for segmentation models.
    
    Args:
        model_type: 'small', 'standard', or 'large'
        in_channels: Number of input channels
        num_classes: Number of output classes
        
    Returns:
        SegmentationUNet model
    """
    if model_type == 'small':
        return SegmentationUNetSmall(in_channels, num_classes)
    elif model_type == 'standard':
        config = UNetConfig(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=32,
            **kwargs
        )
        return SegmentationUNet(config)
    elif model_type == 'large':
        return SegmentationUNetLarge(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
