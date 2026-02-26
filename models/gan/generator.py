# models/gan/generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Configuration for STABLE Generator."""
    in_channels: int = 4  # 4 MRI modalities
    out_channels: int = 4
    base_channels: int = 64
    num_residual_blocks: int = 9
    use_dropout: bool = True
    dropout_rate: float = 0.5
    norm_type: str = 'instance'  # 'instance', 'batch', 'group'
    padding_type: str = 'reflect'  # 'reflect', 'replicate', 'zero'


class ReflectionPad2d(nn.Module):
    """Reflection padding layer."""
    
    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.padding,) * 4, mode='reflect')


class ResidualBlock(nn.Module):
    """
    Residual block with InstanceNorm for style normalization.
    
    Architecture:
        ReflectPad -> Conv -> Norm -> ReLU -> [Dropout] -> ReflectPad -> Conv -> Norm + Skip
    """
    
    def __init__(
        self,
        channels: int,
        norm_type: str = 'instance',
        use_dropout: bool = True,
        dropout_rate: float = 0.5,
        padding_type: str = 'reflect'
    ):
        super().__init__()
        
        layers = []
        
        # First conv block
        if padding_type == 'reflect':
            layers.append(ReflectionPad2d(1))
        elif padding_type == 'replicate':
            layers.append(nn.ReplicationPad2d(1))
        else:
            layers.append(nn.ZeroPad2d(1))
        
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=0))
        layers.append(self._get_norm_layer(norm_type, channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Dropout
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        
        # Second conv block
        if padding_type == 'reflect':
            layers.append(ReflectionPad2d(1))
        elif padding_type == 'replicate':
            layers.append(nn.ReplicationPad2d(1))
        else:
            layers.append(nn.ZeroPad2d(1))
        
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=0))
        layers.append(self._get_norm_layer(norm_type, channels))
        
        self.block = nn.Sequential(*layers)
    
    def _get_norm_layer(self, norm_type: str, channels: int) -> nn.Module:
        if norm_type == 'instance':
            return nn.InstanceNorm2d(channels)
        elif norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'group':
            return nn.GroupNorm(min(32, channels), channels)
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class STABLEGenerator(nn.Module):
    """
    STABLE Generator based on ResNet-9 architecture.
    
    Designed for medical image-to-image translation with:
    - Spatial information preservation
    - Quantitative information preservation
    
    Architecture:
        Input (4, 240, 240)
        -> 7x7 Conv (64) -> InstanceNorm -> ReLU
        -> 3x3 Conv stride 2 (128) -> InstanceNorm -> ReLU
        -> 3x3 Conv stride 2 (256) -> InstanceNorm -> ReLU
        -> 9x ResidualBlocks (256)
        -> 3x3 TransposeConv stride 2 (128) -> InstanceNorm -> ReLU
        -> 3x3 TransposeConv stride 2 (64) -> InstanceNorm -> ReLU
        -> 7x7 Conv (4) -> Tanh
        Output (4, 240, 240)
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        super().__init__()
        
        if config is None:
            config = GeneratorConfig()
        
        self.config = config
        
        # Initial convolution block
        self.initial = nn.Sequential(
            ReflectionPad2d(3),
            nn.Conv2d(config.in_channels, config.base_channels, kernel_size=7, padding=0),
            self._get_norm_layer(config.norm_type, config.base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = self._make_downsample_block(
            config.base_channels, config.base_channels * 2, config.norm_type
        )
        self.down2 = self._make_downsample_block(
            config.base_channels * 2, config.base_channels * 4, config.norm_type
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(
                config.base_channels * 4,
                config.norm_type,
                config.use_dropout,
                config.dropout_rate,
                config.padding_type
            )
            for _ in range(config.num_residual_blocks)
        ])
        
        # Upsampling
        self.up1 = self._make_upsample_block(
            config.base_channels * 4, config.base_channels * 2, config.norm_type
        )
        self.up2 = self._make_upsample_block(
            config.base_channels * 2, config.base_channels, config.norm_type
        )
        
        # Final convolution
        self.final = nn.Sequential(
            ReflectionPad2d(3),
            nn.Conv2d(config.base_channels, config.out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _get_norm_layer(self, norm_type: str, channels: int) -> nn.Module:
        if norm_type == 'instance':
            return nn.InstanceNorm2d(channels)
        elif norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_type == 'group':
            return nn.GroupNorm(min(32, channels), channels)
        else:
            return nn.Identity()
    
    def _make_downsample_block(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            self._get_norm_layer(norm_type, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_upsample_block(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            self._get_norm_layer(norm_type, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image (B, in_channels, H, W)
            
        Returns:
            Generated image (B, out_channels, H, W)
        """
        # Encoder
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        # Residual blocks
        x = self.residual_blocks(x)
        
        # Decoder
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        
        return x


class STABLEGeneratorWithSkip(nn.Module):
    """
    STABLE Generator with skip connections for better spatial preservation.
    Similar to U-Net style connections.
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        super().__init__()
        
        if config is None:
            config = GeneratorConfig()
        
        self.config = config
        
        # Initial
        self.initial = nn.Sequential(
            ReflectionPad2d(3),
            nn.Conv2d(config.in_channels, config.base_channels, kernel_size=7, padding=0),
            nn.InstanceNorm2d(config.base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.down1 = self._make_downsample_block(config.base_channels, config.base_channels * 2)
        self.down2 = self._make_downsample_block(config.base_channels * 2, config.base_channels * 4)
        
        # Bottleneck
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(config.base_channels * 4, 'instance', config.use_dropout, config.dropout_rate)
            for _ in range(config.num_residual_blocks)
        ])
        
        # Decoder with skip connections
        self.up1 = self._make_upsample_block(config.base_channels * 4, config.base_channels * 2)
        self.up1_conv = nn.Sequential(
            nn.Conv2d(config.base_channels * 4, config.base_channels * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(config.base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = self._make_upsample_block(config.base_channels * 2, config.base_channels)
        self.up2_conv = nn.Sequential(
            nn.Conv2d(config.base_channels * 2, config.base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(config.base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final
        self.final = nn.Sequential(
            ReflectionPad2d(3),
            nn.Conv2d(config.base_channels, config.out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _make_downsample_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_upsample_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.initial(x)  # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        
        # Bottleneck
        x = self.residual_blocks(x3)  # (B, 256, H/4, W/4)
        
        # Decoder with skip connections
        x = self.up1(x)  # (B, 128, H/2, W/2)
        x = torch.cat([x, x2], dim=1)  # (B, 256, H/2, W/2)
        x = self.up1_conv(x)  # (B, 128, H/2, W/2)
        
        x = self.up2(x)  # (B, 64, H, W)
        x = torch.cat([x, x1], dim=1)  # (B, 128, H, W)
        x = self.up2_conv(x)  # (B, 64, H, W)
        
        x = self.final(x)  # (B, 4, H, W)
        
        return x


class STABLEGeneratorSmall(nn.Module):
    """
    Smaller STABLE Generator for faster training.
    Uses 6 residual blocks instead of 9.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 32,
        num_residual_blocks: int = 6,
        use_dropout: bool = True,  
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Initial
        self.initial = nn.Sequential(
            ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=0),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_channels * 4, 'instance', use_dropout=use_dropout, dropout_rate=dropout_rate)
            for _ in range(num_residual_blocks)
        ])
        
        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final
        self.final = nn.Sequential(
            ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residual_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x


def create_generator(
    model_type: str = 'small',
    **kwargs
) -> nn.Module:
    """
    Factory function to create generators.
    
    Args:
        model_type: 'standard', 'skip', or 'small'
        **kwargs: Additional arguments
        
    Returns:
        Generator model
    """
    if model_type == 'standard':
        config = GeneratorConfig(**kwargs)
        return STABLEGenerator(config)
    elif model_type == 'skip':
        config = GeneratorConfig(**kwargs)
        return STABLEGeneratorWithSkip(config)
    elif model_type == 'small':
        return STABLEGeneratorSmall(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
