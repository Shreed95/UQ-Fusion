# models/gan/discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class DiscriminatorConfig:
    """Configuration for PatchGAN Discriminator."""
    in_channels: int = 8  # Source + Generated concatenated (4 + 4)
    base_channels: int = 64
    num_layers: int = 3
    norm_type: str = 'instance'  # 'instance', 'batch', 'spectral'
    use_spectral_norm: bool = True


class SpectralNorm(nn.Module):
    """Spectral normalization wrapper."""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = nn.utils.spectral_norm(module)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field).
    
    Classifies overlapping patches as real or fake.
    Uses spectral normalization for training stability.
    
    Architecture:
        Input: (B, 8, H, W) - concatenated source and target
        -> 4x4 Conv stride 2 (64) -> LeakyReLU
        -> 4x4 Conv stride 2 (128) -> [SpectralNorm/InstanceNorm] -> LeakyReLU
        -> 4x4 Conv stride 2 (256) -> [SpectralNorm/InstanceNorm] -> LeakyReLU
        -> 4x4 Conv stride 1 (512) -> [SpectralNorm/InstanceNorm] -> LeakyReLU
        -> 4x4 Conv stride 1 (1) -> Output
    """
    
    def __init__(self, config: Optional[DiscriminatorConfig] = None):
        super().__init__()
        
        if config is None:
            config = DiscriminatorConfig()
        
        self.config = config
        
        # Build discriminator
        layers = []
        
        # First layer (no normalization)
        layers.append(
            nn.Conv2d(config.in_channels, config.base_channels, kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        in_ch = config.base_channels
        for i in range(1, config.num_layers):
            out_ch = min(config.base_channels * (2 ** i), 512)
            stride = 2 if i < config.num_layers - 1 else 1
            
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
            
            if config.use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.append(conv)
            
            if not config.use_spectral_norm:
                if config.norm_type == 'instance':
                    layers.append(nn.InstanceNorm2d(out_ch))
                elif config.norm_type == 'batch':
                    layers.append(nn.BatchNorm2d(out_ch))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            in_ch = out_ch
        
        # Final layer
        out_ch = min(config.base_channels * (2 ** config.num_layers), 512)
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=1, padding=1)
        if config.use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        layers.append(conv)
        
        if not config.use_spectral_norm:
            if config.norm_type == 'instance':
                layers.append(nn.InstanceNorm2d(out_ch))
            elif config.norm_type == 'batch':
                layers.append(nn.BatchNorm2d(out_ch))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer
        layers.append(nn.Conv2d(out_ch, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
               Usually concatenation of source and target images
               
        Returns:
            Patch predictions (B, 1, H', W')
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN Discriminator.
    
    Uses multiple discriminators at different scales for better
    texture and structure assessment.
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 64,
        num_discriminators: int = 2,
        num_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.num_discriminators = num_discriminators
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            config = DiscriminatorConfig(
                in_channels=in_channels,
                base_channels=base_channels,
                num_layers=num_layers,
                use_spectral_norm=use_spectral_norm
            )
            self.discriminators.append(PatchGANDiscriminator(config))
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all discriminators.
        
        Args:
            x: Input tensor
            
        Returns:
            List of outputs from each scale
        """
        outputs = []
        
        for i, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(x))
            
            if i < self.num_discriminators - 1:
                x = self.downsample(x)
        
        return outputs


class ConditionalPatchGAN(nn.Module):
    """
    Conditional PatchGAN that takes source and target as separate inputs.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        # Discriminator takes concatenated input
        config = DiscriminatorConfig(
            in_channels=in_channels * 2,  # Source + Target
            base_channels=base_channels,
            num_layers=num_layers,
            use_spectral_norm=use_spectral_norm
        )
        self.discriminator = PatchGANDiscriminator(config)
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with source and target.
        
        Args:
            source: Source image (B, C, H, W)
            target: Target image (real or fake) (B, C, H, W)
            
        Returns:
            Patch predictions
        """
        # Concatenate along channel dimension
        x = torch.cat([source, target], dim=1)
        return self.discriminator(x)


class PatchGANDiscriminatorSmall(nn.Module):
    """
    Smaller PatchGAN for faster training.
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 32,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        def make_conv(in_ch, out_ch, stride):
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            return conv
        
        self.model = nn.Sequential(
            # Layer 1: No norm
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            make_conv(base_channels, base_channels * 2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            make_conv(base_channels * 2, base_channels * 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            make_conv(base_channels * 4, base_channels * 4, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_discriminator(
    model_type: str = 'small',
    **kwargs
) -> nn.Module:
    """
    Factory function to create discriminators.
    
    Args:
        model_type: 'standard', 'multiscale', 'conditional', or 'small'
        **kwargs: Additional arguments
        
    Returns:
        Discriminator model
    """
    if model_type == 'standard':
        config = DiscriminatorConfig(**kwargs)
        return PatchGANDiscriminator(config)
    elif model_type == 'multiscale':
        return MultiScaleDiscriminator(**kwargs)
    elif model_type == 'conditional':
        return ConditionalPatchGAN(**kwargs)
    elif model_type == 'small':
        return PatchGANDiscriminatorSmall(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
