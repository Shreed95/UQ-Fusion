# models/diffusion/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .blocks import (
    TimeEmbedding,
    ResidualBlock,
    AttentionBlock,
    DownBlock,
    UpBlock,
    MiddleBlock
)


@dataclass
class UNetConfig:
    """Configuration for Diffusion U-Net."""
    in_channels: int = 4  # Latent channels from VAE
    out_channels: int = 4  # Predicted noise channels
    base_channels: int = 128
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)  # For 60x60 -> 30x30 -> 15x15
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (1, 2)  # Apply attention at 30x30 and 15x15
    time_embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    use_conditioning: bool = True  # For image-to-image translation


class DiffusionUNet(nn.Module):
    """
    U-Net for diffusion model denoising.
    
    Operates in latent space (60x60) from VAE.
    Takes noisy latent + timestep -> predicts noise (ε-prediction).
    
    Architecture:
        - Encoder: Progressive downsampling with residual blocks + attention
        - Middle: Self-attention at lowest resolution
        - Decoder: Progressive upsampling with skip connections
    """
    
    def __init__(self, config: Optional[UNetConfig] = None):
        super().__init__()
        
        if config is None:
            config = UNetConfig()
        
        self.config = config
        
        # Channel setup
        channels = [config.base_channels * m for m in config.channel_multipliers]
        
        # Time embedding
        self.time_embedding = TimeEmbedding(
            time_dim=config.base_channels,
            embed_dim=config.time_embed_dim
        )
        
        # Input convolution
        if config.use_conditioning:
            # Concatenate noisy latent with source latent
            self.conv_in = nn.Conv2d(config.in_channels * 2, config.base_channels, kernel_size=3, padding=1)
        else:
            self.conv_in = nn.Conv2d(config.in_channels, config.base_channels, kernel_size=3, padding=1)
        
        # Encoder (Downsampling path)
        self.down_blocks = nn.ModuleList()
        in_ch = config.base_channels
        
        for level, out_ch in enumerate(channels):
            use_attn = level in config.attention_resolutions
            is_last = level == len(channels) - 1
            
            self.down_blocks.append(
                DownBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_embed_dim=config.time_embed_dim,
                    num_res_blocks=config.num_res_blocks,
                    use_attention=use_attn,
                    num_heads=config.num_heads,
                    downsample=not is_last,
                    dropout=config.dropout
                )
            )
            in_ch = out_ch
        
        # Middle block
        self.middle_block = MiddleBlock(
            channels=channels[-1],
            time_embed_dim=config.time_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Decoder (Upsampling path)
        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        
        for level, out_ch in enumerate(reversed_channels):
            in_ch = reversed_channels[level]
            skip_ch = reversed_channels[level]  # Skip connection channels
            
            # Output channels for this level
            if level < len(reversed_channels) - 1:
                up_out_ch = reversed_channels[level + 1]
            else:
                up_out_ch = config.base_channels
            
            use_attn = (len(channels) - 1 - level) in config.attention_resolutions
            is_last = level == len(reversed_channels) - 1
            
            self.up_blocks.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=up_out_ch,
                    skip_channels=skip_ch * config.num_res_blocks,  # Multiple skips per level
                    time_embed_dim=config.time_embed_dim,
                    num_res_blocks=config.num_res_blocks,
                    use_attention=use_attn,
                    num_heads=config.num_heads,
                    upsample=not is_last,
                    dropout=config.dropout
                )
            )
        
        # Output convolution
        self.norm_out = nn.GroupNorm(32, config.base_channels)
        self.conv_out = nn.Conv2d(config.base_channels, config.out_channels, kernel_size=3, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero initialize output conv for stable training
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for noise prediction.
        
        Args:
            x: Noisy latent (B, C, H, W)
            timesteps: Diffusion timesteps (B,)
            condition: Optional conditioning latent (B, C, H, W) for image-to-image
            
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        t_embed = self.time_embedding(timesteps)
        
        # Concatenate condition if provided
        if condition is not None and self.config.use_conditioning:
            x = torch.cat([x, condition], dim=1)
        
        # Input convolution
        x = self.conv_in(x)
        
        # Encoder with skip connections
        all_skips = []
        for down_block in self.down_blocks:
            x, skips = down_block(x, t_embed)
            all_skips.extend(skips)
        
        # Middle
        x = self.middle_block(x, t_embed)
        
        # Decoder with skip connections
        for up_block in self.up_blocks:
            # Get skips for this level
            num_skips = self.config.num_res_blocks
            level_skips = all_skips[-num_skips:]
            all_skips = all_skips[:-num_skips]
            
            x = up_block(x, t_embed, level_skips)
        
        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x


class DiffusionUNetSmall(nn.Module):
    """
    Smaller U-Net for faster training/inference.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 64,
        time_embed_dim: int = 256,
        use_conditioning: bool = True
    ):
        super().__init__()
        
        self.use_conditioning = use_conditioning
        
        # Time embedding
        self.time_embedding = TimeEmbedding(base_channels, time_embed_dim)
        
        # Input
        in_ch = in_channels * 2 if use_conditioning else in_channels
        self.conv_in = nn.Conv2d(in_ch, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = ResidualBlock(base_channels, base_channels, time_embed_dim)
        self.enc2 = ResidualBlock(base_channels, base_channels * 2, time_embed_dim)
        self.down1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 2, time_embed_dim)
        self.enc4 = ResidualBlock(base_channels * 2, base_channels * 4, time_embed_dim)
        self.down2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1)
        
        # Middle with attention
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_embed_dim)
        self.mid_attn = AttentionBlock(base_channels * 4, num_heads=4)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_embed_dim)
        
        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)
        )
        self.dec1 = ResidualBlock(base_channels * 8, base_channels * 2, time_embed_dim)  # Skip connection
        self.dec2 = ResidualBlock(base_channels * 2, base_channels * 2, time_embed_dim)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)
        )
        self.dec3 = ResidualBlock(base_channels * 4, base_channels, time_embed_dim)  # Skip connection
        self.dec4 = ResidualBlock(base_channels, base_channels, time_embed_dim)
        
        # Output
        self.norm_out = nn.GroupNorm(8, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Zero init output
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Time embedding
        t = self.time_embedding(timesteps)
        
        # Concat condition
        if condition is not None and self.use_conditioning:
            x = torch.cat([x, condition], dim=1)
        
        # Input
        x = self.conv_in(x)
        
        # Encoder
        x = self.enc1(x, t)
        x = self.enc2(x, t)
        skip1 = x
        x = self.down1(x)
        
        x = self.enc3(x, t)
        x = self.enc4(x, t)
        skip2 = x
        x = self.down2(x)
        
        # Middle
        x = self.mid1(x, t)
        x = self.mid_attn(x)
        x = self.mid2(x, t)
        
        # Decoder
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec1(x, t)
        x = self.dec2(x, t)
        
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec3(x, t)
        x = self.dec4(x, t)
        
        # Output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return x


def create_unet(
    model_type: str = 'small',
    **kwargs
) -> nn.Module:
    """
    Factory function to create U-Net models.
    
    Args:
        model_type: 'standard' or 'small'
        **kwargs: Additional arguments
        
    Returns:
        U-Net model
    """
    if model_type == 'standard':
        config = UNetConfig(**kwargs)
        return DiffusionUNet(config)
    elif model_type == 'small':
        return DiffusionUNetSmall(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
