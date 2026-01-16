# models/vae/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .encoder import ResidualBlock, AttentionBlock


class UpsampleBlock(nn.Module):
    """Upsampling block with nearest neighbor + convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class DecoderBlock(nn.Module):
    """Decoder block with residual blocks and optional attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_groups: int = 32,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        
        # First residual block handles channel change
        self.res_blocks.append(ResidualBlock(in_channels, out_channels, num_groups))
        
        # Remaining residual blocks
        for _ in range(num_res_blocks - 1):
            self.res_blocks.append(ResidualBlock(out_channels, out_channels, num_groups))
        
        # Optional attention
        self.attention = None
        if use_attention:
            self.attention = AttentionBlock(out_channels, num_heads, num_groups)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class VAEDecoder(nn.Module):
    """
    VAE Decoder for reconstructing images from latent space.
    
    Architecture:
        Latent (B, latent_channels, 60, 60) -> Multiple decoder blocks with upsampling
        -> Output (B, 4, 240, 240)
    
    Decompression: 4x spatial upsampling (60 -> 240)
    """
    
    def __init__(
        self,
        out_channels: int = 4,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (4, 4, 2, 1),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (0,),  # Apply attention at 60x60
        num_groups: int = 32,
        num_heads: int = 8
    ):
        """
        Initialize VAE Decoder.
        
        Args:
            out_channels: Number of output channels (4 for MRI modalities)
            latent_channels: Number of latent channels
            base_channels: Base number of channels
            channel_multipliers: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per level
            attention_resolutions: Resolution levels to apply attention (indexed from 0)
            num_groups: Number of groups for GroupNorm
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        
        channels = [base_channels * m for m in channel_multipliers]
        
        # Initial convolution from latent space
        self.conv_in = nn.Conv2d(latent_channels, channels[0], kernel_size=3, padding=1)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        in_ch = channels[0]
        
        for level, out_ch in enumerate(channels):
            use_attn = level in attention_resolutions
            
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    num_groups=num_groups,
                    num_heads=num_heads
                )
            )
            
            # Add upsampling except for last level
            if level < len(channels) - 1:
                self.upsample_blocks.append(
                    UpsampleBlock(out_ch, out_ch, num_groups)
                )
            
            in_ch = out_ch
        
        # Final layers
        self.norm_out = nn.GroupNorm(num_groups=min(num_groups, channels[-1]), num_channels=channels[-1])
        self.activation = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent tensor (B, latent_channels, H, W)
            
        Returns:
            Reconstructed tensor (B, out_channels, H*4, W*4)
        """
        # Initial convolution
        x = self.conv_in(z)
        
        # Decoder blocks with upsampling
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x)
            
            if i < len(self.upsample_blocks):
                x = self.upsample_blocks[i](x)
        
        # Final processing
        x = self.norm_out(x)
        x = self.activation(x)
        x = self.conv_out(x)
        
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Alias for forward pass."""
        return self.forward(z)


class VAEDecoderSmall(nn.Module):
    """
    Smaller VAE Decoder for faster training/inference.
    
    Architecture:
        Latent (B, latent_channels, 60, 60) -> 3 decoder blocks
        -> Output (B, 4, 240, 240)
    """
    
    def __init__(
        self,
        out_channels: int = 4,
        latent_channels: int = 4,
        hidden_channels: List[int] = [256, 128, 64]
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # 60 -> 60
            nn.Conv2d(latent_channels, hidden_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_channels[0]),
            nn.SiLU(),
            ResidualBlock(hidden_channels[0], hidden_channels[0]),
            
            # Attention at 60x60
            AttentionBlock(hidden_channels[0], num_heads=8),
            
            # 60 -> 120
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_channels[1]),
            nn.SiLU(),
            ResidualBlock(hidden_channels[1], hidden_channels[1]),
            
            # 120 -> 240
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels[2]),
            nn.SiLU(),
            ResidualBlock(hidden_channels[2], hidden_channels[2]),
            
            # Output
            nn.GroupNorm(8, hidden_channels[2]),
            nn.SiLU(),
            nn.Conv2d(hidden_channels[2], out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)


class VAEDecoderWithSkip(nn.Module):
    """
    VAE Decoder with skip connections for better reconstruction.
    Requires encoder features to be passed during decoding.
    """
    
    def __init__(
        self,
        out_channels: int = 4,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (4, 4, 2, 1),
        num_res_blocks: int = 2,
        num_groups: int = 32
    ):
        super().__init__()
        
        channels = [base_channels * m for m in channel_multipliers]
        
        # Initial convolution
        self.conv_in = nn.Conv2d(latent_channels, channels[0], kernel_size=3, padding=1)
        
        # Decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        in_ch = channels[0]
        
        for level, out_ch in enumerate(channels):
            # Skip connection projection
            if level > 0:
                skip_ch = channels[level - 1]
                self.skip_convs.append(
                    nn.Conv2d(in_ch + skip_ch, in_ch, kernel_size=1)
                )
            
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_res_blocks=num_res_blocks,
                    use_attention=(level == 0),
                    num_groups=num_groups
                )
            )
            
            if level < len(channels) - 1:
                self.upsample_blocks.append(
                    UpsampleBlock(out_ch, out_ch, num_groups)
                )
            
            in_ch = out_ch
        
        # Final layers
        self.norm_out = nn.GroupNorm(num_groups=min(num_groups, channels[-1]), num_channels=channels[-1])
        self.activation = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
    
    def forward(
        self,
        z: torch.Tensor,
        encoder_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional skip connections.
        
        Args:
            z: Latent tensor (B, latent_channels, H, W)
            encoder_features: Optional list of encoder features for skip connections
            
        Returns:
            Reconstructed tensor (B, out_channels, H*4, W*4)
        """
        x = self.conv_in(z)
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Add skip connection if available
            if encoder_features is not None and i > 0 and i - 1 < len(encoder_features):
                skip = encoder_features[-(i)]
                if skip.shape[2:] == x.shape[2:]:
                    x = torch.cat([x, skip], dim=1)
                    x = self.skip_convs[i - 1](x)
            
            x = decoder_block(x)
            
            if i < len(self.upsample_blocks):
                x = self.upsample_blocks[i](x)
        
        x = self.norm_out(x)
        x = self.activation(x)
        x = self.conv_out(x)
        
        return x
    
    def decode(
        self,
        z: torch.Tensor,
        encoder_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        return self.forward(z, encoder_features)
