# models/vae/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        return self.activation(x + residual)


class AttentionBlock(nn.Module):
    """Self-attention block for feature refinement."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        out = self.proj(out)
        
        return out + residual


class DownsampleBlock(nn.Module):
    """Downsampling block with strided convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1
        )
        self.norm = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class EncoderBlock(nn.Module):
    """Encoder block with residual blocks and optional attention."""
    
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


class VAEEncoder(nn.Module):
    """
    VAE Encoder for compressing images to latent space.
    
    Architecture:
        Input (B, 4, 240, 240) -> Multiple encoder blocks with downsampling
        -> Latent (B, latent_channels*2, 60, 60) [mean and log_var]
    
    Compression: 4x spatial downsampling (240 -> 60)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (2,),  # Apply attention at 60x60
        num_groups: int = 32,
        num_heads: int = 8
    ):
        """
        Initialize VAE Encoder.
        
        Args:
            in_channels: Number of input channels (4 for MRI modalities)
            latent_channels: Number of latent channels
            base_channels: Base number of channels
            channel_multipliers: Channel multipliers for each resolution level
            num_res_blocks: Number of residual blocks per level
            attention_resolutions: Resolution levels to apply attention (indexed from 0)
            num_groups: Number of groups for GroupNorm
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        channels = [base_channels * m for m in channel_multipliers]
        in_ch = base_channels
        
        for level, out_ch in enumerate(channels):
            use_attn = level in attention_resolutions
            
            self.encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    num_groups=num_groups,
                    num_heads=num_heads
                )
            )
            
            # Add downsampling except for last level
            if level < len(channels) - 1:
                self.downsample_blocks.append(
                    DownsampleBlock(out_ch, out_ch, num_groups)
                )
            
            in_ch = out_ch
        
        # Final layers
        self.norm_out = nn.GroupNorm(num_groups=min(num_groups, channels[-1]), num_channels=channels[-1])
        self.activation = nn.SiLU()
        
        # Output convolutions for mean and log_var
        self.conv_out = nn.Conv2d(channels[-1], latent_channels * 2, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
            
        Returns:
            Tuple of (mean, log_var) each of shape (B, latent_channels, H//4, W//4)
        """
        # Initial convolution
        x = self.conv_in(x)
        
        # Encoder blocks with downsampling
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            
            if i < len(self.downsample_blocks):
                x = self.downsample_blocks[i](x)
        
        # Final processing
        x = self.norm_out(x)
        x = self.activation(x)
        x = self.conv_out(x)
        
        # Split into mean and log_var
        mean, log_var = torch.chunk(x, 2, dim=1)
        
        return mean, log_var
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for forward pass."""
        return self.forward(x)


class VAEEncoderSmall(nn.Module):
    """
    Smaller VAE Encoder for faster training/inference.
    
    Architecture:
        Input (B, 4, 240, 240) -> 3 encoder blocks
        -> Latent (B, latent_channels*2, 60, 60)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 4,
        hidden_channels: List[int] = [64, 128, 256]
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 240 -> 120
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, hidden_channels[0]),
            nn.SiLU(),
            ResidualBlock(hidden_channels[0], hidden_channels[0]),
            
            # 120 -> 60
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, hidden_channels[1]),
            nn.SiLU(),
            ResidualBlock(hidden_channels[1], hidden_channels[1]),
            
            # 60 -> 60 (no downsampling)
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, hidden_channels[2]),
            nn.SiLU(),
            ResidualBlock(hidden_channels[2], hidden_channels[2]),
            
            # Attention at 60x60
            AttentionBlock(hidden_channels[2], num_heads=8),
            
            # Output
            nn.GroupNorm(32, hidden_channels[2]),
            nn.SiLU(),
            nn.Conv2d(hidden_channels[2], latent_channels * 2, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        return mean, log_var
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)
