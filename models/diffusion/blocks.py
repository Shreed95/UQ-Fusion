# models/diffusion/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.
    Maps timestep t to a high-dimensional embedding.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of timesteps
            
        Returns:
            (B, dim) tensor of embeddings
        """
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        
        return embeddings


class TimeEmbedding(nn.Module):
    """
    Time embedding module that projects sinusoidal embeddings through MLPs.
    """
    
    def __init__(self, time_dim: int, embed_dim: int):
        super().__init__()
        
        self.sinusoidal = SinusoidalPositionEmbeddings(time_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of timesteps
            
        Returns:
            (B, embed_dim) tensor of time embeddings
        """
        t_embed = self.sinusoidal(timesteps)
        return self.mlp(t_embed)


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning for diffusion U-Net.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_groups: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
        
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            t_embed: (B, time_embed_dim) time embedding
            
        Returns:
            (B, out_channels, H, W) output tensor
        """
        residual = self.skip(x)
        
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        # Add time embedding
        t = self.time_proj(t_embed)[:, :, None, None]
        x = x + t
        
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        return x + residual


class AttentionBlock(nn.Module):
    """
    Self-attention block for diffusion U-Net.
    """
    
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
        
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            
        Returns:
            (B, C, H, W) output tensor
        """
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


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning on source images.
    """
    
    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 8,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.norm_context = nn.GroupNorm(min(num_groups, context_dim), context_dim)
        
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(context_dim, channels, kernel_size=1)
        self.v = nn.Conv2d(context_dim, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            context: (B, context_dim, H, W) context tensor
            
        Returns:
            (B, C, H, W) output tensor
        """
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        context = self.norm_context(context)
        
        q = self.q(x).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = self.k(context).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = self.v(context).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        out = self.proj(out)
        
        return out + residual


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 8,
        downsample: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, time_embed_dim, dropout=dropout)
            )
            
            if use_attention:
                self.attn_blocks.append(AttentionBlock(out_channels, num_heads))
            else:
                self.attn_blocks.append(nn.Identity())
        
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: Input tensor
            t_embed: Time embedding
            
        Returns:
            Tuple of (output, list of skip connections)
        """
        skips = []
        
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, t_embed)
            x = attn_block(x)
            skips.append(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, skips


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        num_heads: int = 8,
        upsample: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        for i in range(num_res_blocks):
            in_ch = in_channels + skip_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(in_ch, out_channels, time_embed_dim, dropout=dropout)
            )
            
            if use_attention:
                self.attn_blocks.append(AttentionBlock(out_channels, num_heads))
            else:
                self.attn_blocks.append(nn.Identity())
        
        self.upsample = None
        if upsample:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        skips: list
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            t_embed: Time embedding
            skips: List of skip connections from encoder
            
        Returns:
            Output tensor
        """
        if self.upsample is not None:
            x = self.upsample(x)
        
        for i, (res_block, attn_block) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            if i < len(skips):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            
            x = res_block(x, t_embed)
            x = attn_block(x)
        
        return x


class MiddleBlock(nn.Module):
    """
    Middle block of U-Net (bottleneck).
    """
    
    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.res1 = ResidualBlock(channels, channels, time_embed_dim, dropout=dropout)
        self.attn = AttentionBlock(channels, num_heads)
        self.res2 = ResidualBlock(channels, channels, time_embed_dim, dropout=dropout)
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_embed)
        x = self.attn(x)
        x = self.res2(x, t_embed)
        return x
