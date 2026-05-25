# models/diffusion/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeEmbedding(nn.Module):
    """Time embedding module."""
    def __init__(self, time_dim: int, embed_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(timesteps))


class ResidualBlock(nn.Module):
    """Residual block with time conditioning."""
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int,
                 num_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, out_channels))
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = x + self.time_proj(t_embed)[:, :, None, None]
        x = self.act(self.norm2(x))
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class AttentionBlock(nn.Module):
    """
    Self-attention block using F.scaled_dot_product_attention.
    Memory-efficient — never materializes the full (HW × HW) attention matrix.
    """
    def __init__(self, channels: int, num_heads: int = 8, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out) + residual


class CrossAttentionBlock(nn.Module):
    """Cross-attention block using F.scaled_dot_product_attention."""
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8, num_groups: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.norm_context = nn.GroupNorm(min(num_groups, context_dim), context_dim)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(context_dim, channels, 1)
        self.v = nn.Conv2d(context_dim, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        context = self.norm_context(context)
        q = self.q(x).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = self.k(context).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = self.v(context).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out) + residual


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder."""
    def __init__(self, in_channels, out_channels, time_embed_dim, num_res_blocks=2,
                 use_attention=False, num_heads=8, downsample=True, dropout=0.1):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            in_ch = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResidualBlock(in_ch, out_channels, time_embed_dim, dropout=dropout))
            self.attn_blocks.append(AttentionBlock(out_channels, num_heads) if use_attention else nn.Identity())
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1) if downsample else None

    def forward(self, x, t_embed):
        skips = []
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x, t_embed)
            x = attn(x)
            skips.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skips

class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder."""
    def __init__(self, in_channels, out_channels, skip_channels, time_embed_dim,
                 num_res_blocks=2, use_attention=False, num_heads=8, upsample=True, dropout=0.1):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        
        single_skip_ch = skip_channels // num_res_blocks if num_res_blocks > 0 else skip_channels

        for i in range(num_res_blocks):
            in_ch = in_channels + single_skip_ch if i == 0 else out_channels + single_skip_ch
            
            self.res_blocks.append(ResidualBlock(in_ch, out_channels, time_embed_dim, dropout=dropout))
            self.attn_blocks.append(AttentionBlock(out_channels, num_heads) if use_attention else nn.Identity())
            
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        ) if upsample else None

    def forward(self, x, t_embed, skips):
        # 1. Process skip connections and residual blocks FIRST
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            # THE FIX: Check if the list has items, not relying on the dynamic index 'i'
            if len(skips) > 0: 
                skip = skips.pop()
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
                
                x = torch.cat([x, skip], dim=1)
                
            x = res(x, t_embed)
            x = attn(x)
            
        # 2. Upsample at the END of the block
        if self.upsample is not None:
            x = self.upsample(x)
            
        return x


class MiddleBlock(nn.Module):
    """Middle block of U-Net (bottleneck)."""
    def __init__(self, channels, time_embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.res1 = ResidualBlock(channels, channels, time_embed_dim, dropout=dropout)
        self.attn = AttentionBlock(channels, num_heads)
        self.res2 = ResidualBlock(channels, channels, time_embed_dim, dropout=dropout)

    def forward(self, x, t_embed):
        x = self.res1(x, t_embed)
        x = self.attn(x)
        return self.res2(x, t_embed)