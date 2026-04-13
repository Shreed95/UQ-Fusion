# models/vae/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    O(C) compute instead of O(HW^2) for spatial attention.
    Perfect for MPS where spatial attention is extremely slow.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        ng = min(num_groups, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(ng, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(ng, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class VAEEncoder(nn.Module):
    """
    VAE Encoder: (B, 4, 240, 240) -> (B, latent*2, 60, 60)
    
    3-level downsampling with residual blocks and SE channel attention.
    No spatial self-attention — runs fast on MPS.
    """
    def __init__(
        self,
        in_channels: int = 4,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        use_se: bool = True,
        **kwargs  # Accept and ignore extra kwargs for compatibility
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_multipliers]
        blocks = nn.ModuleList()
        in_ch = base_channels

        for level, out_ch in enumerate(channels):
            for i in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch if i == 0 else out_ch, out_ch))
            if use_se:
                blocks.append(SEBlock(out_ch))
            # Downsample except last level
            if level < len(channels) - 1:
                blocks.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch

        self.blocks = blocks
        self.norm_out = nn.GroupNorm(min(32, channels[-1]), channels[-1])
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[-1], latent_channels * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.act(self.norm_out(x))
        x = self.conv_out(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        return mean, log_var

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x)


# Backward compatibility aliases
VAEEncoderSmall = VAEEncoder