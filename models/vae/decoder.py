# models/vae/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .encoder import ResidualBlock, SEBlock


class VAEDecoder(nn.Module):
    """
    VAE Decoder: (B, latent, 60, 60) -> (B, 4, 240, 240)
    
    Symmetric to encoder. Uses nearest-neighbor upsample + conv.
    SE channel attention, no spatial self-attention.
    """
    def __init__(
        self,
        out_channels: int = 4,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (4, 2, 1),
        num_res_blocks: int = 2,
        use_se: bool = True,
        **kwargs  # Accept and ignore extra kwargs for compatibility
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]

        self.conv_in = nn.Conv2d(latent_channels, channels[0], 3, padding=1)

        blocks = nn.ModuleList()
        in_ch = channels[0]

        for level, out_ch in enumerate(channels):
            for i in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch if i == 0 else out_ch, out_ch))
            if use_se:
                blocks.append(SEBlock(out_ch))
            # Upsample except last level
            if level < len(channels) - 1:
                blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1)
                ))
            in_ch = out_ch

        self.blocks = blocks
        self.norm_out = nn.GroupNorm(min(32, channels[-1]), channels[-1])
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[-1], out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        for block in self.blocks:
            x = block(x)
        x = self.act(self.norm_out(x))
        return self.conv_out(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)


# Backward compatibility aliases
VAEDecoderSmall = VAEDecoder