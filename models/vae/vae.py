# models/vae/vae.py

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from .encoder import VAEEncoder
from .decoder import VAEDecoder
from .losses import VAELoss, CombinedVAELoss


@dataclass
class VAEConfig:
    """Configuration for VAE model."""
    in_channels: int = 4
    out_channels: int = 4
    latent_channels: int = 4
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    use_se: bool = True

    # Loss config (kept for compatibility)
    recon_loss_type: str = 'l1'
    recon_weight: float = 1.0
    kl_weight: float = 0.0001
    use_perceptual: bool = False
    perceptual_weight: float = 0.0

    # Legacy fields accepted but ignored
    attention_resolutions: Tuple[int, ...] = ()
    num_groups: int = 32
    num_heads: int = 8


class VAE(nn.Module):
    """
    Variational Autoencoder for medical image compression.

    Input:  (B, 4, 240, 240) — 4 MRI modalities
    Latent: (B, latent_channels, 60, 60) — 4x spatial compression
    Output: (B, 4, 240, 240) — reconstructed modalities

    Uses SE channel attention instead of spatial self-attention
    for fast training on MPS/CPU.
    """

    def __init__(self, config: Optional[VAEConfig] = None):
        super().__init__()
        if config is None:
            config = VAEConfig()
        self.config = config
        self.latent_channels = config.latent_channels
        self.scale_factor = 0.18215

        self.encoder = VAEEncoder(
            in_channels=config.in_channels,
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            use_se=config.use_se,
        )

        dec_mults = tuple(reversed(config.channel_multipliers))
        self.decoder = VAEDecoder(
            out_channels=config.out_channels,
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            channel_multipliers=dec_mults,
            num_res_blocks=config.num_res_blocks,
            use_se=config.use_se,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor,
                       deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return mean
        return mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(x)
        log_var = torch.clamp(log_var, -30.0, 20.0)
        z = self.reparameterize(mean, log_var, deterministic)
        recon = self.decode(z)
        return recon, mean, log_var

    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var, deterministic)
        return z * self.scale_factor

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z / self.scale_factor)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        recon, _, _ = self.forward(x, deterministic=True)
        return recon

    def sample(self, num_samples: int, device: torch.device,
               latent_size: Tuple[int, int] = (60, 60)) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_channels, *latent_size, device=device)
        return self.decode(z)

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                    num_steps: int = 10) -> torch.Tensor:
        m1, _ = self.encode(x1)
        m2, _ = self.encode(x2)
        alphas = torch.linspace(0, 1, num_steps, device=x1.device)
        return torch.cat([self.decode((1 - a) * m1 + a * m2) for a in alphas], dim=0)


# Backward compatibility — VAESmall is now the same as VAE
VAESmall = VAE


class VAEWithLoss(nn.Module):
    """VAE with built-in loss computation."""
    def __init__(self, config: Optional[VAEConfig] = None):
        super().__init__()
        if config is None:
            config = VAEConfig()
        self.vae = VAE(config)
        self.loss_fn = CombinedVAELoss(
            recon_weight=config.recon_weight, kl_weight=config.kl_weight)

    def forward(self, x, compute_loss=True):
        recon, mean, log_var = self.vae(x)
        result = {'reconstruction': recon, 'mean': mean, 'log_var': log_var}
        if compute_loss:
            _, loss_dict = self.loss_fn(recon, x, mean, log_var)
            result.update(loss_dict)
        return result

    def encode(self, x): return self.vae.encode(x)
    def decode(self, z): return self.vae.decode(z)
    def get_latent(self, x, deterministic=True): return self.vae.get_latent(x, deterministic)
    def decode_latent(self, z): return self.vae.decode_latent(z)
    def reconstruct(self, x): return self.vae.reconstruct(x)


def create_vae(model_type: str = 'standard', **kwargs) -> nn.Module:
    """Factory function to create VAE models."""
    config = VAEConfig(**{k: v for k, v in kwargs.items() if hasattr(VAEConfig, k)})
    if model_type == 'with_loss':
        return VAEWithLoss(config)
    return VAE(config)