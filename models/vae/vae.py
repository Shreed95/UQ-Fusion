# models/vae/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from .encoder import VAEEncoder, VAEEncoderSmall
from .decoder import VAEDecoder, VAEDecoderSmall
from .losses import VAELoss, CombinedVAELoss


@dataclass
class VAEConfig:
    """Configuration for VAE model."""
    in_channels: int = 4
    out_channels: int = 4
    latent_channels: int = 4
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (2,)
    num_groups: int = 32
    num_heads: int = 8
    
    # Loss config
    recon_loss_type: str = 'l1'
    recon_weight: float = 1.0
    kl_weight: float = 0.0001
    use_perceptual: bool = False
    perceptual_weight: float = 0.0


class VAE(nn.Module):
    """
    Variational Autoencoder for medical image compression.
    
    Compresses 240x240 images to 60x60 latent representations with 4x spatial compression.
    Uses reparameterization trick for training.
    
    Input: (B, 4, 240, 240) - 4 MRI modalities
    Latent: (B, latent_channels, 60, 60)
    Output: (B, 4, 240, 240) - Reconstructed modalities
    """
    
    def __init__(self, config: Optional[VAEConfig] = None):
        """
        Initialize VAE.
        
        Args:
            config: VAE configuration
        """
        super().__init__()
        
        if config is None:
            config = VAEConfig()
        
        self.config = config
        self.latent_channels = config.latent_channels
        
        # Encoder
        self.encoder = VAEEncoder(
            in_channels=config.in_channels,
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            num_groups=config.num_groups,
            num_heads=config.num_heads
        )
        
        # Decoder
        # Reverse channel multipliers for decoder
        decoder_multipliers = tuple(reversed(config.channel_multipliers))
        decoder_attention = tuple(len(config.channel_multipliers) - 1 - a 
                                  for a in config.attention_resolutions)
        
        self.decoder = VAEDecoder(
            out_channels=config.out_channels,
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            channel_multipliers=decoder_multipliers,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=decoder_attention,
            num_groups=config.num_groups,
            num_heads=config.num_heads
        )
        
        # Scaling factor for latent space (helps with training stability)
        self.scale_factor = 0.18215
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of (mean, log_var)
        """
        mean, log_var = self.encoder(x)
        return mean, log_var
    
    def reparameterize(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = mean + std * epsilon.
        
        Args:
            mean: Mean of latent distribution
            log_var: Log variance of latent distribution
            deterministic: If True, return mean without sampling
            
        Returns:
            Sampled latent vector
        """
        if deterministic:
            return mean
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + std * eps
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent tensor (B, latent_channels, H, W)
            
        Returns:
            Reconstructed image
        """
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor (B, C, H, W)
            deterministic: If True, use mean instead of sampling
            
        Returns:
            Tuple of (reconstruction, mean, log_var)
        """
        # Encode
        mean, log_var = self.encode(x)
        
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, -30.0, 20.0)
        
        # Sample latent
        z = self.reparameterize(mean, log_var, deterministic)
        
        # Decode
        recon = self.decode(z)
        
        return recon, mean, log_var
    
    def get_latent(
        self,
        x: torch.Tensor,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Get latent representation.
        
        Args:
            x: Input tensor (B, C, H, W)
            deterministic: If True, return mean
            
        Returns:
            Latent tensor (B, latent_channels, H/4, W/4)
        """
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var, deterministic)
        return z * self.scale_factor
    
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from scaled latent.
        
        Args:
            z: Scaled latent tensor
            
        Returns:
            Reconstructed image
        """
        return self.decode(z / self.scale_factor)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input (deterministic).
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed tensor
        """
        recon, _, _ = self.forward(x, deterministic=True)
        return recon
    
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        latent_size: Tuple[int, int] = (60, 60)
    ) -> torch.Tensor:
        """
        Sample from prior and decode.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to use
            latent_size: Spatial size of latent
            
        Returns:
            Generated samples
        """
        # Sample from prior (standard normal)
        z = torch.randn(
            num_samples,
            self.latent_channels,
            latent_size[0],
            latent_size[1],
            device=device
        )
        
        # Decode
        samples = self.decode(z)
        
        return samples
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two images in latent space.
        
        Args:
            x1: First image (B, C, H, W)
            x2: Second image (B, C, H, W)
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated images (num_steps, C, H, W)
        """
        # Encode both images
        mean1, _ = self.encode(x1)
        mean2, _ = self.encode(x2)
        
        # Interpolate in latent space
        alphas = torch.linspace(0, 1, num_steps, device=x1.device)
        
        interpolations = []
        for alpha in alphas:
            z = (1 - alpha) * mean1 + alpha * mean2
            recon = self.decode(z)
            interpolations.append(recon)
        
        return torch.cat(interpolations, dim=0)


class VAESmall(nn.Module):
    """
    Smaller VAE for faster training and inference.
    Same interface as VAE but with reduced capacity.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        latent_channels: int = 4,
        hidden_channels: List[int] = [64, 128, 256]
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        
        self.encoder = VAEEncoderSmall(
            in_channels=in_channels,
            latent_channels=latent_channels,
            hidden_channels=hidden_channels
        )
        
        self.decoder = VAEDecoderSmall(
            out_channels=out_channels,
            latent_channels=latent_channels,
            hidden_channels=list(reversed(hidden_channels))
        )
        
        self.scale_factor = 0.18215
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
    
    def reparameterize(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        if deterministic:
            return mean
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False
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


class VAEWithLoss(nn.Module):
    """
    VAE with built-in loss computation for training convenience.
    """
    
    def __init__(
        self,
        config: Optional[VAEConfig] = None,
        use_combined_loss: bool = True
    ):
        super().__init__()
        
        if config is None:
            config = VAEConfig()
        
        self.vae = VAE(config)
        
        if use_combined_loss:
            self.loss_fn = CombinedVAELoss(
                recon_weight=config.recon_weight,
                kl_weight=config.kl_weight
            )
        else:
            self.loss_fn = VAELoss(
                recon_loss_type=config.recon_loss_type,
                recon_weight=config.recon_weight,
                kl_weight=config.kl_weight,
                perceptual_weight=config.perceptual_weight,
                use_perceptual=config.use_perceptual
            )
    
    def forward(
        self,
        x: torch.Tensor,
        compute_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.
        
        Args:
            x: Input tensor
            compute_loss: Whether to compute loss
            
        Returns:
            Dictionary with reconstruction, latent params, and optionally loss
        """
        recon, mean, log_var = self.vae(x)
        
        result = {
            'reconstruction': recon,
            'mean': mean,
            'log_var': log_var
        }
        
        if compute_loss:
            loss, loss_dict = self.loss_fn(recon, x, mean, log_var)
            result.update(loss_dict)
        
        return result
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vae.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)
    
    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.vae.get_latent(x, deterministic)
    
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode_latent(z)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.reconstruct(x)


def create_vae(
    model_type: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create VAE models.
    
    Args:
        model_type: Type of VAE ('standard', 'small', 'with_loss')
        **kwargs: Additional arguments for VAE config
        
    Returns:
        VAE model
    """
    if model_type == 'standard':
        config = VAEConfig(**kwargs)
        return VAE(config)
    elif model_type == 'small':
        return VAESmall(
            in_channels=kwargs.get('in_channels', 4),
            out_channels=kwargs.get('out_channels', 4),
            latent_channels=kwargs.get('latent_channels', 4)
        )
    elif model_type == 'with_loss':
        config = VAEConfig(**kwargs)
        return VAEWithLoss(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
