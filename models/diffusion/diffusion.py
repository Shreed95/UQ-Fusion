# models/diffusion/diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass

from .unet import DiffusionUNet, DiffusionUNetSmall, UNetConfig
from .scheduler import NoiseScheduler, DDPMScheduler, DDIMScheduler, SchedulerConfig
from .sampler import DDPMSampler, DDIMSampler, ImageToImageSampler


@dataclass
class LatentDiffusionConfig:
    """Configuration for Latent Diffusion Model."""
    # U-Net config
    latent_channels: int = 4
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (1, 2)
    time_embed_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    
    # Scheduler config
    num_timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    
    # Training
    use_conditioning: bool = True
    
    # VAE scaling
    scale_factor: float = 0.18215


class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model for medical image synthesis.
    
    Operates in the compressed latent space from VAE.
    Supports image-to-image translation with conditioning.
    
    Architecture:
        Source Image -> VAE Encoder -> Latent (60x60)
        Latent + Noise + Time -> U-Net -> Predicted Noise
        Denoised Latent -> VAE Decoder -> Generated Image
    """
    
    def __init__(
        self,
        config: Optional[LatentDiffusionConfig] = None,
        vae: Optional[nn.Module] = None
    ):
        """
        Initialize Latent Diffusion Model.
        
        Args:
            config: Model configuration
            vae: Pre-trained VAE (optional, can be set later)
        """
        super().__init__()
        
        if config is None:
            config = LatentDiffusionConfig()
        
        self.config = config
        self.scale_factor = config.scale_factor
        
        # VAE (frozen, used for encoding/decoding)
        self.vae = vae
        if vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False
        
        # U-Net for denoising
        unet_config = UNetConfig(
            in_channels=config.latent_channels,
            out_channels=config.latent_channels,
            base_channels=config.base_channels,
            channel_multipliers=config.channel_multipliers,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            time_embed_dim=config.time_embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            use_conditioning=config.use_conditioning
        )
        self.unet = DiffusionUNet(unet_config)
        
        # Noise scheduler
        scheduler_config = SchedulerConfig(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type
        )
        self.scheduler = NoiseScheduler(scheduler_config)
    
    def set_vae(self, vae: nn.Module):
        """Set VAE and freeze its parameters."""
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space using VAE.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            Latent representation (B, latent_C, H/4, W/4)
        """
        if self.vae is None:
            raise ValueError("VAE not set. Call set_vae() first.")
        
        self.vae.eval()
        z = self.vae.get_latent(x, deterministic=True)
        return z
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image using VAE.
        
        Args:
            z: Latent representation
            
        Returns:
            Decoded image
        """
        if self.vae is None:
            raise ValueError("VAE not set. Call set_vae() first.")
        
        self.vae.eval()
        x = self.vae.decode_latent(z)
        return x
    
    def forward(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            x_0: Clean latent (B, C, H, W)
            condition: Conditioning latent (B, C, H, W)
            noise: Optional pre-generated noise
            timesteps: Optional specific timesteps
            
        Returns:
            Tuple of (noise_pred, noise_target, timesteps)
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.num_timesteps,
                (batch_size,), device=device, dtype=torch.long
            )
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Add noise to latent
        x_t, _ = self.scheduler.q_sample(x_0, timesteps, noise)
        
        # Predict noise
        noise_pred = self.unet(x_t, timesteps, condition)
        
        return noise_pred, noise, timesteps
    
    def get_loss(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x_0: Clean latent
            condition: Conditioning latent
            noise: Optional pre-generated noise
            
        Returns:
            Dictionary with loss values
        """
        noise_pred, noise_target, _ = self.forward(x_0, condition, noise)
        
        # MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise_target)
        
        return {
            'loss': loss,
            'mse_loss': loss
        }
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: Optional[torch.Tensor] = None,
        latent_shape: Tuple[int, int] = (60, 60),
        num_inference_steps: int = 50,
        use_ddim: bool = True,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            batch_size: Number of samples to generate
            condition: Optional conditioning latent
            latent_shape: Spatial size of latent
            num_inference_steps: Number of denoising steps
            use_ddim: Whether to use DDIM (faster) or DDPM
            show_progress: Whether to show progress bar
            
        Returns:
            Generated latents (B, C, H, W)
        """
        self.unet.eval()
        device = next(self.unet.parameters()).device
        
        shape = (batch_size, self.config.latent_channels, *latent_shape)
        
        if use_ddim:
            sampler = DDIMSampler(self.scheduler, self.unet, num_inference_steps)
        else:
            sampler = DDPMSampler(self.scheduler, self.unet)
        
        latents = sampler.sample(shape, condition, device, show_progress)
        
        return latents
    
    @torch.no_grad()
    def generate(
        self,
        source_images: torch.Tensor,
        num_inference_steps: int = 50,
        strength: float = 0.8,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate images from source images (image-to-image).
        
        Args:
            source_images: Source images (B, C, H, W)
            num_inference_steps: Number of denoising steps
            strength: Transformation strength (0-1)
            show_progress: Whether to show progress bar
            
        Returns:
            Generated images (B, C, H, W)
        """
        # Encode source to latent
        source_latent = self.encode(source_images)
        
        # Create sampler
        sampler = ImageToImageSampler(
            self.scheduler,
            self.unet,
            strength=strength,
            num_inference_steps=num_inference_steps
        )
        
        # Generate
        generated_latent = sampler.sample(
            source_latent,
            condition=source_latent,
            show_progress=show_progress
        )
        
        # Decode to image
        generated_images = self.decode(generated_latent)
        
        return generated_images


class LatentDiffusionModelSmall(nn.Module):
    """
    Smaller Latent Diffusion Model for faster training.
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 64,
        num_timesteps: int = 1000,
        vae: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.scale_factor = 0.18215
        
        # VAE
        self.vae = vae
        if vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False
        
        # U-Net
        self.unet = DiffusionUNetSmall(
            in_channels=latent_channels,
            out_channels=latent_channels,
            base_channels=base_channels,
            time_embed_dim=base_channels * 4,
            use_conditioning=True
        )
        
        # Scheduler
        scheduler_config = SchedulerConfig(
            num_timesteps=num_timesteps,
            beta_schedule="linear"
        )
        self.scheduler = NoiseScheduler(scheduler_config)
    
    def set_vae(self, vae: nn.Module):
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return x  # Assume already encoded
        self.vae.eval()
        return self.vae.get_latent(x, deterministic=True)
    
    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return z  # Assume no decoding needed
        self.vae.eval()
        return self.vae.decode_latent(z)
    
    def forward(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x_0.shape[0]
        device = x_0.device
        
        if timesteps is None:
            timesteps = torch.randint(
                0, self.scheduler.num_timesteps,
                (batch_size,), device=device, dtype=torch.long
            )
        
        if noise is None:
            noise = torch.randn_like(x_0)
        
        x_t, _ = self.scheduler.q_sample(x_0, timesteps, noise)
        noise_pred = self.unet(x_t, timesteps, condition)
        
        return noise_pred, noise, timesteps
    
    def get_loss(
        self,
        x_0: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        noise_pred, noise_target, _ = self.forward(x_0, condition, noise)
        loss = F.mse_loss(noise_pred, noise_target)
        return {'loss': loss, 'mse_loss': loss}
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: Optional[torch.Tensor] = None,
        latent_shape: Tuple[int, int] = (60, 60),
        num_inference_steps: int = 50,
        show_progress: bool = True
    ) -> torch.Tensor:
        self.unet.eval()
        device = next(self.unet.parameters()).device
        
        shape = (batch_size, self.latent_channels, *latent_shape)
        sampler = DDIMSampler(self.scheduler, self.unet, num_inference_steps)
        
        return sampler.sample(shape, condition, device, show_progress)
    
    @torch.no_grad()
    def generate(
        self,
        source_images: torch.Tensor,
        num_inference_steps: int = 50,
        strength: float = 0.8,
        show_progress: bool = True
    ) -> torch.Tensor:
        source_latent = self.encode(source_images)
        
        sampler = ImageToImageSampler(
            self.scheduler,
            self.unet,
            strength=strength,
            num_inference_steps=num_inference_steps
        )
        
        generated_latent = sampler.sample(
            source_latent,
            condition=source_latent,
            show_progress=show_progress
        )
        
        return self.decode(generated_latent)


def create_latent_diffusion(
    model_type: str = 'small',
    vae: Optional[nn.Module] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create Latent Diffusion Model.
    
    Args:
        model_type: 'standard' or 'small'
        vae: Optional pre-trained VAE
        **kwargs: Additional arguments
        
    Returns:
        Latent Diffusion Model
    """
    if model_type == 'standard':
        config = LatentDiffusionConfig(**kwargs)
        return LatentDiffusionModel(config, vae)
    elif model_type == 'small':
        return LatentDiffusionModelSmall(
            latent_channels=kwargs.get('latent_channels', 4),
            base_channels=kwargs.get('base_channels', 64),
            num_timesteps=kwargs.get('num_timesteps', 1000),
            vae=vae
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
