# models/diffusion/sampler.py

import torch
import torch.nn as nn
from typing import Optional, Callable, List, Union
from tqdm import tqdm

from .scheduler import NoiseScheduler, DDIMScheduler


class DDPMSampler:
    """
    DDPM sampler for diffusion models.
    Standard sampling with all timesteps.
    """
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        model: nn.Module
    ):
        """
        Initialize DDPM sampler.
        
        Args:
            scheduler: Noise scheduler
            model: Denoising U-Net model
        """
        self.scheduler = scheduler
        self.model = model
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        condition: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        show_progress: bool = True,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate samples using DDPM sampling.
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            condition: Optional conditioning tensor
            device: Device to use
            show_progress: Whether to show progress bar
            return_intermediates: Whether to return intermediate samples
            
        Returns:
            Generated samples or list of intermediate samples
        """
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        intermediates = []
        
        # Reverse diffusion
        timesteps = list(range(self.scheduler.num_timesteps))[::-1]
        
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            x = self.scheduler.p_sample(self.model, x, t_batch, condition)
            
            if return_intermediates and t % 100 == 0:
                intermediates.append(x.clone())
        
        if return_intermediates:
            intermediates.append(x)
            return intermediates
        
        return x
    
    @torch.no_grad()
    def sample_loop(
        self,
        shape: tuple,
        condition: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        callback: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Generate samples with optional callback.
        
        Args:
            shape: Shape of samples
            condition: Optional conditioning
            device: Device to use
            callback: Optional callback function(x, t)
            
        Returns:
            Generated samples
        """
        self.model.eval()
        
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.scheduler.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.scheduler.p_sample(self.model, x, t_batch, condition)
            
            if callback is not None:
                callback(x, t)
        
        return x


class DDIMSampler:
    """
    DDIM sampler for faster deterministic sampling.
    Uses subset of timesteps for accelerated generation.
    """
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        model: nn.Module,
        num_inference_steps: int = 50,
        eta: float = 0.0
    ):
        """
        Initialize DDIM sampler.
        
        Args:
            scheduler: Noise scheduler
            model: Denoising U-Net model
            num_inference_steps: Number of denoising steps
            eta: Noise scale (0 for deterministic, 1 for DDPM)
        """
        self.scheduler = scheduler
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        # Setup timesteps
        self._setup_timesteps()
    
    def _setup_timesteps(self):
        """Setup inference timesteps (subset of training timesteps)."""
        step_ratio = self.scheduler.num_timesteps // self.num_inference_steps
        self.timesteps = list(range(0, self.scheduler.num_timesteps, step_ratio))[::-1]
    
    def _get_prev_timestep(self, t: int) -> int:
        """Get previous timestep."""
        idx = self.timesteps.index(t)
        if idx < len(self.timesteps) - 1:
            return self.timesteps[idx + 1]
        return 0
    
    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Single DDIM denoising step.
        
        Args:
            model_output: Predicted noise
            t: Current timestep
            x: Current noisy sample
            
        Returns:
            Denoised sample
        """
        prev_t = self._get_prev_timestep(t)
        
        # Get alpha values
        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(x.device)
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t].to(x.device) if prev_t > 0 else torch.tensor(1.0, device=x.device)
        
        # Predict x_0
        if self.scheduler.prediction_type == "epsilon":
            pred_x_0 = (x - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        else:
            pred_x_0 = model_output
        
        # Clip x_0
        pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev = self.eta * torch.sqrt(variance)
        
        # Direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev ** 2) * model_output
        
        # x_{t-1}
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_x_0 + pred_sample_direction
        
        if self.eta > 0:
            noise = torch.randn_like(x)
            prev_sample = prev_sample + std_dev * noise
        
        return prev_sample
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        condition: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        show_progress: bool = True,
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate samples using DDIM sampling.
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            condition: Optional conditioning tensor
            device: Device to use
            show_progress: Whether to show progress bar
            return_intermediates: Whether to return intermediate samples
            
        Returns:
            Generated samples or list of intermediate samples
        """
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        intermediates = []
        
        timesteps = self.timesteps
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"DDIM Sampling ({self.num_inference_steps} steps)")
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Get model prediction
            model_output = self.model(x, t_batch, condition)
            
            # DDIM step
            x = self.step(model_output, t, x)
            
            if return_intermediates:
                intermediates.append(x.clone())
        
        if return_intermediates:
            return intermediates
        
        return x
    
    def set_timesteps(self, num_inference_steps: int):
        """Update number of inference steps."""
        self.num_inference_steps = num_inference_steps
        self._setup_timesteps()


class GuidedSampler:
    """
    Sampler with classifier-free guidance for conditional generation.
    """
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        model: nn.Module,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ):
        """
        Initialize guided sampler.
        
        Args:
            scheduler: Noise scheduler
            model: Denoising U-Net model
            guidance_scale: Guidance scale (higher = more conditioning)
            num_inference_steps: Number of denoising steps
        """
        self.scheduler = scheduler
        self.model = model
        self.guidance_scale = guidance_scale
        
        self.ddim_sampler = DDIMSampler(scheduler, model, num_inference_steps)
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        condition: torch.Tensor,
        device: str = 'cuda',
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples with classifier-free guidance.
        
        Args:
            shape: Shape of samples to generate
            condition: Conditioning tensor
            device: Device to use
            show_progress: Whether to show progress bar
            
        Returns:
            Generated samples
        """
        self.model.eval()
        
        x = torch.randn(shape, device=device)
        
        timesteps = self.ddim_sampler.timesteps
        if show_progress:
            timesteps = tqdm(timesteps, desc="Guided Sampling")
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Conditional prediction
            noise_pred_cond = self.model(x, t_batch, condition)
            
            # Unconditional prediction (with zero/null condition)
            null_condition = torch.zeros_like(condition)
            noise_pred_uncond = self.model(x, t_batch, null_condition)
            
            # Classifier-free guidance
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # DDIM step
            x = self.ddim_sampler.step(noise_pred, t, x)
        
        return x


class ImageToImageSampler:
    """
    Sampler for image-to-image translation.
    Starts from noised source image instead of pure noise.
    """
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        model: nn.Module,
        strength: float = 0.8,
        num_inference_steps: int = 50
    ):
        """
        Initialize image-to-image sampler.
        
        Args:
            scheduler: Noise scheduler
            model: Denoising U-Net model
            strength: How much to transform (0 = no change, 1 = full generation)
            num_inference_steps: Number of denoising steps
        """
        self.scheduler = scheduler
        self.model = model
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        
        self.ddim_sampler = DDIMSampler(scheduler, model, num_inference_steps)
    
    @torch.no_grad()
    def sample(
        self,
        source: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples from source image.
        
        Args:
            source: Source image/latent to transform
            condition: Optional additional conditioning
            show_progress: Whether to show progress bar
            
        Returns:
            Generated samples
        """
        self.model.eval()
        device = source.device
        batch_size = source.shape[0]
        
        # Determine starting timestep based on strength
        start_step = int(self.num_inference_steps * (1 - self.strength))
        timesteps = self.ddim_sampler.timesteps[start_step:]
        
        # Add noise to source at starting timestep
        if len(timesteps) > 0:
            start_t = timesteps[0]
            t_batch = torch.full((batch_size,), start_t, device=device, dtype=torch.long)
            
            noise = torch.randn_like(source)
            x, _ = self.scheduler.q_sample(source, t_batch, noise)
        else:
            x = source
        
        # Use source as condition if not provided
        if condition is None:
            condition = source
        
        if show_progress:
            timesteps = tqdm(timesteps, desc="Image-to-Image")
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get model prediction
            noise_pred = self.model(x, t_batch, condition)
            
            # DDIM step
            x = self.ddim_sampler.step(noise_pred, t, x)
        
        return x
