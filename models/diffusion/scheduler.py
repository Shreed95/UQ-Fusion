# models/diffusion/scheduler.py

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class BetaSchedule(Enum):
    """Beta schedule types for diffusion."""
    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"
    SIGMOID = "sigmoid"


@dataclass
class SchedulerConfig:
    """Configuration for noise scheduler."""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    prediction_type: str = "epsilon"  # "epsilon", "v_prediction", "sample"


class NoiseScheduler:
    """
    Noise scheduler for diffusion models.
    
    Manages the forward (noising) and reverse (denoising) diffusion processes.
    Supports multiple beta schedules and prediction types.
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize noise scheduler.
        
        Args:
            config: Scheduler configuration
        """
        if config is None:
            config = SchedulerConfig()
        
        self.config = config
        self.num_timesteps = config.num_timesteps
        self.prediction_type = config.prediction_type
        
        # Compute beta schedule
        betas = self._get_beta_schedule(
            config.beta_schedule,
            config.num_timesteps,
            config.beta_start,
            config.beta_end
        )
        
        # Convert to torch tensors
        self.betas = torch.tensor(betas, dtype=torch.float32)
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])
        
        # Pre-compute values for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Pre-compute values for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        
        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _get_beta_schedule(
        self,
        schedule_type: str,
        num_timesteps: int,
        beta_start: float,
        beta_end: float
    ) -> np.ndarray:
        """
        Get beta schedule.
        
        Args:
            schedule_type: Type of schedule
            num_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            
        Returns:
            Array of beta values
        """
        if schedule_type == "linear":
            return np.linspace(beta_start, beta_end, num_timesteps)
        
        elif schedule_type == "cosine":
            # Cosine schedule from "Improved DDPM"
            steps = num_timesteps + 1
            s = 0.008
            x = np.linspace(0, num_timesteps, steps)
            alphas_cumprod = np.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0.0001, 0.9999)
        
        elif schedule_type == "quadratic":
            return np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        
        elif schedule_type == "sigmoid":
            betas = np.linspace(-6, 6, num_timesteps)
            betas = (beta_end - beta_start) / (1 + np.exp(-betas)) + beta_start
            return betas
        
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_type}")
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """
        Extract values from tensor a at indices t.
        
        Args:
            a: Source tensor (T,)
            t: Indices (B,)
            x_shape: Shape to broadcast to
            
        Returns:
            Extracted values broadcast to x_shape
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0).
        Add noise to x_0 according to timestep t.
        
        Args:
            x_0: Original sample (B, C, H, W)
            t: Timesteps (B,)
            noise: Optional pre-generated noise
            
        Returns:
            Tuple of (noisy_sample, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Move scheduler tensors to device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_0.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_0.device)
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha_prod = self._extract(sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_prod = self._extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        
        return x_t, noise
    
    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior distribution q(x_{t-1} | x_t, x_0).
        
        Args:
            x_0: Original sample
            x_t: Noisy sample at timestep t
            t: Timesteps
            
        Returns:
            Tuple of (posterior_mean, posterior_variance)
        """
        posterior_mean_coef1 = self.posterior_mean_coef1.to(x_0.device)
        posterior_mean_coef2 = self.posterior_mean_coef2.to(x_0.device)
        posterior_variance = self.posterior_variance.to(x_0.device)
        
        coef1 = self._extract(posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(posterior_mean_coef2, t, x_t.shape)
        
        posterior_mean = coef1 * x_0 + coef2 * x_t
        posterior_variance = self._extract(posterior_variance, t, x_t.shape)
        
        return posterior_mean, posterior_variance
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from noise prediction.
        
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        
        Args:
            x_t: Noisy sample
            t: Timesteps
            noise: Predicted noise
            
        Returns:
            Predicted x_0
        """
        sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(x_t.device)
        sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(x_t.device)
        
        sqrt_recip = self._extract(sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        x_0 = sqrt_recip * x_t - sqrt_recipm1 * noise
        
        return x_0
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).
        
        Args:
            model: Denoising model
            x_t: Noisy sample
            t: Timesteps
            condition: Optional conditioning
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            Tuple of (model_mean, posterior_variance, posterior_log_variance, pred_x_0)
        """
        # Get model prediction
        model_output = model(x_t, t, condition)
        
        # Predict x_0 based on prediction type
        if self.prediction_type == "epsilon":
            pred_x_0 = self.predict_start_from_noise(x_t, t, model_output)
        elif self.prediction_type == "sample":
            pred_x_0 = model_output
        elif self.prediction_type == "v_prediction":
            # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_0
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_t.device)
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)
            
            sqrt_alpha = self._extract(sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alpha = self._extract(sqrt_one_minus, t, x_t.shape)
            
            pred_x_0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Clip predicted x_0
        if clip_denoised:
            pred_x_0 = torch.clamp(pred_x_0, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        # Compute posterior
        model_mean, posterior_variance = self.q_posterior(pred_x_0, x_t, t)
        posterior_log_variance = self.posterior_log_variance_clipped.to(x_t.device)
        posterior_log_variance = self._extract(posterior_log_variance, t, x_t.shape)
        
        return model_mean, posterior_variance, posterior_log_variance, pred_x_0
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single denoising step: sample x_{t-1} from p(x_{t-1} | x_t).
        
        Args:
            model: Denoising model
            x_t: Noisy sample at timestep t
            t: Timesteps
            condition: Optional conditioning
            
        Returns:
            Denoised sample x_{t-1}
        """
        model_mean, _, posterior_log_variance, _ = self.p_mean_variance(
            model, x_t, t, condition
        )
        
        # No noise at t=0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        # x_{t-1} = model_mean + sqrt(variance) * noise
        x_prev = model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
        
        return x_prev
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples (same as q_sample but different interface).
        """
        noisy, _ = self.q_sample(original_samples, timesteps, noise)
        return noisy
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity for v-prediction.
        
        v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * sample
        """
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(sample.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.to(sample.device)
        
        sqrt_alpha = self._extract(sqrt_alphas_cumprod, timesteps, sample.shape)
        sqrt_one_minus_alpha = self._extract(sqrt_one_minus, timesteps, sample.shape)
        
        velocity = sqrt_alpha * noise - sqrt_one_minus_alpha * sample
        
        return velocity


class DDPMScheduler(NoiseScheduler):
    """DDPM scheduler with standard sampling."""
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        if config is None:
            config = SchedulerConfig(
                num_timesteps=1000,
                beta_schedule="linear"
            )
        super().__init__(config)


class DDIMScheduler(NoiseScheduler):
    """DDIM scheduler for faster deterministic sampling."""
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0  # 0 for deterministic, 1 for DDPM
    ):
        if config is None:
            config = SchedulerConfig(
                num_timesteps=1000,
                beta_schedule="linear"
            )
        super().__init__(config)
        
        self.eta = eta
        self.set_timesteps(num_inference_steps)
    
    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference (subset of training timesteps)."""
        self.num_inference_steps = num_inference_steps
        
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = np.arange(0, num_inference_steps) * step_ratio
        timesteps = np.flip(timesteps).copy()  # Reverse for denoising
        
        self.timesteps = torch.from_numpy(timesteps).long()
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True
    ) -> Union[torch.Tensor, dict]:
        """
        DDIM denoising step.
        
        Args:
            model_output: Predicted noise from model
            timestep: Current timestep
            sample: Current noisy sample
            
        Returns:
            Denoised sample
        """
        # Current and previous timestep
        t = timestep
        prev_t = timestep - self.num_timesteps // self.num_inference_steps
        prev_t = max(prev_t, 0)
        
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        # Move to device
        alpha_prod_t = alpha_prod_t.to(sample.device)
        alpha_prod_t_prev = alpha_prod_t_prev.to(sample.device)
        
        # Predict x_0
        if self.prediction_type == "epsilon":
            pred_x_0 = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            pred_x_0 = model_output
        else:
            pred_x_0 = model_output
        
        # Clip
        if self.config.clip_sample:
            pred_x_0 = torch.clamp(pred_x_0, -self.config.clip_sample_range, self.config.clip_sample_range)
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev = self.eta * torch.sqrt(variance)
        
        # Compute "direction pointing to x_t"
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev ** 2) * model_output
        
        # Compute x_{t-1}
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_x_0 + pred_sample_direction
        
        if self.eta > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev * noise
        
        if return_dict:
            return {"prev_sample": prev_sample, "pred_original_sample": pred_x_0}
        return prev_sample
