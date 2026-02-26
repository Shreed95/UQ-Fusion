# models/uncertainty/aleatoric.py

"""
Aleatoric Uncertainty Estimation Module.

Aleatoric uncertainty represents inherent data uncertainty that cannot be
reduced with more training data. It captures noise, ambiguity, and variability
in the input images.

Estimation Method: Train network to predict both output and variance.
The variance head outputs σ² for each pixel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class AleatoricConfig:
    """Configuration for aleatoric uncertainty estimation."""
    min_log_var: float = -10.0  # Minimum log variance (for numerical stability)
    max_log_var: float = 10.0   # Maximum log variance
    learn_variance: bool = True  # Whether to learn variance
    variance_channels: int = 4   # Number of variance output channels


class VarianceHead(nn.Module):
    """
    Variance prediction head for aleatoric uncertainty.
    
    Outputs log-variance for numerical stability.
    Actual variance = exp(log_variance)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0
    ):
        super().__init__()
        
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )
        
        # Initialize to predict low variance
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.constant_(self.conv[-1].bias, -5.0)  # Start with low variance
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict log-variance.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Clamped log-variance (B, out_channels, H, W)
        """
        log_var = self.conv(x)
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)
        return log_var


class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic loss for aleatoric uncertainty estimation.
    
    Loss = (1/2σ²) * ||y - ŷ||² + (1/2) * log(σ²)
    
    The first term weights the MSE by inverse variance.
    The second term regularizes to prevent variance from growing unbounded.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute heteroscedastic loss.
        
        Args:
            pred: Predicted output (B, C, H, W)
            target: Ground truth (B, C, H, W)
            log_var: Log variance prediction (B, C, H, W)
            
        Returns:
            Loss value
        """
        # Precision-weighted MSE
        precision = torch.exp(-log_var)
        mse = (pred - target) ** 2
        
        # Heteroscedastic loss
        loss = 0.5 * precision * mse + 0.5 * log_var
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AleatoricUncertaintyEstimator(nn.Module):
    """
    Aleatoric uncertainty estimator that wraps a base model.
    
    Adds variance prediction capability to any model that outputs
    feature maps.
    """
    
    def __init__(
        self,
        feature_channels: int,
        output_channels: int = 4,
        config: Optional[AleatoricConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = AleatoricConfig()
        
        self.config = config
        
        self.variance_head = VarianceHead(
            in_channels=feature_channels,
            out_channels=output_channels,
            min_log_var=config.min_log_var,
            max_log_var=config.max_log_var
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate aleatoric uncertainty from features.
        
        Args:
            features: Feature tensor from base model (B, C, H, W)
            
        Returns:
            Log-variance map (B, out_C, H, W)
        """
        return self.variance_head(features)
    
    def get_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get uncertainty as standard deviation.
        
        Args:
            features: Feature tensor
            
        Returns:
            Standard deviation map (uncertainty)
        """
        log_var = self.forward(features)
        std = torch.exp(0.5 * log_var)
        return std


class DiffusionWithAleatoric(nn.Module):
    """
    Diffusion U-Net with aleatoric uncertainty estimation.
    
    Predicts both noise and variance at each denoising step.
    """
    
    def __init__(self, base_unet: nn.Module, latent_channels: int = 4):
        super().__init__()
        
        self.unet = base_unet
        
        # Get the output channels from U-Net's final conv
        # Assuming U-Net outputs latent_channels
        self.variance_head = VarianceHead(
            in_channels=latent_channels,
            out_channels=latent_channels
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        return_variance: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with variance prediction.
        
        Args:
            x: Noisy input
            timesteps: Diffusion timesteps
            condition: Optional conditioning
            return_variance: Whether to return variance
            
        Returns:
            Tuple of (noise_pred, log_variance)
        """
        # Get noise prediction from base U-Net
        noise_pred = self.unet(x, timesteps, condition)
        
        if return_variance:
            # Predict variance from the noise prediction
            log_var = self.variance_head(noise_pred)
            return noise_pred, log_var
        
        return noise_pred, None


class GANGeneratorWithAleatoric(nn.Module):
    """
    GAN Generator with aleatoric uncertainty estimation.
    
    Outputs both generated image and uncertainty map.
    """
    
    def __init__(self, base_generator: nn.Module, output_channels: int = 4):
        super().__init__()
        
        self.generator = base_generator
        
        # Variance head operates on generator output
        self.variance_head = VarianceHead(
            in_channels=output_channels,
            out_channels=output_channels
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_variance: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with variance prediction.
        
        Args:
            x: Input image
            return_variance: Whether to return variance
            
        Returns:
            Tuple of (generated_image, log_variance)
        """
        # Generate image
        generated = self.generator(x)
        
        if return_variance:
            # Predict variance from generated image
            log_var = self.variance_head(generated)
            return generated, log_var
        
        return generated, None


class AleatoricLoss(nn.Module):
    """
    Combined loss for training with aleatoric uncertainty.
    
    Combines task loss with heteroscedastic uncertainty weighting.
    """
    
    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        uncertainty_weight: float = 1.0
    ):
        super().__init__()
        
        self.base_loss = base_loss if base_loss else nn.MSELoss(reduction='none')
        self.hetero_loss = HeteroscedasticLoss(reduction='none')
        self.uncertainty_weight = uncertainty_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_var: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: Prediction
            target: Ground truth
            log_var: Log variance
            
        Returns:
            Dictionary of losses
        """
        # Base task loss (unweighted)
        base_loss = self.base_loss(pred, target).mean()
        
        # Heteroscedastic loss
        hetero_loss = self.hetero_loss(pred, target, log_var)
        
        # Combined
        total = base_loss + self.uncertainty_weight * hetero_loss
        
        return {
            'total_loss': total,
            'base_loss': base_loss,
            'hetero_loss': hetero_loss,
            'mean_var': torch.exp(log_var).mean()
        }


def compute_aleatoric_uncertainty(
    log_variance: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute aleatoric uncertainty from log-variance.
    
    Args:
        log_variance: Log variance prediction (B, C, H, W)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Uncertainty map (B, 1, H, W) or (B, C, H, W)
    """
    # Convert log-variance to standard deviation
    std = torch.exp(0.5 * log_variance)
    
    # Average across channels if multi-channel
    if std.shape[1] > 1:
        uncertainty = std.mean(dim=1, keepdim=True)
    else:
        uncertainty = std
    
    if normalize:
        # Normalize to [0, 1] per sample
        B = uncertainty.shape[0]
        uncertainty_flat = uncertainty.view(B, -1)
        min_val = uncertainty_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = uncertainty_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        uncertainty = (uncertainty - min_val) / (max_val - min_val + 1e-8)
    
    return uncertainty
