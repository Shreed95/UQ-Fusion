# models/uncertainty/epistemic.py

"""
Epistemic Uncertainty Estimation Module.

Epistemic uncertainty represents model uncertainty due to limited training
data or model capacity. It is reducible with more data or better models.

Estimation Methods:
1. Monte Carlo Dropout: Keep dropout active during inference
2. Deep Ensembles: Train multiple models with different initializations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import copy


@dataclass
class EpistemicConfig:
    """Configuration for epistemic uncertainty estimation."""
    method: str = 'mc_dropout'  # 'mc_dropout' or 'ensemble'
    num_samples: int = 10       # Number of forward passes (MC) or ensemble members
    dropout_rate: float = 0.1   # Dropout rate for MC Dropout
    normalize: bool = True      # Whether to normalize uncertainty


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout module.
    
    Keeps dropout active during inference to estimate epistemic uncertainty
    through multiple stochastic forward passes.
    """
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Always apply dropout (even in eval mode)."""
        return F.dropout(x, p=self.p, training=True)


class MCDropoutWrapper(nn.Module):
    """
    Wraps a model to add MC Dropout for epistemic uncertainty estimation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
        # Add dropout layers after activations
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Add MC Dropout after activation functions in the model."""
        # This is a simplified version - in practice, you might want to
        # modify specific layers or use existing dropout layers
        self.dropout = MCDropout(self.dropout_rate)
    
    def enable_dropout(self):
        """Enable dropout for all modules."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def forward_with_uncertainty(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with epistemic uncertainty estimation.
        
        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments for model
            
        Returns:
            Tuple of (mean_prediction, variance, all_predictions)
        """
        self.enable_dropout()
        
        predictions = []
        
        for _ in range(self.num_samples):
            # Apply dropout before model (if not built-in)
            x_dropout = self.dropout(x)
            pred = self.model(x_dropout, *args, **kwargs)
            
            # Handle tuple outputs
            if isinstance(pred, tuple):
                pred = pred[0]
            
            predictions.append(pred)
        
        # Stack predictions: (num_samples, B, C, H, W)
        predictions = torch.stack(predictions, dim=0)
        
        # Compute statistics
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return mean, variance, predictions


class EpistemicUncertaintyEstimator(nn.Module):
    """
    Epistemic uncertainty estimator using MC Dropout.
    
    Performs multiple forward passes with dropout enabled and
    computes prediction variance as uncertainty estimate.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EpistemicConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = EpistemicConfig()
        
        self.model = model
        self.config = config
        self.num_samples = config.num_samples
    
    def _enable_dropout(self):
        """Enable dropout layers for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate epistemic uncertainty via MC Dropout.
        
        Args:
            x: Input tensor
            *args, **kwargs: Additional model arguments
            
        Returns:
            Tuple of (mean_prediction, uncertainty_map)
        """
        self._enable_dropout()
        
        predictions = []
        
        for i in range(self.num_samples):
            pred = self.model(x, *args, **kwargs)
            
            # Handle tuple outputs (e.g., from models returning variance)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Mean prediction
        mean = predictions.mean(dim=0)
        
        # Variance as uncertainty (average across channels)
        variance = predictions.var(dim=0)
        uncertainty = variance.mean(dim=1, keepdim=True)
        
        if self.config.normalize:
            uncertainty = self._normalize(uncertainty)
        
        return mean, uncertainty
    
    def _normalize(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Normalize uncertainty to [0, 1] per sample."""
        B = uncertainty.shape[0]
        u_flat = uncertainty.view(B, -1)
        min_val = u_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = u_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        return (uncertainty - min_val) / (max_val - min_val + 1e-8)
    
    def get_all_predictions(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Get all MC predictions for analysis."""
        self._enable_dropout()
        
        predictions = []
        for _ in range(self.num_samples):
            pred = self.model(x, *args, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            predictions.append(pred)
        
        return torch.stack(predictions, dim=0)


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for epistemic uncertainty estimation.
    
    Uses multiple independently trained models and measures
    disagreement as uncertainty.
    """
    
    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        num_models: int = 5
    ):
        """
        Args:
            model_fn: Function that returns a new model instance
            num_models: Number of ensemble members
        """
        super().__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([model_fn() for _ in range(num_models)])
    
    def load_checkpoints(self, checkpoint_paths: List[str], device: str = 'cuda'):
        """Load pre-trained weights for each ensemble member."""
        for i, path in enumerate(checkpoint_paths):
            if i < len(self.models):
                checkpoint = torch.load(path, map_location=device)
                self.models[i].load_state_dict(checkpoint['model_state_dict'])
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all ensemble members.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            pred = model(x, *args, **kwargs)
            
            if isinstance(pred, tuple):
                pred = pred[0]
            
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        uncertainty = variance.mean(dim=1, keepdim=True)
        
        return mean, uncertainty
    
    def get_ensemble_predictions(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> List[torch.Tensor]:
        """Get predictions from all ensemble members."""
        predictions = []
        
        for model in self.models:
            model.eval()
            pred = model(x, *args, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            predictions.append(pred)
        
        return predictions


class DiffusionEpistemicEstimator(nn.Module):
    """
    Epistemic uncertainty estimator for diffusion models.
    
    Uses MC Dropout during the denoising process.
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        num_samples: int = 10
    ):
        super().__init__()
        
        self.diffusion_model = diffusion_model
        self.num_samples = num_samples
    
    def _enable_dropout(self):
        """Enable dropout in U-Net."""
        for module in self.diffusion_model.unet.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
    
    @torch.no_grad()
    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate epistemic uncertainty for a single denoising step.
        
        Args:
            x: Noisy input
            timesteps: Diffusion timesteps
            condition: Optional conditioning
            
        Returns:
            Tuple of (mean_noise_pred, uncertainty)
        """
        self._enable_dropout()
        
        predictions = []
        
        for _ in range(self.num_samples):
            noise_pred = self.diffusion_model.unet(x, timesteps, condition)
            predictions.append(noise_pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        uncertainty = variance.mean(dim=1, keepdim=True)
        
        return mean, uncertainty


class GANEpistemicEstimator(nn.Module):
    """
    Epistemic uncertainty estimator for GAN generators.
    
    Uses MC Dropout in the generator.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        num_samples: int = 10
    ):
        super().__init__()
        
        self.generator = generator
        self.num_samples = num_samples
    
    def _enable_dropout(self):
        """Enable dropout in generator."""
        for module in self.generator.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
    
    @torch.no_grad()
    def estimate_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate epistemic uncertainty for generation.
        
        Args:
            x: Input image
            
        Returns:
            Tuple of (mean_generated, uncertainty)
        """
        self._enable_dropout()
        
        predictions = []
        
        for _ in range(self.num_samples):
            generated = self.generator(x)
            predictions.append(generated)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        uncertainty = variance.mean(dim=1, keepdim=True)
        
        return mean, uncertainty


def compute_epistemic_uncertainty(
    predictions: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute epistemic uncertainty from multiple predictions.
    
    Args:
        predictions: Tensor of shape (num_samples, B, C, H, W)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Uncertainty map (B, 1, H, W)
    """
    # Compute variance across samples
    variance = predictions.var(dim=0)
    
    # Average across channels
    uncertainty = variance.mean(dim=1, keepdim=True)
    
    if normalize:
        B = uncertainty.shape[0]
        u_flat = uncertainty.view(B, -1)
        min_val = u_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = u_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        uncertainty = (uncertainty - min_val) / (max_val - min_val + 1e-8)
    
    return uncertainty


def entropy_from_predictions(predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy-based uncertainty from predictions.
    
    For continuous outputs, uses differential entropy approximation.
    
    Args:
        predictions: Tensor of shape (num_samples, B, C, H, W)
        
    Returns:
        Entropy map (B, 1, H, W)
    """
    # Compute standard deviation
    std = predictions.std(dim=0)
    
    # Differential entropy of Gaussian: H = 0.5 * log(2 * pi * e * sigma^2)
    # Simplified: H ∝ log(sigma)
    entropy = torch.log(std + 1e-8)
    
    # Average across channels
    entropy = entropy.mean(dim=1, keepdim=True)
    
    return entropy
