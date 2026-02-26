# models/uncertainty/combined.py

"""
Combined Uncertainty Estimation Module.

Combines aleatoric (data) and epistemic (model) uncertainty into
a unified uncertainty estimate for use in uncertainty-guided fusion.

Total Uncertainty = w_ale * U_aleatoric + w_epi * U_epistemic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from .aleatoric import (
    AleatoricUncertaintyEstimator,
    compute_aleatoric_uncertainty,
    VarianceHead
)
from .epistemic import (
    EpistemicUncertaintyEstimator,
    compute_epistemic_uncertainty,
    MCDropout
)


@dataclass
class CombinedUncertaintyConfig:
    """Configuration for combined uncertainty estimation."""
    # Aleatoric settings
    use_aleatoric: bool = True
    min_log_var: float = -10.0
    max_log_var: float = 10.0
    
    # Epistemic settings
    use_epistemic: bool = True
    num_mc_samples: int = 10
    dropout_rate: float = 0.1
    
    # Combination weights
    aleatoric_weight: float = 0.5
    epistemic_weight: float = 0.5
    
    # Output settings
    normalize: bool = True
    smooth_sigma: float = 0.0  # Gaussian smoothing (0 = no smoothing)


class CombinedUncertaintyEstimator(nn.Module):
    """
    Combined uncertainty estimator that computes both aleatoric
    and epistemic uncertainty and combines them.
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_channels: int,
        output_channels: int = 4,
        config: Optional[CombinedUncertaintyConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = CombinedUncertaintyConfig()
        
        self.config = config
        self.model = model
        
        # Aleatoric uncertainty
        if config.use_aleatoric:
            self.variance_head = VarianceHead(
                in_channels=output_channels,
                out_channels=output_channels,
                min_log_var=config.min_log_var,
                max_log_var=config.max_log_var
            )
        else:
            self.variance_head = None
        
        # MC Dropout for epistemic uncertainty
        if config.use_epistemic:
            self.mc_dropout = MCDropout(config.dropout_rate)
            self.num_mc_samples = config.num_mc_samples
        else:
            self.mc_dropout = None
    
    def _enable_dropout(self):
        """Enable dropout in model for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined uncertainty.
        
        Args:
            x: Input tensor
            *args, **kwargs: Additional model arguments
            
        Returns:
            Dictionary containing:
                - 'prediction': Mean prediction
                - 'aleatoric': Aleatoric uncertainty map
                - 'epistemic': Epistemic uncertainty map
                - 'total': Combined uncertainty map
        """
        results = {}
        
        # Epistemic uncertainty via MC Dropout
        if self.config.use_epistemic:
            self._enable_dropout()
            
            mc_predictions = []
            mc_log_vars = []
            
            for _ in range(self.num_mc_samples):
                pred = self.model(x, *args, **kwargs)
                
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                mc_predictions.append(pred)
                
                if self.variance_head is not None:
                    log_var = self.variance_head(pred)
                    mc_log_vars.append(log_var)
            
            # Stack predictions
            mc_predictions = torch.stack(mc_predictions, dim=0)
            
            # Mean prediction
            mean_pred = mc_predictions.mean(dim=0)
            results['prediction'] = mean_pred
            
            # Epistemic uncertainty (variance across MC samples)
            epistemic = compute_epistemic_uncertainty(mc_predictions, normalize=False)
            results['epistemic'] = epistemic
            
            # Aleatoric uncertainty (mean of predicted variances)
            if self.variance_head is not None and len(mc_log_vars) > 0:
                mc_log_vars = torch.stack(mc_log_vars, dim=0)
                mean_log_var = mc_log_vars.mean(dim=0)
                aleatoric = compute_aleatoric_uncertainty(mean_log_var, normalize=False)
                results['aleatoric'] = aleatoric
            else:
                results['aleatoric'] = torch.zeros_like(epistemic)
        
        else:
            # Single forward pass
            pred = self.model(x, *args, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            results['prediction'] = pred
            results['epistemic'] = torch.zeros(
                pred.shape[0], 1, pred.shape[2], pred.shape[3],
                device=pred.device
            )
            
            if self.variance_head is not None:
                log_var = self.variance_head(pred)
                aleatoric = compute_aleatoric_uncertainty(log_var, normalize=False)
                results['aleatoric'] = aleatoric
            else:
                results['aleatoric'] = torch.zeros_like(results['epistemic'])
        
        # Combined uncertainty
        total = (
            self.config.aleatoric_weight * results['aleatoric'] +
            self.config.epistemic_weight * results['epistemic']
        )
        
        # Normalize
        if self.config.normalize:
            results['aleatoric'] = self._normalize(results['aleatoric'])
            results['epistemic'] = self._normalize(results['epistemic'])
            total = self._normalize(total)
        
        # Smooth if requested
        if self.config.smooth_sigma > 0:
            total = self._smooth(total, self.config.smooth_sigma)
        
        results['total'] = total
        
        return results
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0, 1] per sample."""
        B = x.shape[0]
        x_flat = x.view(B, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        return (x - min_val) / (max_val - min_val + 1e-8)
    
    def _smooth(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
        
        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, device=x.device).float() - kernel_size // 2
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        # Apply
        padding = kernel_size // 2
        return F.conv2d(x, kernel_2d, padding=padding)


class BranchUncertaintyEstimator(nn.Module):
    """
    Uncertainty estimator for a single generation branch (diffusion or GAN).
    
    Produces uncertainty maps that can be used for fusion.
    """
    
    def __init__(
        self,
        branch_type: str,  # 'diffusion' or 'gan'
        model: nn.Module,
        output_channels: int = 4,
        config: Optional[CombinedUncertaintyConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = CombinedUncertaintyConfig()
        
        self.branch_type = branch_type
        self.config = config
        self.model = model
        
        # Variance head for aleatoric uncertainty
        self.variance_head = VarianceHead(
            in_channels=output_channels,
            out_channels=output_channels,
            min_log_var=config.min_log_var,
            max_log_var=config.max_log_var
        )
        
        self.num_mc_samples = config.num_mc_samples
    
    def _enable_dropout(self):
        """Enable dropout in model."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty for this branch.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with prediction and uncertainty maps
        """
        self._enable_dropout()
        
        predictions = []
        log_variances = []
        
        for _ in range(self.num_mc_samples):
            if self.branch_type == 'diffusion':
                # For diffusion, model expects additional arguments
                pred = self.model(x, *args, **kwargs)
            else:
                # For GAN, just input
                pred = self.model(x)
            
            if isinstance(pred, tuple):
                pred = pred[0]
            
            predictions.append(pred)
            log_variances.append(self.variance_head(pred))
        
        predictions = torch.stack(predictions, dim=0)
        log_variances = torch.stack(log_variances, dim=0)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Aleatoric uncertainty (from variance prediction)
        mean_log_var = log_variances.mean(dim=0)
        aleatoric = compute_aleatoric_uncertainty(mean_log_var, normalize=False)
        
        # Epistemic uncertainty (variance across MC samples)
        epistemic = compute_epistemic_uncertainty(predictions, normalize=False)
        
        # Combined
        total = (
            self.config.aleatoric_weight * aleatoric +
            self.config.epistemic_weight * epistemic
        )
        
        # Normalize
        if self.config.normalize:
            aleatoric = self._normalize(aleatoric)
            epistemic = self._normalize(epistemic)
            total = self._normalize(total)
        
        return {
            'prediction': mean_pred,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total': total,
            'log_variance': mean_log_var
        }
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0, 1]."""
        B = x.shape[0]
        x_flat = x.view(B, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        return (x - min_val) / (max_val - min_val + 1e-8)


class DualBranchUncertaintyEstimator(nn.Module):
    """
    Uncertainty estimator for both diffusion and GAN branches.
    
    Produces uncertainty maps for both branches that can be used
    in the uncertainty-guided fusion module.
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        gan_model: nn.Module,
        output_channels: int = 4,
        config: Optional[CombinedUncertaintyConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = CombinedUncertaintyConfig()
        
        self.config = config
        
        # Branch estimators
        self.diffusion_estimator = BranchUncertaintyEstimator(
            'diffusion', diffusion_model, output_channels, config
        )
        self.gan_estimator = BranchUncertaintyEstimator(
            'gan', gan_model, output_channels, config
        )
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        diffusion_args: dict = None,
        gan_args: dict = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute uncertainty for both branches.
        
        Args:
            x: Input tensor
            diffusion_args: Additional arguments for diffusion model
            gan_args: Additional arguments for GAN model
            
        Returns:
            Dictionary with results for each branch
        """
        if diffusion_args is None:
            diffusion_args = {}
        if gan_args is None:
            gan_args = {}
        
        diffusion_results = self.diffusion_estimator(x, **diffusion_args)
        gan_results = self.gan_estimator(x, **gan_args)
        
        return {
            'diffusion': diffusion_results,
            'gan': gan_results
        }


def combine_uncertainties(
    aleatoric: torch.Tensor,
    epistemic: torch.Tensor,
    aleatoric_weight: float = 0.5,
    epistemic_weight: float = 0.5,
    normalize: bool = True
) -> torch.Tensor:
    """
    Combine aleatoric and epistemic uncertainties.
    
    Args:
        aleatoric: Aleatoric uncertainty map
        epistemic: Epistemic uncertainty map
        aleatoric_weight: Weight for aleatoric
        epistemic_weight: Weight for epistemic
        normalize: Whether to normalize output
        
    Returns:
        Combined uncertainty map
    """
    combined = aleatoric_weight * aleatoric + epistemic_weight * epistemic
    
    if normalize:
        B = combined.shape[0]
        c_flat = combined.view(B, -1)
        min_val = c_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = c_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        combined = (combined - min_val) / (max_val - min_val + 1e-8)
    
    return combined


def uncertainty_quality_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    uncertainty: torch.Tensor
) -> Dict[str, float]:
    """
    Compute quality scores that relate uncertainty to actual error.
    
    High-quality uncertainty should:
    - Be high where error is high
    - Be low where error is low
    
    Args:
        prediction: Model prediction
        target: Ground truth
        uncertainty: Uncertainty map
        
    Returns:
        Dictionary of quality metrics
    """
    # Compute error
    error = torch.abs(prediction - target).mean(dim=1, keepdim=True)
    
    # Flatten for correlation
    error_flat = error.view(-1)
    uncertainty_flat = uncertainty.view(-1)
    
    # Correlation between error and uncertainty
    correlation = torch.corrcoef(
        torch.stack([error_flat, uncertainty_flat])
    )[0, 1].item()
    
    # Mean uncertainty in high-error regions
    error_threshold = error_flat.quantile(0.75)
    high_error_mask = error_flat > error_threshold
    high_error_uncertainty = uncertainty_flat[high_error_mask].mean().item()
    low_error_uncertainty = uncertainty_flat[~high_error_mask].mean().item()
    
    return {
        'correlation': correlation,
        'high_error_uncertainty': high_error_uncertainty,
        'low_error_uncertainty': low_error_uncertainty,
        'uncertainty_separation': high_error_uncertainty - low_error_uncertainty
    }
