# models/fusion/fusion_module.py

"""
Uncertainty-Guided Fusion Module.

Combines outputs from diffusion and GAN branches based on their
respective uncertainty estimates, giving higher weight to the branch
with lower uncertainty at each spatial location.

Fusion Weight Computation:
    W_diff = 1/(U_diff + ε)
    W_gan = 1/(U_gan + ε)
    α = W_diff / (W_diff + W_gan)
    β = W_gan / (W_diff + W_gan)
    
Output:
    I_fused = α ⊙ I_diff + β ⊙ I_gan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """Configuration for fusion module."""
    fusion_method: str = 'uncertainty'  # 'uncertainty', 'learned', 'adaptive', 'average'
    epsilon: float = 1e-6              # Numerical stability
    smooth_weights: bool = True        # Apply Gaussian smoothing to weights
    smooth_sigma: float = 2.0          # Gaussian sigma for smoothing
    temperature: float = 1.0           # Temperature for weight sharpening
    learnable_epsilon: bool = False    # Learn epsilon parameter


class UncertaintyGuidedFusion(nn.Module):
    """
    Basic uncertainty-guided fusion using inverse uncertainty weighting.
    
    Higher uncertainty → Lower weight
    Lower uncertainty → Higher weight
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        
        if config is None:
            config = FusionConfig()
        
        self.config = config
        
        if config.learnable_epsilon:
            self.log_epsilon = nn.Parameter(torch.tensor(-6.0))  # ~1e-6
        else:
            self.register_buffer('epsilon', torch.tensor(config.epsilon))
        
        # Gaussian smoothing kernel
        if config.smooth_weights:
            self._create_smoothing_kernel(config.smooth_sigma)
    
    def _create_smoothing_kernel(self, sigma: float):
        """Create Gaussian smoothing kernel."""
        kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
        
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        self.register_buffer('smooth_kernel', kernel_2d)
        self.kernel_size = kernel_size
    
    def _get_epsilon(self) -> torch.Tensor:
        """Get epsilon value."""
        if self.config.learnable_epsilon:
            return torch.exp(self.log_epsilon)
        return self.epsilon
    
    def _smooth_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to weight maps."""
        if not self.config.smooth_weights:
            return weights
        
        B, C, H, W = weights.shape
        padding = self.kernel_size // 2
        
        # Apply smoothing per channel
        smoothed = F.conv2d(
            weights.view(B * C, 1, H, W),
            self.smooth_kernel,
            padding=padding
        ).view(B, C, H, W)
        
        return smoothed
    
    def compute_fusion_weights(
        self,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fusion weights from uncertainty maps.
        
        Args:
            U_diff: Diffusion uncertainty (B, 1, H, W)
            U_gan: GAN uncertainty (B, 1, H, W)
            
        Returns:
            Tuple of (alpha, beta) weights for diffusion and GAN
        """
        eps = self._get_epsilon()
        
        # Inverse uncertainty weighting
        W_diff = 1.0 / (U_diff + eps)
        W_gan = 1.0 / (U_gan + eps)
        
        # Apply temperature for sharpening/softening
        if self.config.temperature != 1.0:
            W_diff = W_diff ** (1.0 / self.config.temperature)
            W_gan = W_gan ** (1.0 / self.config.temperature)
        
        # Normalize to sum to 1
        W_total = W_diff + W_gan
        alpha = W_diff / W_total
        beta = W_gan / W_total
        
        # Smooth weights for spatial consistency
        alpha = self._smooth_weights(alpha)
        beta = self._smooth_weights(beta)
        
        # Re-normalize after smoothing
        W_total = alpha + beta
        alpha = alpha / W_total
        beta = beta / W_total
        
        return alpha, beta
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform uncertainty-guided fusion.
        
        Args:
            I_diff: Diffusion generated image (B, C, H, W)
            I_gan: GAN generated image (B, C, H, W)
            U_diff: Diffusion uncertainty (B, 1, H, W)
            U_gan: GAN uncertainty (B, 1, H, W)
            
        Returns:
            Dictionary containing:
                - 'fused': Fused image
                - 'alpha': Diffusion weight map
                - 'beta': GAN weight map
        """
        # Compute fusion weights
        alpha, beta = self.compute_fusion_weights(U_diff, U_gan)
        
        # Expand weights to match image channels
        if alpha.shape[1] == 1 and I_diff.shape[1] > 1:
            alpha = alpha.expand(-1, I_diff.shape[1], -1, -1)
            beta = beta.expand(-1, I_gan.shape[1], -1, -1)
        
        # Fuse images
        I_fused = alpha * I_diff + beta * I_gan
        
        return {
            'fused': I_fused,
            'alpha': alpha[:, 0:1],  # Return single-channel weights
            'beta': beta[:, 0:1]
        }


class AverageFusion(nn.Module):
    """Simple average fusion (baseline)."""
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor = None,
        U_gan: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Average the two branches."""
        I_fused = 0.5 * I_diff + 0.5 * I_gan
        
        B, C, H, W = I_diff.shape
        alpha = torch.full((B, 1, H, W), 0.5, device=I_diff.device)
        beta = torch.full((B, 1, H, W), 0.5, device=I_diff.device)
        
        return {
            'fused': I_fused,
            'alpha': alpha,
            'beta': beta
        }


class SoftmaxFusion(nn.Module):
    """
    Softmax-based fusion with temperature scaling.
    
    Converts uncertainties to weights using softmax over negative uncertainty.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Softmax fusion."""
        # Stack negative uncertainties (lower uncertainty = higher weight)
        neg_uncertainties = torch.stack([-U_diff, -U_gan], dim=-1)
        
        # Softmax with temperature
        weights = F.softmax(neg_uncertainties / self.temperature, dim=-1)
        
        alpha = weights[..., 0]
        beta = weights[..., 1]
        
        # Expand for multi-channel images
        if alpha.shape[1] == 1 and I_diff.shape[1] > 1:
            alpha_exp = alpha.expand(-1, I_diff.shape[1], -1, -1)
            beta_exp = beta.expand(-1, I_gan.shape[1], -1, -1)
        else:
            alpha_exp = alpha
            beta_exp = beta
        
        I_fused = alpha_exp * I_diff + beta_exp * I_gan
        
        return {
            'fused': I_fused,
            'alpha': alpha,
            'beta': beta
        }


class ConfidenceGatedFusion(nn.Module):
    """
    Confidence-gated fusion using hard selection based on uncertainty.
    
    Selects the branch with lower uncertainty at each pixel location.
    """
    
    def __init__(self, soft_gate: bool = True, temperature: float = 0.1):
        super().__init__()
        self.soft_gate = soft_gate
        self.temperature = temperature
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Gated fusion."""
        if self.soft_gate:
            # Soft gate using sigmoid
            gate = torch.sigmoid((U_gan - U_diff) / self.temperature)
        else:
            # Hard gate
            gate = (U_diff < U_gan).float()
        
        # Expand gate for multi-channel
        if gate.shape[1] == 1 and I_diff.shape[1] > 1:
            gate_exp = gate.expand(-1, I_diff.shape[1], -1, -1)
        else:
            gate_exp = gate
        
        I_fused = gate_exp * I_diff + (1 - gate_exp) * I_gan
        
        return {
            'fused': I_fused,
            'alpha': gate,
            'beta': 1 - gate
        }


class RegionAdaptiveFusion(nn.Module):
    """
    Region-adaptive fusion that applies different strategies
    to different image regions based on uncertainty patterns.
    """
    
    def __init__(
        self,
        high_uncertainty_threshold: float = 0.7,
        low_uncertainty_threshold: float = 0.3
    ):
        super().__init__()
        self.high_threshold = high_uncertainty_threshold
        self.low_threshold = low_uncertainty_threshold
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Region-adaptive fusion:
        - High confidence regions: Use lower uncertainty branch
        - Low confidence regions: Average both branches
        - Medium confidence: Weighted average
        """
        # Identify regions
        both_confident = (U_diff < self.low_threshold) & (U_gan < self.low_threshold)
        both_uncertain = (U_diff > self.high_threshold) & (U_gan > self.high_threshold)
        
        # Compute base weights using inverse uncertainty
        eps = 1e-6
        W_diff = 1.0 / (U_diff + eps)
        W_gan = 1.0 / (U_gan + eps)
        W_total = W_diff + W_gan
        alpha = W_diff / W_total
        beta = W_gan / W_total
        
        # In regions where both are uncertain, use equal weights
        alpha = torch.where(both_uncertain, torch.full_like(alpha, 0.5), alpha)
        beta = torch.where(both_uncertain, torch.full_like(beta, 0.5), beta)
        
        # Expand for fusion
        if alpha.shape[1] == 1 and I_diff.shape[1] > 1:
            alpha_exp = alpha.expand(-1, I_diff.shape[1], -1, -1)
            beta_exp = beta.expand(-1, I_gan.shape[1], -1, -1)
        else:
            alpha_exp = alpha
            beta_exp = beta
        
        I_fused = alpha_exp * I_diff + beta_exp * I_gan
        
        return {
            'fused': I_fused,
            'alpha': alpha,
            'beta': beta,
            'both_confident': both_confident.float(),
            'both_uncertain': both_uncertain.float()
        }


class UQFusionModule(nn.Module):
    """
    Complete UQ-Fusion module that wraps all fusion strategies.
    
    This is the main interface for the fusion component of the UQ-Fusion framework.
    """
    
    def __init__(
        self,
        fusion_method: str = 'uncertainty',
        config: Optional[FusionConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = FusionConfig(fusion_method=fusion_method)
        
        self.config = config
        self.fusion_method = fusion_method
        
        # Initialize appropriate fusion module
        if fusion_method == 'uncertainty':
            self.fusion = UncertaintyGuidedFusion(config)
        elif fusion_method == 'average':
            self.fusion = AverageFusion()
        elif fusion_method == 'softmax':
            self.fusion = SoftmaxFusion(temperature=config.temperature)
        elif fusion_method == 'gated':
            self.fusion = ConfidenceGatedFusion(soft_gate=True, temperature=config.temperature)
        elif fusion_method == 'region_adaptive':
            self.fusion = RegionAdaptiveFusion()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Perform fusion.
        
        Args:
            I_diff: Diffusion generated image (B, C, H, W)
            I_gan: GAN generated image (B, C, H, W)
            U_diff: Diffusion uncertainty (B, 1, H, W)
            U_gan: GAN uncertainty (B, 1, H, W)
            
        Returns:
            Dictionary with fused image and weight maps
        """
        return self.fusion(I_diff, I_gan, U_diff, U_gan)
    
    def fuse_with_models(
        self,
        dual_branch_model,
        source_images: torch.Tensor,
        diffusion_steps: int = 50,
        diffusion_strength: float = 0.8
    ) -> Dict[str, torch.Tensor]:
        """
        Generate and fuse using uncertainty-aware dual branch model.
        
        Args:
            dual_branch_model: UncertaintyAwareDualBranch model
            source_images: Source images
            diffusion_steps: Diffusion inference steps
            diffusion_strength: Diffusion strength
            
        Returns:
            Dictionary with fused result and all intermediate outputs
        """
        # Get inputs from dual branch model
        inputs = dual_branch_model.get_fusion_inputs(
            source_images,
            diffusion_steps=diffusion_steps,
            diffusion_strength=diffusion_strength
        )
        
        # Perform fusion
        fusion_result = self.forward(
            inputs['I_diff'],
            inputs['I_gan'],
            inputs['U_diff'],
            inputs['U_gan']
        )
        
        # Combine all outputs
        result = {
            'fused': fusion_result['fused'],
            'alpha': fusion_result['alpha'],
            'beta': fusion_result['beta'],
            'I_diff': inputs['I_diff'],
            'I_gan': inputs['I_gan'],
            'U_diff': inputs['U_diff'],
            'U_gan': inputs['U_gan']
        }
        
        return result


def create_fusion_module(
    method: str = 'uncertainty',
    **kwargs
) -> UQFusionModule:
    """
    Factory function to create fusion modules.
    
    Args:
        method: Fusion method ('uncertainty', 'average', 'softmax', 'gated', 'region_adaptive')
        **kwargs: Additional config arguments
        
    Returns:
        UQFusionModule instance
    """
    config = FusionConfig(fusion_method=method, **kwargs)
    return UQFusionModule(fusion_method=method, config=config)
