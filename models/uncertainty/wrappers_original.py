# models/uncertainty/wrappers.py

"""
Uncertainty-Aware Model Wrappers.

Wraps diffusion and GAN models to provide uncertainty estimation
capabilities while maintaining the original model interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass

from .aleatoric import VarianceHead, compute_aleatoric_uncertainty
from .epistemic import compute_epistemic_uncertainty
from .combined import CombinedUncertaintyConfig


@dataclass
class UncertaintyWrapperConfig:
    """Configuration for uncertainty wrappers."""
    estimate_aleatoric: bool = True
    estimate_epistemic: bool = True
    num_mc_samples: int = 10
    dropout_rate: float = 0.1
    min_log_var: float = -10.0
    max_log_var: float = 10.0
    normalize_uncertainty: bool = True


class UncertaintyAwareDiffusion(nn.Module):
    """
    Uncertainty-aware wrapper for Latent Diffusion Model.
    
    Adds uncertainty estimation to the diffusion generation process.
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        config: Optional[UncertaintyWrapperConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = UncertaintyWrapperConfig()
        
        self.config = config
        self.diffusion_model = diffusion_model
        
        # Variance head for aleatoric uncertainty - operates on image space
        if config.estimate_aleatoric:
            output_channels = 4  # MRI modalities
            self.variance_head = VarianceHead(
                in_channels=output_channels,
                out_channels=output_channels,
                min_log_var=config.min_log_var,
                max_log_var=config.max_log_var
            )
        else:
            self.variance_head = None
    
    def _enable_dropout(self):
        """Enable dropout in U-Net for MC sampling."""
        for module in self.diffusion_model.unet.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
    
    @torch.no_grad()
    def generate_with_uncertainty(
        self,
        source_images: torch.Tensor,
        num_inference_steps: int = 50,
        strength: float = 0.8
    ) -> Dict[str, torch.Tensor]:
        """
        Generate images with uncertainty estimation.
        
        Args:
            source_images: Source images for image-to-image
            num_inference_steps: Number of denoising steps
            strength: Transformation strength
            
        Returns:
            Dictionary with generated images and uncertainty maps
        """
        self._enable_dropout()
        
        all_generated = []
        all_log_vars = []
        
        # Multiple MC forward passes
        for _ in range(self.config.num_mc_samples):
            # Generate using diffusion
            generated = self.diffusion_model.generate(
                source_images,
                num_inference_steps=num_inference_steps,
                strength=strength,
                show_progress=False
            )
            
            all_generated.append(generated)
            
            # Estimate aleatoric uncertainty from generated image (image space)
            if self.variance_head is not None:
                log_var = self.variance_head(generated)
                all_log_vars.append(log_var)
        
        # Stack predictions
        all_generated = torch.stack(all_generated, dim=0)
        
        # Mean prediction
        mean_generated = all_generated.mean(dim=0)
        
        # Epistemic uncertainty (from variance across MC samples)
        if self.config.estimate_epistemic:
            epistemic = compute_epistemic_uncertainty(
                all_generated, normalize=self.config.normalize_uncertainty
            )
        else:
            epistemic = torch.zeros(
                mean_generated.shape[0], 1,
                mean_generated.shape[2], mean_generated.shape[3],
                device=mean_generated.device
            )
        
        # Aleatoric uncertainty (from variance prediction)
        if self.variance_head is not None and len(all_log_vars) > 0:
            all_log_vars = torch.stack(all_log_vars, dim=0)
            mean_log_var = all_log_vars.mean(dim=0)
            aleatoric = compute_aleatoric_uncertainty(
                mean_log_var, normalize=self.config.normalize_uncertainty
            )
        else:
            aleatoric = torch.zeros_like(epistemic)
        
        # Combined uncertainty (both are now in image space 240x240)
        total_uncertainty = 0.5 * aleatoric + 0.5 * epistemic
        if self.config.normalize_uncertainty:
            total_uncertainty = self._normalize(total_uncertainty)
        
        return {
            'generated': mean_generated,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total_uncertainty': total_uncertainty,
            'all_samples': all_generated
        }
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.view(B, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        return (x - min_val) / (max_val - min_val + 1e-8)
    
    def forward(self, *args, **kwargs):
        """Standard forward pass (delegates to base model)."""
        return self.diffusion_model(*args, **kwargs)


class UncertaintyAwareGAN(nn.Module):
    """
    Uncertainty-aware wrapper for STABLE-GAN Generator.
    
    Adds uncertainty estimation to the GAN generation process.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        config: Optional[UncertaintyWrapperConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = UncertaintyWrapperConfig()
        
        self.config = config
        self.generator = generator
        
        # Get output channels from generator
        output_channels = 4  # Default for BraTS
        
        # Variance head for aleatoric uncertainty
        if config.estimate_aleatoric:
            self.variance_head = VarianceHead(
                in_channels=output_channels,
                out_channels=output_channels,
                min_log_var=config.min_log_var,
                max_log_var=config.max_log_var
            )
        else:
            self.variance_head = None
    
    def _enable_dropout(self):
        """Enable dropout in generator for MC sampling."""
        for module in self.generator.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
    
    @torch.no_grad()
    def generate_with_uncertainty(
        self,
        source_images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Generate images with uncertainty estimation.
        
        Args:
            source_images: Source images
            
        Returns:
            Dictionary with generated images and uncertainty maps
        """
        self._enable_dropout()
        
        all_generated = []
        all_log_vars = []
        
        # Multiple MC forward passes
        for _ in range(self.config.num_mc_samples):
            generated = self.generator(source_images)
            all_generated.append(generated)
            
            if self.variance_head is not None:
                log_var = self.variance_head(generated)
                all_log_vars.append(log_var)
        
        # Stack predictions
        all_generated = torch.stack(all_generated, dim=0)
        
        # Mean prediction
        mean_generated = all_generated.mean(dim=0)
        
        # Epistemic uncertainty
        if self.config.estimate_epistemic:
            epistemic = compute_epistemic_uncertainty(
                all_generated, normalize=self.config.normalize_uncertainty
            )
        else:
            epistemic = torch.zeros(
                mean_generated.shape[0], 1,
                mean_generated.shape[2], mean_generated.shape[3],
                device=mean_generated.device
            )
        
        # Aleatoric uncertainty
        if self.variance_head is not None and len(all_log_vars) > 0:
            all_log_vars = torch.stack(all_log_vars, dim=0)
            mean_log_var = all_log_vars.mean(dim=0)
            aleatoric = compute_aleatoric_uncertainty(
                mean_log_var, normalize=self.config.normalize_uncertainty
            )
        else:
            aleatoric = torch.zeros_like(epistemic)
        
        # Combined uncertainty
        total_uncertainty = 0.5 * aleatoric + 0.5 * epistemic
        if self.config.normalize_uncertainty:
            total_uncertainty = self._normalize(total_uncertainty)
        
        return {
            'generated': mean_generated,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'total_uncertainty': total_uncertainty,
            'all_samples': all_generated
        }
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.view(B, -1)
        min_val = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        max_val = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        return (x - min_val) / (max_val - min_val + 1e-8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (delegates to generator)."""
        return self.generator(x)


class UncertaintyAwareDualBranch(nn.Module):
    """
    Wrapper for both diffusion and GAN branches with uncertainty.
    
    Provides a unified interface for uncertainty-aware generation
    from both branches, which feeds into the fusion module.
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        gan_generator: nn.Module,
        config: Optional[UncertaintyWrapperConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = UncertaintyWrapperConfig()
        
        self.config = config
        
        self.diffusion_branch = UncertaintyAwareDiffusion(diffusion_model, config)
        self.gan_branch = UncertaintyAwareGAN(gan_generator, config)
    
    @torch.no_grad()
    def generate_both_with_uncertainty(
        self,
        source_images: torch.Tensor,
        diffusion_steps: int = 50,
        diffusion_strength: float = 0.8
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate from both branches with uncertainty.
        
        Args:
            source_images: Source images
            diffusion_steps: Number of diffusion steps
            diffusion_strength: Diffusion transformation strength
            
        Returns:
            Dictionary with results from both branches
        """
        # Diffusion branch
        diffusion_results = self.diffusion_branch.generate_with_uncertainty(
            source_images,
            num_inference_steps=diffusion_steps,
            strength=diffusion_strength
        )
        
        # GAN branch
        gan_results = self.gan_branch.generate_with_uncertainty(source_images)
        
        return {
            'diffusion': diffusion_results,
            'gan': gan_results
        }
    
    def get_fusion_inputs(
        self,
        source_images: torch.Tensor,
        diffusion_steps: int = 50,
        diffusion_strength: float = 0.8
    ) -> Dict[str, torch.Tensor]:
        """
        Get inputs ready for the fusion module.
        
        Args:
            source_images: Source images
            
        Returns:
            Dictionary with:
                - I_diff: Diffusion generated image
                - I_gan: GAN generated image
                - U_diff: Diffusion uncertainty
                - U_gan: GAN uncertainty
        """
        results = self.generate_both_with_uncertainty(
            source_images, diffusion_steps, diffusion_strength
        )
        
        return {
            'I_diff': results['diffusion']['generated'],
            'I_gan': results['gan']['generated'],
            'U_diff': results['diffusion']['total_uncertainty'],
            'U_gan': results['gan']['total_uncertainty'],
            'U_diff_aleatoric': results['diffusion']['aleatoric'],
            'U_diff_epistemic': results['diffusion']['epistemic'],
            'U_gan_aleatoric': results['gan']['aleatoric'],
            'U_gan_epistemic': results['gan']['epistemic']
        }


def load_uncertainty_aware_models(
    diffusion_checkpoint: str,
    gan_checkpoint: str,
    vae_checkpoint: str,
    device: str = 'cuda',
    config: Optional[UncertaintyWrapperConfig] = None
) -> UncertaintyAwareDualBranch:
    """
    Load pre-trained models and wrap with uncertainty estimation.
    
    Args:
        diffusion_checkpoint: Path to diffusion checkpoint
        gan_checkpoint: Path to GAN checkpoint
        vae_checkpoint: Path to VAE checkpoint
        device: Device to use
        config: Uncertainty wrapper config
        
    Returns:
        UncertaintyAwareDualBranch model
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from models.vae import VAESmall
    from models.diffusion import LatentDiffusionModelSmall
    from models.gan import STABLEGeneratorSmall
    
    # Load VAE
    vae = VAESmall(in_channels=4, out_channels=4, latent_channels=4)
    vae_ckpt = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    
    # Load Diffusion
    diffusion = LatentDiffusionModelSmall(latent_channels=4, base_channels=64, num_timesteps=1000)
    diff_ckpt = torch.load(diffusion_checkpoint, map_location=device)
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.eval()
    
    # Load GAN
    gan_ckpt = torch.load(gan_checkpoint, map_location=device)
    gan_config = gan_ckpt.get('config', {})
    generator = STABLEGeneratorSmall(
        in_channels=gan_config.get('in_channels', 4),
        out_channels=gan_config.get('out_channels', 4),
        base_channels=gan_config.get('base_channels_g', 32),
        num_residual_blocks=gan_config.get('num_residual_blocks', 6)
    )
    generator.load_state_dict(gan_ckpt['generator_state_dict'])
    generator.eval()
    
    # Move to device
    diffusion = diffusion.to(device)
    generator = generator.to(device)
    
    # Wrap with uncertainty
    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, config)
    dual_branch = dual_branch.to(device)
    
    return dual_branch