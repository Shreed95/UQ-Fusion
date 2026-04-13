# models/uncertainty/wrappers.py

"""
Uncertainty-Aware Model Wrappers (v3 — Hybrid Uncertainty).

Hybrid Uncertainty = λ × MC_variance_norm + (1-λ) × reconstruction_error_norm

This correctly identifies:
  - Healthy tissue: GAN has low error → low U_gan → high GAN weight
  - Tumor regions: GAN has higher error → higher U_gan → more diffusion
  - Blurry regions: Diffusion has high error → high U_diff → more GAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass

from .epistemic import compute_epistemic_uncertainty


@dataclass
class UncertaintyWrapperConfig:
    """Configuration for uncertainty wrappers."""
    num_mc_samples: int = 10
    gan_perturbation_std: float = 0.02
    lambda_mc_variance: float = 0.0  # 0 = pure quality, 1 = pure MC variance
    normalize_uncertainty: bool = False  # Joint normalization in get_fusion_inputs()
    # Backward compatibility (ignored)
    estimate_aleatoric: bool = False
    estimate_epistemic: bool = True
    dropout_rate: float = 0.1
    inject_dropout: bool = False
    inject_dropout_rate: float = 0.1
    min_log_var: float = -10.0
    max_log_var: float = 10.0


class UncertaintyAwareDiffusion(nn.Module):
    """Uncertainty-aware wrapper for Latent Diffusion Model."""

    def __init__(self, diffusion_model, config=None):
        super().__init__()
        if config is None:
            config = UncertaintyWrapperConfig()
        self.config = config
        self.diffusion_model = diffusion_model

    @torch.no_grad()
    def generate_with_uncertainty(self, source_images, num_inference_steps=50, strength=0.8):
        all_generated = []
        for _ in range(self.config.num_mc_samples):
            generated = self.diffusion_model.generate(
                source_images, num_inference_steps=num_inference_steps,
                strength=strength, show_progress=False)
            all_generated.append(generated)

        all_generated = torch.stack(all_generated, dim=0)
        mean_generated = all_generated.mean(dim=0)
        mc_variance = compute_epistemic_uncertainty(all_generated, normalize=False)

        return {'generated': mean_generated, 'mc_variance': mc_variance, 'all_samples': all_generated}

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class UncertaintyAwareGAN(nn.Module):
    """Uncertainty-aware wrapper for GAN. Uses input perturbation for MC variance."""

    def __init__(self, generator, config=None):
        super().__init__()
        if config is None:
            config = UncertaintyWrapperConfig()
        self.config = config
        self.generator = generator
        self.perturbation_std = config.gan_perturbation_std
        print(f"[UncertaintyAwareGAN] Input perturbation σ={self.perturbation_std}")

    @torch.no_grad()
    def generate_with_uncertainty(self, source_images):
        all_generated = []
        for _ in range(self.config.num_mc_samples):
            noise = torch.randn_like(source_images) * self.perturbation_std
            perturbed = (source_images + noise).clamp(0, 1)
            generated = self.generator(perturbed)
            all_generated.append(generated)

        all_generated = torch.stack(all_generated, dim=0)
        clean_output = self.generator(source_images)
        mc_variance = compute_epistemic_uncertainty(all_generated, normalize=False)

        return {'generated': clean_output, 'mc_variance': mc_variance, 'all_samples': all_generated}

    def forward(self, x):
        return self.generator(x)


class UncertaintyAwareDualBranch(nn.Module):
    """
    Dual branch wrapper with HYBRID uncertainty estimation.

    Hybrid Uncertainty = λ × MC_variance_norm + (1-λ) × recon_error_norm
    Joint normalization preserves cross-branch relative magnitudes.
    """

    def __init__(self, diffusion_model, gan_generator, config=None):
        super().__init__()
        if config is None:
            config = UncertaintyWrapperConfig()
        self.config = config

        branch_config = UncertaintyWrapperConfig(
            num_mc_samples=config.num_mc_samples,
            gan_perturbation_std=config.gan_perturbation_std,
            normalize_uncertainty=False, estimate_aleatoric=False)

        self.diffusion_branch = UncertaintyAwareDiffusion(diffusion_model, branch_config)
        self.gan_branch = UncertaintyAwareGAN(gan_generator, branch_config)
        self.lambda_mc = config.lambda_mc_variance

    @staticmethod
    def _compute_reconstruction_error(generated, source):
        """Per-pixel MSE as quality-based uncertainty. Returns (B, 1, H, W)."""
        return (generated - source).pow(2).mean(dim=1, keepdim=True)

    @staticmethod
    def _joint_normalize_pair(U_a, U_b):
        """Jointly normalize using percentile-based bounds (2nd-98th)."""
        B = U_a.shape[0]
        U_a_norm = torch.zeros_like(U_a)
        U_b_norm = torch.zeros_like(U_b)

        for i in range(B):
            joint = torch.cat([U_a[i].flatten(), U_b[i].flatten()])
            p2 = torch.quantile(joint, 0.02).item()
            p98 = torch.quantile(joint, 0.98).item()
            scale = p98 - p2
            if scale < 1e-10:
                U_a_norm[i] = 0.5
                U_b_norm[i] = 0.5
            else:
                U_a_norm[i] = ((U_a[i] - p2) / scale).clamp(0, 1)
                U_b_norm[i] = ((U_b[i] - p2) / scale).clamp(0, 1)

        return U_a_norm, U_b_norm

    @torch.no_grad()
    def generate_both_with_uncertainty(self, source_images, diffusion_steps=50, diffusion_strength=0.8):
        """Generate from both branches with MC variance."""
        diff_results = self.diffusion_branch.generate_with_uncertainty(
            source_images, num_inference_steps=diffusion_steps, strength=diffusion_strength)
        gan_results = self.gan_branch.generate_with_uncertainty(source_images)
        return {'diffusion': diff_results, 'gan': gan_results}

    def get_fusion_inputs(self, source_images, diffusion_steps=50, diffusion_strength=0.8):
        """
        Get fusion inputs with hybrid uncertainty.

        Returns dict with: I_diff, I_gan, U_diff, U_gan, and diagnostics.
        """
        results = self.generate_both_with_uncertainty(source_images, diffusion_steps, diffusion_strength)

        I_diff = results['diffusion']['generated']
        I_gan = results['gan']['generated']

        # Component 1: MC variance
        mc_diff = results['diffusion']['mc_variance']
        mc_gan = results['gan']['mc_variance']
        mc_diff_norm, mc_gan_norm = self._joint_normalize_pair(mc_diff, mc_gan)

        # Component 2: Reconstruction error
        recon_diff = self._compute_reconstruction_error(I_diff, source_images)
        recon_gan = self._compute_reconstruction_error(I_gan, source_images)
        recon_diff_norm, recon_gan_norm = self._joint_normalize_pair(recon_diff, recon_gan)

        # Combine
        lam = self.lambda_mc
        U_diff = lam * mc_diff_norm + (1 - lam) * recon_diff_norm
        U_gan = lam * mc_gan_norm + (1 - lam) * recon_gan_norm

        return {
            'I_diff': I_diff, 'I_gan': I_gan, 'U_diff': U_diff, 'U_gan': U_gan,
            'U_diff_mc_raw': mc_diff, 'U_gan_mc_raw': mc_gan,
            'U_diff_mc_norm': mc_diff_norm, 'U_gan_mc_norm': mc_gan_norm,
            'U_diff_recon': recon_diff_norm, 'U_gan_recon': recon_gan_norm,
        }

    def verify_uncertainty(self, source_images, diffusion_steps=50, diffusion_strength=0.8):
        """Diagnostic: verify hybrid uncertainties are meaningful."""
        inputs = self.get_fusion_inputs(source_images, diffusion_steps, diffusion_strength)

        stats = {}
        stats['mc_diff_raw_mean'] = inputs['U_diff_mc_raw'].mean().item()
        stats['mc_gan_raw_mean'] = inputs['U_gan_mc_raw'].mean().item()
        stats['mc_raw_ratio'] = stats['mc_diff_raw_mean'] / (stats['mc_gan_raw_mean'] + 1e-10)
        stats['mc_diff_norm_mean'] = inputs['U_diff_mc_norm'].mean().item()
        stats['mc_gan_norm_mean'] = inputs['U_gan_mc_norm'].mean().item()
        stats['recon_diff_norm_mean'] = inputs['U_diff_recon'].mean().item()
        stats['recon_gan_norm_mean'] = inputs['U_gan_recon'].mean().item()

        mse_diff = (inputs['I_diff'] - source_images).pow(2).mean().item()
        mse_gan = (inputs['I_gan'] - source_images).pow(2).mean().item()
        stats['psnr_diff'] = 10 * torch.log10(torch.tensor(1.0 / (mse_diff + 1e-10))).item()
        stats['psnr_gan'] = 10 * torch.log10(torch.tensor(1.0 / (mse_gan + 1e-10))).item()

        stats['hybrid_diff_mean'] = inputs['U_diff'].mean().item()
        stats['hybrid_gan_mean'] = inputs['U_gan'].mean().item()

        U_d, U_g, eps = stats['hybrid_diff_mean'], stats['hybrid_gan_mean'], 1e-6
        W_d, W_g = 1.0 / (U_d + eps), 1.0 / (U_g + eps)
        stats['expected_alpha'] = W_d / (W_d + W_g)

        print("\n" + "=" * 60)
        print("HYBRID UNCERTAINTY VERIFICATION")
        print("=" * 60)
        print(f"\n  MC Variance (λ={self.lambda_mc})")
        print(f"    Diff raw: {stats['mc_diff_raw_mean']:.8f} | GAN raw: {stats['mc_gan_raw_mean']:.8f}")
        print(f"  Reconstruction Error (weight={1 - self.lambda_mc})")
        print(f"    Diff PSNR: {stats['psnr_diff']:.2f} dB | GAN PSNR: {stats['psnr_gan']:.2f} dB")
        print(f"  Hybrid: U_diff={stats['hybrid_diff_mean']:.4f} | U_gan={stats['hybrid_gan_mean']:.4f}")
        print(f"  Expected α={stats['expected_alpha']:.4f} β={1 - stats['expected_alpha']:.4f}")

        checks = 0
        if stats['recon_gan_norm_mean'] < stats['recon_diff_norm_mean']:
            print("  ✓ GAN lower recon error"); checks += 1
        else:
            print("  ✗ Diffusion lower recon error")
        if stats['hybrid_gan_mean'] < stats['hybrid_diff_mean']:
            print("  ✓ GAN lower hybrid uncertainty"); checks += 1
        else:
            print("  ✗ GAN higher hybrid uncertainty")
        if stats['expected_alpha'] < 0.5:
            print("  ✓ Fusion favors GAN"); checks += 1
        else:
            print("  ✗ Fusion favors diffusion")
        print(f"  Result: {checks}/3 checks passed")
        print("=" * 60 + "\n")
        return stats


def load_uncertainty_aware_models(
    diffusion_checkpoint, gan_checkpoint, vae_checkpoint,
    device='mps', config=None
):
    """Load pre-trained models and wrap with hybrid uncertainty."""
    from models.vae import VAE, VAEConfig
    from models.diffusion import LatentDiffusionModelSmall
    from models.gan import STABLEGeneratorSmall

    # Load VAE (read config from checkpoint)
    vae_ckpt = torch.load(vae_checkpoint, map_location='cpu')
    vae_cfg = vae_ckpt.get('config', {})
    vae = VAE(VAEConfig(
        in_channels=4, out_channels=4,
        latent_channels=vae_cfg.get('latent_channels', 4),
        base_channels=vae_cfg.get('base_channels', 64)))
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()

    # Load Diffusion
    diff_ckpt = torch.load(diffusion_checkpoint, map_location='cpu')
    diff_cfg = diff_ckpt.get('config', {})
    diffusion = LatentDiffusionModelSmall(
        latent_channels=diff_cfg.get('latent_channels', 4),
        base_channels=diff_cfg.get('base_channels', 64),
        num_timesteps=diff_cfg.get('num_timesteps', 1000))
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.eval()

    # Load GAN (as-trained, no dropout modification)
    gan_ckpt = torch.load(gan_checkpoint, map_location='cpu')
    gan_cfg = gan_ckpt.get('config', {})
    generator = STABLEGeneratorSmall(
        in_channels=gan_cfg.get('in_channels', 4),
        out_channels=gan_cfg.get('out_channels', 4),
        base_channels=gan_cfg.get('base_channels_g', 32),
        num_residual_blocks=gan_cfg.get('num_residual_blocks', 6))
    generator.load_state_dict(gan_ckpt['generator_state_dict'])
    generator.eval()

    # Move to device
    diffusion = diffusion.to(device)
    generator = generator.to(device)

    if config is None:
        config = UncertaintyWrapperConfig()

    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, config)
    return dual_branch.to(device)