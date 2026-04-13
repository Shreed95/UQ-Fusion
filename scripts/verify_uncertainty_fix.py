#!/usr/bin/env python
# scripts/verify_uncertainty_fix.py

"""
Verify hybrid uncertainty fix works before full regeneration.

Expected output:
  - GAN PSNR ~33 dB (model loads correctly)
  - GAN reconstruction error < Diffusion reconstruction error
  - Hybrid U_gan < U_diff
  - α < 0.5 (GAN correctly favored)
  - Fused PSNR > Diffusion PSNR (and close to GAN PSNR)

Usage:
    python scripts/verify_uncertainty_fix.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data \
        --device mps
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from data import BraTSSliceDataset
from models.vae import VAE, VAEConfig
from models.diffusion import LatentDiffusionModelSmall
from models.gan import STABLEGeneratorSmall
from models.uncertainty import UncertaintyAwareDualBranch, UncertaintyWrapperConfig
from models.fusion import create_fusion_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_mc_samples', type=int, default=10)
    parser.add_argument('--lambda_mc', type=float, default=0.3,
                        help='MC variance weight in hybrid (0=pure quality, 1=pure MC)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else 'cpu')

    print("=" * 60)
    print("HYBRID UNCERTAINTY FIX VERIFICATION")
    print("=" * 60)

    # --- Load models ---
    print("\n[1/4] Loading models...")

    vae = VAE(VAEConfig())
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=str(device), weights_only=False)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.to(device).eval()

    diffusion = LatentDiffusionModelSmall(latent_channels=4, base_channels=64, num_timesteps=1000)
    diff_ckpt = torch.load(args.diffusion_checkpoint, map_location=str(device), weights_only=False)
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.to(device).eval()

    gan_ckpt = torch.load(args.gan_checkpoint, map_location=str(device), weights_only=False)
    gan_config = gan_ckpt.get('config', {})
    generator = STABLEGeneratorSmall(
        in_channels=gan_config.get('in_channels', 4),
        out_channels=gan_config.get('out_channels', 4),
        base_channels=gan_config.get('base_channels_g', 32),
        num_residual_blocks=gan_config.get('num_residual_blocks', 6),
        use_dropout=False  # As trained
    )
    generator.load_state_dict(gan_ckpt['generator_state_dict'])
    generator.to(device).eval()

    # --- Create hybrid uncertainty wrapper ---
    print("\n[2/4] Creating hybrid uncertainty wrapper...")
    config = UncertaintyWrapperConfig(
        num_mc_samples=args.num_mc_samples,
        gan_perturbation_std=0.02,
        lambda_mc_variance=args.lambda_mc,
        normalize_uncertainty=False,
        estimate_aleatoric=False,
    )
    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, config)
    dual_branch = dual_branch.to(device)

    # --- Load test batch ---
    print("\n[3/4] Loading test batch...")
    data_dir = Path(args.data_dir)
    test_dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "test_metadata.json",
        augmentor=None,
        return_segmentation=False
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(test_loader))
    images = batch['image'].to(device)

    # --- Run verification ---
    print(f"\n[4/4] Running hybrid verification on {images.shape[0]} samples "
          f"(λ_mc={args.lambda_mc})...")
    stats = dual_branch.verify_uncertainty(
        images, diffusion_steps=50, diffusion_strength=0.8
    )

    # --- Test actual fusion ---
    print("Testing fusion with hybrid uncertainties...")
    fusion_inputs = dual_branch.get_fusion_inputs(images, diffusion_steps=50, diffusion_strength=0.8)
    fusion_module = create_fusion_module(method='uncertainty').to(device)
    fusion_result = fusion_module(
        fusion_inputs['I_diff'],
        fusion_inputs['I_gan'],
        fusion_inputs['U_diff'],
        fusion_inputs['U_gan']
    )

    alpha = fusion_result['alpha']
    beta = fusion_result['beta']

    print(f"\nActual fusion weights (after smoothing):")
    print(f"  α (diff weight) mean: {alpha.mean().item():.4f}")
    print(f"  β (GAN weight) mean:  {beta.mean().item():.4f}")
    print(f"  α range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")

    # PSNR comparison
    source = images.cpu()
    I_diff = fusion_inputs['I_diff'].clamp(0, 1).cpu()
    I_gan = fusion_inputs['I_gan'].clamp(0, 1).cpu()
    I_fused = fusion_result['fused'].clamp(0, 1).cpu()

    psnr_diff = 10 * np.log10(1.0 / ((I_diff - source) ** 2).mean().item())
    psnr_gan = 10 * np.log10(1.0 / ((I_gan - source) ** 2).mean().item())
    psnr_fused = 10 * np.log10(1.0 / ((I_fused - source) ** 2).mean().item())

    print(f"\nPSNR comparison:")
    print(f"  Diffusion: {psnr_diff:.2f} dB")
    print(f"  GAN:       {psnr_gan:.2f} dB")
    print(f"  Fused:     {psnr_fused:.2f} dB")
    print(f"  Fused vs GAN: {psnr_fused - psnr_gan:+.2f} dB")

    # Final verdict
    print("\n" + "=" * 60)
    if beta.mean().item() > 0.5 and psnr_fused > psnr_diff:
        print("✓ FIX WORKING!")
        print("  GAN correctly favored in fusion.")
        if psnr_fused >= psnr_gan - 1.0:
            print("  Fused PSNR within 1 dB of GAN — good fusion quality.")
        print("  → Proceed with full dataset regeneration.")
    elif psnr_fused > psnr_diff:
        print("⚠ PARTIAL SUCCESS")
        print("  Fused is better than diffusion alone.")
        print(f"  But β={beta.mean().item():.3f} — consider tuning λ_mc")
        print(f"  Try: --lambda_mc 0.1 or --lambda_mc 0.2")
    else:
        print("✗ FIX NOT WORKING")
        print("  Fused PSNR <= Diffusion PSNR.")
    print("=" * 60)


if __name__ == "__main__":
    main()