#!/usr/bin/env python
# scripts/evaluate_uncertainty.py

"""
Evaluate hybrid uncertainty estimation for diffusion and GAN branches.
Uses the v3 hybrid uncertainty: λ × MC_variance + (1-λ) × reconstruction_error.

Usage:
    python scripts/evaluate_uncertainty.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data --device mps
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import BraTSSliceDataset
from models.uncertainty import (
    UncertaintyAwareDualBranch,
    UncertaintyWrapperConfig,
    load_uncertainty_aware_models
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Uncertainty')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_mc_samples', type=int, default=10)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--lambda_mc', type=float, default=0.0,
                        help='Hybrid balance: 0=pure recon error, 1=pure MC variance')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/uncertainty')
    parser.add_argument('--num_samples', type=int, default=8)

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)
    return parser.parse_args()


def compute_psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 50.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


@torch.no_grad()
def evaluate_uncertainty(model, dataloader, device, args):
    """Evaluate hybrid uncertainty on test data."""
    all_results = {
        'diff_psnr': [], 'gan_psnr': [],
        'U_diff_mean': [], 'U_gan_mean': [],
        'recon_diff_mean': [], 'recon_gan_mean': [],
        'mc_diff_mean': [], 'mc_gan_mean': [],
    }
    collected = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        if collected >= args.num_samples:
            break

        images = batch['image'].to(device)
        inputs = model.get_fusion_inputs(
            images, diffusion_steps=args.diffusion_steps, diffusion_strength=0.8)

        for i in range(images.shape[0]):
            if collected >= args.num_samples:
                break

            all_results['diff_psnr'].append(compute_psnr(inputs['I_diff'][i], images[i]))
            all_results['gan_psnr'].append(compute_psnr(inputs['I_gan'][i], images[i]))
            all_results['U_diff_mean'].append(inputs['U_diff'][i].mean().item())
            all_results['U_gan_mean'].append(inputs['U_gan'][i].mean().item())
            all_results['recon_diff_mean'].append(inputs['U_diff_recon'][i].mean().item())
            all_results['recon_gan_mean'].append(inputs['U_gan_recon'][i].mean().item())
            all_results['mc_diff_mean'].append(inputs['U_diff_mc_raw'][i].mean().item())
            all_results['mc_gan_mean'].append(inputs['U_gan_mc_raw'][i].mean().item())
            collected += 1

        # Free memory
        del images, inputs
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

    return all_results


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Hybrid Uncertainty Evaluation")
    print("=" * 60)
    print(f"Diffusion: {args.diffusion_checkpoint}")
    print(f"GAN: {args.gan_checkpoint}")
    print(f"VAE: {args.vae_checkpoint}")
    print(f"MC samples: {args.num_mc_samples} | λ_mc: {args.lambda_mc}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load models
    print("\n[1/4] Loading models...")
    config = UncertaintyWrapperConfig(
        num_mc_samples=args.num_mc_samples,
        lambda_mc_variance=args.lambda_mc)
    model = load_uncertainty_aware_models(
        args.diffusion_checkpoint, args.gan_checkpoint,
        args.vae_checkpoint, str(device), config)

    # Load data
    print("\n[2/4] Loading data...")
    data_dir = Path(args.data_dir)
    test_ds = BraTSSliceDataset(
        data_dir / "slices", data_dir / "splits" / "test_metadata.json",
        augmentor=None, return_segmentation=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # Run verification on first batch
    print("\n[3/4] Running verification...")
    first_batch = next(iter(test_loader))
    first_images = first_batch['image'][:4].to(device)
    stats = model.verify_uncertainty(first_images, diffusion_steps=args.diffusion_steps)

    # Full evaluation
    print("\n[4/4] Evaluating on all samples...")
    results = evaluate_uncertainty(model, test_loader, device, args)

    # Summary
    print("\n" + "=" * 60)
    print("UNCERTAINTY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {len(results['diff_psnr'])}")
    print(f"\nGeneration Quality:")
    print(f"  Diffusion PSNR: {np.mean(results['diff_psnr']):.2f} ± {np.std(results['diff_psnr']):.2f} dB")
    print(f"  GAN PSNR:       {np.mean(results['gan_psnr']):.2f} ± {np.std(results['gan_psnr']):.2f} dB")
    print(f"\nHybrid Uncertainty (λ_mc={args.lambda_mc}):")
    print(f"  U_diff mean: {np.mean(results['U_diff_mean']):.4f}")
    print(f"  U_gan mean:  {np.mean(results['U_gan_mean']):.4f}")
    print(f"\nComponents:")
    print(f"  Recon error (diff): {np.mean(results['recon_diff_mean']):.4f}")
    print(f"  Recon error (GAN):  {np.mean(results['recon_gan_mean']):.4f}")
    print(f"  MC variance (diff): {np.mean(results['mc_diff_mean']):.6f}")
    print(f"  MC variance (GAN):  {np.mean(results['mc_gan_mean']):.6f}")

    # Expected fusion weights
    U_d = np.mean(results['U_diff_mean'])
    U_g = np.mean(results['U_gan_mean'])
    eps = 1e-6
    alpha = (1 / (U_d + eps)) / (1 / (U_d + eps) + 1 / (U_g + eps))
    print(f"\nExpected fusion weights: α(diff)={alpha:.4f} β(GAN)={1 - alpha:.4f}")
    print("=" * 60)

    # Save
    metrics = {
        'diff_psnr_mean': float(np.mean(results['diff_psnr'])),
        'diff_psnr_std': float(np.std(results['diff_psnr'])),
        'gan_psnr_mean': float(np.mean(results['gan_psnr'])),
        'gan_psnr_std': float(np.std(results['gan_psnr'])),
        'U_diff_mean': float(np.mean(results['U_diff_mean'])),
        'U_gan_mean': float(np.mean(results['U_gan_mean'])),
        'recon_diff_mean': float(np.mean(results['recon_diff_mean'])),
        'recon_gan_mean': float(np.mean(results['recon_gan_mean'])),
        'mc_diff_mean': float(np.mean(results['mc_diff_mean'])),
        'mc_gan_mean': float(np.mean(results['mc_gan_mean'])),
        'expected_alpha': float(alpha),
        'expected_beta': float(1 - alpha),
        'lambda_mc': args.lambda_mc,
        'num_mc_samples': args.num_mc_samples,
        'num_samples': len(results['diff_psnr']),
        'verification': stats,
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()