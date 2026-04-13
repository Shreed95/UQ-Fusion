#!/usr/bin/env python
# scripts/evaluate_diffusion.py

"""
Evaluate trained Latent Diffusion Model.

Usage:
    python scripts/evaluate_diffusion.py \
        --checkpoint ./outputs/checkpoints/diffusion/best.pth \
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import BraTSSliceDataset
from models.vae import VAE, VAEConfig
from models.diffusion import LatentDiffusionModelSmall, DDIMSampler


def compute_psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 50.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def load_models(diffusion_path, vae_path, device):
    """Load VAE and Diffusion models."""
    # Load VAE
    vae_ckpt = torch.load(vae_path, map_location='cpu')
    vae_cfg = vae_ckpt.get('config', {})
    vae = VAE(VAEConfig(
        in_channels=4, out_channels=4,
        latent_channels=vae_cfg.get('latent_channels', 4),
        base_channels=vae_cfg.get('base_channels', 64),
    ))
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.to(device).eval()

    # Load Diffusion
    diff_ckpt = torch.load(diffusion_path, map_location='cpu')
    diff_cfg = diff_ckpt.get('config', {})
    diffusion = LatentDiffusionModelSmall(
        latent_channels=diff_cfg.get('latent_channels', 4),
        base_channels=diff_cfg.get('base_channels', 64),
        num_timesteps=diff_cfg.get('num_timesteps', 1000))
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.to(device)
    diffusion.unet.eval()

    return vae, diffusion, diff_ckpt


@torch.no_grad()
def generate_samples(model, dataloader, device, num_samples, num_inference_steps):
    """Generate samples using image-to-image translation."""
    model.unet.eval()
    sources, generateds = [], []
    collected = 0

    for batch in dataloader:
        if collected >= num_samples:
            break
        images = batch['image'].to(device)

        source_latents = model.encode(images)
        sampler = DDIMSampler(model.scheduler, model.unet, num_inference_steps)
        gen_latents = sampler.sample(
            shape=source_latents.shape, condition=source_latents,
            device=device, show_progress=False)
        generated = model.decode(gen_latents)

        sources.append(images.cpu())
        generateds.append(generated.cpu())
        collected += images.shape[0]

        del images, source_latents, gen_latents, generated

    return torch.cat(sources)[:num_samples], torch.cat(generateds)[:num_samples]


def main():
    parser = argparse.ArgumentParser(description='Evaluate Diffusion Model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/diffusion')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--num_inference_steps', type=int, default=50)

    if torch.backends.mps.is_available():
        default_device = 'mps'
    elif torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    parser.add_argument('--device', type=str, default=default_device)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Diffusion Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"VAE: {args.vae_checkpoint}")
    print(f"Device: {device} | Steps: {args.num_inference_steps}")
    print("=" * 60)

    # Load models
    print("\n[1/4] Loading models...")
    vae, diffusion, ckpt = load_models(args.checkpoint, args.vae_checkpoint, device)
    print(f"Loaded from epoch {ckpt.get('epoch', '?') + 1}")

    # Load data
    print("\n[2/4] Loading data...")
    data_dir = Path(args.data_dir)
    test_ds = BraTSSliceDataset(
        data_dir / "slices", data_dir / "splits" / "test_metadata.json",
        augmentor=None, return_segmentation=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # Generate
    print(f"\n[3/4] Generating {args.num_samples} samples...")
    source, generated = generate_samples(
        diffusion, test_loader, device, args.num_samples, args.num_inference_steps)
    print(f"Generated {generated.shape[0]} samples")

    # Metrics
    print("\n[4/4] Computing metrics...")
    psnrs = [compute_psnr(generated[i], source[i]) for i in range(source.shape[0])]

    print(f"\nDiffusion Generation Quality:")
    print(f"  PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")

    metrics = {
        'psnr_mean': float(np.mean(psnrs)),
        'psnr_std': float(np.std(psnrs)),
        'num_samples': len(psnrs),
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()