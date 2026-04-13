#!/usr/bin/env python
# scripts/evaluate_gan.py

"""
Evaluate trained STABLE-GAN.

Usage:
    python scripts/evaluate_gan.py --checkpoint ./outputs/checkpoints/gan/best.pth --data_dir ./data --device mps
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
from models.gan import STABLEGeneratorSmall, STABLEGenerator, GeneratorConfig


def compute_psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 50.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def load_generator(checkpoint_path, device):
    """Load trained generator."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt.get('config', {})
    model_type = cfg.get('model_type', 'small')

    if model_type == 'small':
        generator = STABLEGeneratorSmall(
            in_channels=cfg.get('in_channels', 4),
            out_channels=cfg.get('out_channels', 4),
            base_channels=cfg.get('base_channels_g', 32),
            num_residual_blocks=cfg.get('num_residual_blocks', 6))
    else:
        gen_config = GeneratorConfig(
            in_channels=cfg.get('in_channels', 4),
            out_channels=cfg.get('out_channels', 4),
            base_channels=cfg.get('base_channels_g', 64),
            num_residual_blocks=cfg.get('num_residual_blocks', 9))
        generator = STABLEGenerator(gen_config)

    generator.load_state_dict(ckpt['generator_state_dict'])
    generator.to(device).eval()
    return generator, ckpt


@torch.no_grad()
def generate_and_evaluate(model, dataloader, device, num_samples):
    """Generate samples and compute metrics."""
    model.eval()
    psnrs = []
    collected = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        if collected >= num_samples:
            break
        images = batch['image'].to(device)
        generated = model(images).clamp(0, 1)

        for i in range(images.shape[0]):
            if collected >= num_samples:
                break
            psnrs.append(compute_psnr(generated[i], images[i]))
            collected += 1

        del images, generated

    return {'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
            'num_samples': len(psnrs)}


def main():
    parser = argparse.ArgumentParser(description='Evaluate STABLE-GAN')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/gan')
    parser.add_argument('--num_samples', type=int, default=8)

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
    print("STABLE-GAN Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load
    print("\n[1/3] Loading model...")
    generator, ckpt = load_generator(args.checkpoint, device)
    print(f"Loaded from epoch {ckpt.get('epoch', '?') + 1}")

    # Data
    print("\n[2/3] Loading data...")
    data_dir = Path(args.data_dir)
    test_ds = BraTSSliceDataset(
        data_dir / "slices", data_dir / "splits" / "test_metadata.json",
        augmentor=None, return_segmentation=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # Evaluate
    print(f"\n[3/3] Evaluating {args.num_samples} samples...")
    metrics = generate_and_evaluate(generator, test_loader, device, args.num_samples)

    print(f"\nGAN Generation Quality:")
    print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()