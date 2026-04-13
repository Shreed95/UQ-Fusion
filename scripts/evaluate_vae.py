#!/usr/bin/env python
# scripts/evaluate_vae.py

"""
Evaluate VAE reconstruction quality.

Usage:
    python scripts/evaluate_vae.py --checkpoint ./outputs/checkpoints/vae/best.pth --data_dir ./data --device mps
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


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 50.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim_simple(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM without large conv kernels (memory-friendly)."""
    import torch.nn.functional as F
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    # Use average pooling instead of gaussian conv for speed
    k = 7
    pad = k // 2
    mu1 = F.avg_pool2d(pred, k, stride=1, padding=pad)
    mu2 = F.avg_pool2d(target, k, stride=1, padding=pad)
    s1 = F.avg_pool2d(pred * pred, k, stride=1, padding=pad) - mu1 * mu1
    s2 = F.avg_pool2d(target * target, k, stride=1, padding=pad) - mu2 * mu2
    s12 = F.avg_pool2d(pred * target, k, stride=1, padding=pad) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return ssim_map.mean().item()


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit evaluation to N samples (None=all)')
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != 'mps' or torch.backends.mps.is_available()) else 'cpu')
    data_dir = Path(args.data_dir)

    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg = ckpt.get('config', {})

    model = VAE(VAEConfig(
        base_channels=cfg.get('base_channels', 64),
        latent_channels=cfg.get('latent_channels', 4),
    ))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Trained for {ckpt.get('epoch', '?')+1} epochs")

    # Dataset
    test_ds = BraTSSliceDataset(
        data_dir / "slices", data_dir / "splits" / "test_metadata.json",
        augmentor=None, return_segmentation=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Evaluate
    print(f"\nEvaluating on {len(test_ds)} test slices...")
    psnrs, ssims = [], []
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch['image'].to(device)
            recon = model.reconstruct(images).clamp(0, 1)

            for i in range(images.shape[0]):
                psnrs.append(compute_psnr(recon[i], images[i]))
                ssims.append(compute_ssim_simple(recon[i:i+1], images[i:i+1]))
                total += 1

            del recon, images
            if args.num_samples and total >= args.num_samples:
                break

    # Results
    print("\n" + "=" * 60)
    print("VAE Reconstruction Quality")
    print("=" * 60)
    print(f"Samples evaluated: {len(psnrs)}")
    print(f"PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    print(f"SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"Target: PSNR > 30 dB, SSIM > 0.95")
    print("=" * 60)

    if np.mean(psnrs) >= 30:
        print("✓ PSNR target met!")
    else:
        print(f"✗ PSNR below target (need {30 - np.mean(psnrs):.1f} dB more)")

    if np.mean(ssims) >= 0.95:
        print("✓ SSIM target met!")
    else:
        print(f"✗ SSIM below target")

    # Save results
    out_dir = Path("./outputs/evaluation/vae")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)), 'ssim_std': float(np.std(ssims)),
        'num_samples': len(psnrs),
        'checkpoint': args.checkpoint,
    }
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()