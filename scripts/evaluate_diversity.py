#!/usr/bin/env python
# scripts/evaluate_diversity.py

"""
Compute diversity and quality metrics between synthetic and real images:
  1. LPIPS (synthetic vs source): Per-pair perceptual distance — higher = more diverse
  2. FID (synthetic set vs real set): Distribution-level quality — lower = better quality
  3. Intra-set LPIPS: Compare diversity within synthetic vs within real sets

Usage:
    python scripts/evaluate_diversity.py \
        --synthetic_dir ./outputs/expanded_dataset/accepted \
        --data_dir ./data1 \
        --device mps

Requires: pip install lpips --break-system-packages
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import gc
from scipy import linalg


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate synthetic image diversity')
    parser.add_argument('--synthetic_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data1')
    parser.add_argument('--max_samples', type=int, default=500,
                        help='Max samples for FID computation (memory-limited)')
    parser.add_argument('--num_lpips_pairs', type=int, default=200,
                        help='Number of pairs for LPIPS evaluation')
    parser.add_argument('--num_intra_pairs', type=int, default=200,
                        help='Number of random pairs for intra-set diversity')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/diversity')

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)
    return parser.parse_args()


def to_3channel(img_4ch):
    """
    Convert 4-channel BraTS image to 3-channel pseudo-RGB for perceptual metrics.
    Uses T1ce (ch1), T2 (ch2), FLAIR (ch3) — the 3 most diagnostically informative.
    Input: (4, H, W) tensor in [0, 1]
    Output: (3, H, W) tensor in [-1, 1] (LPIPS/Inception expected range)
    """
    img_3ch = img_4ch[1:4]  # channels 1,2,3 = T1ce, T2, FLAIR
    return img_3ch * 2.0 - 1.0  # [0,1] → [-1,1]


def to_3channel_inception(img_4ch):
    """
    Convert 4-channel to 3-channel for Inception (expects [0,1] range, 299×299).
    """
    img_3ch = img_4ch[1:4]  # T1ce, T2, FLAIR
    # Resize to 299×299 for InceptionV3
    img_3ch = torch.nn.functional.interpolate(
        img_3ch.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False
    ).squeeze(0)
    return img_3ch


# ─────────────────────────────────────────────────────────
# LPIPS COMPUTATION
# ─────────────────────────────────────────────────────────

def compute_lpips_scores(synthetic_files, real_dataset, device, num_pairs=200):
    """
    Compute LPIPS between each synthetic image and its source.
    Higher LPIPS = more perceptual difference = more diversity.
    """
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    num_pairs = min(num_pairs, len(synthetic_files), len(real_dataset))
    scores = []

    print(f"  Computing LPIPS for {num_pairs} synthetic-source pairs...")
    for i in tqdm(range(num_pairs), desc="  LPIPS", leave=False):
        # Load synthetic
        syn_data = np.load(synthetic_files[i], allow_pickle=True)
        syn_img = syn_data['image']
        if syn_img.ndim == 4:
            syn_img = syn_img.squeeze(0)
        syn_img = torch.from_numpy(syn_img).float()

        # Load corresponding source (synthetic_i was generated from source_i)
        source_idx = i % len(real_dataset)
        source_sample = real_dataset[source_idx]
        source_img = source_sample['image'].float()

        # Convert to 3-channel [-1, 1]
        syn_3ch = to_3channel(syn_img).unsqueeze(0).to(device)
        src_3ch = to_3channel(source_img).unsqueeze(0).to(device)

        with torch.no_grad():
            score = lpips_fn(syn_3ch, src_3ch).item()
        scores.append(score)

        # Memory cleanup
        del syn_3ch, src_3ch
        if i % 50 == 0 and device.type == 'mps':
            torch.mps.empty_cache()

    del lpips_fn
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    return np.array(scores)


def compute_intra_lpips(images_list, device, num_pairs=200, label="set"):
    """
    Compute LPIPS between random pairs WITHIN a set.
    Measures internal diversity of the set.
    """
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    lpips_fn.eval()

    n = len(images_list)
    num_pairs = min(num_pairs, n * (n - 1) // 2)
    scores = []

    np.random.seed(123)
    pair_indices = set()
    while len(pair_indices) < num_pairs:
        i, j = np.random.randint(0, n, 2)
        if i != j and (min(i, j), max(i, j)) not in pair_indices:
            pair_indices.add((min(i, j), max(i, j)))

    print(f"  Computing intra-{label} LPIPS for {num_pairs} pairs...")
    for i, j in tqdm(pair_indices, desc=f"  Intra-{label}", leave=False):
        img_i = to_3channel(images_list[i]).unsqueeze(0).to(device)
        img_j = to_3channel(images_list[j]).unsqueeze(0).to(device)

        with torch.no_grad():
            score = lpips_fn(img_i, img_j).item()
        scores.append(score)

        del img_i, img_j

    del lpips_fn
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    return np.array(scores)


# ─────────────────────────────────────────────────────────
# FID COMPUTATION
# ─────────────────────────────────────────────────────────

class InceptionFeatureExtractor(nn.Module):
    """Extract pool3 features from InceptionV3 for FID computation."""

    def __init__(self, device):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # We want features before the final FC layer
        # pool3 output = 2048-dimensional
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        # x: (B, 3, 299, 299) in [0, 1]
        # Inception expects ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        features = self.blocks(x)
        return features.squeeze(-1).squeeze(-1)  # (B, 2048)


def extract_features(images_list, extractor, device, batch_size=8):
    """Extract Inception features from a list of 4-channel images."""
    all_features = []

    for i in tqdm(range(0, len(images_list), batch_size), desc="  Features", leave=False):
        batch = images_list[i:i + batch_size]
        # Convert to 3ch 299×299
        batch_3ch = torch.stack([to_3channel_inception(img) for img in batch]).to(device)

        with torch.no_grad():
            feats = extractor(batch_3ch).cpu().numpy()
        all_features.append(feats)

        del batch_3ch
        if device.type == 'mps' and i % 40 == 0:
            torch.mps.empty_cache()

    return np.concatenate(all_features, axis=0)


def compute_fid(mu1, sigma1, mu2, sigma2):
    """Compute Frechet Inception Distance between two sets of statistics."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical stability
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print("  Warning: imaginary component in sqrtm, taking real part")
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_fid_score(synthetic_images, real_images, device, batch_size=8):
    """Full FID computation pipeline."""
    print("  Loading InceptionV3...")
    extractor = InceptionFeatureExtractor(device)

    print(f"  Extracting features from {len(real_images)} real images...")
    real_feats = extract_features(real_images, extractor, device, batch_size)

    print(f"  Extracting features from {len(synthetic_images)} synthetic images...")
    syn_feats = extract_features(synthetic_images, extractor, device, batch_size)

    # Compute statistics
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    mu_syn = np.mean(syn_feats, axis=0)
    sigma_syn = np.cov(syn_feats, rowvar=False)

    print("  Computing FID...")
    fid = compute_fid(mu_real, sigma_real, mu_syn, sigma_syn)

    del extractor
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    return fid


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Synthetic Image Diversity Evaluation")
    print("=" * 60)
    print(f"Synthetic: {args.synthetic_dir}")
    print(f"Real data: {args.data_dir}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load synthetic files
    syn_dir = Path(args.synthetic_dir)
    syn_files = sorted(syn_dir.glob('synthetic_*.npz'))
    print(f"\nSynthetic files found: {len(syn_files)}")

    # Load real dataset
    from data import BraTSSliceDataset
    data_dir = Path(args.data_dir)
    real_dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "train_metadata.json",
        augmentor=None,
        return_segmentation=False
    )
    print(f"Real training samples: {len(real_dataset)}")

    # Limit samples for memory
    num_samples = min(args.max_samples, len(syn_files), len(real_dataset))
    print(f"Using {num_samples} samples for evaluation")

    # Pre-load images into lists
    print("\n[1/4] Loading images...")
    real_images = []
    for i in tqdm(range(num_samples), desc="  Real", leave=False):
        sample = real_dataset[i]
        real_images.append(sample['image'].float())

    syn_images = []
    for i in tqdm(range(num_samples), desc="  Synthetic", leave=False):
        data = np.load(syn_files[i], allow_pickle=True)
        img = data['image']
        if img.ndim == 4:
            img = img.squeeze(0)
        syn_images.append(torch.from_numpy(img).float())

    results = {}

    # ── LPIPS: Synthetic vs Source ──
    print("\n[2/4] Computing LPIPS (synthetic vs source)...")
    lpips_scores = compute_lpips_scores(
        syn_files, real_dataset, device,
        num_pairs=min(args.num_lpips_pairs, num_samples)
    )
    results['lpips_syn_vs_source'] = {
        'mean': float(np.mean(lpips_scores)),
        'std': float(np.std(lpips_scores)),
        'median': float(np.median(lpips_scores)),
        'min': float(np.min(lpips_scores)),
        'max': float(np.max(lpips_scores)),
        'num_pairs': len(lpips_scores)
    }
    print(f"\n  LPIPS (synthetic vs source): {np.mean(lpips_scores):.4f} ± {np.std(lpips_scores):.4f}")
    print(f"  Range: [{np.min(lpips_scores):.4f}, {np.max(lpips_scores):.4f}]")

    # ── Intra-set LPIPS ──
    print("\n[3/4] Computing intra-set LPIPS (diversity comparison)...")

    intra_real = compute_intra_lpips(
        real_images, device, num_pairs=args.num_intra_pairs, label="real"
    )
    results['lpips_intra_real'] = {
        'mean': float(np.mean(intra_real)),
        'std': float(np.std(intra_real)),
        'num_pairs': len(intra_real)
    }

    intra_syn = compute_intra_lpips(
        syn_images, device, num_pairs=args.num_intra_pairs, label="synthetic"
    )
    results['lpips_intra_synthetic'] = {
        'mean': float(np.mean(intra_syn)),
        'std': float(np.std(intra_syn)),
        'num_pairs': len(intra_syn)
    }

    print(f"\n  Intra-real LPIPS:      {np.mean(intra_real):.4f} ± {np.std(intra_real):.4f}")
    print(f"  Intra-synthetic LPIPS: {np.mean(intra_syn):.4f} ± {np.std(intra_syn):.4f}")

    diversity_ratio = np.mean(intra_syn) / np.mean(intra_real)
    results['diversity_ratio'] = float(diversity_ratio)
    print(f"  Diversity ratio (syn/real): {diversity_ratio:.3f}")
    if diversity_ratio > 0.8:
        print(f"  ✓ Synthetic set has comparable diversity to real set")
    else:
        print(f"  ✗ Synthetic set is less diverse than real set")

    # ── FID ──
    print("\n[4/4] Computing FID (synthetic vs real)...")
    fid_score = compute_fid_score(syn_images, real_images, device, args.batch_size)
    results['fid'] = float(fid_score)
    print(f"\n  FID: {fid_score:.2f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DIVERSITY EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n1. LPIPS (synthetic vs source):  {results['lpips_syn_vs_source']['mean']:.4f} ± "
          f"{results['lpips_syn_vs_source']['std']:.4f}")
    print(f"   → Higher = more perceptual difference from source (not memorizing)")
    print(f"   → Range: [{results['lpips_syn_vs_source']['min']:.4f}, "
          f"{results['lpips_syn_vs_source']['max']:.4f}]")

    print(f"\n2. Intra-set LPIPS (internal diversity):")
    print(f"   Real set:      {results['lpips_intra_real']['mean']:.4f} ± "
          f"{results['lpips_intra_real']['std']:.4f}")
    print(f"   Synthetic set: {results['lpips_intra_synthetic']['mean']:.4f} ± "
          f"{results['lpips_intra_synthetic']['std']:.4f}")
    print(f"   Ratio (syn/real): {results['diversity_ratio']:.3f}")

    print(f"\n3. FID (synthetic vs real): {results['fid']:.2f}")
    print(f"   → Lower = synthetic distribution closer to real (better quality)")

    print("\n" + "=" * 60)

    # Interpretation for paper
    print("\nPAPER INTERPRETATION:")
    lpips_mean = results['lpips_syn_vs_source']['mean']
    if lpips_mean > 0.05:
        print(f"  LPIPS = {lpips_mean:.4f} indicates synthetic images are perceptually")
        print(f"  distinct from their sources — the model is generating novel content,")
        print(f"  not memorizing/reconstructing the training set.")
    else:
        print(f"  LPIPS = {lpips_mean:.4f} is relatively low, indicating synthetic images")
        print(f"  are close to sources. This is expected at strength 0.95 with a")
        print(f"  VAE-bottlenecked diffusion model.")

    if diversity_ratio > 0.8:
        print(f"  Diversity ratio = {diversity_ratio:.3f} confirms the synthetic set")
        print(f"  maintains comparable internal diversity to the real training set.")

    # Save results
    with open(output_dir / 'diversity_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_dir / 'diversity_metrics.json'}")


if __name__ == "__main__":
    main()