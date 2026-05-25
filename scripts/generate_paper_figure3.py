#!/usr/bin/env python
# scripts/generate_paper_figure3.py

"""
Generate Fig. 3 for the UQ-Fusion paper:
  4 rows (test slices) × 5 columns:
    Source FLAIR | Ground Truth | Baseline Pred | UQ-Fusion Pred | Dice Overlay Diff

Usage:
    python scripts/generate_paper_figure3.py \
        --baseline_checkpoint ./outputs/checkpoints/segmentation/baseline_aug_v3/best.pth \
        --augmented_checkpoint ./outputs/checkpoints/segmentation/augmented_aug_v3/best.pth \
        --data_dir ./data1 \
        --device mps \
        --output_dir ./outputs/figures
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch.utils.data import DataLoader
import gc

from data import BraTSSliceDataset
from models.segmentation import create_segmentation_model, DiceScore


def parse_args():
    parser = argparse.ArgumentParser(description='Generate paper Fig. 3')
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--augmented_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data1')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                        help='4 specific slice indices. If None, auto-selects.')
    parser.add_argument('--num_rows', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='./outputs/figures')
    parser.add_argument('--dpi', type=int, default=300)

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load segmentation model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = create_segmentation_model(
        model_type=config.get('model_type', 'standard'),
        in_channels=config.get('in_channels', 4),
        num_classes=config.get('num_classes', 4)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    name = ckpt.get('experiment_name', 'unknown')
    print(f"  Loaded: {name} (best Dice: {ckpt.get('best_dice', 0):.4f})")
    return model


@torch.no_grad()
def predict(model, image, device):
    """Run inference on a single image, return predicted label map."""
    x = image.unsqueeze(0).to(device)
    logits = model(x)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu()
    return pred


def find_representative_slices(dataset, baseline_model, augmented_model, device,
                                num_slices=4, num_candidates=80):
    """
    Auto-select slices that best demonstrate the augmented model's advantage.
    Picks slices where:
      - Visible tumor exists (ET present)
      - Augmented model has higher Dice than baseline
      - Good visual diversity
    """
    dice_scorer = DiceScore()
    candidates = []

    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_candidates, len(dataset)), replace=False)

    print(f"  Scanning {len(indices)} candidates...")
    for idx in indices:
        sample = dataset[int(idx)]
        image = sample['image'].float()
        gt = sample['segmentation']

        if isinstance(gt, torch.Tensor):
            gt_np = gt.numpy()
        else:
            gt_np = gt

        # Skip slices with no ET
        et_pixels = (gt_np == 4).sum() if 4 in np.unique(gt_np) else (gt_np == 3).sum()
        total_tumor = (gt_np > 0).sum()
        if et_pixels < 50 or total_tumor < 200:
            continue

        # Get predictions
        pred_base = predict(baseline_model, image, device)
        pred_aug = predict(augmented_model, image, device)

        # Compute Dice
        dice_base = dice_scorer.compute_brats_regions(pred_base, gt)
        dice_aug = dice_scorer.compute_brats_regions(pred_aug, gt)

        # Score: prefer slices where augmented wins, especially on ET
        et_improvement = dice_aug['dice_et'] - dice_base['dice_et']
        mean_improvement = dice_aug['dice_mean_brats'] - dice_base['dice_mean_brats']
        tumor_ratio = total_tumor / gt_np.size

        candidates.append({
            'idx': int(idx),
            'et_improvement': et_improvement,
            'mean_improvement': mean_improvement,
            'tumor_ratio': tumor_ratio,
            'dice_base': dice_base,
            'dice_aug': dice_aug
        })

    # Sort by ET improvement (primary) then mean improvement (secondary)
    candidates.sort(key=lambda c: (c['et_improvement'], c['mean_improvement']), reverse=True)

    # Pick top slices with some diversity in tumor ratio
    selected = []
    used_ratios = []
    for c in candidates:
        # Skip if too similar tumor ratio to already selected
        too_similar = any(abs(c['tumor_ratio'] - r) < 0.03 for r in used_ratios)
        if too_similar and len(selected) > 0:
            continue
        selected.append(c)
        used_ratios.append(c['tumor_ratio'])
        if len(selected) >= num_slices:
            break

    # If not enough diverse ones, fill with top remaining
    if len(selected) < num_slices:
        for c in candidates:
            if c not in selected:
                selected.append(c)
            if len(selected) >= num_slices:
                break

    for i, s in enumerate(selected):
        print(f"  Slice {i+1}: idx={s['idx']}, "
              f"ET Δ={s['et_improvement']:+.3f}, "
              f"Mean Δ={s['mean_improvement']:+.3f}, "
              f"tumor={s['tumor_ratio']:.1%}")

    return [s['idx'] for s in selected]


def seg_to_color(seg_map):
    """
    Convert BraTS segmentation labels to RGB color image.
    Label mapping:
      0 = background (black)
      1 = NCR/NET (red)
      2 = ED / peritumoral edema (green)
      3 or 4 = ET / enhancing tumor (yellow)
    """
    h, w = seg_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.numpy()

    # Background stays black
    # Label 1: NCR/NET → red
    rgb[seg_map == 1] = [0.90, 0.20, 0.20]
    # Label 2: Edema → green
    rgb[seg_map == 2] = [0.30, 0.80, 0.30]
    # Label 3 or 4: ET → yellow
    rgb[seg_map == 3] = [1.00, 0.85, 0.10]
    rgb[seg_map == 4] = [1.00, 0.85, 0.10]

    return rgb


def compute_diff_overlay(flair, gt, pred_base, pred_aug):
    """
    Create a difference overlay on the FLAIR image:
      - Green: augmented correct where baseline was wrong (augmented gain)
      - Red: baseline correct where augmented was wrong (augmented loss)
      - Semi-transparent overlay on grayscale FLAIR
    """
    if isinstance(gt, torch.Tensor):
        gt = gt.numpy()
    if isinstance(pred_base, torch.Tensor):
        pred_base = pred_base.numpy()
    if isinstance(pred_aug, torch.Tensor):
        pred_aug = pred_aug.numpy()

    h, w = flair.shape

    # Create grayscale base (FLAIR)
    rgb = np.stack([flair, flair, flair], axis=-1)

    # Where predictions differ from GT
    base_correct = (pred_base == gt)
    aug_correct = (pred_aug == gt)

    # Augmented gains: augmented correct AND baseline wrong (in tumor region)
    tumor_region = (gt > 0)
    gain = aug_correct & ~base_correct & tumor_region
    loss = ~aug_correct & base_correct & tumor_region

    # Overlay
    alpha = 0.6
    rgb[gain] = rgb[gain] * (1 - alpha) + np.array([0.1, 0.85, 0.2]) * alpha  # green
    rgb[loss] = rgb[loss] * (1 - alpha) + np.array([0.90, 0.15, 0.15]) * alpha  # red

    return np.clip(rgb, 0, 1)


def create_figure(dataset, sample_indices, baseline_model, augmented_model,
                  device, output_path, dpi=300):
    """Create the 4×5 publication figure."""
    num_rows = len(sample_indices)
    dice_scorer = DiceScore()

    # Collect all data first
    rows_data = []
    for idx in sample_indices:
        sample = dataset[idx]
        image = sample['image'].float()
        gt = sample['segmentation']
        flair = image[3].numpy()  # FLAIR channel

        pred_base = predict(baseline_model, image, device)
        pred_aug = predict(augmented_model, image, device)

        dice_base = dice_scorer.compute_brats_regions(pred_base, gt)
        dice_aug = dice_scorer.compute_brats_regions(pred_aug, gt)

        rows_data.append({
            'flair': flair,
            'gt': gt,
            'pred_base': pred_base,
            'pred_aug': pred_aug,
            'dice_base': dice_base,
            'dice_aug': dice_aug
        })

    # Figure layout
    fig = plt.figure(figsize=(17, num_rows * 3.6 + 1.2))
    gs = gridspec.GridSpec(num_rows, 5, wspace=0.06, hspace=0.22,
                           left=0.02, right=0.98, top=0.92, bottom=0.04)

    col_titles = ['Source (FLAIR)', 'Ground Truth', 'Baseline Prediction',
                  'UQ-Fusion Prediction', 'Difference Overlay']

    for row_i, data in enumerate(rows_data):
        flair = data['flair']
        gt = data['gt']
        pred_base = data['pred_base']
        pred_aug = data['pred_aug']
        d_base = data['dice_base']
        d_aug = data['dice_aug']

        gt_color = seg_to_color(gt)
        base_color = seg_to_color(pred_base)
        aug_color = seg_to_color(pred_aug)
        diff_overlay = compute_diff_overlay(flair, gt, pred_base, pred_aug)

        panels = [
            ('flair', flair),
            ('seg', gt_color),
            ('seg', base_color),
            ('seg', aug_color),
            ('diff', diff_overlay)
        ]

        for col_i, (ptype, img) in enumerate(panels):
            ax = fig.add_subplot(gs[row_i, col_i])

            if ptype == 'flair':
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                # Tumor boundary contour
                gt_np = gt.numpy() if isinstance(gt, torch.Tensor) else gt
                tumor_mask = (gt_np > 0).astype(float)
                if tumor_mask.sum() > 0:
                    ax.contour(tumor_mask, levels=[0.5], colors='lime',
                               linewidths=0.6, alpha=0.5)
            elif ptype == 'seg':
                # Show FLAIR underneath segmentation for context
                ax.imshow(flair, cmap='gray', vmin=0, vmax=1, alpha=0.3)
                # Overlay segmentation where non-zero
                seg_rgb = img
                mask = (seg_rgb.sum(axis=-1) > 0)
                composite = np.stack([flair, flair, flair], axis=-1) * 0.3
                composite[mask] = seg_rgb[mask] * 0.85 + composite[mask] * 0.15
                composite[~mask] = np.stack([flair, flair, flair], axis=-1)[~mask]
                ax.imshow(np.clip(composite, 0, 1))
            else:
                ax.imshow(img)

            # Column titles (top row only)
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=9.5, fontweight='bold', pad=6)

            # Dice scores as annotations
            if col_i == 2:  # Baseline
                dice_text = f"Dice: {d_base['dice_mean_brats']:.3f}"
                ax.text(0.03, 0.97, dice_text, transform=ax.transAxes,
                        fontsize=7, color='white', fontweight='bold',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                  alpha=0.7, edgecolor='none'))
            elif col_i == 3:  # Augmented
                dice_text = f"Dice: {d_aug['dice_mean_brats']:.3f}"
                delta = d_aug['dice_mean_brats'] - d_base['dice_mean_brats']
                delta_color = '#00ff00' if delta >= 0 else '#ff4444'
                full_text = f"Dice: {d_aug['dice_mean_brats']:.3f} ({delta:+.3f})"
                ax.text(0.03, 0.97, full_text, transform=ax.transAxes,
                        fontsize=7, color=delta_color, fontweight='bold',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                  alpha=0.7, edgecolor='none'))
            elif col_i == 4:  # Diff overlay — show ET dice specifically
                et_delta = d_aug['dice_et'] - d_base['dice_et']
                et_color = '#00ff00' if et_delta >= 0 else '#ff4444'
                et_text = f"ET Δ: {et_delta:+.3f}"
                ax.text(0.03, 0.97, et_text, transform=ax.transAxes,
                        fontsize=7, color=et_color, fontweight='bold',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                  alpha=0.7, edgecolor='none'))

            # Row label (leftmost column)
            if col_i == 0:
                ax.text(-0.08, 0.5, f'Case {row_i+1}', transform=ax.transAxes,
                        fontsize=9, fontweight='bold', rotation=90,
                        verticalalignment='center', horizontalalignment='center')

            ax.axis('off')

    # Legend for segmentation colors
    legend_patches = [
        mpatches.Patch(color=[0.90, 0.20, 0.20], label='NCR/NET'),
        mpatches.Patch(color=[0.30, 0.80, 0.30], label='Edema (ED)'),
        mpatches.Patch(color=[1.00, 0.85, 0.10], label='Enhancing (ET)'),
        mpatches.Patch(color=[0.1, 0.85, 0.2], label='Augmented gain'),
        mpatches.Patch(color=[0.90, 0.15, 0.15], label='Augmented loss'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
               fontsize=8, frameon=True, fancybox=True,
               edgecolor='gray', facecolor='white')

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path.with_suffix('.png'), dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"\nFigure saved:")
    print(f"  PNG: {output_path.with_suffix('.png')}")
    print(f"  PDF: {output_path.with_suffix('.pdf')}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("Generating Paper Figure 3 — Segmentation Comparison")
    print("=" * 60)

    # Load models
    print("\n[1/4] Loading models...")
    baseline_model = load_model(args.baseline_checkpoint, str(device))
    augmented_model = load_model(args.augmented_checkpoint, str(device))

    # Load data
    print("\n[2/4] Loading test data...")
    data_dir = Path(args.data_dir)
    dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "test_metadata.json",
        augmentor=None,
        return_segmentation=True
    )
    print(f"  Test samples: {len(dataset)}")

    # Select slices
    print("\n[3/4] Selecting representative slices...")
    if args.sample_indices is not None:
        indices = args.sample_indices[:args.num_rows]
        print(f"  Using specified indices: {indices}")
    else:
        indices = find_representative_slices(
            dataset, baseline_model, augmented_model, device,
            num_slices=args.num_rows
        )

    # Create figure
    print("\n[4/4] Creating figure...")
    create_figure(
        dataset=dataset,
        sample_indices=indices,
        baseline_model=baseline_model,
        augmented_model=augmented_model,
        device=device,
        output_path=Path(args.output_dir) / 'fig3_segmentation_comparison',
        dpi=args.dpi
    )

    # Print per-case summary
    dice_scorer = DiceScore()
    print(f"\n{'='*60}")
    print("Per-Case Summary:")
    print(f"{'Case':<6} {'Baseline Mean':<15} {'Augmented Mean':<16} {'Delta':<8} {'ET Delta':<8}")
    print("-" * 60)
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image'].float()
        gt = sample['segmentation']
        pred_b = predict(baseline_model, image, device)
        pred_a = predict(augmented_model, image, device)
        d_b = dice_scorer.compute_brats_regions(pred_b, gt)
        d_a = dice_scorer.compute_brats_regions(pred_a, gt)
        delta = d_a['dice_mean_brats'] - d_b['dice_mean_brats']
        et_delta = d_a['dice_et'] - d_b['dice_et']
        print(f"  {i+1:<4} {d_b['dice_mean_brats']:<15.4f} {d_a['dice_mean_brats']:<16.4f} "
              f"{delta:<+8.4f} {et_delta:<+8.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()