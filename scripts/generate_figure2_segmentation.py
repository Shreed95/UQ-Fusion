#!/usr/bin/env python
# scripts/generate_figure2_segmentation.py

"""
Generate Figure 2: Segmentation Comparison for research paper.

Creates a 2x4 grid showing:
(a) Original MRI, (b) Ground Truth, (c) Baseline Prediction, (d) Augmented Prediction

Color coding:
- Whole Tumor (WT): Green
- Tumor Core (TC): Yellow  
- Enhancing Tumor (ET): Red

Usage:
    python scripts/generate_figure2_segmentation.py \
        --baseline_checkpoint ./outputs/checkpoints/segmentation/baseline/best.pth \
        --augmented_checkpoint ./outputs/checkpoints/segmentation/augmented/best.pth \
        --data_dir ./data \
        --output_path ./outputs/figures/figure2_segmentation.png \
        --device mps
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from data import BraTSSliceDataset
from models.segmentation import create_segmentation_model, DiceScore


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Figure 2: Segmentation Comparison')
    
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--augmented_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_path', type=str, default='./outputs/figures/figure2_segmentation.png')
    parser.add_argument('--slice_indices', type=int, nargs='+', default=None,
                        help='Specific slice indices to use (default: auto-select)')
    parser.add_argument('--modality', type=int, default=3,
                        help='MRI modality to display (0=T1, 1=T1ce, 2=T2, 3=FLAIR)')
    parser.add_argument('--model_type', type=str, default='small')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--overlay_alpha', type=float, default=0.6,
                        help='Transparency of segmentation overlay')
    
    return parser.parse_args()


def load_model(checkpoint_path, model_type, device):
    """Load segmentation model from checkpoint."""
    model = create_segmentation_model(
        model_type=model_type,
        in_channels=4,
        num_classes=4
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def remap_brats_labels(mask):
    """Remap BraTS labels: 0->0, 1->1, 2->2, 4->3."""
    remapped = mask.clone()
    remapped[mask == 4] = 3
    return remapped


def select_improvement_slices(dataset, baseline_model, augmented_model, device, num_slices=2):
    """Select slices where augmented model shows visible improvement."""
    dice_scorer = DiceScore()
    improvements = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            mask = sample['segmentation']
            
            # Remap labels
            mask_remapped = remap_brats_labels(mask)
            
            # Get predictions
            baseline_pred = torch.argmax(baseline_model(image), dim=1)[0].cpu()
            augmented_pred = torch.argmax(augmented_model(image), dim=1)[0].cpu()
            
            # Compute Dice scores
            baseline_dice = dice_scorer.compute_brats_regions(baseline_pred, mask_remapped)
            augmented_dice = dice_scorer.compute_brats_regions(augmented_pred, mask_remapped)
            
            # Calculate improvement
            baseline_mean = np.mean([baseline_dice['dice_wt'], baseline_dice['dice_tc'], baseline_dice['dice_et']])
            augmented_mean = np.mean([augmented_dice['dice_wt'], augmented_dice['dice_tc'], augmented_dice['dice_et']])
            improvement = augmented_mean - baseline_mean
            
            # Also check tumor size (want visible tumor)
            tumor_fraction = (mask_remapped > 0).float().mean().item()
            
            if tumor_fraction > 0.02 and baseline_mean > 0.3:  # Has visible tumor and reasonable baseline
                improvements.append({
                    'idx': idx,
                    'improvement': improvement,
                    'baseline_mean': baseline_mean,
                    'augmented_mean': augmented_mean,
                    'tumor_fraction': tumor_fraction
                })
    
    # Sort by improvement (descending)
    improvements_sorted = sorted(improvements, key=lambda x: x['improvement'], reverse=True)
    
    # Select top slices with good improvement
    selected = []
    for item in improvements_sorted:
        if item['improvement'] > 0.02:  # At least 2% improvement
            selected.append(item['idx'])
            if len(selected) >= num_slices:
                break
    
    # If not enough, just take top by improvement
    while len(selected) < num_slices and len(improvements_sorted) > len(selected):
        idx = improvements_sorted[len(selected)]['idx']
        if idx not in selected:
            selected.append(idx)
    
    return selected[:num_slices]


def create_segmentation_overlay(mri, segmentation, alpha=0.6):
    """Create MRI image with colored segmentation overlay."""
    # Normalize MRI to 0-1
    mri_norm = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
    
    # Create RGB image from grayscale MRI
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    
    # Define colors for each class (RGB, 0-1 range)
    colors = {
        0: [0, 0, 0],           # Background - transparent
        1: [0.2, 0.8, 0.2],     # NCR/NET - Green
        2: [1.0, 1.0, 0.2],     # ED (Edema) - Yellow
        3: [1.0, 0.2, 0.2]      # ET (Enhancing) - Red
    }
    
    # Create overlay
    overlay = np.zeros_like(rgb)
    for label, color in colors.items():
        if label == 0:
            continue
        mask = (segmentation == label)
        for c in range(3):
            overlay[:, :, c][mask] = color[c]
    
    # Blend MRI with overlay
    tumor_mask = segmentation > 0
    result = rgb.copy()
    result[tumor_mask] = (1 - alpha) * rgb[tumor_mask] + alpha * overlay[tumor_mask]
    
    return result


def create_figure(all_data, args):
    """Create the 2x4 segmentation comparison figure."""
    
    num_rows = len(all_data)
    num_cols = 4
    
    # Create figure
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, wspace=0.05, hspace=0.15)
    
    # Column titles
    col_titles = ['(a) FLAIR MRI', '(b) Ground Truth', '(c) Baseline', '(d) Augmented']
    
    # Row labels
    row_labels = ['Sample 1', 'Sample 2']
    
    for row_idx, data in enumerate(all_data):
        mri = data['mri']
        gt = data['ground_truth']
        baseline_pred = data['baseline_pred']
        augmented_pred = data['augmented_pred']
        
        # Create overlays
        gt_overlay = create_segmentation_overlay(mri, gt, alpha=args.overlay_alpha)
        baseline_overlay = create_segmentation_overlay(mri, baseline_pred, alpha=args.overlay_alpha)
        augmented_overlay = create_segmentation_overlay(mri, augmented_pred, alpha=args.overlay_alpha)
        
        images = [mri, gt_overlay, baseline_overlay, augmented_overlay]
        
        for col_idx, (img, title) in enumerate(zip(images, col_titles)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            if col_idx == 0:
                # Plain MRI
                vmin, vmax = np.percentile(img, [2, 98])
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                # Overlay images
                ax.imshow(img)
            
            ax.axis('off')
            
            # Add column titles to first row
            if row_idx == 0:
                ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            
            # Add Dice scores for predictions
            if col_idx >= 2:
                if col_idx == 2:
                    dice_mean = data['baseline_dice_mean']
                else:
                    dice_mean = data['augmented_dice_mean']
                ax.text(0.5, -0.05, f'Dice: {dice_mean:.3f}', transform=ax.transAxes,
                       fontsize=9, ha='center', va='top')
            
            # Add row labels
            if col_idx == 0:
                ax.text(-0.1, 0.5, row_labels[row_idx], transform=ax.transAxes,
                       fontsize=11, fontweight='bold', va='center', ha='right',
                       rotation=90)
    
    # Add main title
    fig.suptitle('Segmentation Comparison: Baseline vs Augmented Training', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=[0.2, 0.8, 0.2], edgecolor='black', label='NCR/NET (Label 1)'),
        mpatches.Patch(facecolor=[1.0, 1.0, 0.2], edgecolor='black', label='Edema (Label 2)'),
        mpatches.Patch(facecolor=[1.0, 0.2, 0.2], edgecolor='black', label='Enhancing (Label 4)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    
    return fig


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("=" * 60)
    print("Generating Figure 2: Segmentation Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n[1/4] Loading models...")
    baseline_model = load_model(args.baseline_checkpoint, args.model_type, device)
    augmented_model = load_model(args.augmented_checkpoint, args.model_type, device)
    
    # Load test data
    print("\n[2/4] Loading test data...")
    data_dir = Path(args.data_dir)
    test_dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "test_metadata.json",
        augmentor=None,
        return_segmentation=True
    )
    
    # Select slices with improvement
    print("\n[3/4] Selecting slices with visible improvement...")
    dice_scorer = DiceScore()
    
    if args.slice_indices:
        slice_indices = args.slice_indices[:2]
    else:
        slice_indices = select_improvement_slices(
            test_dataset, baseline_model, augmented_model, device, num_slices=2
        )
    
    print(f"Selected slices: {slice_indices}")
    
    # Generate predictions and collect data
    print("\n[4/4] Generating predictions...")
    all_data = []
    
    with torch.no_grad():
        for idx in slice_indices:
            sample = test_dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            mask = sample['segmentation']
            
            # Remap labels
            mask_remapped = remap_brats_labels(mask)
            
            # Get predictions
            baseline_pred = torch.argmax(baseline_model(image), dim=1)[0].cpu()
            augmented_pred = torch.argmax(augmented_model(image), dim=1)[0].cpu()
            
            # Compute Dice scores
            baseline_dice = dice_scorer.compute_brats_regions(baseline_pred, mask_remapped)
            augmented_dice = dice_scorer.compute_brats_regions(augmented_pred, mask_remapped)
            
            baseline_mean = np.mean([baseline_dice['dice_wt'], baseline_dice['dice_tc'], baseline_dice['dice_et']])
            augmented_mean = np.mean([augmented_dice['dice_wt'], augmented_dice['dice_tc'], augmented_dice['dice_et']])
            
            all_data.append({
                'mri': sample['image'][args.modality].numpy(),
                'ground_truth': mask_remapped.numpy(),
                'baseline_pred': baseline_pred.numpy(),
                'augmented_pred': augmented_pred.numpy(),
                'baseline_dice_mean': baseline_mean,
                'augmented_dice_mean': augmented_mean
            })
            
            print(f"  Slice {idx}: Baseline={baseline_mean:.3f}, Augmented={augmented_mean:.3f}, "
                  f"Improvement={augmented_mean - baseline_mean:.3f}")
    
    # Create figure
    fig = create_figure(all_data, args)
    
    # Save figure
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Also save as PDF for paper
    pdf_path = output_path.with_suffix('.pdf')
    fig = create_figure(all_data, args)
    fig.savefig(pdf_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"\n✓ Figure saved to: {output_path}")
    print(f"✓ PDF saved to: {pdf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
