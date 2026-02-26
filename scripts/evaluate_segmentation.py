#!/usr/bin/env python
# scripts/evaluate_segmentation.py

"""
Evaluate trained segmentation model.

Usage:
    python scripts/evaluate_segmentation.py \
        --checkpoint ./outputs/checkpoints/segmentation/baseline/best.pth \
        --data_dir ./data
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from data import BraTSSliceDataset
from models.segmentation import (
    create_segmentation_model,
    SegmentationMetrics,
    DiceScore,
    HausdorffDistance
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Segmentation Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/segmentation')
    parser.add_argument('--num_visualize', type=int, default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--compute_hd', action='store_true',
                        help='Compute Hausdorff distance (slow)')
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'small')
    in_channels = config.get('in_channels', 4)
    num_classes = config.get('num_classes', 4)
    
    model = create_segmentation_model(
        model_type=model_type,
        in_channels=in_channels,
        num_classes=num_classes
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    experiment_name = checkpoint.get('experiment_name', 'unknown')
    best_dice = checkpoint.get('best_dice', 0.0)
    
    print(f"Loaded model: {experiment_name}")
    print(f"Training best Dice: {best_dice:.4f}")
    
    return model, experiment_name


@torch.no_grad()
def evaluate(model, dataloader, device, compute_hd=False):
    """Evaluate model on dataset."""
    dice_scorer = DiceScore()
    hd_scorer = HausdorffDistance() if compute_hd else None
    
    all_metrics = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)
        
        logits = model(images)
        pred = torch.argmax(logits, dim=1)
        
        for i in range(pred.shape[0]):
            # Dice scores
            dice_scores = dice_scorer.compute_brats_regions(pred[i], masks[i])
            
            metrics = {
                'dice_wt': dice_scores['dice_wt'],
                'dice_tc': dice_scores['dice_tc'],
                'dice_et': dice_scores['dice_et'],
                'dice_mean': dice_scores['dice_mean_brats']
            }
            
            # Hausdorff distance
            if hd_scorer is not None:
                pred_np = pred[i].cpu().numpy()
                mask_np = masks[i].cpu().numpy()
                hd_scores = hd_scorer.compute_brats_regions(pred_np, mask_np)
                metrics.update(hd_scores)
            
            all_metrics.append(metrics)
    
    return all_metrics


def compute_statistics(metrics_list):
    """Compute aggregate statistics."""
    stats = {}
    
    keys = metrics_list[0].keys()
    for key in keys:
        values = [m[key] for m in metrics_list if not np.isinf(m[key])]
        if values:
            stats[f'{key}_mean'] = float(np.mean(values))
            stats[f'{key}_std'] = float(np.std(values))
            stats[f'{key}_median'] = float(np.median(values))
            stats[f'{key}_min'] = float(np.min(values))
            stats[f'{key}_max'] = float(np.max(values))
    
    return stats


def visualize_predictions(model, dataloader, device, output_dir, num_samples=8):
    """Visualize segmentation predictions."""
    model.eval()
    
    samples_shown = 0
    
    for batch in dataloader:
        if samples_shown >= num_samples:
            break
        
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)
        
        with torch.no_grad():
            logits = model(images)
            pred = torch.argmax(logits, dim=1)
        
        for i in range(min(images.shape[0], num_samples - samples_shown)):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Row 1: Input modalities
            modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
            for j in range(4):
                axes[0, j].imshow(images[i, j].cpu().numpy(), cmap='gray')
                axes[0, j].set_title(modality_names[j])
                axes[0, j].axis('off')
            
            # Row 2: Ground truth, prediction, overlay, difference
            # Ground truth
            axes[1, 0].imshow(masks[i].cpu().numpy(), cmap='tab10', vmin=0, vmax=4)
            axes[1, 0].set_title('Ground Truth')
            axes[1, 0].axis('off')
            
            # Prediction
            axes[1, 1].imshow(pred[i].cpu().numpy(), cmap='tab10', vmin=0, vmax=4)
            axes[1, 1].set_title('Prediction')
            axes[1, 1].axis('off')
            
            # Overlay on FLAIR
            axes[1, 2].imshow(images[i, 3].cpu().numpy(), cmap='gray')
            pred_overlay = np.ma.masked_where(pred[i].cpu().numpy() == 0, pred[i].cpu().numpy())
            axes[1, 2].imshow(pred_overlay, cmap='hot', alpha=0.5, vmin=1, vmax=4)
            axes[1, 2].set_title('Overlay on FLAIR')
            axes[1, 2].axis('off')
            
            # Difference (errors)
            diff = (pred[i] != masks[i]).cpu().numpy().astype(float)
            axes[1, 3].imshow(diff, cmap='Reds')
            axes[1, 3].set_title('Errors')
            axes[1, 3].axis('off')
            
            # Compute Dice for this sample
            dice_scorer = DiceScore()
            dice = dice_scorer.compute_brats_regions(pred[i], masks[i])
            
            fig.suptitle(f'Sample {samples_shown + 1} - Dice WT: {dice["dice_wt"]:.3f}, '
                        f'TC: {dice["dice_tc"]:.3f}, ET: {dice["dice_et"]:.3f}')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'prediction_{samples_shown}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            samples_shown += 1


def visualize_dice_distribution(metrics_list, output_dir):
    """Visualize Dice score distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    regions = ['wt', 'tc', 'et']
    titles = ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
    
    for i, (region, title) in enumerate(zip(regions, titles)):
        values = [m[f'dice_{region}'] for m in metrics_list]
        
        axes[i].hist(values, bins=20, edgecolor='black', alpha=0.7)
        axes[i].axvline(np.mean(values), color='r', linestyle='--',
                       label=f'Mean: {np.mean(values):.3f}')
        axes[i].set_xlabel('Dice Score')
        axes[i].set_ylabel('Count')
        axes[i].set_title(title)
        axes[i].legend()
        axes[i].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dice_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Segmentation Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, experiment_name = load_model(args.checkpoint, str(device))
    
    # Load test data
    print("\n[2/4] Loading data...")
    data_dir = Path(args.data_dir)
    
    test_dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "test_metadata.json",
        augmentor=None,
        return_segmentation=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\n[3/4] Evaluating...")
    metrics_list = evaluate(model, test_loader, device, compute_hd=args.compute_hd)
    
    # Compute statistics
    stats = compute_statistics(metrics_list)
    
    print("\nSegmentation Results:")
    print(f"  Dice Whole Tumor: {stats['dice_wt_mean']:.4f} ± {stats['dice_wt_std']:.4f}")
    print(f"  Dice Tumor Core: {stats['dice_tc_mean']:.4f} ± {stats['dice_tc_std']:.4f}")
    print(f"  Dice Enhancing Tumor: {stats['dice_et_mean']:.4f} ± {stats['dice_et_std']:.4f}")
    print(f"  Dice Mean: {stats['dice_mean_mean']:.4f} ± {stats['dice_mean_std']:.4f}")
    
    if args.compute_hd:
        print(f"\n  HD95 Whole Tumor: {stats.get('hd95_wt_mean', 'N/A'):.2f} ± {stats.get('hd95_wt_std', 0):.2f}")
        print(f"  HD95 Tumor Core: {stats.get('hd95_tc_mean', 'N/A'):.2f} ± {stats.get('hd95_tc_std', 0):.2f}")
        print(f"  HD95 Enhancing Tumor: {stats.get('hd95_et_mean', 'N/A'):.2f} ± {stats.get('hd95_et_std', 0):.2f}")
    
    # Visualize
    print("\n[4/4] Generating visualizations...")
    visualize_predictions(model, test_loader, device, output_dir, args.num_visualize)
    visualize_dice_distribution(metrics_list, output_dir)
    
    # Save results
    results = {
        'experiment': experiment_name,
        'checkpoint': args.checkpoint,
        'num_samples': len(test_dataset),
        'statistics': stats,
        'per_sample_metrics': metrics_list
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
