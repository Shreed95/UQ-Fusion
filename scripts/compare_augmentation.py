#!/usr/bin/env python
# scripts/compare_augmentation.py

"""
Compare segmentation performance: Baseline vs Augmented training.

Usage:
    python scripts/compare_augmentation.py \
        --baseline_checkpoint ./outputs/checkpoints/segmentation/baseline/best.pth \
        --augmented_checkpoint ./outputs/checkpoints/segmentation/augmented/best.pth \
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
from models.segmentation import create_segmentation_model, DiceScore


def parse_args():
    parser = argparse.ArgumentParser(description='Compare Baseline vs Augmented')
    
    parser.add_argument('--baseline_checkpoint', type=str, required=True)
    parser.add_argument('--augmented_checkpoint', type=str, required=True)
    parser.add_argument('--synthetic_checkpoint', type=str, default=None,
                        help='Optional: synthetic-only checkpoint for ablation')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/comparison')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu') 
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'small')
    
    model = create_segmentation_model(
        model_type=model_type,
        in_channels=config.get('in_channels', 4),
        num_classes=config.get('num_classes', 4)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('experiment_name', 'unknown')


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model and return per-sample metrics."""
    dice_scorer = DiceScore()
    metrics_list = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)
        
        logits = model(images)
        pred = torch.argmax(logits, dim=1)
        
        for i in range(pred.shape[0]):
            dice = dice_scorer.compute_brats_regions(pred[i], masks[i])
            metrics_list.append({
                'dice_wt': dice['dice_wt'],
                'dice_tc': dice['dice_tc'],
                'dice_et': dice['dice_et'],
                'dice_mean': dice['dice_mean_brats']
            })
    
    return metrics_list


def compute_statistics(metrics_list):
    """Compute aggregate statistics."""
    stats = {}
    for key in ['dice_wt', 'dice_tc', 'dice_et', 'dice_mean']:
        values = [m[key] for m in metrics_list]
        stats[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values))
        }
    return stats


def compute_improvement(baseline_stats, augmented_stats):
    """Compute improvement percentages."""
    improvements = {}
    for key in ['dice_wt', 'dice_tc', 'dice_et', 'dice_mean']:
        base = baseline_stats[key]['mean']
        aug = augmented_stats[key]['mean']
        
        absolute = aug - base
        relative = (absolute / base) * 100 if base > 0 else 0
        
        improvements[key] = {
            'absolute': absolute,
            'relative_percent': relative
        }
    
    return improvements


def visualize_comparison(results, output_dir):
    """Create comparison visualizations."""
    experiments = list(results.keys())
    metrics = ['dice_wt', 'dice_tc', 'dice_et', 'dice_mean']
    titles = ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor', 'Mean']
    
    # Bar chart comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    x = np.arange(len(experiments))
    width = 0.6
    
    colors = ['#2ecc71', '#3498db', '#9b59b6'][:len(experiments)]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        means = [results[exp]['statistics'][metric]['mean'] for exp in experiments]
        stds = [results[exp]['statistics'][metric]['std'] for exp in experiments]
        
        bars = axes[i].bar(x, means, width, yerr=stds, capsize=5, color=colors)
        axes[i].set_ylabel('Dice Score')
        axes[i].set_title(title)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels([exp.capitalize() for exp in experiments], rotation=45)
        axes[i].set_ylim(0, 1)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dice_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Box plot comparison
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        data = [
            [m[metric] for m in results[exp]['metrics']]
            for exp in experiments
        ]
        
        bp = axes[i].boxplot(data, tick_labels=[exp.capitalize() for exp in experiments],
                             patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[i].set_ylabel('Dice Score')
        axes[i].set_title(title)
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dice_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_improvement(improvements, output_dir):
    """Visualize improvements."""
    metrics = ['dice_wt', 'dice_tc', 'dice_et', 'dice_mean']
    titles = ['Whole Tumor', 'Tumor Core', 'Enhancing Tumor', 'Mean']
    
    abs_improvements = [improvements[m]['absolute'] * 100 for m in metrics]
    rel_improvements = [improvements[m]['relative_percent'] for m in metrics]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Absolute improvement
    colors = ['green' if v >= 0 else 'red' for v in abs_improvements]
    axes[0].bar(titles, abs_improvements, color=colors, alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_ylabel('Absolute Improvement (%)')
    axes[0].set_title('Absolute Dice Improvement')
    
    for i, v in enumerate(abs_improvements):
        axes[0].text(i, v + 0.2, f'{v:+.2f}%', ha='center', fontsize=10)
    
    # Relative improvement
    colors = ['green' if v >= 0 else 'red' for v in rel_improvements]
    axes[1].bar(titles, rel_improvements, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Relative Improvement (%)')
    axes[1].set_title('Relative Dice Improvement')
    
    for i, v in enumerate(rel_improvements):
        axes[1].text(i, v + 0.5, f'{v:+.2f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Baseline vs Augmented Comparison")
    print("=" * 60)
    print(f"Baseline: {args.baseline_checkpoint}")
    print(f"Augmented: {args.augmented_checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load test data
    print("\n[1/4] Loading data...")
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
    
    # Evaluate models
    results = {}
    
    print("\n[2/4] Evaluating baseline model...")
    baseline_model, baseline_name = load_model(args.baseline_checkpoint, str(device))
    baseline_metrics = evaluate_model(baseline_model, test_loader, device)
    results['baseline'] = {
        'metrics': baseline_metrics,
        'statistics': compute_statistics(baseline_metrics)
    }
    
    print("\n[3/4] Evaluating augmented model...")
    augmented_model, augmented_name = load_model(args.augmented_checkpoint, str(device))
    augmented_metrics = evaluate_model(augmented_model, test_loader, device)
    results['augmented'] = {
        'metrics': augmented_metrics,
        'statistics': compute_statistics(augmented_metrics)
    }
    
    # Optional: synthetic-only ablation
    if args.synthetic_checkpoint:
        print("\n[3.5/4] Evaluating synthetic-only model...")
        synthetic_model, _ = load_model(args.synthetic_checkpoint, str(device))
        synthetic_metrics = evaluate_model(synthetic_model, test_loader, device)
        results['synthetic'] = {
            'metrics': synthetic_metrics,
            'statistics': compute_statistics(synthetic_metrics)
        }
    
    # Compute improvements
    improvements = compute_improvement(
        results['baseline']['statistics'],
        results['augmented']['statistics']
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nBaseline Performance:")
    stats = results['baseline']['statistics']
    print(f"  Dice WT: {stats['dice_wt']['mean']:.4f} ± {stats['dice_wt']['std']:.4f}")
    print(f"  Dice TC: {stats['dice_tc']['mean']:.4f} ± {stats['dice_tc']['std']:.4f}")
    print(f"  Dice ET: {stats['dice_et']['mean']:.4f} ± {stats['dice_et']['std']:.4f}")
    print(f"  Dice Mean: {stats['dice_mean']['mean']:.4f} ± {stats['dice_mean']['std']:.4f}")
    
    print("\nAugmented Performance:")
    stats = results['augmented']['statistics']
    print(f"  Dice WT: {stats['dice_wt']['mean']:.4f} ± {stats['dice_wt']['std']:.4f}")
    print(f"  Dice TC: {stats['dice_tc']['mean']:.4f} ± {stats['dice_tc']['std']:.4f}")
    print(f"  Dice ET: {stats['dice_et']['mean']:.4f} ± {stats['dice_et']['std']:.4f}")
    print(f"  Dice Mean: {stats['dice_mean']['mean']:.4f} ± {stats['dice_mean']['std']:.4f}")
    
    if 'synthetic' in results:
        print("\nSynthetic-Only Performance:")
        stats = results['synthetic']['statistics']
        print(f"  Dice WT: {stats['dice_wt']['mean']:.4f} ± {stats['dice_wt']['std']:.4f}")
        print(f"  Dice TC: {stats['dice_tc']['mean']:.4f} ± {stats['dice_tc']['std']:.4f}")
        print(f"  Dice ET: {stats['dice_et']['mean']:.4f} ± {stats['dice_et']['std']:.4f}")
        print(f"  Dice Mean: {stats['dice_mean']['mean']:.4f} ± {stats['dice_mean']['std']:.4f}")
    
    print("\nImprovement (Augmented vs Baseline):")
    for key, title in [('dice_wt', 'WT'), ('dice_tc', 'TC'), ('dice_et', 'ET'), ('dice_mean', 'Mean')]:
        imp = improvements[key]
        print(f"  Dice {title}: {imp['absolute']*100:+.2f}% absolute, {imp['relative_percent']:+.2f}% relative")
    
    # Visualize
    print("\n[4/4] Generating visualizations...")
    visualize_comparison(results, output_dir)
    visualize_improvement(improvements, output_dir)
    
    # Save results
    save_results = {
        'baseline': {
            'checkpoint': args.baseline_checkpoint,
            'statistics': results['baseline']['statistics']
        },
        'augmented': {
            'checkpoint': args.augmented_checkpoint,
            'statistics': results['augmented']['statistics']
        },
        'improvements': improvements
    }
    
    if 'synthetic' in results:
        save_results['synthetic'] = {
            'checkpoint': args.synthetic_checkpoint,
            'statistics': results['synthetic']['statistics']
        }
    
    with open(output_dir / 'comparison_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
