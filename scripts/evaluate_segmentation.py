#!/usr/bin/env python
# scripts/evaluate_segmentation.py

"""
Evaluate trained segmentation model.

Usage:
    python scripts/evaluate_segmentation.py \
        --checkpoint ./outputs/checkpoints/segmentation/baseline_aug/best.pth \
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
from models.segmentation import create_segmentation_model, DiceScore


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/segmentation')

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model = create_segmentation_model(
        model_type=config.get('model_type', 'small'),
        in_channels=config.get('in_channels', 4),
        num_classes=config.get('num_classes', 4))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    experiment_name = checkpoint.get('experiment_name', 'unknown')
    best_dice = checkpoint.get('best_dice', 0.0)
    print(f"Loaded: {experiment_name} (best Dice: {best_dice:.4f})")
    return model, experiment_name


@torch.no_grad()
def evaluate(model, dataloader, device):
    dice_scorer = DiceScore()
    all_metrics = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)
        logits = model(images)
        pred = torch.argmax(logits, dim=1)

        for i in range(pred.shape[0]):
            dice = dice_scorer.compute_brats_regions(pred[i], masks[i])
            all_metrics.append({
                'dice_wt': dice['dice_wt'],
                'dice_tc': dice['dice_tc'],
                'dice_et': dice['dice_et'],
                'dice_mean': dice['dice_mean_brats']
            })

    return all_metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Segmentation Model Evaluation")
    print("=" * 60)

    print("\n[1/3] Loading model...")
    model, experiment_name = load_model(args.checkpoint, str(device))

    print("\n[2/3] Loading data...")
    data_dir = Path(args.data_dir)
    test_ds = BraTSSliceDataset(
        data_dir / "slices", data_dir / "splits" / "test_metadata.json",
        augmentor=None, return_segmentation=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    print(f"Test samples: {len(test_ds)}")

    print("\n[3/3] Evaluating...")
    metrics_list = evaluate(model, test_loader, device)

    # Statistics
    for key in ['dice_wt', 'dice_tc', 'dice_et', 'dice_mean']:
        vals = [m[key] for m in metrics_list]
        print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    results = {
        'experiment': experiment_name,
        'num_samples': len(test_ds),
        'statistics': {k: {'mean': float(np.mean([m[k] for m in metrics_list])),
                            'std': float(np.std([m[k] for m in metrics_list]))}
                       for k in ['dice_wt', 'dice_tc', 'dice_et', 'dice_mean']}
    }
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()