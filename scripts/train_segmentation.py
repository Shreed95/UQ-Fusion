#!/usr/bin/env python
# scripts/train_segmentation.py

"""
Train tumor segmentation model with optional joint augmentation.

Usage:
    # Baseline (no augmentation)
    python scripts/train_segmentation.py --data_dir ./data --experiment baseline --device mps

    # Baseline with augmentation
    python scripts/train_segmentation.py --data_dir ./data --experiment baseline_aug --augment --device mps

    # Augmented with augmentation (original + synthetic + transforms)
    python scripts/train_segmentation.py --data_dir ./data --experiment augmented_aug \
        --synthetic_dir ./outputs/expanded_dataset/accepted --augment --device mps
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
import json

from data import BraTSSliceDataset
from models.segmentation import create_segmentation_model
from training.train_segmentation import (
    SegmentationTrainer,
    SegmentationTrainingConfig,
    SegmentationAugmentor,
    AugmentedDataset,
    SyntheticOnlyDataset
)


class OriginalDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper for consistent types + optional joint augmentation."""

    def __init__(self, dataset, augmentor=None):
        self.dataset = dataset
        self.augmentor = augmentor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image'].float()
        seg = sample['segmentation'].long()

        if self.augmentor is not None:
            image, seg = self.augmentor(image, seg)

        return {'image': image, 'segmentation': seg}


def check_synthetic_has_segmentation(synthetic_dir: Path) -> bool:
    """Check if synthetic files have segmentation saved."""
    synthetic_files = list(synthetic_dir.glob('synthetic_*.npz'))
    if not synthetic_files:
        print(f"No synthetic files found in {synthetic_dir}")
        return False
    data = np.load(synthetic_files[0], allow_pickle=True)
    has_seg = 'segmentation' in data.files
    if has_seg:
        print(f"✓ Synthetic files contain segmentation masks ({len(synthetic_files)} files)")
    else:
        print("!" * 60)
        print("WARNING: Synthetic files missing segmentation masks!")
        print("!" * 60)
    return has_seg


def main():
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--synthetic_dir', type=str, default=None)
    parser.add_argument('--synthetic_ratio', type=float, default=1.0)
    parser.add_argument('--synthetic_only', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--model_type', type=str, default='small', choices=['small', 'standard', 'large'])
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dice_weight', type=float, default=0.7)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--experiment', type=str, default='baseline')
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/segmentation')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs/segmentation')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--augment', action='store_true',
                        help='Enable joint image+mask augmentation (flips, rotation, intensity)')

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)

    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("Tumor Segmentation Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Model: {args.model_type} | Augmentation: {'ON' if args.augment else 'OFF'}")
    print(f"Device: {device}")
    print("=" * 60)

    # Create augmentor if requested
    augmentor = SegmentationAugmentor() if args.augment else None

    # Load datasets
    data_dir = Path(args.data_dir)
    train_dataset_orig = BraTSSliceDataset(
        data_dir / "slices", data_dir / "splits" / "train_metadata.json",
        augmentor=None, return_segmentation=True)
    val_dataset_orig = BraTSSliceDataset(
        data_dir / "slices", data_dir / "splits" / "val_metadata.json",
        augmentor=None, return_segmentation=True)

    # Validation NEVER gets augmented (fair evaluation)
    val_dataset = OriginalDatasetWrapper(val_dataset_orig, augmentor=None)

    # Limit training samples
    if args.max_train_samples and args.max_train_samples < len(train_dataset_orig):
        indices = list(range(args.max_train_samples))
        train_dataset_orig = torch.utils.data.Subset(train_dataset_orig, indices)
        print(f"Limited original training to {args.max_train_samples} samples")

    # Create training dataset
    if args.synthetic_only and args.synthetic_dir:
        print(f"\nMode: Synthetic only" + (" + augmentation" if args.augment else ""))
        check_synthetic_has_segmentation(Path(args.synthetic_dir))
        train_dataset = SyntheticOnlyDataset(
            Path(args.synthetic_dir), original_dataset=train_dataset_orig,
            augmentor=augmentor)
    elif args.synthetic_dir:
        print(f"\nMode: Original + Synthetic" + (" + augmentation" if args.augment else ""))
        check_synthetic_has_segmentation(Path(args.synthetic_dir))
        train_dataset = AugmentedDataset(
            train_dataset_orig, Path(args.synthetic_dir),
            synthetic_ratio=args.synthetic_ratio, augmentor=augmentor)
    else:
        print(f"\nMode: Original only (Baseline)" + (" + augmentation" if args.augment else ""))
        train_dataset = OriginalDatasetWrapper(train_dataset_orig, augmentor=augmentor)

    print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # Config
    config = SegmentationTrainingConfig(
        model_type=args.model_type, num_classes=args.num_classes,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        dice_weight=args.dice_weight, ce_weight=args.ce_weight,
        checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir,
        device=str(device))

    # Model + Trainer
    model = create_segmentation_model(model_type=config.model_type,
                                       in_channels=config.in_channels, num_classes=config.num_classes)
    trainer = SegmentationTrainer(model, train_loader, val_loader, config, args.experiment)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    print("\n" + "=" * 60)
    print(f"Best validation Dice: {trainer.best_dice:.4f}")
    print(f"Results saved to: {trainer.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()