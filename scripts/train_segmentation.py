#!/usr/bin/env python
# scripts/train_segmentation.py

"""
Train tumor segmentation model.

Supports:
- Baseline training (original data only)
- Augmented training (original + synthetic)
- Synthetic-only training (ablation)

Usage:
    # Baseline
    python scripts/train_segmentation.py --data_dir ./data --experiment baseline
    
    # Augmented (after regenerating dataset with segmentations)
    python scripts/train_segmentation.py --data_dir ./data --experiment augmented \
        --synthetic_dir ./outputs/expanded_dataset/accepted
    
    # Synthetic only
    python scripts/train_segmentation.py --data_dir ./data --experiment synthetic_only \
        --synthetic_dir ./outputs/expanded_dataset/accepted --synthetic_only
    
    # Limited data experiment (1000 slices)
    python scripts/train_segmentation.py --data_dir ./data --experiment baseline_1k \
        --max_train_samples 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import json

from data import BraTSSliceDataset
from models.segmentation import create_segmentation_model
from training.train_segmentation import (
    SegmentationTrainer,
    SegmentationTrainingConfig,
    AugmentedDataset,
    SyntheticOnlyDataset
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--synthetic_dir', type=str, default=None,
                        help='Directory with synthetic data for augmentation')
    parser.add_argument('--synthetic_ratio', type=float, default=1.0,
                        help='Ratio of synthetic to original data')
    parser.add_argument('--synthetic_only', action='store_true',
                        help='Train only on synthetic data')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit original training samples (for data-scarce experiments)')
    
    # Model
    parser.add_argument('--model_type', type=str, default='small',
                        choices=['small', 'standard', 'large'])
    parser.add_argument('--num_classes', type=int, default=4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # Loss
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=0.5)
    
    # Experiment
    parser.add_argument('--experiment', type=str, default='baseline',
                        help='Experiment name (baseline, augmented, synthetic_only)')
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/segmentation')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs/segmentation')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


class OriginalDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to ensure consistent tensor types from original dataset."""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            'image': sample['image'].float(),
            'segmentation': sample['segmentation'].long()
        }


def check_synthetic_has_segmentation(synthetic_dir: Path) -> bool:
    """Check if synthetic files have segmentation saved."""
    synthetic_files = list(synthetic_dir.glob('synthetic_*.npz'))
    if not synthetic_files:
        print(f"No synthetic files found in {synthetic_dir}")
        return False
    
    # Check first file
    data = np.load(synthetic_files[0], allow_pickle=True)
    has_seg = 'segmentation' in data.files
    
    if not has_seg:
        print("\n" + "!" * 60)
        print("WARNING: Synthetic files do not contain segmentation masks!")
        print("This will cause incorrect training!")
        print("")
        print("Please regenerate the dataset using the fixed generate_dataset.py:")
        print("  1. Delete existing expanded dataset:")
        print("     rm -rf ./outputs/expanded_dataset/accepted/*")
        print("  2. Regenerate with fixed script:")
        print("     python scripts/generate_dataset.py ...")
        print("!" * 60 + "\n")
    else:
        print(f"✓ Synthetic files contain segmentation masks ({len(synthetic_files)} files)")
    
    return has_seg


def main():
    args = parse_args()
    
    # Handle device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon MPS GPU")
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("=" * 60)
    print("Tumor Segmentation Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load original dataset
    data_dir = Path(args.data_dir)
    
    train_dataset_orig = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "train_metadata.json",
        augmentor=None,
        return_segmentation=True
    )
    
    val_dataset_orig = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "val_metadata.json",
        augmentor=None,
        return_segmentation=True
    )
    
    # Wrap datasets for consistent types
    train_dataset_wrapped = OriginalDatasetWrapper(train_dataset_orig)
    val_dataset = OriginalDatasetWrapper(val_dataset_orig)
    
    # Limit training samples if specified
    if args.max_train_samples and args.max_train_samples < len(train_dataset_wrapped):
        indices = list(range(args.max_train_samples))
        train_dataset_wrapped = torch.utils.data.Subset(train_dataset_wrapped, indices)
        train_dataset_orig = torch.utils.data.Subset(train_dataset_orig, indices)
        print(f"\nLimited original training to {args.max_train_samples} samples")
    
    # Create training dataset based on experiment type
    if args.synthetic_only and args.synthetic_dir:
        print("\nTraining mode: Synthetic only")
        synthetic_dir = Path(args.synthetic_dir)
        check_synthetic_has_segmentation(synthetic_dir)
        train_dataset = SyntheticOnlyDataset(
            synthetic_dir,
            original_dataset=train_dataset_orig
        )
    elif args.synthetic_dir:
        print("\nTraining mode: Original + Synthetic (Augmented)")
        synthetic_dir = Path(args.synthetic_dir)
        check_synthetic_has_segmentation(synthetic_dir)
        train_dataset = AugmentedDataset(
            train_dataset_orig,
            synthetic_dir,
            synthetic_ratio=args.synthetic_ratio
        )
    else:
        print("\nTraining mode: Original only (Baseline)")
        train_dataset = train_dataset_wrapped
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    # Create config
    config = SegmentationTrainingConfig(
        model_type=args.model_type,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dice_weight=args.dice_weight,
        ce_weight=args.ce_weight,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=str(device)
    )
    
    # Create model
    model = create_segmentation_model(
        model_type=config.model_type,
        in_channels=config.in_channels,
        num_classes=config.num_classes
    )
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_name=args.experiment
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation Dice: {trainer.best_dice:.4f}")
    print(f"Results saved to: {trainer.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()