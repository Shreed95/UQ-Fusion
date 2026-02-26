#!/usr/bin/env python
# scripts/train_gan.py

"""
Script to train STABLE-GAN on BraTS 2020 dataset.

Usage:
    python scripts/train_gan.py --data_dir ./data --epochs 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data import BraTSSliceDataset, MedicalImageAugmentor, AugmentationConfig
from models.gan import STABLEGeneratorSmall, PatchGANDiscriminatorSmall
from training.train_gan import GANTrainer, GANTrainingConfig, create_gan_models


def parse_args():
    parser = argparse.ArgumentParser(description='Train STABLE-GAN on BraTS 2020')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to preprocessed data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model
    parser.add_argument('--model_type', type=str, default='small',
                        choices=['standard', 'small'],
                        help='Model type')
    parser.add_argument('--base_channels_g', type=int, default=32,
                        help='Base channels for generator')
    parser.add_argument('--base_channels_d', type=int, default=32,
                        help='Base channels for discriminator')
    parser.add_argument('--num_residual_blocks', type=int, default=6,
                        help='Number of residual blocks in generator')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr_g', type=float, default=2e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4,
                        help='Discriminator learning rate')
    
    # Loss weights
    parser.add_argument('--adv_weight', type=float, default=1.0,
                        help='Adversarial loss weight')
    parser.add_argument('--l1_weight', type=float, default=10.0,
                        help='L1 loss weight')
    parser.add_argument('--spatial_weight', type=float, default=5.0,
                        help='Spatial preservation loss weight')
    parser.add_argument('--quantitative_weight', type=float, default=2.0,
                        help='Quantitative preservation loss weight')
    parser.add_argument('--identity_weight', type=float, default=0.5,
                        help='Identity loss weight')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/gan',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs/gan',
                        help='Log directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Paths
    data_dir = Path(args.data_dir)
    slices_dir = data_dir / "slices"
    splits_dir = data_dir / "splits"
    
    print("=" * 60)
    print("STABLE-GAN Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Generator base channels: {args.base_channels_g}")
    print(f"Discriminator base channels: {args.base_channels_d}")
    print(f"Residual blocks: {args.num_residual_blocks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate (G): {args.lr_g}")
    print(f"Learning rate (D): {args.lr_d}")
    print("=" * 60)
    
    # Create datasets
    print("\n[1/3] Creating datasets...")
    
    # Augmentor for training
    aug_config = AugmentationConfig(
        rotation_range=15.0,
        horizontal_flip=True,
        brightness_range=0.1,
        noise_std_range=(0.01, 0.03)
    )
    augmentor = MedicalImageAugmentor(aug_config, geometric_prob=0.5, intensity_prob=0.5)
    
    train_dataset = BraTSSliceDataset(
        slices_dir=slices_dir,
        metadata_file=splits_dir / "train_metadata.json",
        augmentor=augmentor,
        return_segmentation=False
    )
    
    val_dataset = BraTSSliceDataset(
        slices_dir=slices_dir,
        metadata_file=splits_dir / "val_metadata.json",
        augmentor=None,
        return_segmentation=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create config
    config = GANTrainingConfig(
        model_type=args.model_type,
        in_channels=4,
        out_channels=4,
        base_channels_g=args.base_channels_g,
        base_channels_d=args.base_channels_d,
        num_residual_blocks=args.num_residual_blocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        adv_weight=args.adv_weight,
        l1_weight=args.l1_weight,
        spatial_weight=args.spatial_weight,
        quantitative_weight=args.quantitative_weight,
        identity_weight=args.identity_weight,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # Create models
    print("\n[2/3] Creating models...")
    generator, discriminator = create_gan_models(config)
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # Create trainer
    print("\n[3/3] Starting training...")
    trainer = GANTrainer(generator, discriminator, train_loader, val_loader, config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
