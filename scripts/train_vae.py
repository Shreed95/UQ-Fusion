#!/usr/bin/env python
# scripts/train_vae.py

"""
Script to train VAE model on BraTS 2020 dataset.

Usage:
    python scripts/train_vae.py --data_dir ./data --epochs 100 --batch_size 8
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data import BraTSSliceDataset, MedicalImageAugmentor, AugmentationConfig
from models.vae import VAE, VAESmall, VAEConfig
from training.train_vae import VAETrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE on BraTS 2020')
    
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
                        help='VAE model type')
    parser.add_argument('--latent_channels', type=int, default=4,
                        help='Number of latent channels')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    # Loss
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--kl_weight', type=float, default=0.0001,
                        help='KL divergence weight')
    parser.add_argument('--kl_warmup', type=int, default=10,
                        help='KL warmup epochs')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/vae',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs/vae',
                        help='Log directory')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable EMA')
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
    print("VAE Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Latent channels: {args.latent_channels}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"KL weight: {args.kl_weight}")
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
    
    # Create model
    print("\n[2/3] Creating model...")
    
    if args.model_type == 'small':
        model = VAESmall(
            in_channels=4,
            out_channels=4,
            latent_channels=args.latent_channels
        )
    else:
        vae_config = VAEConfig(
            in_channels=4,
            out_channels=4,
            latent_channels=args.latent_channels,
            base_channels=args.base_channels
        )
        model = VAE(vae_config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create training config
    training_config = TrainingConfig(
        model_type=args.model_type,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        kl_warmup_epochs=args.kl_warmup,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_ema=not args.no_ema,
        device=args.device,
        mixed_precision=args.mixed_precision
    )
    
    # Create trainer
    print("\n[3/3] Starting training...")
    trainer = VAETrainer(model, train_loader, val_loader, training_config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Logs saved to: {args.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
