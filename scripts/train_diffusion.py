#!/usr/bin/env python
# scripts/train_diffusion.py

"""
Script to train Latent Diffusion Model on BraTS 2020 dataset.

Usage:
    python scripts/train_diffusion.py --data_dir ./data --vae_checkpoint ./outputs/checkpoints/vae/best.pth --epochs 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from data import BraTSSliceDataset, MedicalImageAugmentor, AugmentationConfig
from models.vae import VAE, VAESmall
from models.diffusion import LatentDiffusionModelSmall, LatentDiffusionModel, LatentDiffusionConfig
from training.train_diffusion import DiffusionTrainer, DiffusionTrainingConfig, load_vae


def parse_args():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model on BraTS 2020')
    
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
                        help='Diffusion model type')
    parser.add_argument('--latent_channels', type=int, default=4,
                        help='Number of latent channels')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels for U-Net')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    
    # VAE
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to pre-trained VAE checkpoint')
    parser.add_argument('--vae_type', type=str, default='small',
                        choices=['standard', 'small'],
                        help='VAE model type')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/diffusion',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs/diffusion',
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
    print("Latent Diffusion Model Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Base channels: {args.base_channels}")
    print(f"Timesteps: {args.num_timesteps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # Create datasets
    print("\n[1/4] Creating datasets...")
    
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
    
    # Load VAE
    print("\n[2/4] Loading pre-trained VAE...")
    vae = load_vae(args.vae_checkpoint, args.vae_type, args.device)
    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"VAE parameters: {vae_params:,} (frozen)")
    
    # Create diffusion model
    print("\n[3/4] Creating diffusion model...")
    
    if args.model_type == 'small':
        model = LatentDiffusionModelSmall(
            latent_channels=args.latent_channels,
            base_channels=args.base_channels,
            num_timesteps=args.num_timesteps
        )
    else:
        config = LatentDiffusionConfig(
            latent_channels=args.latent_channels,
            base_channels=args.base_channels,
            num_timesteps=args.num_timesteps
        )
        model = LatentDiffusionModel(config)
    
    unet_params = sum(p.numel() for p in model.unet.parameters())
    print(f"U-Net parameters: {unet_params:,}")
    
    # Create training config
    training_config = DiffusionTrainingConfig(
        model_type=args.model_type,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_timesteps=args.num_timesteps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        vae_checkpoint=args.vae_checkpoint,
        vae_type=args.vae_type,
        use_ema=not args.no_ema,
        device=args.device,
        mixed_precision=args.mixed_precision
    )
    
    # Create trainer
    print("\n[4/4] Starting training...")
    trainer = DiffusionTrainer(model, vae, train_loader, val_loader, training_config)
    
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
