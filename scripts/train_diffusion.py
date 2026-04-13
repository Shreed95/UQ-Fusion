#!/usr/bin/env python
# scripts/train_diffusion.py

"""
Train Latent Diffusion Model on BraTS 2020 dataset.

Usage:
    python scripts/train_diffusion.py \
        --data_dir ./data \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --epochs 50 --batch_size 8 --device mps
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from data import BraTSSliceDataset, MedicalImageAugmentor, AugmentationConfig
from models.diffusion import LatentDiffusionModelSmall, LatentDiffusionModel, LatentDiffusionConfig
from training.train_diffusion import DiffusionTrainer, DiffusionTrainingConfig, load_vae


def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='small', choices=['standard', 'small'])
    parser.add_argument('--latent_channels', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--vae_type', type=str, default='standard')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/diffusion')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs/diffusion')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    if torch.backends.mps.is_available():
        default_device = 'mps'
    elif torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    parser.add_argument('--device', type=str, default=default_device)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    slices_dir = data_dir / "slices"
    splits_dir = data_dir / "splits"

    print("=" * 60)
    print("Latent Diffusion Model Training (MPS-Optimized)")
    print("=" * 60)
    print(f"Data: {data_dir} | VAE: {args.vae_checkpoint}")
    print(f"Model: {args.model_type} | Base ch: {args.base_channels} | Timesteps: {args.num_timesteps}")
    print(f"Batch: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Datasets
    print("\n[1/4] Creating datasets...")
    aug = MedicalImageAugmentor(
        AugmentationConfig(rotation_range=15.0, horizontal_flip=True,
                           brightness_range=0.1, noise_std_range=(0.01, 0.03)),
        geometric_prob=0.5, intensity_prob=0.5)

    train_ds = BraTSSliceDataset(slices_dir, splits_dir / "train_metadata.json",
                                  augmentor=aug, return_segmentation=False)
    val_ds = BraTSSliceDataset(slices_dir, splits_dir / "val_metadata.json",
                                augmentor=None, return_segmentation=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # Load VAE
    print("\n[2/4] Loading pre-trained VAE...")
    vae = load_vae(args.vae_checkpoint, args.vae_type, args.device)
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,} (frozen)")

    # Create diffusion model
    print("\n[3/4] Creating diffusion model...")
    if args.model_type == 'small':
        model = LatentDiffusionModelSmall(
            latent_channels=args.latent_channels,
            base_channels=args.base_channels,
            num_timesteps=args.num_timesteps)
    else:
        config = LatentDiffusionConfig(
            latent_channels=args.latent_channels,
            base_channels=args.base_channels,
            num_timesteps=args.num_timesteps)
        model = LatentDiffusionModel(config)

    print(f"U-Net parameters: {sum(p.numel() for p in model.unet.parameters()):,}")

    # Train
    print("\n[4/4] Starting training...")
    tc = DiffusionTrainingConfig(
        model_type=args.model_type,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_timesteps=args.num_timesteps,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs, save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir,
        vae_checkpoint=args.vae_checkpoint, vae_type=args.vae_type,
        use_ema=not args.no_ema, device=args.device)

    trainer = DiffusionTrainer(model, vae, train_loader, val_loader, tc)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

    print("\n" + "=" * 60)
    print(f"Best val loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()