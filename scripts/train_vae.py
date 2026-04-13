#!/usr/bin/env python
# scripts/train_vae.py

"""
Train VAE on BraTS 2020 preprocessed slices.

Usage:
    python scripts/train_vae.py --data_dir ./data --epochs 100 --batch_size 8 --device mps
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from data import BraTSSliceDataset, MedicalImageAugmentor, AugmentationConfig
from models.vae import VAE, VAEConfig
from training.train_vae import VAETrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description='Train VAE on BraTS 2020')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kl_weight', type=float, default=0.0001)
    parser.add_argument('--ssim_weight', type=float, default=0.0)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--latent_channels', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/vae')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs/vae')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    # Accept --model_type for backward compat but ignore it
    parser.add_argument('--model_type', type=str, default='standard')

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
    print("VAE Training (MPS-Optimized)")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Base channels: {args.base_channels} | Latent: {args.latent_channels}")
    print(f"Batch size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"LR: {args.lr} | KL weight: {args.kl_weight}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Datasets
    print("\n[1/3] Creating datasets...")
    aug = MedicalImageAugmentor(
        AugmentationConfig(rotation_range=15.0, horizontal_flip=True,
                           brightness_range=0.1, noise_std_range=(0.01, 0.03)),
        geometric_prob=0.5, intensity_prob=0.5)

    train_ds = BraTSSliceDataset(slices_dir, splits_dir / "train_metadata.json",
                                  augmentor=aug, return_segmentation=False)
    val_ds = BraTSSliceDataset(slices_dir, splits_dir / "val_metadata.json",
                                augmentor=None, return_segmentation=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # num_workers=0 is safest on MPS, avoids multiprocessing overhead
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # Model
    print("\n[2/3] Creating model...")
    config = VAEConfig(
        in_channels=4, out_channels=4,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=2, use_se=True
    )
    model = VAE(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Training
    print("\n[3/3] Starting training...")
    tc = TrainingConfig(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, kl_weight=args.kl_weight,
        ssim_weight=args.ssim_weight,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir, device=args.device,
    )

    trainer = VAETrainer(model, train_loader, val_loader, tc)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

    print("\n" + "=" * 60)
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()