#!/usr/bin/env python
# scripts/train_gan.py

"""
Train STABLE-GAN on BraTS 2020 dataset.

Usage:
    python scripts/train_gan.py --data_dir ./data --epochs 50 --batch_size 8 --device mps
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


def main():
    parser = argparse.ArgumentParser(description='Train STABLE-GAN')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='small', choices=['standard', 'small'])
    parser.add_argument('--base_channels_g', type=int, default=32)
    parser.add_argument('--base_channels_d', type=int, default=32)
    parser.add_argument('--num_residual_blocks', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--adv_weight', type=float, default=1.0)
    parser.add_argument('--l1_weight', type=float, default=10.0)
    parser.add_argument('--spatial_weight', type=float, default=5.0)
    parser.add_argument('--quantitative_weight', type=float, default=2.0)
    parser.add_argument('--identity_weight', type=float, default=0.5)
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs/checkpoints/gan')
    parser.add_argument('--log_dir', type=str, default='./outputs/logs/gan')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
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
    print("STABLE-GAN Training (MPS-Optimized)")
    print("=" * 60)
    print(f"Data: {data_dir} | Model: {args.model_type}")
    print(f"G ch: {args.base_channels_g} | D ch: {args.base_channels_d} | ResBlocks: {args.num_residual_blocks}")
    print(f"Batch: {args.batch_size} | Epochs: {args.epochs}")
    print(f"LR G: {args.lr_g} | LR D: {args.lr_d}")
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # Models
    print("\n[2/3] Creating models...")
    config = GANTrainingConfig(
        model_type=args.model_type, in_channels=4, out_channels=4,
        base_channels_g=args.base_channels_g, base_channels_d=args.base_channels_d,
        num_residual_blocks=args.num_residual_blocks,
        epochs=args.epochs, batch_size=args.batch_size,
        lr_g=args.lr_g, lr_d=args.lr_d,
        adv_weight=args.adv_weight, l1_weight=args.l1_weight,
        spatial_weight=args.spatial_weight, quantitative_weight=args.quantitative_weight,
        identity_weight=args.identity_weight,
        save_every=args.save_every, checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir, device=args.device)

    generator, discriminator = create_gan_models(config)
    print(f"Generator: {sum(p.numel() for p in generator.parameters()):,} params")
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} params")

    # Train
    print("\n[3/3] Starting training...")
    trainer = GANTrainer(generator, discriminator, train_loader, val_loader, config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()

    print("\n" + "=" * 60)
    print(f"Best val G loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()