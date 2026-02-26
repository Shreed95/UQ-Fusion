#!/usr/bin/env python
# scripts/evaluate_gan.py

"""
Script to evaluate trained STABLE-GAN.

Usage:
    python scripts/evaluate_gan.py --checkpoint ./outputs/checkpoints/gan/best.pth --data_dir ./data
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
from models.gan import STABLEGeneratorSmall, STABLEGenerator, GeneratorConfig
from validation.metrics import MetricsCalculator,MetricsConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate STABLE-GAN')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to GAN checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/gan',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def load_generator(checkpoint_path: str, device: str):
    """Load trained generator."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'small')
    
    if model_type == 'small':
        generator = STABLEGeneratorSmall(
            in_channels=config.get('in_channels', 4),
            out_channels=config.get('out_channels', 4),
            base_channels=config.get('base_channels_g', 32),
            num_residual_blocks=config.get('num_residual_blocks', 6)
        )
    else:
        gen_config = GeneratorConfig(
            in_channels=config.get('in_channels', 4),
            out_channels=config.get('out_channels', 4),
            base_channels=config.get('base_channels_g', 64),
            num_residual_blocks=config.get('num_residual_blocks', 9)
        )
        generator = STABLEGenerator(gen_config)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    return generator, checkpoint


@torch.no_grad()
def generate_samples(model, dataloader, device, num_samples):
    """Generate samples."""
    model.eval()
    
    source_images = []
    generated_images = []
    
    samples_collected = 0
    
    for batch in dataloader:
        if samples_collected >= num_samples:
            break
        
        images = batch['image'].to(device)
        
        generated = model(images)
        
        source_images.append(images.cpu())
        generated_images.append(generated.cpu())
        
        samples_collected += images.shape[0]
    
    source_images = torch.cat(source_images, dim=0)[:num_samples]
    generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    
    return source_images, generated_images


def visualize_results(source, generated, output_dir, num_show=8):
    """Visualize results."""
    num_show = min(num_show, source.shape[0])
    
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    # Comparison figure
    fig, axes = plt.subplots(num_show, 8, figsize=(24, num_show * 3))
    
    for i in range(num_show):
        for j in range(4):
            axes[i, j].imshow(source[i, j].numpy(), cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, j].set_title(f'Src {modality_names[j]}')
            axes[i, j].axis('off')
        
        for j in range(4):
            gen_img = (generated[i, j].numpy() + 1) / 2  # Convert from [-1, 1] to [0, 1]
            axes[i, 4 + j].imshow(gen_img.clip(0, 1), cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 4 + j].set_title(f'Gen {modality_names[j]}')
            axes[i, 4 + j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Individual samples
    for i in range(min(4, num_show)):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for j in range(4):
            axes[0, j].imshow(source[i, j].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, j].set_title(f'Source {modality_names[j]}')
            axes[0, j].axis('off')
            
            gen_img = (generated[i, j].numpy() + 1) / 2
            axes[1, j].imshow(gen_img.clip(0, 1), cmap='gray', vmin=0, vmax=1)
            axes[1, j].set_title(f'Generated {modality_names[j]}')
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()


def compute_metrics(source, generated, device):
    """Compute quality metrics."""
    
    # create config properly
    config = MetricsConfig(device=device)
    calculator = MetricsCalculator(config)

    # Convert generated from [-1,1] → [0,1]
    generated = ((generated + 1) / 2).clamp(0, 1)

    source = source.to(device)
    generated = generated.to(device)

    # compute all metrics at once (batch)
    metrics = calculator.compute_all(
        pred=generated,
        target=source,
        include_lpips=False,   # keep False for now (faster)
        include_fid=False
    )

    return {
        "psnr_mean": metrics["psnr_mean"],
        "psnr_std": metrics["psnr_std"],
        "ssim_mean": metrics["ssim_mean"],
        "ssim_std": metrics["ssim_std"]
    }


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STABLE-GAN Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    generator, checkpoint = load_generator(args.checkpoint, str(device))
    print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load data
    print("\n[2/4] Loading data...")
    data_dir = Path(args.data_dir)
    
    test_dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "test_metadata.json",
        augmentor=None,
        return_segmentation=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate samples
    print("\n[3/4] Generating samples...")
    source, generated = generate_samples(generator, test_loader, device, args.num_samples)
    print(f"Generated {generated.shape[0]} samples")
    
    # Compute metrics
    print("\n[4/4] Computing metrics...")
    metrics = compute_metrics(source, generated, str(device))
    
    print("\nGeneration Metrics (Source vs Generated):")
    print(f"  PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"  SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_results(source, generated, output_dir)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
