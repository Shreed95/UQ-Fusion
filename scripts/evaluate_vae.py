#!/usr/bin/env python
# scripts/evaluate_vae.py

"""
Script to evaluate trained VAE model.

Usage:
    python scripts/evaluate_vae.py --checkpoint ./outputs/checkpoints/vae/best.pth --data_dir ./data
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

from data import BraTSSliceDataset
from models.vae import VAE, VAESmall
from validation.metrics import MetricsCalculator, compute_reconstruction_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VAE model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--model_type', type=str, default='small',
                        choices=['standard', 'small'],
                        help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, model_type: str, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'small':
        model = VAESmall(in_channels=4, out_channels=4, latent_channels=4)
    else:
        from models.vae import VAEConfig
        config = VAEConfig()
        model = VAE(config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def visualize_reconstructions(
    model,
    dataloader,
    device,
    num_samples: int,
    output_dir: Path
):
    """Generate reconstruction visualizations."""
    model.eval()
    
    images_list = []
    recons_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            recons = model.reconstruct(images)
            
            images_list.append(images.cpu())
            recons_list.append(recons.cpu())
            
            if sum(img.shape[0] for img in images_list) >= num_samples:
                break
    
    images = torch.cat(images_list, dim=0)[:num_samples]
    recons = torch.cat(recons_list, dim=0)[:num_samples]
    
    # Create visualization
    n_cols = 4  # T1, T1ce, T2, FLAIR
    n_rows = num_samples * 2  # Original + Reconstruction
    
    fig, axes = plt.subplots(num_samples, n_cols * 2 + 1, figsize=(n_cols * 4 + 2, num_samples * 2))
    
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    for i in range(num_samples):
        # Original
        for j in range(4):
            axes[i, j].imshow(images[i, j].numpy(), cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, j].set_title(f'Orig {modality_names[j]}')
            axes[i, j].axis('off')
        
        # Separator
        axes[i, 4].axis('off')
        
        # Reconstruction
        for j in range(4):
            axes[i, 5 + j].imshow(recons[i, j].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 5 + j].set_title(f'Recon {modality_names[j]}')
            axes[i, 5 + j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reconstructions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstructions to {output_dir / 'reconstructions.png'}")
    
    # Save individual comparison images
    for i in range(min(4, num_samples)):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for j in range(4):
            axes[0, j].imshow(images[i, j].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, j].set_title(f'Original {modality_names[j]}')
            axes[0, j].axis('off')
            
            axes[1, j].imshow(recons[i, j].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
            axes[1, j].set_title(f'Reconstructed {modality_names[j]}')
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'comparison_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()


def visualize_latent_space(
    model,
    dataloader,
    device,
    output_dir: Path
):
    """Visualize latent space statistics."""
    model.eval()
    
    all_means = []
    all_log_vars = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing latent statistics"):
            images = batch['image'].to(device)
            mean, log_var = model.encode(images)
            
            all_means.append(mean.cpu())
            all_log_vars.append(log_var.cpu())
    
    means = torch.cat(all_means, dim=0)
    log_vars = torch.cat(all_log_vars, dim=0)
    
    # Statistics
    print("\nLatent Space Statistics:")
    print(f"  Mean - mean: {means.mean().item():.4f}, std: {means.std().item():.4f}")
    print(f"  Log_var - mean: {log_vars.mean().item():.4f}, std: {log_vars.std().item():.4f}")
    
    # Visualize mean distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(means.flatten().numpy(), bins=100, density=True, alpha=0.7)
    axes[0].set_title('Latent Mean Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    
    axes[1].hist(log_vars.flatten().numpy(), bins=100, density=True, alpha=0.7)
    axes[1].set_title('Latent Log-Variance Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latent_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved latent distribution to {output_dir / 'latent_distribution.png'}")


def visualize_interpolation(
    model,
    dataloader,
    device,
    output_dir: Path,
    num_steps: int = 10
):
    """Visualize interpolation between two images."""
    model.eval()
    
    # Get two random images
    batch = next(iter(dataloader))
    images = batch['image'][:2].to(device)
    
    # Interpolate
    interpolations = model.interpolate(images[0:1], images[1:2], num_steps)
    
    # Visualize (first channel only for simplicity)
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
    
    for i in range(num_steps):
        axes[i].imshow(interpolations[i, 0].cpu().numpy().clip(0, 1), cmap='gray')
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title('Start')
        elif i == num_steps - 1:
            axes[i].set_title('End')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'interpolation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved interpolation to {output_dir / 'interpolation.png'}")


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("VAE Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    print("\n[1/5] Loading model...")
    model, checkpoint = load_model(args.checkpoint, args.model_type, device)
    print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load data
    print("\n[2/5] Loading data...")
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
    
    # Compute metrics
    print("\n[3/5] Computing reconstruction metrics...")
    metrics = compute_reconstruction_metrics(model, test_loader, str(device))
    
    print("\nReconstruction Metrics:")
    print(f"  PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    
    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Visualize reconstructions
    print("\n[4/5] Generating visualizations...")
    visualize_reconstructions(model, test_loader, device, args.num_samples, output_dir)
    visualize_latent_space(model, test_loader, device, output_dir)
    visualize_interpolation(model, test_loader, device, output_dir)
    
    # Check quality thresholds
    print("\n[5/5] Quality Assessment...")
    psnr_pass = metrics['psnr'] > 25
    ssim_pass = metrics['ssim'] > 0.90
    
    print(f"\nQuality Thresholds:")
    print(f"  PSNR > 25 dB: {'✓ PASS' if psnr_pass else '✗ FAIL'} ({metrics['psnr']:.2f} dB)")
    print(f"  SSIM > 0.90: {'✓ PASS' if ssim_pass else '✗ FAIL'} ({metrics['ssim']:.4f})")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
