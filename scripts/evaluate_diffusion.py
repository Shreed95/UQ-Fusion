#!/usr/bin/env python
# scripts/evaluate_diffusion.py

"""
Script to evaluate trained Latent Diffusion Model.

Usage:
    python scripts/evaluate_diffusion.py --checkpoint ./outputs/checkpoints/diffusion/best.pth --vae_checkpoint ./outputs/checkpoints/vae/best.pth --data_dir ./data
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
from models.vae import VAESmall
from models.diffusion import LatentDiffusionModelSmall, DDIMSampler
from validation.metrics import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Diffusion Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to diffusion model checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/diffusion',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to generate')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of DDIM inference steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def load_models(diffusion_path: str, vae_path: str, device: str):
    """Load VAE and Diffusion models."""
    # Load VAE
    vae_checkpoint = torch.load(vae_path, map_location=device)
    vae = VAESmall(in_channels=4, out_channels=4, latent_channels=4)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.to(device)
    vae.eval()
    
    # Load Diffusion
    diffusion_checkpoint = torch.load(diffusion_path, map_location=device)
    diffusion = LatentDiffusionModelSmall(latent_channels=4, base_channels=64, num_timesteps=1000)
    diffusion.unet.load_state_dict(diffusion_checkpoint['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.to(device)
    diffusion.unet.eval()
    
    return vae, diffusion, diffusion_checkpoint


@torch.no_grad()
def generate_samples(
    model,
    dataloader,
    device,
    num_samples: int,
    num_inference_steps: int
):
    """Generate samples using image-to-image translation."""
    model.unet.eval()
    
    source_images = []
    generated_images = []
    
    samples_collected = 0
    
    for batch in dataloader:
        if samples_collected >= num_samples:
            break
        
        images = batch['image'].to(device)
        batch_size = images.shape[0]
        
        # Encode source
        source_latents = model.encode(images)
        
        # Generate
        sampler = DDIMSampler(model.scheduler, model.unet, num_inference_steps)
        generated_latents = sampler.sample(
            shape=source_latents.shape,
            condition=source_latents,
            device=device,
            show_progress=False
        )
        
        # Decode
        generated = model.decode(generated_latents)
        
        source_images.append(images.cpu())
        generated_images.append(generated.cpu())
        
        samples_collected += batch_size
    
    source_images = torch.cat(source_images, dim=0)[:num_samples]
    generated_images = torch.cat(generated_images, dim=0)[:num_samples]
    
    return source_images, generated_images


def visualize_results(
    source: torch.Tensor,
    generated: torch.Tensor,
    output_dir: Path,
    num_show: int = 8
):
    """Visualize source and generated images."""
    num_show = min(num_show, source.shape[0])
    
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    # Create comparison figure
    fig, axes = plt.subplots(num_show, 8, figsize=(24, num_show * 3))
    
    for i in range(num_show):
        # Source images
        for j in range(4):
            axes[i, j].imshow(source[i, j].numpy(), cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, j].set_title(f'Src {modality_names[j]}')
            axes[i, j].axis('off')
        
        # Generated images
        for j in range(4):
            axes[i, 4 + j].imshow(generated[i, j].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
            if i == 0:
                axes[i, 4 + j].set_title(f'Gen {modality_names[j]}')
            axes[i, 4 + j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Individual detailed comparisons
    for i in range(min(4, num_show)):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for j in range(4):
            axes[0, j].imshow(source[i, j].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, j].set_title(f'Source {modality_names[j]}')
            axes[0, j].axis('off')
            
            axes[1, j].imshow(generated[i, j].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
            axes[1, j].set_title(f'Generated {modality_names[j]}')
            axes[1, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()


def compute_metrics(source: torch.Tensor, generated: torch.Tensor, device: str):
    """Compute quality metrics between source and generated."""
    calculator = MetricsCalculator(device)
    
    all_psnr = []
    all_ssim = []
    
    for i in range(source.shape[0]):
        src = source[i:i+1].to(device)
        gen = generated[i:i+1].to(device).clamp(0, 1)
        
        metrics = calculator.calculate(gen, src)
        all_psnr.append(metrics['psnr'])
        all_ssim.append(metrics['ssim'])
    
    return {
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim)
    }


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Diffusion Model Evaluation")
    print("=" * 60)
    print(f"Diffusion checkpoint: {args.checkpoint}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Device: {device}")
    print(f"Inference steps: {args.num_inference_steps}")
    print("=" * 60)
    
    # Load models
    print("\n[1/4] Loading models...")
    vae, diffusion, checkpoint = load_models(args.checkpoint, args.vae_checkpoint, str(device))
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
    source, generated = generate_samples(
        diffusion,
        test_loader,
        device,
        args.num_samples,
        args.num_inference_steps
    )
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
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
