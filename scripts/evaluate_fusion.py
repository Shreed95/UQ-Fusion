#!/usr/bin/env python
# scripts/evaluate_fusion.py

"""
Script to evaluate uncertainty-guided fusion.

Usage:
    python scripts/evaluate_fusion.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data
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
from models.vae import VAE, VAEConfig
from models.diffusion import LatentDiffusionModelSmall
from models.gan import STABLEGeneratorSmall
from models.uncertainty import UncertaintyAwareDualBranch, UncertaintyWrapperConfig
from models.fusion import (
    UQFusionModule,
    create_fusion_module,
    FusionQualityMetrics
)
from validation.metrics import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Uncertainty-Guided Fusion')
    
    # Checkpoints
    parser.add_argument('--diffusion_checkpoint', type=str, required=True,
                        help='Path to diffusion checkpoint')
    parser.add_argument('--gan_checkpoint', type=str, required=True,
                        help='Path to GAN checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    
    # Fusion settings
    parser.add_argument('--fusion_method', type=str, default='uncertainty',
                        choices=['uncertainty', 'average', 'softmax', 'gated', 'region_adaptive'],
                        help='Fusion method')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for weight scaling')
    parser.add_argument('--smooth_weights', action='store_true',
                        help='Apply Gaussian smoothing to weights')
    
    # Uncertainty settings
    parser.add_argument('--num_mc_samples', type=int, default=10,
                        help='Number of MC samples')
    parser.add_argument('--diffusion_steps', type=int, default=50,
                        help='Number of diffusion steps')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/fusion',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def load_models(args, device):
    """Load all models."""
    # VAE
    vae = VAE(VAEConfig())
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.to(device)
    vae.eval()
    
    # Diffusion
    diffusion = LatentDiffusionModelSmall(latent_channels=4, base_channels=64, num_timesteps=1000)
    diff_ckpt = torch.load(args.diffusion_checkpoint, map_location=device)
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.to(device)
    diffusion.eval()
    
    # GAN
    gan_ckpt = torch.load(args.gan_checkpoint, map_location=device)
    gan_config = gan_ckpt.get('config', {})
    generator = STABLEGeneratorSmall(
        in_channels=gan_config.get('in_channels', 4),
        out_channels=gan_config.get('out_channels', 4),
        base_channels=gan_config.get('base_channels_g', 32),
        num_residual_blocks=gan_config.get('num_residual_blocks', 6),
        use_dropout=False
    )
    generator.load_state_dict(gan_ckpt['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    return vae, diffusion, generator


def create_models(diffusion, generator, fusion_args, args):
    """Create uncertainty-aware dual branch and fusion module."""
    # Uncertainty wrapper
    unc_config = UncertaintyWrapperConfig(
        num_mc_samples=args.num_mc_samples,
        normalize_uncertainty=True
    )
    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, unc_config)
    
    # Fusion module
    fusion_module = create_fusion_module(
        method=args.fusion_method,
        temperature=args.temperature,
        smooth_weights=args.smooth_weights
    )
    
    return dual_branch, fusion_module


@torch.no_grad()
def evaluate_fusion(dual_branch, fusion_module, dataloader, device, args, num_samples):
    """Evaluate fusion."""
    results = {
        'source': [],
        'I_diff': [],
        'I_gan': [],
        'I_fused': [],
        'U_diff': [],
        'U_gan': [],
        'alpha': [],
        'beta': []
    }
    
    all_metrics = []
    samples_collected = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        if samples_collected >= num_samples:
            break
        
        images = batch['image'].to(device)
        
        # Get fusion inputs
        fusion_inputs = dual_branch.get_fusion_inputs(
            images,
            diffusion_steps=args.diffusion_steps,
            diffusion_strength=0.8
        )
        
        # Perform fusion
        fusion_result = fusion_module(
            fusion_inputs['I_diff'],
            fusion_inputs['I_gan'],
            fusion_inputs['U_diff'],
            fusion_inputs['U_gan']
        )
        
        # Compute metrics
        for i in range(images.shape[0]):
            metrics = FusionQualityMetrics.compute_all(
                fusion_result['fused'][i:i+1],
                images[i:i+1],
                fusion_inputs['I_diff'][i:i+1],
                fusion_inputs['I_gan'][i:i+1]
            )
            all_metrics.append(metrics)
        
        # Store results
        results['source'].append(images.cpu())
        results['I_diff'].append(fusion_inputs['I_diff'].cpu())
        results['I_gan'].append(fusion_inputs['I_gan'].cpu())
        results['I_fused'].append(fusion_result['fused'].cpu())
        results['U_diff'].append(fusion_inputs['U_diff'].cpu())
        results['U_gan'].append(fusion_inputs['U_gan'].cpu())
        results['alpha'].append(fusion_result['alpha'].cpu())
        results['beta'].append(fusion_result['beta'].cpu())
        
        samples_collected += images.shape[0]
    
    # Concatenate
    for key in results:
        results[key] = torch.cat(results[key], dim=0)[:num_samples]
    
    return results, all_metrics


def visualize_fusion(results, output_dir, num_show=4):
    """Visualize fusion results."""
    num_show = min(num_show, results['source'].shape[0])
    
    for i in range(num_show):
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        
        source = results['source'][i]
        I_diff = results['I_diff'][i]
        I_gan = results['I_gan'][i]
        I_fused = results['I_fused'][i]
        alpha = results['alpha'][i]
        beta = results['beta'][i]
        U_diff = results['U_diff'][i]
        U_gan = results['U_gan'][i]
        
        # Normalize GAN output
        I_gan_norm = I_gan.clamp(0, 1)
        
        # Row 0: Images (first modality)
        axes[0, 0].imshow(source[0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Source')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(I_diff[0].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Diffusion')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(I_gan_norm[0].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title('GAN')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(I_fused[0].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, 3].set_title('Fused')
        axes[0, 3].axis('off')
        
        # Error map
        error_fused = torch.abs(I_fused - source).mean(dim=0)
        im = axes[0, 4].imshow(error_fused.numpy(), cmap='hot')
        axes[0, 4].set_title('Fusion Error')
        axes[0, 4].axis('off')
        plt.colorbar(im, ax=axes[0, 4], fraction=0.046)
        
        # Row 1: Uncertainties and weights
        im1 = axes[1, 0].imshow(U_diff[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Diffusion Uncertainty')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        im2 = axes[1, 1].imshow(U_gan[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('GAN Uncertainty')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        im3 = axes[1, 2].imshow(alpha[0].numpy(), cmap='Blues', vmin=0, vmax=1)
        axes[1, 2].set_title('α (Diffusion Weight)')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
        
        im4 = axes[1, 3].imshow(beta[0].numpy(), cmap='Oranges', vmin=0, vmax=1)
        axes[1, 3].set_title('β (GAN Weight)')
        axes[1, 3].axis('off')
        plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
        
        # Weight difference
        weight_diff = (alpha - beta)[0].numpy()
        im5 = axes[1, 4].imshow(weight_diff, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 4].set_title('α - β')
        axes[1, 4].axis('off')
        plt.colorbar(im5, ax=axes[1, 4], fraction=0.046)
        
        # Row 2: Error comparisons
        error_diff = torch.abs(I_diff - source).mean(dim=0)
        error_gan = torch.abs(I_gan_norm - source).mean(dim=0)
        
        im6 = axes[2, 0].imshow(error_diff.numpy(), cmap='hot')
        axes[2, 0].set_title('Diffusion Error')
        axes[2, 0].axis('off')
        plt.colorbar(im6, ax=axes[2, 0], fraction=0.046)
        
        im7 = axes[2, 1].imshow(error_gan.numpy(), cmap='hot')
        axes[2, 1].set_title('GAN Error')
        axes[2, 1].axis('off')
        plt.colorbar(im7, ax=axes[2, 1], fraction=0.046)
        
        # Best possible (oracle)
        oracle = torch.where(error_diff < error_gan, I_diff, I_gan_norm)
        error_oracle = torch.abs(oracle - source).mean(dim=0)
        im8 = axes[2, 2].imshow(error_oracle.numpy(), cmap='hot')
        axes[2, 2].set_title('Oracle Error')
        axes[2, 2].axis('off')
        plt.colorbar(im8, ax=axes[2, 2], fraction=0.046)
        
        # Improvement over best single branch
        best_single = torch.minimum(error_diff, error_gan)
        improvement = best_single - error_fused
        im9 = axes[2, 3].imshow(improvement.numpy(), cmap='RdYlGn', vmin=-0.1, vmax=0.1)
        axes[2, 3].set_title('Improvement over Best')
        axes[2, 3].axis('off')
        plt.colorbar(im9, ax=axes[2, 3], fraction=0.046)
        
        axes[2, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'fusion_sample_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()


def compute_aggregate_metrics(all_metrics):
    """Compute aggregate metrics."""
    agg = {}
    
    keys = all_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in all_metrics]
        agg[f'{key}_mean'] = float(np.mean(values))
        agg[f'{key}_std'] = float(np.std(values))
    
    return agg


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Uncertainty-Guided Fusion Evaluation")
    print("=" * 60)
    print(f"Fusion method: {args.fusion_method}")
    print(f"MC samples: {args.num_mc_samples}")
    print(f"Diffusion steps: {args.diffusion_steps}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load models
    print("\n[1/5] Loading models...")
    vae, diffusion, generator = load_models(args, str(device))
    
    # Create dual branch and fusion
    print("\n[2/5] Creating fusion pipeline...")
    dual_branch, fusion_module = create_models(diffusion, generator, args, args)
    dual_branch = dual_branch.to(device)
    fusion_module = fusion_module.to(device)
    
    # Load data
    print("\n[3/5] Loading data...")
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
        pin_memory=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\n[4/5] Evaluating fusion...")
    results, all_metrics = evaluate_fusion(
        dual_branch, fusion_module, test_loader, device, args, args.num_samples
    )
    
    # Compute aggregate metrics
    print("\n[5/5] Computing metrics...")
    agg_metrics = compute_aggregate_metrics(all_metrics)
    
    print("\nFusion Quality Metrics:")
    print(f"  Fused PSNR: {agg_metrics['psnr_mean']:.2f} ± {agg_metrics['psnr_std']:.2f} dB")
    print(f"  Diffusion PSNR: {agg_metrics['psnr_diff_mean']:.2f} ± {agg_metrics['psnr_diff_std']:.2f} dB")
    print(f"  GAN PSNR: {agg_metrics['psnr_gan_mean']:.2f} ± {agg_metrics['psnr_gan_std']:.2f} dB")
    print(f"  PSNR Improvement: {agg_metrics['psnr_improvement_mean']:+.2f} ± {agg_metrics['psnr_improvement_std']:.2f} dB")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_fusion(results, output_dir)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(agg_metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
