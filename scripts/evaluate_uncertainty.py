#!/usr/bin/env python
# scripts/evaluate_uncertainty.py

"""
Script to evaluate uncertainty estimation for diffusion and GAN models.

Usage:
    python scripts/evaluate_uncertainty.py \
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
from models.uncertainty import (
    UncertaintyAwareDiffusion,
    UncertaintyAwareGAN,
    UncertaintyAwareDualBranch,
    UncertaintyWrapperConfig,
    uncertainty_quality_score
)
from validation.metrics import MetricsCalculator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Uncertainty Estimation')
    
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
    
    # Uncertainty settings
    parser.add_argument('--num_mc_samples', type=int, default=10,
                        help='Number of MC samples for epistemic uncertainty')
    parser.add_argument('--diffusion_steps', type=int, default=50,
                        help='Number of diffusion steps')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/uncertainty',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


def load_models(args, device):
    """Load all models."""
    # Load VAE
    vae = VAE(VAEConfig(in_channels=4, out_channels=4, latent_channels=4))
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.to(device)
    vae.eval()
    
    # Load Diffusion
    diffusion = LatentDiffusionModelSmall(latent_channels=4, base_channels=64, num_timesteps=1000)
    diff_ckpt = torch.load(args.diffusion_checkpoint, map_location=device)
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.to(device)
    diffusion.eval()
    
    # Load GAN
    gan_ckpt = torch.load(args.gan_checkpoint, map_location=device)
    gan_config = gan_ckpt.get('config', {})
    generator = STABLEGeneratorSmall(
        in_channels=gan_config.get('in_channels', 4),
        out_channels=gan_config.get('out_channels', 4),
        base_channels=gan_config.get('base_channels_g', 32),
        num_residual_blocks=gan_config.get('num_residual_blocks', 6),
        use_dropout=True,
        dropout_rate=0.05
    )
    raw_state = gan_ckpt['generator_state_dict']
    remapped_state = {}
    for k, v in raw_state.items():
        if 'residual_blocks' in k and '.block.5.' in k:
            remapped_state[k.replace('.block.5.', '.block.6.')] = v
        else:
            remapped_state[k] = v
    generator.load_state_dict(remapped_state, strict=False)
    generator.to(device)
    generator.eval()
    
    return vae, diffusion, generator


def create_uncertainty_wrapper(diffusion, generator, args):
    """Create uncertainty-aware wrapper."""
    config = UncertaintyWrapperConfig(
        estimate_aleatoric=True,
        estimate_epistemic=True,
        num_mc_samples=args.num_mc_samples,
        dropout_rate=0.1,
        normalize_uncertainty=True
    )
    
    return UncertaintyAwareDualBranch(diffusion, generator, config)


@torch.no_grad()
def evaluate_uncertainty(model, dataloader, device, args, num_samples):
    """Evaluate uncertainty estimation."""
    results = {
        'diffusion': {'generated': [], 'uncertainty': [], 'aleatoric': [], 'epistemic': []},
        'gan': {'generated': [], 'uncertainty': [], 'aleatoric': [], 'epistemic': []},
        'source': []
    }
    
    samples_collected = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        if samples_collected >= num_samples:
            break
        
        images = batch['image'].to(device)
        
        # Generate with uncertainty
        outputs = model.generate_both_with_uncertainty(
            images,
            diffusion_steps=args.diffusion_steps,
            diffusion_strength=0.8
        )
        
        results['source'].append(images.cpu())
        
        # Diffusion results
        results['diffusion']['generated'].append(outputs['diffusion']['generated'].cpu())
        results['diffusion']['uncertainty'].append(outputs['diffusion']['total_uncertainty'].cpu())
        results['diffusion']['aleatoric'].append(outputs['diffusion']['aleatoric'].cpu())
        results['diffusion']['epistemic'].append(outputs['diffusion']['epistemic'].cpu())
        
        # GAN results
        results['gan']['generated'].append(outputs['gan']['generated'].cpu())
        results['gan']['uncertainty'].append(outputs['gan']['total_uncertainty'].cpu())
        results['gan']['aleatoric'].append(outputs['gan']['aleatoric'].cpu())
        results['gan']['epistemic'].append(outputs['gan']['epistemic'].cpu())
        
        samples_collected += images.shape[0]
    
    # Concatenate
    for key in ['source']:
        results[key] = torch.cat(results[key], dim=0)[:num_samples]
    
    for branch in ['diffusion', 'gan']:
        for key in results[branch]:
            results[branch][key] = torch.cat(results[branch][key], dim=0)[:num_samples]
    
    return results


def visualize_uncertainty(results, output_dir, num_show=4):
    """Visualize uncertainty maps."""
    num_show = min(num_show, results['source'].shape[0])
    
    for i in range(num_show):
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        
        source = results['source'][i]
        
        # Row 0: Source images (first modality)
        axes[0, 0].imshow(source[0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Source (T1)')
        axes[0, 0].axis('off')
        
        # Diffusion results
        diff_gen = results['diffusion']['generated'][i]
        diff_total = results['diffusion']['uncertainty'][i]
        diff_ale = results['diffusion']['aleatoric'][i]
        diff_epi = results['diffusion']['epistemic'][i]
        
        axes[0, 1].imshow(diff_gen[0].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('Diffusion Generated')
        axes[0, 1].axis('off')
        
        im1 = axes[0, 2].imshow(diff_total[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('Diff Total Uncertainty')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)
        
        im2 = axes[0, 3].imshow(diff_ale[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[0, 3].set_title('Diff Aleatoric')
        axes[0, 3].axis('off')
        plt.colorbar(im2, ax=axes[0, 3], fraction=0.046)
        
        im3 = axes[0, 4].imshow(diff_epi[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[0, 4].set_title('Diff Epistemic')
        axes[0, 4].axis('off')
        plt.colorbar(im3, ax=axes[0, 4], fraction=0.046)
        
        # GAN results
        gan_gen = results['gan']['generated'][i]
        gan_gen_normalized = gan_gen.clamp(0, 1)
        gan_total = results['gan']['uncertainty'][i]
        gan_ale = results['gan']['aleatoric'][i]
        gan_epi = results['gan']['epistemic'][i]
        
        axes[1, 0].imshow(source[0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('Source (T1)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(gan_gen_normalized[0].numpy().clip(0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('GAN Generated')
        axes[1, 1].axis('off')
        
        im4 = axes[1, 2].imshow(gan_total[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title('GAN Total Uncertainty')
        axes[1, 2].axis('off')
        plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)
        
        im5 = axes[1, 3].imshow(gan_ale[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1, 3].set_title('GAN Aleatoric')
        axes[1, 3].axis('off')
        plt.colorbar(im5, ax=axes[1, 3], fraction=0.046)
        
        im6 = axes[1, 4].imshow(gan_epi[0].numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1, 4].set_title('GAN Epistemic')
        axes[1, 4].axis('off')
        plt.colorbar(im6, ax=axes[1, 4], fraction=0.046)
        
        # Row 2: All modalities comparison
        for j in range(4):
            axes[2, j].imshow(source[j].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[2, j].set_title(f'Source Ch{j}')
            axes[2, j].axis('off')
        axes[2, 4].axis('off')
        
        # Row 3: Difference maps
        diff_error = torch.abs(diff_gen - source).mean(dim=0)
        gan_error = torch.abs(gan_gen_normalized - source).mean(dim=0)
        
        im7 = axes[3, 0].imshow(diff_error.numpy(), cmap='viridis')
        axes[3, 0].set_title('Diffusion Error')
        axes[3, 0].axis('off')
        plt.colorbar(im7, ax=axes[3, 0], fraction=0.046)
        
        im8 = axes[3, 1].imshow(gan_error.numpy(), cmap='viridis')
        axes[3, 1].set_title('GAN Error')
        axes[3, 1].axis('off')
        plt.colorbar(im8, ax=axes[3, 1], fraction=0.046)
        
        # Uncertainty vs Error correlation
        axes[3, 2].scatter(diff_total[0].numpy().flatten()[::100], 
                          diff_error.numpy().flatten()[::100], alpha=0.5, s=1)
        axes[3, 2].set_xlabel('Uncertainty')
        axes[3, 2].set_ylabel('Error')
        axes[3, 2].set_title('Diff: U vs E')
        
        axes[3, 3].scatter(gan_total[0].numpy().flatten()[::100], 
                          gan_error.numpy().flatten()[::100], alpha=0.5, s=1)
        axes[3, 3].set_xlabel('Uncertainty')
        axes[3, 3].set_ylabel('Error')
        axes[3, 3].set_title('GAN: U vs E')
        
        axes[3, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'uncertainty_sample_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()


def compute_uncertainty_metrics(results):
    """Compute metrics for uncertainty quality."""
    metrics = {'diffusion': {}, 'gan': {}}
    
    for branch in ['diffusion', 'gan']:
        generated = results[branch]['generated']
        source = results['source']
        uncertainty = results[branch]['uncertainty']
        
        # Normalize GAN output
        if branch == 'gan':
            generated = generated.clamp(0, 1)
        
        # Compute error
        error = torch.abs(generated - source).mean(dim=1, keepdim=True)
        
        # Correlation between uncertainty and error
        u_flat = uncertainty.view(-1).numpy()
        e_flat = error.view(-1).numpy()
        
        correlation = np.corrcoef(u_flat, e_flat)[0, 1]
        
        # Mean uncertainty
        mean_uncertainty = uncertainty.mean().item()
        
        # Uncertainty in high-error regions
        error_threshold = np.percentile(e_flat, 75)
        high_error_mask = e_flat > error_threshold
        high_error_uncertainty = u_flat[high_error_mask].mean()
        low_error_uncertainty = u_flat[~high_error_mask].mean()
        
        metrics[branch] = {
            'correlation': float(correlation),
            'mean_uncertainty': float(mean_uncertainty),
            'high_error_uncertainty': float(high_error_uncertainty),
            'low_error_uncertainty': float(low_error_uncertainty),
            'uncertainty_separation': float(high_error_uncertainty - low_error_uncertainty),
            'mean_aleatoric': float(results[branch]['aleatoric'].mean().item()),
            'mean_epistemic': float(results[branch]['epistemic'].mean().item())
        }
    
    return metrics


def main():
    args = parse_args()
    
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Uncertainty Estimation Evaluation")
    print("=" * 60)
    print(f"Diffusion checkpoint: {args.diffusion_checkpoint}")
    print(f"GAN checkpoint: {args.gan_checkpoint}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"MC samples: {args.num_mc_samples}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load models
    print("\n[1/5] Loading models...")
    vae, diffusion, generator = load_models(args, str(device))
    
    # Create uncertainty wrapper
    print("\n[2/5] Creating uncertainty wrapper...")
    model = create_uncertainty_wrapper(diffusion, generator, args)
    model = model.to(device)
    
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
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\n[4/5] Evaluating uncertainty...")
    results = evaluate_uncertainty(model, test_loader, device, args, args.num_samples)
    
    # Compute metrics
    print("\n[5/5] Computing metrics...")
    metrics = compute_uncertainty_metrics(results)
    
    print("\nUncertainty Quality Metrics:")
    for branch in ['diffusion', 'gan']:
        print(f"\n{branch.upper()}:")
        print(f"  Error-Uncertainty Correlation: {metrics[branch]['correlation']:.4f}")
        print(f"  Mean Uncertainty: {metrics[branch]['mean_uncertainty']:.4f}")
        print(f"  High-Error Region Uncertainty: {metrics[branch]['high_error_uncertainty']:.4f}")
        print(f"  Low-Error Region Uncertainty: {metrics[branch]['low_error_uncertainty']:.4f}")
        print(f"  Uncertainty Separation: {metrics[branch]['uncertainty_separation']:.4f}")
        print(f"  Mean Aleatoric: {metrics[branch]['mean_aleatoric']:.4f}")
        print(f"  Mean Epistemic: {metrics[branch]['mean_epistemic']:.4f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_uncertainty(results, output_dir)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
