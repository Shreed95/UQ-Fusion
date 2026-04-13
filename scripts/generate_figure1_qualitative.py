#!/usr/bin/env python
# scripts/generate_figure1_qualitative.py

"""
Generate Figure 1: Qualitative Results Grid for research paper.

Creates a 2x5 grid showing:
(a) Original, (b) Diffusion Output, (c) GAN Output, (d) Fusion Weights, (e) Fused Output

Usage:
    python scripts/generate_figure1_qualitative.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data \
        --output_path ./outputs/figures/figure1_qualitative.png \
        --device mps
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

from data import BraTSSliceDataset
from models.vae import VAE,VAEConfig
from models.diffusion import LatentDiffusionModelSmall
from models.gan import STABLEGeneratorSmall
from models.uncertainty import UncertaintyAwareDualBranch, UncertaintyWrapperConfig
from models.fusion import create_fusion_module


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Figure 1: Qualitative Results')
    
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_path', type=str, default='./outputs/figures/figure1_qualitative.png')
    parser.add_argument('--slice_indices', type=int, nargs='+', default=None,
                        help='Specific slice indices to use (default: auto-select)')
    parser.add_argument('--modality', type=int, default=3,
                        help='MRI modality to display (0=T1, 1=T1ce, 2=T2, 3=FLAIR)')
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--diffusion_strength', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dpi', type=int, default=300)
    
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
    generator.load_state_dict(gan_ckpt['generator_state_dict'], strict=False)
    generator.to(device)
    generator.eval()
    
    return vae, diffusion, generator


def select_representative_slices(dataset, device, num_slices=2):
    """Select representative slices with clear tumor and complex boundaries."""
    tumor_sizes = []
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        seg = sample['segmentation'].numpy()
        
        # Calculate tumor size (non-zero pixels in segmentation)
        tumor_pixels = (seg > 0).sum()
        total_pixels = seg.size
        tumor_fraction = tumor_pixels / total_pixels
        
        # Calculate boundary complexity (edge pixels / tumor pixels)
        if tumor_pixels > 100:  # Minimum tumor size
            # Simple edge detection using gradient
            dy = np.abs(np.diff(seg, axis=0, prepend=seg[0:1]))
            dx = np.abs(np.diff(seg, axis=1, prepend=seg[:, 0:1]))
            edges = dy + dx
            edge_pixels = (edges > 0).sum()
            complexity = edge_pixels / tumor_pixels
        else:
            complexity = 0
        
        tumor_sizes.append({
            'idx': idx,
            'tumor_fraction': tumor_fraction,
            'complexity': complexity,
            'has_all_classes': len(np.unique(seg)) >= 3
        })
    
    # Filter slices with reasonable tumor content
    valid_slices = [s for s in tumor_sizes if 0.02 < s['tumor_fraction'] < 0.3 and s['has_all_classes']]
    
    if len(valid_slices) < num_slices:
        valid_slices = sorted(tumor_sizes, key=lambda x: x['tumor_fraction'], reverse=True)[:20]
    
    # Sort by complexity
    valid_slices_sorted = sorted(valid_slices, key=lambda x: x['complexity'])
    
    # Select one with clear tumor (medium size, low complexity)
    clear_idx = len(valid_slices_sorted) // 4
    clear_tumor_idx = valid_slices_sorted[clear_idx]['idx']
    
    # Select one with complex boundary (high complexity)
    complex_idx = -len(valid_slices_sorted) // 4 - 1
    complex_boundary_idx = valid_slices_sorted[complex_idx]['idx']
    
    # Ensure different slices
    if clear_tumor_idx == complex_boundary_idx:
        complex_boundary_idx = valid_slices_sorted[-1]['idx']
    
    return [clear_tumor_idx, complex_boundary_idx]


@torch.no_grad()
def generate_outputs(image, diffusion, generator, fusion_module, dual_branch, args, device):
    """Generate all outputs for visualization."""
    image = image.to(device)
    
    # Get fusion inputs from dual branch
    fusion_inputs = dual_branch.get_fusion_inputs(
        image,
        diffusion_steps=args.diffusion_steps,
        diffusion_strength=args.diffusion_strength
    )
    
    diff_output = fusion_inputs['I_diff']
    gan_output = fusion_inputs['I_gan']
    diff_uncertainty = fusion_inputs['U_diff']
    gan_uncertainty = fusion_inputs['U_gan']
    
    # Compute fusion
    fusion_result = fusion_module(
        diff_output, gan_output,
        diff_uncertainty, gan_uncertainty
    )
    
    fused_output = fusion_result['fused']
    
    # Compute fusion weights (inverse uncertainty weighting)
    epsilon = 1e-6
    w_diff = 1.0 / (diff_uncertainty + epsilon)
    w_gan = 1.0 / (gan_uncertainty + epsilon)
    w_total = w_diff + w_gan
    fusion_weights = w_diff / w_total  # Weight for diffusion (0-1)
    
    return {
        'original': image,
        'diffusion': diff_output,
        'gan': gan_output,
        'weights': fusion_weights,
        'fused': fused_output,
        'diff_uncertainty': diff_uncertainty,
        'gan_uncertainty': gan_uncertainty
    }


def create_figure(all_outputs, args):
    """Create the 2x5 qualitative results figure."""
    
    modality = args.modality
    num_rows = len(all_outputs)
    num_cols = 5
    
    # Create figure
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, wspace=0.05, hspace=0.15)
    
    # Column titles
    col_titles = ['(a) Original', '(b) Diffusion', '(c) GAN', '(d) Fusion Weights', '(e) Fused']
    
    # Row labels
    row_labels = ['Clear Tumor', 'Complex Boundary']
    
    axes_list = []
    
    for row_idx, outputs in enumerate(all_outputs):
        # Extract single modality for visualization
        original = outputs['original'][0, modality].cpu().numpy()
        diffusion = outputs['diffusion'][0, modality].cpu().numpy()
        gan = outputs['gan'][0, modality].cpu().numpy()
        weights = outputs['weights'][0, 0].cpu().numpy()  # Fusion weight map
        fused = outputs['fused'][0, modality].cpu().numpy()
        
        images = [original, diffusion, gan, weights, fused]
        
        for col_idx, (img, title) in enumerate(zip(images, col_titles)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            axes_list.append(ax)
            
            if col_idx == 3:  # Fusion weights
                # Use diverging colormap for weights
                im = ax.imshow(img, cmap='RdYlBu', vmin=0, vmax=1)
                if row_idx == 0:
                    # Add colorbar only for first row
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Diff. Weight', fontsize=8)
                    cbar.ax.tick_params(labelsize=7)
            else:
                # Use grayscale for MRI images
                vmin, vmax = np.percentile(img, [2, 98])
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            
            ax.axis('off')
            
            # Add column titles to first row
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            
            # Add row labels
            if col_idx == 0:
                ax.text(-0.15, 0.5, row_labels[row_idx], transform=ax.transAxes,
                       fontsize=10, fontweight='bold', va='center', ha='right',
                       rotation=90)
    
    # Add main title
    fig.suptitle('UQ-Fusion: Qualitative Generation Results', fontsize=14, fontweight='bold', y=1.02)
    
    # Add caption/legend
    caption = "Fusion weights show per-pixel contribution from diffusion (blue) vs GAN (red). Yellow indicates balanced fusion."
    fig.text(0.5, -0.02, caption, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    return fig


def main():
    args = parse_args()
    
    # Setup device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("=" * 60)
    print("Generating Figure 1: Qualitative Results Grid")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n[1/4] Loading models...")
    vae, diffusion, generator = load_models(args, device)
    
    # Create dual branch and fusion module
    unc_config = UncertaintyWrapperConfig(num_mc_samples=5, normalize_uncertainty=True)
    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, unc_config)
    dual_branch = dual_branch.to(device)
    
    fusion_module = create_fusion_module(method='uncertainty')
    fusion_module = fusion_module.to(device)
    
    # Load test data
    print("\n[2/4] Loading test data...")
    data_dir = Path(args.data_dir)
    test_dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "test_metadata.json",
        augmentor=None,
        return_segmentation=True
    )
    
    # Select representative slices
    print("\n[3/4] Selecting representative slices...")
    if args.slice_indices:
        slice_indices = args.slice_indices[:2]
    else:
        slice_indices = select_representative_slices(test_dataset, device, num_slices=2)
    
    print(f"Selected slices: {slice_indices}")
    
    # Generate outputs for each slice
    print("\n[4/4] Generating outputs...")
    all_outputs = []
    for idx in slice_indices:
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0)
        outputs = generate_outputs(image, diffusion, generator, fusion_module, dual_branch, args, device)
        all_outputs.append(outputs)
    
    # Create figure
    fig = create_figure(all_outputs, args)
    
    # Save figure
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Also save as PDF for paper
    pdf_path = output_path.with_suffix('.pdf')
    fig = create_figure(all_outputs, args)
    fig.savefig(pdf_path, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"\n✓ Figure saved to: {output_path}")
    print(f"✓ PDF saved to: {pdf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
