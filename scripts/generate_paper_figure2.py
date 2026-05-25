#!/usr/bin/env python
# scripts/generate_paper_figure2.py

"""
Generate Fig. 2 for the UQ-Fusion paper (2 rows × 5 columns):
  Source FLAIR | Diffusion output | GAN output | Fused output | Fusion weight α

Usage:
    python scripts/generate_paper_figure2.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data1 \
        --device mps \
        --output_dir ./outputs/figures

    # Or specify exact indices:
    python scripts/generate_paper_figure2.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data1 \
        --device mps \
        --sample_indices 42 156
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import gc


def parse_args():
    parser = argparse.ArgumentParser(description='Generate paper Fig. 2')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data1')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                        help='2 specific slice indices. If None, auto-selects.')
    parser.add_argument('--num_mc_samples', type=int, default=10)
    parser.add_argument('--diffusion_strength', type=float, default=0.95)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./outputs/figures')
    parser.add_argument('--dpi', type=int, default=300)

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)
    return parser.parse_args()


def load_vae(checkpoint_path, device):
    """Load VAE model."""
    from models.vae import VAE, VAEConfig
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    model = VAE(VAEConfig(
        in_channels=4, out_channels=4,
        latent_channels=config.get('latent_channels', 4),
        base_channels=config.get('base_channels', 64)
    ))

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  VAE loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_diffusion(checkpoint_path, vae_checkpoint_path, device):
    """Load standard diffusion model."""
    from models.diffusion.diffusion import LatentDiffusionModel, LatentDiffusionConfig
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    diff_cfg = ckpt.get('config', {})

    diff_config_obj = LatentDiffusionConfig(
        latent_channels=diff_cfg.get('latent_channels', 4),
        base_channels=diff_cfg.get('base_channels', 64),
        num_timesteps=diff_cfg.get('num_timesteps', 1000)
    )
    model = LatentDiffusionModel(config=diff_config_obj)
    model.unet.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  Diffusion loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_gan(checkpoint_path, device):
    """Load GAN generator."""
    from models.gan import STABLEGeneratorSmall
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gan_cfg = ckpt.get('config', {})

    generator = STABLEGeneratorSmall(
        in_channels=gan_cfg.get('in_channels', 4),
        out_channels=gan_cfg.get('out_channels', 4),
        base_channels=gan_cfg.get('base_channels_g', 32),
        num_residual_blocks=gan_cfg.get('num_residual_blocks', 6)
    )

    generator.load_state_dict(ckpt['generator_state_dict'])
    generator.to(device).eval()
    print(f"  GAN loaded (epoch {ckpt.get('epoch', '?')})")
    return generator


@torch.no_grad()
def generate_diffusion_output(source, vae, diffusion_model, strength, steps, device):
    """Generate diffusion branch output via SDEdit."""
    source_dev = source.unsqueeze(0).to(device)

    latent = diffusion_model.encode(source_dev)

    num_timesteps = diffusion_model.num_timesteps if hasattr(diffusion_model, 'num_timesteps') else 1000
    start_step = int(num_timesteps * strength)

    betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    t = torch.tensor([start_step - 1], device=device)
    noise = torch.randn_like(latent)
    sqrt_alpha = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    noisy_latent = sqrt_alpha * latent + sqrt_one_minus * noise

    step_size = max(1, start_step // steps)
    current = noisy_latent
    for i in range(start_step - 1, -1, -step_size):
        t_tensor = torch.tensor([i], device=device)
        pred_noise = diffusion_model.unet(current, t_tensor, latent)

        alpha_t = alphas_cumprod[i]
        alpha_prev = alphas_cumprod[i - step_size] if i >= step_size else torch.tensor(1.0, device=device)

        pred_x0 = (current - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
        pred_x0 = pred_x0.clamp(-3, 3)

        if i > step_size:
            noise_new = torch.randn_like(current)
            current = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * noise_new
        else:
            current = pred_x0

    decoded = diffusion_model.decode(current)
    return decoded.squeeze(0).cpu().clamp(0, 1)


@torch.no_grad()
def generate_gan_output(source, generator, device):
    """Generate GAN branch output."""
    source_dev = source.unsqueeze(0).to(device)
    output = generator(source_dev)
    return output.squeeze(0).cpu().clamp(0, 1)


@torch.no_grad()
def compute_fusion(source, diff_out, gan_out):
    """Compute uncertainty-guided fusion and per-pixel alpha map."""
    u_diff = ((diff_out - source) ** 2).mean(dim=0)
    u_gan = ((gan_out - source) ** 2).mean(dim=0)

    all_u = torch.cat([u_diff.flatten(), u_gan.flatten()])
    p2 = torch.quantile(all_u, 0.02)
    p98 = torch.quantile(all_u, 0.98)
    denom = (p98 - p2).clamp(min=1e-8)

    u_diff_norm = ((u_diff - p2) / denom).clamp(0, 1)
    u_gan_norm = ((u_gan - p2) / denom).clamp(0, 1)

    eps = 1e-6
    w_diff = 1.0 / (u_diff_norm + eps)
    w_gan = 1.0 / (u_gan_norm + eps)
    alpha = w_diff / (w_diff + w_gan)

    fused = alpha.unsqueeze(0) * diff_out + (1 - alpha.unsqueeze(0)) * gan_out
    return fused.clamp(0, 1), alpha


def find_good_samples(dataset, num_samples=2, num_candidates=40):
    """Find slices with visible tumor and diverse tumor ratios."""
    candidates = []

    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_candidates, len(dataset)), replace=False)
    for idx in indices:
        sample = dataset[int(idx)]
        seg = sample['segmentation']
        if isinstance(seg, torch.Tensor):
            seg = seg.numpy()
        tumor_pixels = (seg > 0).sum()
        total_pixels = seg.size
        ratio = tumor_pixels / total_pixels

        if 0.05 < ratio < 0.30:
            candidates.append({'idx': int(idx), 'ratio': ratio})

    candidates.sort(key=lambda c: c['ratio'])
    if len(candidates) >= 2:
        small = candidates[len(candidates) // 4]
        large = candidates[3 * len(candidates) // 4]
        selected = [small, large]
    elif len(candidates) == 1:
        selected = [candidates[0], candidates[0]]
    else:
        selected = [{'idx': 0, 'ratio': 0}, {'idx': 1, 'ratio': 0}]

    for i, s in enumerate(selected):
        print(f"  Sample {i+1}: idx={s['idx']}, tumor ratio={s['ratio']:.1%}")

    return [s['idx'] for s in selected]


def create_figure(rows_data, output_path, dpi=300):
    """Create publication-quality 2-row × 5-column figure with shared colorbar."""
    num_rows = len(rows_data)

    colors_list = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                   '#ffffff',
                   '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    fusion_cmap = LinearSegmentedColormap.from_list('fusion', colors_list, N=256)

    fig = plt.figure(figsize=(18, num_rows * 3.8 + 0.8))
    gs = gridspec.GridSpec(num_rows, 6,
                           width_ratios=[1, 1, 1, 1, 1, 0.06],
                           wspace=0.06, hspace=0.15,
                           left=0.03, right=0.95,
                           top=0.92, bottom=0.04)

    titles = ['Source (FLAIR)', 'Diffusion Branch', 'GAN Branch',
              'UQ-Fusion Output', 'Fusion Weight α']

    last_im = None  # for shared colorbar

    for row_i, data in enumerate(rows_data):
        source_img = data['source']
        diff_img = data['diff']
        gan_img = data['gan']
        fused_img = data['fused']
        alpha_map = data['alpha']
        seg_mask = data['seg']

        flair_source = source_img[3].numpy()
        flair_diff = diff_img[3].numpy()
        flair_gan = gan_img[3].numpy()
        flair_fused = fused_img[3].numpy()
        alpha_np = alpha_map.numpy()

        if isinstance(seg_mask, torch.Tensor):
            seg_np = seg_mask.numpy()
        else:
            seg_np = seg_mask

        images = [flair_source, flair_diff, flair_gan, flair_fused]

        # Compute PSNR for annotations
        diff_psnr = 10 * np.log10(1.0 / ((diff_img - source_img) ** 2).mean().item())
        gan_psnr = 10 * np.log10(1.0 / ((gan_img - source_img) ** 2).mean().item())
        fused_psnr = 10 * np.log10(1.0 / ((fused_img - source_img) ** 2).mean().item())
        psnr_vals = [None, diff_psnr, gan_psnr, fused_psnr]

        # Panels 1-4: grayscale images
        for col_i, (img, title) in enumerate(zip(images, titles[:4])):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)

            if col_i == 0 and seg_np is not None:
                tumor_mask = (seg_np > 0).astype(float)
                if tumor_mask.sum() > 0:
                    ax.contour(tumor_mask, levels=[0.5], colors='lime',
                               linewidths=0.8, alpha=0.7)

            if row_i == 0:
                ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

            if col_i == 0:
                ax.text(-0.08, 0.5, f'Case {row_i+1}', transform=ax.transAxes,
                        fontsize=10, fontweight='bold', rotation=90,
                        verticalalignment='center', horizontalalignment='center')

            # PSNR annotation
            if psnr_vals[col_i] is not None:
                ax.text(0.03, 0.97, f'{psnr_vals[col_i]:.1f} dB',
                        transform=ax.transAxes, fontsize=7.5, color='white',
                        fontweight='bold', verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                  alpha=0.7, edgecolor='none'))

            ax.axis('off')

        # Panel 5: alpha fusion weight map
        ax5 = fig.add_subplot(gs[row_i, 4])
        last_im = ax5.imshow(alpha_np, cmap=fusion_cmap, vmin=0, vmax=1)

        if seg_np is not None:
            tumor_mask = (seg_np > 0).astype(float)
            if tumor_mask.sum() > 0:
                ax5.contour(tumor_mask, levels=[0.5], colors='black',
                            linewidths=0.8, linestyles='dashed', alpha=0.8)

        if row_i == 0:
            ax5.set_title(titles[4], fontsize=10, fontweight='bold', pad=6)

        ax5.text(0.03, 0.97, f'α={alpha_np.mean():.2f}', transform=ax5.transAxes,
                 fontsize=7.5, color='white', fontweight='bold',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                           alpha=0.7, edgecolor='none'))
        ax5.axis('off')

    # Shared colorbar spanning both rows
    cax = fig.add_subplot(gs[:, 5])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(['GAN\ndominates', 'Equal', 'Diffusion\ndominates'],
                         fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path.with_suffix('.png'), dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(output_path.with_suffix('.pdf'), dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"\nFigure saved:")
    print(f"  PNG: {output_path.with_suffix('.png')}")
    print(f"  PDF: {output_path.with_suffix('.pdf')}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Paper Figure 2 (2 rows)")
    print("=" * 60)

    # Load models
    print("\n[1/5] Loading models...")
    vae = load_vae(args.vae_checkpoint, str(device))
    diffusion = load_diffusion(args.diffusion_checkpoint, args.vae_checkpoint, str(device))
    diffusion.set_vae(vae)
    gan = load_gan(args.gan_checkpoint, str(device))

    # Load data
    print("\n[2/5] Loading test data...")
    from data import BraTSSliceDataset
    data_dir = Path(args.data_dir)
    dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "test_metadata.json",
        augmentor=None,
        return_segmentation=True
    )
    print(f"  Test samples: {len(dataset)}")

    # Select 2 samples
    print("\n[3/5] Selecting samples...")
    if args.sample_indices is not None:
        indices = args.sample_indices[:2]
        print(f"  Using specified indices: {indices}")
    else:
        print("  Auto-selecting 2 samples with visible tumor...")
        indices = find_good_samples(dataset, num_samples=2)

    # Generate outputs for each sample
    rows_data = []
    for i, idx in enumerate(indices):
        print(f"\n[4/5] Processing sample {i+1}/2 (index {idx})...")
        sample = dataset[idx]
        source = sample['image'].float()
        seg = sample['segmentation']

        print(f"  Generating diffusion output...")
        diff_out = generate_diffusion_output(
            source, vae, diffusion,
            strength=args.diffusion_strength,
            steps=args.diffusion_steps,
            device=device
        )

        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

        print(f"  Generating GAN output...")
        gan_out = generate_gan_output(source, gan, device)

        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

        print(f"  Computing fusion...")
        fused, alpha = compute_fusion(source, diff_out, gan_out)

        diff_psnr = 10 * np.log10(1.0 / ((diff_out - source) ** 2).mean().item())
        gan_psnr = 10 * np.log10(1.0 / ((gan_out - source) ** 2).mean().item())
        fused_psnr = 10 * np.log10(1.0 / ((fused - source) ** 2).mean().item())
        print(f"  Diff PSNR: {diff_psnr:.2f} | GAN PSNR: {gan_psnr:.2f} | "
              f"Fused PSNR: {fused_psnr:.2f} | α mean: {alpha.mean():.3f}")

        rows_data.append({
            'source': source, 'diff': diff_out, 'gan': gan_out,
            'fused': fused, 'alpha': alpha, 'seg': seg
        })

    # Create figure
    print("\n[5/5] Creating figure...")
    create_figure(
        rows_data=rows_data,
        output_path=output_dir / 'fig2_qualitative_comparison',
        dpi=args.dpi
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Quality Summary:")
    for i, data in enumerate(rows_data):
        s, d, g, f = data['source'], data['diff'], data['gan'], data['fused']
        dp = 10 * np.log10(1.0 / ((d - s) ** 2).mean().item())
        gp = 10 * np.log10(1.0 / ((g - s) ** 2).mean().item())
        fp = 10 * np.log10(1.0 / ((f - s) ** 2).mean().item())
        print(f"  Case {i+1} (idx {indices[i]}): "
              f"Diff {dp:.2f} dB | GAN {gp:.2f} dB | Fused {fp:.2f} dB | "
              f"α={data['alpha'].mean():.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()