#!/usr/bin/env python
# scripts/compute_branch_ssim.py

"""
Compute PSNR and SSIM for each generation branch + fused output.
Fills in the missing SSIM values for the paper's Table II.

Usage:
    python scripts/compute_branch_ssim.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data1 \
        --num_samples 50 \
        --device mps
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim_fn
import json
import gc


def parse_args():
    parser = argparse.ArgumentParser(description='Compute branch SSIM for paper table')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data1')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of test samples to evaluate')
    parser.add_argument('--diffusion_strength', type=float, default=0.95)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/branch_metrics')

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)
    return parser.parse_args()


def load_vae(checkpoint_path, device):
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
    return model


def load_diffusion(checkpoint_path, device):
    from models.diffusion.diffusion import LatentDiffusionModel, LatentDiffusionConfig
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    diff_cfg = ckpt.get('config', {})
    config_obj = LatentDiffusionConfig(
        latent_channels=diff_cfg.get('latent_channels', 4),
        base_channels=diff_cfg.get('base_channels', 64),
        num_timesteps=diff_cfg.get('num_timesteps', 1000)
    )
    model = LatentDiffusionModel(config=config_obj)
    model.unet.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


def load_gan(checkpoint_path, device):
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
    return generator


def compute_psnr(img1, img2):
    """Compute PSNR between two numpy arrays in [0,1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 50.0
    return float(10 * np.log10(1.0 / mse))


def compute_ssim(img1, img2):
    """
    Compute SSIM between two multi-channel images.
    img1, img2: (C, H, W) numpy arrays in [0, 1]
    Returns mean SSIM across channels.
    """
    c, h, w = img1.shape
    ssim_vals = []
    for ch in range(c):
        val = ssim_fn(img1[ch], img2[ch], data_range=1.0)
        ssim_vals.append(val)
    return float(np.mean(ssim_vals))


@torch.no_grad()
def generate_diffusion(source, diffusion_model, strength, steps, device):
    """Generate diffusion output via SDEdit."""
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
def generate_gan(source, generator, device):
    """Generate GAN output."""
    source_dev = source.unsqueeze(0).to(device)
    output = generator(source_dev)
    return output.squeeze(0).cpu().clamp(0, 1)


@torch.no_grad()
def compute_fusion(source, diff_out, gan_out):
    """Compute uncertainty-guided fused output."""
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
    return fused.clamp(0, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Branch PSNR + SSIM Computation (for Paper Table)")
    print("=" * 60)

    # Load models
    print("\n[1/3] Loading models...")
    vae = load_vae(args.vae_checkpoint, str(device))
    diffusion = load_diffusion(args.diffusion_checkpoint, str(device))
    diffusion.set_vae(vae)
    gan = load_gan(args.gan_checkpoint, str(device))
    print("  All models loaded.")

    # Load data
    print("\n[2/3] Loading test data...")
    from data import BraTSSliceDataset
    data_dir = Path(args.data_dir)
    dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "test_metadata.json",
        augmentor=None,
        return_segmentation=False
    )
    num_samples = min(args.num_samples, len(dataset))
    print(f"  Test samples: {len(dataset)}, evaluating: {num_samples}")

    # Evaluate
    print(f"\n[3/3] Evaluating {num_samples} samples...")
    metrics = {
        'vae': {'psnr': [], 'ssim': []},
        'diffusion': {'psnr': [], 'ssim': []},
        'gan': {'psnr': [], 'ssim': []},
        'fused': {'psnr': [], 'ssim': []}
    }

    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample = dataset[i]
        source = sample['image'].float()
        source_np = source.numpy()

        # VAE reconstruction
        source_dev = source.unsqueeze(0).to(device)
        with torch.no_grad():
            vae_out = vae(source_dev)
            if isinstance(vae_out, dict):
                vae_recon = vae_out['reconstruction']
            else:
                vae_recon = vae_out[0]
        vae_recon = vae_recon.squeeze(0).cpu().clamp(0, 1)
        vae_np = vae_recon.numpy()
        metrics['vae']['psnr'].append(compute_psnr(source_np, vae_np))
        metrics['vae']['ssim'].append(compute_ssim(source_np, vae_np))

        # Diffusion
        diff_out = generate_diffusion(source, diffusion, args.diffusion_strength,
                                       args.diffusion_steps, device)
        diff_np = diff_out.numpy()
        metrics['diffusion']['psnr'].append(compute_psnr(source_np, diff_np))
        metrics['diffusion']['ssim'].append(compute_ssim(source_np, diff_np))

        # GAN
        gan_out = generate_gan(source, gan, device)
        gan_np = gan_out.numpy()
        metrics['gan']['psnr'].append(compute_psnr(source_np, gan_np))
        metrics['gan']['ssim'].append(compute_ssim(source_np, gan_np))

        # Fused
        fused = compute_fusion(source, diff_out, gan_out)
        fused_np = fused.numpy()
        metrics['fused']['psnr'].append(compute_psnr(source_np, fused_np))
        metrics['fused']['ssim'].append(compute_ssim(source_np, fused_np))

        # Memory cleanup
        del source_dev, vae_recon, diff_out, gan_out, fused
        if device.type == 'mps' and i % 10 == 0:
            gc.collect()
            torch.mps.empty_cache()

    # Results
    print("\n" + "=" * 70)
    print("RESULTS — Copy these into your paper table")
    print("=" * 70)
    print(f"\n{'Stage':<30} {'PSNR (dB)':<20} {'SSIM':<20}")
    print("-" * 70)

    results = {}
    for branch, label in [('vae', 'VAE reconstruction'),
                           ('diffusion', 'Latent diffusion branch'),
                           ('gan', 'STABLE-GAN branch'),
                           ('fused', 'UQ-Fusion (fused)')]:
        psnr_mean = np.mean(metrics[branch]['psnr'])
        psnr_std = np.std(metrics[branch]['psnr'])
        ssim_mean = np.mean(metrics[branch]['ssim'])
        ssim_std = np.std(metrics[branch]['ssim'])

        print(f"{label:<30} {psnr_mean:.2f} ± {psnr_std:.2f}       "
              f"{ssim_mean:.4f} ± {ssim_std:.4f}")

        results[branch] = {
            'psnr_mean': float(psnr_mean), 'psnr_std': float(psnr_std),
            'ssim_mean': float(ssim_mean), 'ssim_std': float(ssim_std),
            'num_samples': num_samples
        }

    print("-" * 70)

    # LaTeX-ready output
    print("\nLaTeX-ready table rows:")
    print(f"VAE reconstruction (ceiling) & "
          f"${results['vae']['psnr_mean']:.2f} \\pm {results['vae']['psnr_std']:.2f}$ & "
          f"${results['vae']['ssim_mean']:.3f} \\pm {results['vae']['ssim_std']:.3f}$ & "
          f"upper bound \\\\")
    print(f"Latent diffusion branch & "
          f"${results['diffusion']['psnr_mean']:.2f} \\pm {results['diffusion']['psnr_std']:.2f}$ & "
          f"${results['diffusion']['ssim_mean']:.3f} \\pm {results['diffusion']['ssim_std']:.3f}$ & "
          f"diverse \\\\")
    print(f"STABLE-GAN branch & "
          f"${results['gan']['psnr_mean']:.2f} \\pm {results['gan']['psnr_std']:.2f}$ & "
          f"${results['gan']['ssim_mean']:.3f} \\pm {results['gan']['ssim_std']:.3f}$ & "
          f"high fidelity \\\\")
    print(f"\\textbf{{UQ-Fusion (ours)}} & "
          f"$\\mathbf{{{results['fused']['psnr_mean']:.2f} \\pm {results['fused']['psnr_std']:.2f}}}$ & "
          f"$\\mathbf{{{results['fused']['ssim_mean']:.3f} \\pm {results['fused']['ssim_std']:.3f}}}$ & "
          f"adaptive \\\\")

    # Save
    with open(output_dir / 'branch_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_dir / 'branch_metrics.json'}")


if __name__ == "__main__":
    main()