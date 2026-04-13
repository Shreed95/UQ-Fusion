#!/usr/bin/env python
# scripts/evaluate_quality.py

"""
Evaluate quality of generated images and make accept/reject decisions.

Usage:
    python scripts/evaluate_quality.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data --device mps
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import BraTSSliceDataset
from models.vae import VAE, VAEConfig
from models.diffusion import LatentDiffusionModelSmall
from models.gan import STABLEGeneratorSmall
from models.uncertainty import UncertaintyAwareDualBranch, UncertaintyWrapperConfig
from models.fusion import UQFusionModule, create_fusion_module
from validation import (
    MetricsCalculator, MetricsConfig,
    StatisticalValidator, ValidationThresholds,
    QualityDecisionEngine, QualityDecisionConfig, QualityWeights,
    DatasetExpansionValidator
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Image Quality')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--acceptance_threshold', type=float, default=0.70)
    parser.add_argument('--psnr_weight', type=float, default=0.20)
    parser.add_argument('--ssim_weight', type=float, default=0.25)
    parser.add_argument('--uncertainty_weight', type=float, default=0.20)
    parser.add_argument('--fusion_method', type=str, default='uncertainty')
    parser.add_argument('--num_mc_samples', type=int, default=5)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/quality')
    parser.add_argument('--num_samples', type=int, default=50)

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)
    return parser.parse_args()


def load_models(args, device):
    """Load all models with proper config from checkpoints."""
    # VAE
    vae_ckpt = torch.load(args.vae_checkpoint, map_location='cpu')
    vae_cfg = vae_ckpt.get('config', {})
    vae = VAE(VAEConfig(
        in_channels=4, out_channels=4,
        latent_channels=vae_cfg.get('latent_channels', 4),
        base_channels=vae_cfg.get('base_channels', 64)))
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.to(device).eval()

    # Diffusion
    diff_ckpt = torch.load(args.diffusion_checkpoint, map_location='cpu')
    diff_cfg = diff_ckpt.get('config', {})
    diffusion = LatentDiffusionModelSmall(
        latent_channels=diff_cfg.get('latent_channels', 4),
        base_channels=diff_cfg.get('base_channels', 64),
        num_timesteps=diff_cfg.get('num_timesteps', 1000))
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.to(device).eval()

    # GAN — load as-trained
    gan_ckpt = torch.load(args.gan_checkpoint, map_location='cpu')
    gan_cfg = gan_ckpt.get('config', {})
    generator = STABLEGeneratorSmall(
        in_channels=gan_cfg.get('in_channels', 4),
        out_channels=gan_cfg.get('out_channels', 4),
        base_channels=gan_cfg.get('base_channels_g', 32),
        num_residual_blocks=gan_cfg.get('num_residual_blocks', 6))
    generator.load_state_dict(gan_ckpt['generator_state_dict'])
    generator.to(device).eval()

    return vae, diffusion, generator


def create_pipeline(diffusion, generator, args, device):
    """Create uncertainty + fusion pipeline."""
    unc_config = UncertaintyWrapperConfig(
        num_mc_samples=args.num_mc_samples,
        normalize_uncertainty=False,
        lambda_mc_variance=0.0)
    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, unc_config).to(device)

    fusion_module = create_fusion_module(method=args.fusion_method).to(device)
    return dual_branch, fusion_module


@torch.no_grad()
def evaluate_quality(dual_branch, fusion_module, dataloader, metrics_calculator,
                     decision_engine, device, args, num_samples):
    """Evaluate quality of generated images."""
    all_decisions = []
    processed = 0

    for batch in tqdm(dataloader, desc="Evaluating quality"):
        if processed >= num_samples:
            break

        images = batch['image'].to(device)

        fusion_inputs = dual_branch.get_fusion_inputs(
            images, diffusion_steps=args.diffusion_steps, diffusion_strength=0.8)

        fusion_result = fusion_module(
            fusion_inputs['I_diff'], fusion_inputs['I_gan'],
            fusion_inputs['U_diff'], fusion_inputs['U_gan'])

        fused = fusion_result['fused']

        for i in range(images.shape[0]):
            if processed >= num_samples:
                break

            gen = fused[i:i+1]
            ref = images[i:i+1]
            unc = fusion_inputs['U_diff'][i:i+1].mean().item()

            psnr = metrics_calculator.psnr(gen, ref).item()
            ssim = metrics_calculator.ssim(gen, ref).item()
            mae = metrics_calculator.mae(gen, ref).item()
            nrmse = metrics_calculator.nrmse(gen, ref).item()

            metrics = {'psnr': psnr, 'ssim': ssim, 'mae': mae, 'nrmse': nrmse}
            decision = decision_engine.make_decision(metrics, uncertainty=unc,
                                                      image_id=f"sample_{processed}")
            all_decisions.append(decision)
            processed += 1

        del images, fusion_inputs, fusion_result, fused
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()

    return all_decisions


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Statistical Quality Validation")
    print("=" * 60)
    print(f"Threshold: {args.acceptance_threshold} | Device: {device}")
    print("=" * 60)

    print("\n[1/6] Loading models...")
    vae, diffusion, generator = load_models(args, device)

    print("\n[2/6] Creating pipeline...")
    dual_branch, fusion_module = create_pipeline(diffusion, generator, args, device)

    print("\n[3/6] Initializing metrics...")
    metrics_calculator = MetricsCalculator(MetricsConfig(device=str(device)))

    weights = QualityWeights(psnr=args.psnr_weight, ssim=args.ssim_weight,
                              uncertainty=args.uncertainty_weight)
    decision_config = QualityDecisionConfig(
        acceptance_threshold=args.acceptance_threshold, weights=weights,
        log_path=str(output_dir / 'decisions.csv'))
    decision_engine = QualityDecisionEngine(decision_config)

    print("\n[4/6] Loading data...")
    data_dir = Path(args.data_dir)
    test_ds = BraTSSliceDataset(
        data_dir / "slices", data_dir / "splits" / "test_metadata.json",
        augmentor=None, return_segmentation=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    print("\n[5/6] Evaluating quality...")
    decisions = evaluate_quality(dual_branch, fusion_module, test_loader,
                                  metrics_calculator, decision_engine, device, args, args.num_samples)

    print("\n[6/6] Computing statistics...")
    stats = decision_engine.get_statistics()
    rejection_analysis = decision_engine.get_rejection_analysis()

    print(f"\nResults:")
    print(f"  Total: {stats['total_processed']} | Accepted: {stats['accepted']} | "
          f"Rejected: {stats['rejected']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']*100:.1f}%")
    if 'score_mean' in stats:
        print(f"  Score: {stats['score_mean']:.4f} ± {stats['score_std']:.4f} "
              f"(range: {stats['score_min']:.4f}–{stats['score_max']:.4f})")

    decision_engine.save_report(str(output_dir / 'validation_report.json'))

    summary = {'statistics': stats, 'rejection_analysis': rejection_analysis}
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()