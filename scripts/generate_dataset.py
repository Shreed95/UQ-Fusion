#!/usr/bin/env python
# scripts/generate_dataset.py

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import gc
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import json
from datetime import datetime

from data import BraTSSliceDataset
from models.vae import VAE, VAEConfig
from models.diffusion import LatentDiffusionModelSmall
from models.gan import STABLEGeneratorSmall
from models.uncertainty import UncertaintyAwareDualBranch, UncertaintyWrapperConfig
from models.fusion import create_fusion_module
from validation import (
    MetricsCalculator, MetricsConfig,
    QualityDecisionConfig,
    DatasetExpansionValidator
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Expanded Dataset')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--expansion_factor', type=int, default=2)
    parser.add_argument('--max_attempts_per_sample', type=int, default=3)
    parser.add_argument('--max_source_samples', type=int, default=None)
    parser.add_argument('--fusion_method', type=str, default='uncertainty')
    parser.add_argument('--num_mc_samples', type=int, default=5)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--diffusion_strength', type=float, default=0.8)
    parser.add_argument('--acceptance_threshold', type=float, default=0.5)  # safer default
    parser.add_argument('--output_dir', type=str, default='./outputs/expanded_dataset')
    parser.add_argument('--save_rejected', action='store_true')

    if torch.backends.mps.is_available():
        dd = 'mps'
    elif torch.cuda.is_available():
        dd = 'cuda'
    else:
        dd = 'cpu'
    parser.add_argument('--device', type=str, default=dd)
    return parser.parse_args()


def load_models(args, device):
    vae_ckpt = torch.load(args.vae_checkpoint, map_location='cpu')
    vae_cfg = vae_ckpt.get('config', {})
    vae = VAE(VAEConfig(
        in_channels=4, out_channels=4,
        latent_channels=vae_cfg.get('latent_channels', 4),
        base_channels=vae_cfg.get('base_channels', 64)))
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.to(device).eval()

    diff_ckpt = torch.load(args.diffusion_checkpoint, map_location='cpu')
    diff_cfg = diff_ckpt.get('config', {})
    diffusion = LatentDiffusionModelSmall(
        latent_channels=diff_cfg.get('latent_channels', 4),
        base_channels=diff_cfg.get('base_channels', 64),
        num_timesteps=diff_cfg.get('num_timesteps', 1000))
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion.to(device).eval()

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
    unc_config = UncertaintyWrapperConfig(
        num_mc_samples=args.num_mc_samples,
        normalize_uncertainty=False,
        estimate_aleatoric=False,
        lambda_mc_variance=0.0
    )
    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, unc_config).to(device)
    fusion_module = create_fusion_module(method=args.fusion_method).to(device)
    return dual_branch, fusion_module


@torch.no_grad()
def generate_expanded_dataset(dual_branch, fusion_module, dataloader,
                             validator, output_dir, device, args,
                             metrics_calculator):

    accepted_dir = output_dir / 'accepted'
    accepted_dir.mkdir(parents=True, exist_ok=True)

    original_count = len(dataloader.dataset)
    target_synthetic = original_count * (args.expansion_factor - 1)

    print(f"Original: {original_count} | Target synthetic: {target_synthetic}")

    accepted_count = 0
    rejected_count = 0
    total_generated = 0

    pbar = tqdm(desc="Generating (attempts)")

    epoch = 0
    max_total_attempts = target_synthetic * 10

    while accepted_count < target_synthetic:
        epoch += 1
        if epoch > args.max_attempts_per_sample:
            print(f"\nReached max attempts ({args.max_attempts_per_sample})")
            break

        for batch in dataloader:
            if accepted_count >= target_synthetic:
                break

            if total_generated > max_total_attempts:
                print("\nToo many rejections, stopping early.")
                break

            images = batch['image'].to(device)

            has_seg = 'segmentation' in batch
            if has_seg:
                segmentations = batch['segmentation']

            fusion_inputs = dual_branch.get_fusion_inputs(
                images,
                diffusion_steps=args.diffusion_steps,
                diffusion_strength=args.diffusion_strength
            )

            fusion_result = fusion_module(
                fusion_inputs['I_diff'],
                fusion_inputs['I_gan'],
                fusion_inputs['U_diff'],
                fusion_inputs['U_gan']
            )

            fused = fusion_result['fused']

            for i in range(images.shape[0]):
                total_generated += 1
                pbar.update(1)

                gen = fused[i:i+1]
                ref = images[i:i+1]
                unc = fusion_inputs['U_diff'][i].mean().item()

                # ✅ NEW: FULL METRICS
                metrics_list = metrics_calculator.compute_per_sample(gen, ref)
                metrics = metrics_list[0]

                decision = validator.decision_engine.make_decision(
                    metrics,
                    uncertainty=unc,
                    image_id=f"syn_{epoch}_{total_generated}"
                )

                # DEBUG
                if total_generated % 10 == 0:
                    print(f"[DEBUG] Total: {total_generated} | Accepted: {accepted_count} | Score: {decision['total_score']:.3f}")

                if decision['accepted']:
                    save_dict = {
                        'image': gen.cpu().numpy().squeeze(0),
                        'source': ref.cpu().numpy().squeeze(0),
                        'metrics': metrics,
                        'total_score': decision['total_score']
                    }

                    if has_seg:
                        seg = segmentations[i]
                        if seg.ndim == 3:
                            seg = seg.squeeze(0)
                        save_dict['segmentation'] = seg.cpu().numpy()

                    save_path = accepted_dir / f"synthetic_{accepted_count:06d}.npz"
                    np.savez_compressed(save_path, **save_dict)

                    accepted_count += 1
                else:
                    rejected_count += 1

            del images, fusion_inputs, fusion_result, fused
            gc.collect()
            if device.type == 'mps':
                torch.mps.empty_cache()

    pbar.close()

    return {
        'accepted': accepted_count,
        'rejected': rejected_count,
        'total_generated': total_generated,
        'acceptance_rate': accepted_count / max(1, total_generated),
        'epochs_used': epoch
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UQ-Fusion Dataset Expansion")
    print("=" * 60)

    print("\n[1/5] Loading models...")
    vae, diffusion, generator = load_models(args, device)

    print("\n[2/5] Creating pipeline...")
    dual_branch, fusion_module = create_pipeline(diffusion, generator, args, device)

    print("\n[3/5] Initializing validation...")
    decision_config = QualityDecisionConfig(
        acceptance_threshold=args.acceptance_threshold,
        log_path=str(output_dir / 'generation_log.csv')
    )
    validator = DatasetExpansionValidator(
        decision_config=decision_config,
        output_dir=str(output_dir)
    )

    print("\n[4/5] Initializing metrics...")
    metrics_calculator = MetricsCalculator(
        MetricsConfig(device=str(device))
    )

    print("\n[5/5] Loading training data...")
    data_dir = Path(args.data_dir)
    train_dataset = BraTSSliceDataset(
        data_dir / "slices",
        data_dir / "splits" / "train_metadata.json",
        augmentor=None,
        return_segmentation=True
    )

    if args.max_source_samples and args.max_source_samples < len(train_dataset):
        train_dataset = Subset(train_dataset, list(range(args.max_source_samples)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    print("\nGenerating...")
    start = datetime.now()

    results = generate_expanded_dataset(
        dual_branch,
        fusion_module,
        train_loader,
        validator,
        output_dir,
        device,
        args,
        metrics_calculator
    )

    elapsed = datetime.now() - start

    print("\n" + "=" * 60)
    print(f"Accepted: {results['accepted']} | Rejected: {results['rejected']}")
    print(f"Acceptance Rate: {results['acceptance_rate']*100:.2f}%")
    print(f"Time: {elapsed}")
    print("=" * 60)


if __name__ == "__main__":
    main()