#!/usr/bin/env python
# scripts/generate_dataset.py

"""
Generate and validate expanded dataset using UQ-Fusion.

Usage:
    python scripts/generate_dataset.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data \
        --expansion_factor 2 \
        --acceptance_threshold 0.70
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
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
    MetricsCalculator,
    MetricsConfig,
    QualityDecisionEngine,
    QualityDecisionConfig,
    QualityWeights,
    DatasetExpansionValidator
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Expanded Dataset')
    
    # Checkpoints
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Generation settings
    parser.add_argument('--expansion_factor', type=int, default=2,
                        help='Target expansion factor (e.g., 2 = double the dataset)')
    parser.add_argument('--max_attempts_per_sample', type=int, default=3,
                        help='Max generation attempts per original sample')
    
    # Fusion settings
    parser.add_argument('--fusion_method', type=str, default='uncertainty')
    parser.add_argument('--num_mc_samples', type=int, default=5)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    parser.add_argument('--diffusion_strength', type=float, default=0.8)
    
    # Quality settings
    parser.add_argument('--acceptance_threshold', type=float, default=0.70)
    parser.add_argument('--psnr_min', type=float, default=25.0)
    parser.add_argument('--ssim_min', type=float, default=0.80)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/expanded_dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_rejected', action='store_true',
                        help='Save rejected images for analysis')
    
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


def create_pipeline(diffusion, generator, args, device):
    """Create generation pipeline."""
    unc_config = UncertaintyWrapperConfig(
        num_mc_samples=args.num_mc_samples,
        normalize_uncertainty=True
    )
    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, unc_config)
    dual_branch = dual_branch.to(device)
    
    fusion_module = create_fusion_module(method=args.fusion_method)
    fusion_module = fusion_module.to(device)
    
    return dual_branch, fusion_module


@torch.no_grad()
def generate_expanded_dataset(
    dual_branch,
    fusion_module,
    dataloader,
    metrics_calculator,
    validator,
    output_dir,
    device,
    args
):
    """Generate and validate expanded dataset."""
    
    accepted_dir = output_dir / 'accepted'
    rejected_dir = output_dir / 'rejected'
    accepted_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rejected:
        rejected_dir.mkdir(parents=True, exist_ok=True)
    
    original_count = len(dataloader.dataset)
    target_synthetic = original_count * (args.expansion_factor - 1)
    
    print(f"Original dataset size: {original_count}")
    print(f"Target synthetic images: {target_synthetic}")
    
    accepted_count = 0
    rejected_count = 0
    total_generated = 0
    
    # Track per-original statistics
    generation_stats = []
    
    pbar = tqdm(total=target_synthetic, desc="Generating")
    
    epoch = 0
    while accepted_count < target_synthetic:
        epoch += 1
        if epoch > args.max_attempts_per_sample:
            print(f"\nReached max attempts ({args.max_attempts_per_sample})")
            break
        
        for batch_idx, batch in enumerate(dataloader):
            if accepted_count >= target_synthetic:
                break
            
            images = batch['image'].to(device)
            # Get segmentation masks (keep on CPU for saving)
            segmentations = batch['segmentation']
            
            # Generate with fusion
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
            
            # Validate and save each sample
            for i in range(images.shape[0]):
                if accepted_count >= target_synthetic:
                    break
                
                total_generated += 1
                
                gen = fused[i:i+1]
                ref = images[i:i+1]
                seg = segmentations[i]  # Get corresponding segmentation
                unc = fusion_inputs['U_diff'][i].mean().item()
                
                # Compute metrics
                psnr = metrics_calculator.psnr(gen, ref).item()
                ssim = metrics_calculator.ssim(gen, ref).item()
                mae = metrics_calculator.mae(gen, ref).item()
                nrmse = metrics_calculator.nrmse(gen, ref).item()
                
                metrics = {
                    'psnr': psnr,
                    'ssim': ssim,
                    'mae': mae,
                    'nrmse': nrmse
                }
                
                # Make decision
                decision = validator.decision_engine.make_decision(
                    metrics,
                    uncertainty=unc,
                    image_id=f"syn_{epoch}_{batch_idx}_{i}"
                )
                
                if decision['accepted']:
                    # Save accepted image WITH SEGMENTATION
                    save_path = accepted_dir / f"synthetic_{accepted_count:06d}.npz"
                    np.savez_compressed(
                        save_path,
                        image=gen.cpu().numpy(),
                        segmentation=seg.cpu().numpy(),  # SAVE SEGMENTATION!
                        source=ref.cpu().numpy(),
                        metrics=metrics,
                        uncertainty=unc,
                        quality_score=decision['total_score']
                    )
                    accepted_count += 1
                    pbar.update(1)
                    
                else:
                    rejected_count += 1
                    
                    if args.save_rejected:
                        save_path = rejected_dir / f"rejected_{rejected_count:06d}.npz"
                        np.savez_compressed(
                            save_path,
                            image=gen.cpu().numpy(),
                            segmentation=seg.cpu().numpy(),  # Also save for rejected
                            source=ref.cpu().numpy(),
                            metrics=metrics,
                            rejection_reason=decision['rejection_reason']
                        )
    
    pbar.close()
    
    return {
        'accepted': accepted_count,
        'rejected': rejected_count,
        'total_generated': total_generated,
        'acceptance_rate': accepted_count / total_generated if total_generated > 0 else 0,
        'epochs_used': epoch
    }


def main():
    args = parse_args()
    
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("UQ-Fusion Dataset Expansion")
    print("=" * 60)
    print(f"Expansion factor: {args.expansion_factor}x")
    print(f"Acceptance threshold: {args.acceptance_threshold}")
    print(f"Fusion method: {args.fusion_method}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load models
    print("\n[1/5] Loading models...")
    vae, diffusion, generator = load_models(args, str(device))
    
    # Create pipeline
    print("\n[2/5] Creating generation pipeline...")
    dual_branch, fusion_module = create_pipeline(diffusion, generator, args, device)
    
    # Create metrics and validator
    print("\n[3/5] Initializing validation...")
    metrics_config = MetricsConfig(device=str(device))
    metrics_calculator = MetricsCalculator(metrics_config)
    
    decision_config = QualityDecisionConfig(
        acceptance_threshold=args.acceptance_threshold,
        log_path=str(output_dir / 'generation_log.csv')
    )
    validator = DatasetExpansionValidator(
        decision_config=decision_config,
        output_dir=str(output_dir)
    )
    
    # Load data - WITH SEGMENTATION!
    print("\n[4/5] Loading training data...")
    data_dir = Path(args.data_dir)
    
    train_dataset = BraTSSliceDataset(
        slices_dir=data_dir / "slices",
        metadata_file=data_dir / "splits" / "train_metadata.json",
        augmentor=None,
        return_segmentation=True  # CHANGED: Request segmentation!
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Generate
    print("\n[5/5] Generating expanded dataset...")
    start_time = datetime.now()
    
    results = generate_expanded_dataset(
        dual_branch, fusion_module, train_loader,
        metrics_calculator, validator, output_dir, device, args
    )
    
    elapsed = datetime.now() - start_time
    
    # Get final statistics
    stats = validator.decision_engine.get_statistics()
    
    print("\n" + "=" * 60)
    print("Dataset Expansion Complete!")
    print("=" * 60)
    print(f"Original dataset: {len(train_dataset)} samples")
    print(f"Synthetic accepted: {results['accepted']} samples")
    print(f"Synthetic rejected: {results['rejected']} samples")
    print(f"Total generated: {results['total_generated']} samples")
    print(f"Acceptance rate: {results['acceptance_rate']*100:.1f}%")
    print(f"Expansion achieved: {1 + results['accepted']/len(train_dataset):.2f}x")
    print(f"Time elapsed: {elapsed}")
    print("=" * 60)
    
    # Save summary
    summary = {
        'config': {
            'expansion_factor': args.expansion_factor,
            'acceptance_threshold': args.acceptance_threshold,
            'fusion_method': args.fusion_method,
            'diffusion_steps': args.diffusion_steps,
            'diffusion_strength': args.diffusion_strength
        },
        'results': {
            'original_count': len(train_dataset),
            'synthetic_accepted': results['accepted'],
            'synthetic_rejected': results['rejected'],
            'total_generated': results['total_generated'],
            'acceptance_rate': results['acceptance_rate'],
            'expansion_achieved': 1 + results['accepted']/len(train_dataset),
            'epochs_used': results['epochs_used']
        },
        'quality_statistics': stats,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed.total_seconds()
    }
    
    with open(output_dir / 'expansion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    validator.decision_engine.save_report(str(output_dir / 'validation_report.json'))
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
