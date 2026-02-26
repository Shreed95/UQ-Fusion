#!/usr/bin/env python
# scripts/evaluate_quality.py

"""
Script to evaluate quality of generated images and make accept/reject decisions.

Usage:
    python scripts/evaluate_quality.py \
        --diffusion_checkpoint ./outputs/checkpoints/diffusion/best.pth \
        --gan_checkpoint ./outputs/checkpoints/gan/best.pth \
        --vae_checkpoint ./outputs/checkpoints/vae/best.pth \
        --data_dir ./data \
        --acceptance_threshold 0.70
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
from models.fusion import UQFusionModule, create_fusion_module
from validation import (
    MetricsCalculator,
    MetricsConfig,
    StatisticalValidator,
    ValidationThresholds,
    QualityDecisionEngine,
    QualityDecisionConfig,
    QualityWeights,
    BaselineStatistics,
    DatasetExpansionValidator
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Image Quality')
    
    # Checkpoints
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--gan_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, required=True)
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Quality settings
    parser.add_argument('--acceptance_threshold', type=float, default=0.70)
    parser.add_argument('--psnr_weight', type=float, default=0.20)
    parser.add_argument('--ssim_weight', type=float, default=0.25)
    parser.add_argument('--uncertainty_weight', type=float, default=0.20)
    
    # Validation thresholds
    parser.add_argument('--psnr_min', type=float, default=25.0)
    parser.add_argument('--ssim_min', type=float, default=0.80)
    
    # Generation settings
    parser.add_argument('--fusion_method', type=str, default='uncertainty')
    parser.add_argument('--num_mc_samples', type=int, default=5)
    parser.add_argument('--diffusion_steps', type=int, default=50)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation/quality')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def load_models(args, device):
    """Load all models."""
    # VAE - use VAE (not VAESmall) to match checkpoint architecture
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
    
    # GAN - with dropout enabled for MC uncertainty estimation
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
    """Create uncertainty-aware dual branch and fusion module."""
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
def evaluate_quality(
    dual_branch,
    fusion_module,
    dataloader,
    metrics_calculator,
    decision_engine,
    device,
    args,
    num_samples
):
    """Evaluate quality of generated images."""
    all_decisions = []
    samples_processed = 0
    
    for batch in tqdm(dataloader, desc="Evaluating quality"):
        if samples_processed >= num_samples:
            break
        
        images = batch['image'].to(device)
        
        # Generate with fusion
        fusion_inputs = dual_branch.get_fusion_inputs(
            images,
            diffusion_steps=args.diffusion_steps,
            diffusion_strength=0.8
        )
        
        fusion_result = fusion_module(
            fusion_inputs['I_diff'],
            fusion_inputs['I_gan'],
            fusion_inputs['U_diff'],
            fusion_inputs['U_gan']
        )
        
        fused = fusion_result['fused']
        
        # Compute metrics for each sample
        for i in range(images.shape[0]):
            if samples_processed >= num_samples:
                break
            
            gen = fused[i:i+1]
            ref = images[i:i+1]
            unc = fusion_inputs['U_diff'][i:i+1].mean().item()
            
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
            decision = decision_engine.make_decision(
                metrics,
                uncertainty=unc,
                image_id=f"sample_{samples_processed}"
            )
            
            all_decisions.append(decision)
            samples_processed += 1
    
    return all_decisions


def visualize_quality_distribution(decisions, output_dir):
    """Visualize quality score distribution."""
    scores = [d['total_score'] for d in decisions]
    accepted = [d['total_score'] for d in decisions if d['accepted']]
    rejected = [d['total_score'] for d in decisions if not d['accepted']]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Score distribution
    axes[0, 0].hist(scores, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(decisions[0]['threshold'], color='r', linestyle='--', label='Threshold')
    axes[0, 0].set_xlabel('Quality Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Quality Score Distribution')
    axes[0, 0].legend()
    
    # Accepted vs Rejected
    axes[0, 1].bar(['Accepted', 'Rejected'], [len(accepted), len(rejected)], 
                   color=['green', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Accept/Reject (Rate: {len(accepted)/len(decisions)*100:.1f}%)')
    
    # Metrics distribution
    psnr_vals = [d['metrics']['psnr'] for d in decisions]
    ssim_vals = [d['metrics']['ssim'] for d in decisions]
    
    axes[1, 0].hist(psnr_vals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('PSNR (dB)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('PSNR Distribution')
    
    axes[1, 1].hist(ssim_vals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('SSIM')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('SSIM Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def visualize_score_components(decisions, output_dir):
    """Visualize score component breakdown."""
    components = ['psnr_score', 'ssim_score', 'mae_score', 'nrmse_score', 'uncertainty_score']
    
    accepted_comps = {c: [] for c in components}
    rejected_comps = {c: [] for c in components}
    
    for d in decisions:
        target = accepted_comps if d['accepted'] else rejected_comps
        for c in components:
            if c in d['score_components']:
                target[c].append(d['score_components'][c])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(components))
    width = 0.35
    
    accepted_means = [np.mean(accepted_comps[c]) if accepted_comps[c] else 0 for c in components]
    rejected_means = [np.mean(rejected_comps[c]) if rejected_comps[c] else 0 for c in components]
    
    ax.bar(x - width/2, accepted_means, width, label='Accepted', color='green', alpha=0.7)
    ax.bar(x + width/2, rejected_means, width, label='Rejected', color='red', alpha=0.7)
    
    ax.set_xlabel('Score Component')
    ax.set_ylabel('Mean Score')
    ax.set_title('Score Components: Accepted vs Rejected')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_score', '') for c in components], rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_components.png', dpi=150, bbox_inches='tight')
    plt.close()


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
    print("Statistical Quality Validation")
    print("=" * 60)
    print(f"Acceptance threshold: {args.acceptance_threshold}")
    print(f"PSNR weight: {args.psnr_weight}")
    print(f"SSIM weight: {args.ssim_weight}")
    print(f"Uncertainty weight: {args.uncertainty_weight}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load models
    print("\n[1/6] Loading models...")
    vae, diffusion, generator = load_models(args, str(device))
    
    # Create pipeline
    print("\n[2/6] Creating pipeline...")
    dual_branch, fusion_module = create_pipeline(diffusion, generator, args, device)
    
    # Create metrics calculator
    print("\n[3/6] Initializing metrics...")
    metrics_config = MetricsConfig(device=str(device))
    metrics_calculator = MetricsCalculator(metrics_config)
    
    # Create decision engine
    weights = QualityWeights(
        psnr=args.psnr_weight,
        ssim=args.ssim_weight,
        uncertainty=args.uncertainty_weight
    )
    decision_config = QualityDecisionConfig(
        acceptance_threshold=args.acceptance_threshold,
        weights=weights,
        log_path=str(output_dir / 'decisions.csv')
    )
    decision_engine = QualityDecisionEngine(decision_config)
    
    # Load data
    print("\n[4/6] Loading data...")
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
    print("\n[5/6] Evaluating quality...")
    decisions = evaluate_quality(
        dual_branch, fusion_module, test_loader,
        metrics_calculator, decision_engine, device, args, args.num_samples
    )
    
    # Get statistics
    print("\n[6/6] Computing statistics...")
    stats = decision_engine.get_statistics()
    rejection_analysis = decision_engine.get_rejection_analysis()
    
    print("\nQuality Validation Results:")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']*100:.1f}%")
    
    if 'score_mean' in stats:
        print(f"\n  Score statistics:")
        print(f"    Mean: {stats['score_mean']:.4f}")
        print(f"    Std: {stats['score_std']:.4f}")
        print(f"    Min: {stats['score_min']:.4f}")
        print(f"    Max: {stats['score_max']:.4f}")
    
    if rejection_analysis:
        print(f"\n  Rejection reasons:")
        for reason, count in sorted(rejection_analysis.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_quality_distribution(decisions, output_dir)
    visualize_score_components(decisions, output_dir)
    
    # Save report
    decision_engine.save_report(str(output_dir / 'validation_report.json'))
    
    # Save summary
    summary = {
        'config': {
            'acceptance_threshold': args.acceptance_threshold,
            'weights': {
                'psnr': args.psnr_weight,
                'ssim': args.ssim_weight,
                'uncertainty': args.uncertainty_weight
            }
        },
        'statistics': stats,
        'rejection_analysis': rejection_analysis
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Validation Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()