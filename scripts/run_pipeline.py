#!/usr/bin/env python
# scripts/run_pipeline.py

"""
Run the complete UQ-Fusion pipeline with customizable phases.

Usage:
    # Run all phases
    python scripts/run_pipeline.py --data_dir ./data
    
    # Run specific phases
    python scripts/run_pipeline.py --data_dir ./data --phases vae diffusion gan
    
    # Skip training (use existing checkpoints)
    python scripts/run_pipeline.py --data_dir ./data --skip-training
    
    # Generate and evaluate only
    python scripts/run_pipeline.py --data_dir ./data --generate-only
"""

import argparse
import sys
from pathlib import Path
import subprocess
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Run UQ-Fusion Pipeline')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    # Phases
    parser.add_argument('--phases', nargs='+', 
                        default=['preprocess', 'vae', 'diffusion', 'gan', 
                                'fusion', 'generate', 'segmentation', 'compare'],
                        help='Phases to run')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training phases, use existing checkpoints')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only run generation and evaluation')
    
    # Training settings
    parser.add_argument('--vae_epochs', type=int, default=100)
    parser.add_argument('--diffusion_epochs', type=int, default=100)
    parser.add_argument('--gan_epochs', type=int, default=100)
    parser.add_argument('--seg_epochs', type=int, default=50)
    
    # Generation
    parser.add_argument('--expansion_factor', type=int, default=2)
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=8)
    
    return parser.parse_args()


def get_device(device_str):
    """Get device string."""
    import torch
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_str


def run_command(cmd, description):
    """Run a command and check for errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        return False
    
    return True


def main():
    args = parse_args()
    
    device = get_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    checkpoint_dir = output_dir / 'checkpoints'
    
    print("=" * 70)
    print("UQ-Fusion Complete Pipeline")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Phases: {args.phases}")
    print("=" * 70)
    
    results = {
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'phases': {}
    }
    
    # Phase 1-2: Preprocessing
    if 'preprocess' in args.phases and not args.skip_training:
        slices_dir = data_dir / 'slices'
        if not slices_dir.exists() or len(list(slices_dir.glob('*.npz'))) == 0:
            success = run_command([
                'python', 'scripts/preprocess_dataset.py',
                '--data_dir', str(data_dir)
            ], "Data Preprocessing")
            results['phases']['preprocess'] = {'status': 'completed' if success else 'failed'}
        else:
            print("\nPreprocessed data exists. Skipping...")
            results['phases']['preprocess'] = {'status': 'skipped'}
    
    # Phase 3: VAE Training
    if 'vae' in args.phases and not args.skip_training and not args.generate_only:
        vae_ckpt = checkpoint_dir / 'vae' / 'best.pth'
        if not vae_ckpt.exists():
            success = run_command([
                'python', 'scripts/train_vae.py',
                '--data_dir', str(data_dir),
                '--epochs', str(args.vae_epochs),
                '--batch_size', str(args.batch_size),
                '--device', device
            ], "VAE Training")
            results['phases']['vae'] = {'status': 'completed' if success else 'failed'}
        else:
            print(f"\nVAE checkpoint exists: {vae_ckpt}. Skipping...")
            results['phases']['vae'] = {'status': 'skipped'}
    
    # Phase 4: Diffusion Training
    if 'diffusion' in args.phases and not args.skip_training and not args.generate_only:
        diff_ckpt = checkpoint_dir / 'diffusion' / 'best.pth'
        if not diff_ckpt.exists():
            success = run_command([
                'python', 'scripts/train_diffusion.py',
                '--data_dir', str(data_dir),
                '--vae_checkpoint', str(checkpoint_dir / 'vae' / 'best.pth'),
                '--epochs', str(args.diffusion_epochs),
                '--batch_size', str(args.batch_size),
                '--device', device
            ], "Diffusion Training")
            results['phases']['diffusion'] = {'status': 'completed' if success else 'failed'}
        else:
            print(f"\nDiffusion checkpoint exists: {diff_ckpt}. Skipping...")
            results['phases']['diffusion'] = {'status': 'skipped'}
    
    # Phase 5: GAN Training
    if 'gan' in args.phases and not args.skip_training and not args.generate_only:
        gan_ckpt = checkpoint_dir / 'gan' / 'best.pth'
        if not gan_ckpt.exists():
            success = run_command([
                'python', 'scripts/train_gan.py',
                '--data_dir', str(data_dir),
                '--epochs', str(args.gan_epochs),
                '--batch_size', str(args.batch_size),
                '--device', device
            ], "GAN Training")
            results['phases']['gan'] = {'status': 'completed' if success else 'failed'}
        else:
            print(f"\nGAN checkpoint exists: {gan_ckpt}. Skipping...")
            results['phases']['gan'] = {'status': 'skipped'}
    
    # Phase 6-7: Fusion Evaluation
    if 'fusion' in args.phases:
        success = run_command([
            'python', 'scripts/evaluate_fusion.py',
            '--diffusion_checkpoint', str(checkpoint_dir / 'diffusion' / 'best.pth'),
            '--gan_checkpoint', str(checkpoint_dir / 'gan' / 'best.pth'),
            '--vae_checkpoint', str(checkpoint_dir / 'vae' / 'best.pth'),
            '--data_dir', str(data_dir),
            '--device', device
        ], "Fusion Evaluation")
        results['phases']['fusion'] = {'status': 'completed' if success else 'failed'}
    
    # Phase 8: Dataset Generation
    if 'generate' in args.phases:
        expanded_dir = output_dir / 'expanded_dataset' / 'accepted'
        if not expanded_dir.exists() or len(list(expanded_dir.glob('synthetic_*.npz'))) == 0:
            success = run_command([
                'python', 'scripts/generate_dataset.py',
                '--diffusion_checkpoint', str(checkpoint_dir / 'diffusion' / 'best.pth'),
                '--gan_checkpoint', str(checkpoint_dir / 'gan' / 'best.pth'),
                '--vae_checkpoint', str(checkpoint_dir / 'vae' / 'best.pth'),
                '--data_dir', str(data_dir),
                '--expansion_factor', str(args.expansion_factor),
                '--device', device
            ], "Dataset Generation")
            results['phases']['generate'] = {'status': 'completed' if success else 'failed'}
        else:
            print(f"\nExpanded dataset exists. Skipping...")
            results['phases']['generate'] = {'status': 'skipped'}
    
    # Phase 9: Segmentation Training
    if 'segmentation' in args.phases:
        # Baseline
        baseline_ckpt = checkpoint_dir / 'segmentation' / 'baseline' / 'best.pth'
        if not baseline_ckpt.exists():
            success = run_command([
                'python', 'scripts/train_segmentation.py',
                '--data_dir', str(data_dir),
                '--experiment', 'baseline',
                '--epochs', str(args.seg_epochs),
                '--device', device
            ], "Baseline Segmentation Training")
            results['phases']['segmentation_baseline'] = {'status': 'completed' if success else 'failed'}
        else:
            print(f"\nBaseline checkpoint exists. Skipping...")
            results['phases']['segmentation_baseline'] = {'status': 'skipped'}
        
        # Augmented
        augmented_ckpt = checkpoint_dir / 'segmentation' / 'augmented' / 'best.pth'
        if not augmented_ckpt.exists():
            success = run_command([
                'python', 'scripts/train_segmentation.py',
                '--data_dir', str(data_dir),
                '--experiment', 'augmented',
                '--synthetic_dir', str(output_dir / 'expanded_dataset' / 'accepted'),
                '--epochs', str(args.seg_epochs),
                '--device', device
            ], "Augmented Segmentation Training")
            results['phases']['segmentation_augmented'] = {'status': 'completed' if success else 'failed'}
        else:
            print(f"\nAugmented checkpoint exists. Skipping...")
            results['phases']['segmentation_augmented'] = {'status': 'skipped'}
    
    # Comparison
    if 'compare' in args.phases:
        success = run_command([
            'python', 'scripts/compare_augmentation.py',
            '--baseline_checkpoint', str(checkpoint_dir / 'segmentation' / 'baseline' / 'best.pth'),
            '--augmented_checkpoint', str(checkpoint_dir / 'segmentation' / 'augmented' / 'best.pth'),
            '--data_dir', str(data_dir),
            '--device', device
        ], "Comparison")
        results['phases']['compare'] = {'status': 'completed' if success else 'failed'}
    
    # Save results
    results['end_time'] = datetime.now().isoformat()
    
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    with open(reports_dir / 'pipeline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {reports_dir / 'pipeline_results.json'}")


if __name__ == "__main__":
    main()
