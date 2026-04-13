# UQ-Fusion: Uncertainty-Guided Fusion for Medical Image Dataset Expansion

## Overview

UQ-Fusion is a novel hybrid generative framework for expanding medical image datasets, specifically designed for brain tumor MRI data (BraTS 2020). The framework combines Latent Diffusion Models and STABLE-GANs with uncertainty-guided fusion to generate high-quality synthetic images.

## Key Innovation

**Uncertainty-Guided Fusion**: Uses spatial uncertainty maps (both aleatoric and epistemic) to intelligently weight and combine outputs from dual generative pathways, ensuring that regions requiring high-fidelity synthesis receive appropriate attention from the most suitable generation method.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     UQ-FUSION FRAMEWORK                     │
├─────────────────────────────────────────────────────────────┤
│  [INPUT] BraTS 2020 MRI Slices (4 modalities)               │
│                           │                                 │
│              ┌────────────┴────────────┐                    │
│              ▼                         ▼                    │
│     ┌─────────────────┐       ┌─────────────────┐           │
│     │ Diffusion Branch│       │ STABLE-GAN      │           │
│     │ (VAE + U-Net)   │       │ (ResNet-9)      │           │
│     └────────┬────────┘       └────────┬────────┘           │
│              │ I_diff + U_diff          │ I_gan + U_gan     │
│              └────────────┬─────────────┘                   │
│                           ▼                                 │
│              ┌─────────────────────────┐                    │
│              │ Uncertainty-Guided      │                    │
│              │ Fusion Module           │                    │
│              │ I = α⊙I_diff + β⊙I_gan  │                    │
│              └────────────┬────────────┘                    │
│                           ▼                                 │
│              ┌─────────────────────────┐                    │
│              │ Statistical Validation  │                    │
│              │ (PSNR, SSIM, FID, LPIPS)│                    │
│              └────────────┬────────────┘                    │
│                           ▼                                 │
│  [OUTPUT] Validated Expanded Dataset                        │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
UQ_Fusion/
├── configs/              # Configuration files
├── data/                 # Data loading and preprocessing
├── models/               # Model architectures
│   ├── vae/             # Variational Autoencoder
│   ├── diffusion/       # Latent Diffusion Model
│   ├── gan/             # STABLE-GAN
│   ├── uncertainty/     # Uncertainty estimation
│   ├── fusion/          # Uncertainty-guided fusion
│   └── segmentation/    # Downstream U-Net
├── training/            # Training modules
├── validation/          # Quality validation
├── utils/               # Utilities
├── scripts/             # CLI scripts
├── outputs/             # Results and checkpoints
└── main.py              # Main entry point
```

## Installation

```bash
# Create virtual environment
python -m venv uqfusion_env
source uqfusion_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Complete Pipeline

```bash
python main.py --config configs/config.yaml
```

### Run Individual Phases

```bash
# Phase 1-2: Preprocessing
python scripts/preprocess_dataset.py --data_dir ./data

# Phase 3: VAE Training
python scripts/train_vae.py --data_dir ./data --epochs 100

# Phase 4: Diffusion Training
python scripts/train_diffusion.py --data_dir ./data --epochs 100

# Phase 5: GAN Training
python scripts/train_gan.py --data_dir ./data --epochs 100

# Phase 6-7: Fusion Evaluation
python scripts/evaluate_fusion.py --data_dir ./data

# Phase 8: Dataset Generation
python scripts/generate_dataset.py --expansion_factor 2

# Phase 9: Segmentation Training & Comparison
python scripts/train_segmentation.py --experiment baseline
python scripts/train_segmentation.py --experiment augmented --synthetic_dir ./outputs/expanded_dataset/accepted
python scripts/compare_augmentation.py

# Phase 10: Generate Report
python scripts/generate_report.py
```

## Expected Results

| Metric | Target | Description |
|--------|--------|-------------|
| PSNR | > 25 dB | Pixel-level fidelity |
| SSIM | > 0.80 | Structural similarity |
| FID | < 50 | Distribution similarity |
| Acceptance Rate | > 85% | Quality validation pass rate |
| Dice Improvement | 5-10% | Downstream segmentation gain |

## Phases

1. **Data Preprocessing**: NIfTI loading, normalization, slice extraction
2. **VAE Training**: Latent space compression (4x downsampling)
3. **Diffusion Training**: Latent diffusion for high-fidelity synthesis
4. **GAN Training**: STABLE-GAN for structure preservation
5. **Uncertainty Estimation**: Aleatoric + epistemic uncertainty
6. **Fusion**: Uncertainty-guided spatial weighting
7. **Validation**: Multi-metric quality assessment
8. **Generation**: Validated dataset expansion
9. **Segmentation**: Downstream task validation
10. **Integration**: End-to-end pipeline

## Novel Contributions

1. First hybrid uncertainty-guided diffusion-GAN framework for medical imaging
2. Novel spatial fusion mechanism using per-pixel uncertainty weighting
3. Adaptive validation system with uncertainty-aware quality scoring
4. Comprehensive evaluation framework combining statistical and downstream metrics

## Citation

```bibtex
@thesis{uqfusion2025,
  title={Uncertainty-Guided Fusion for Medical Image Dataset Expansion},
  author={[Author Name]},
  year={2025},
  type={Bachelor of Engineering Thesis}
}
```

## License

This project is for academic purposes as part of a Bachelor of Engineering final year project.
