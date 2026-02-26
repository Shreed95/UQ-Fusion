# training/__init__.py

from .train_vae import (
    VAETrainer,
    TrainingConfig,
    EMA,
    train_vae
)

from .train_diffusion import (
    DiffusionTrainer,
    DiffusionTrainingConfig,
    load_vae,
    train_diffusion
)

from .train_gan import (
    GANTrainer,
    GANTrainingConfig,
    ImagePool,
    create_gan_models,
    train_gan
)

from .train_fusion import (
    FusionTrainer,
    FusionTrainingConfig,
    create_fusion_trainer
)

from .train_segmentation import (
    SegmentationTrainer,
    SegmentationTrainingConfig,
    AugmentedDataset,
    create_segmentation_trainer
)

__all__ = [
    # VAE Training
    'VAETrainer',
    'TrainingConfig',
    'EMA',
    'train_vae',
    
    # Diffusion Training
    'DiffusionTrainer',
    'DiffusionTrainingConfig',
    'load_vae',
    'train_diffusion',
    
    # GAN Training
    'GANTrainer',
    'GANTrainingConfig',
    'ImagePool',
    'create_gan_models',
    'train_gan',
    
    # Fusion Training
    'FusionTrainer',
    'FusionTrainingConfig',
    'create_fusion_trainer',
    
    # Segmentation Training
    'SegmentationTrainer',
    'SegmentationTrainingConfig',
    'AugmentedDataset',
    'create_segmentation_trainer'
]
