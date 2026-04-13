# training/__init__.py

from .train_vae import (
    VAETrainer,
    TrainingConfig,
    EMA,
)

# Re-export train_vae as a function for backward compatibility
def train_vae(train_loader, val_loader, config=None, resume_from=None):
    """Backward-compatible convenience function."""
    from models.vae import VAE, VAEConfig
    if config is None:
        config = TrainingConfig()
    model = VAE(VAEConfig(latent_channels=config.latent_channels,
                           base_channels=config.base_channels))
    trainer = VAETrainer(model, train_loader, val_loader, config)
    if resume_from:
        trainer.load_checkpoint(resume_from)
    trainer.train()
    return trainer

try:
    from .train_diffusion import (
        DiffusionTrainer,
        DiffusionTrainingConfig,
        load_vae,
        train_diffusion
    )
except ImportError:
    pass

try:
    from .train_gan import (
        GANTrainer,
        GANTrainingConfig,
        ImagePool,
        create_gan_models,
        train_gan
    )
except ImportError:
    pass

try:
    from .train_fusion import (
        FusionTrainer,
        FusionTrainingConfig,
        create_fusion_trainer
    )
except ImportError:
    pass

try:
    from .train_segmentation import (
        SegmentationTrainer,
        SegmentationTrainingConfig,
        AugmentedDataset,
        create_segmentation_trainer
    )
except ImportError:
    pass