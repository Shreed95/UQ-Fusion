# models/vae/__init__.py

from .encoder import (
    VAEEncoder,
    VAEEncoderSmall,
    ResidualBlock,
    AttentionBlock,
    DownsampleBlock,
    EncoderBlock
)

from .decoder import (
    VAEDecoder,
    VAEDecoderSmall,
    VAEDecoderWithSkip,
    UpsampleBlock,
    DecoderBlock
)

from .vae import (
    VAE,
    VAESmall,
    VAEWithLoss,
    VAEConfig,
    create_vae
)

from .losses import (
    VAELoss,
    CombinedVAELoss,
    PerceptualLoss,
    SSIMLoss
)

__all__ = [
    # Encoder
    'VAEEncoder',
    'VAEEncoderSmall',
    'ResidualBlock',
    'AttentionBlock',
    'DownsampleBlock',
    'EncoderBlock',
    
    # Decoder
    'VAEDecoder',
    'VAEDecoderSmall',
    'VAEDecoderWithSkip',
    'UpsampleBlock',
    'DecoderBlock',
    
    # VAE
    'VAE',
    'VAESmall',
    'VAEWithLoss',
    'VAEConfig',
    'create_vae',
    
    # Losses
    'VAELoss',
    'CombinedVAELoss',
    'PerceptualLoss',
    'SSIMLoss'
]
