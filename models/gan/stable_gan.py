# models/gan/stable_gan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .generator import STABLEGenerator, STABLEGeneratorSmall, GeneratorConfig, create_generator
from .discriminator import PatchGANDiscriminator, ConditionalDiscriminatorSmall, DiscriminatorConfig, create_discriminator
from .losses import STABLELoss, GANLoss


@dataclass
class STABLEGANConfig:
    """Configuration for STABLE-GAN."""
    # Generator
    in_channels: int = 4
    out_channels: int = 4
    gen_base_channels: int = 64
    num_residual_blocks: int = 9
    
    # Discriminator
    disc_base_channels: int = 64
    num_disc_layers: int = 3
    use_spectral_norm: bool = True
    
    # Loss weights
    lambda_adv: float = 1.0
    lambda_spatial: float = 5.0
    lambda_quantitative: float = 2.0
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    lambda_l1: float = 10.0
    
    # GAN mode
    gan_mode: str = 'lsgan'  # 'vanilla', 'lsgan', 'wgan'


class STABLEGAN(nn.Module):
    """
    STABLE-GAN for medical image-to-image translation.
    
    Combines:
        - ResNet-based Generator with instance normalization
        - PatchGAN Discriminator with spectral normalization
        - STABLE losses (spatial + quantitative preservation)
    
    Key Features:
        - Preserves spatial information (anatomical structures)
        - Preserves quantitative information (intensity distributions)
        - Suitable for unpaired and paired translation
    """
    
    def __init__(self, config: Optional[STABLEGANConfig] = None):
        super().__init__()
        
        if config is None:
            config = STABLEGANConfig()
        
        self.config = config
        
        # Generator
        gen_config = GeneratorConfig(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.gen_base_channels,
            num_residual_blocks=config.num_residual_blocks
        )
        self.generator = STABLEGenerator(gen_config)
        
        # Discriminator (conditional - takes input + output)
        disc_config = DiscriminatorConfig(
            in_channels=config.in_channels + config.out_channels,
            base_channels=config.disc_base_channels,
            num_layers=config.num_disc_layers,
            use_spectral_norm=config.use_spectral_norm
        )
        self.discriminator = PatchGANDiscriminator(disc_config)
        
        # Loss function
        self.loss_fn = STABLELoss(
            lambda_adv=config.lambda_adv,
            lambda_spatial=config.lambda_spatial,
            lambda_quantitative=config.lambda_quantitative,
            lambda_cycle=config.lambda_cycle,
            lambda_identity=config.lambda_identity,
            lambda_l1=config.lambda_l1,
            gan_mode=config.gan_mode
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate output from input.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            Generated image (B, C, H, W)
        """
        return self.generator(x)
    
    def discriminate(
        self,
        input: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Discriminate real/fake.
        
        Args:
            input: Input/condition image
            output: Generated or real output image
            
        Returns:
            Patch predictions
        """
        x = torch.cat([input, output], dim=1)
        return self.discriminator(x)
    
    def compute_generator_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute generator loss.
        
        Args:
            source: Source input image
            target: Target real image
            fake: Generated fake image
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        # Discriminator prediction for fake
        fake_pred = self.discriminate(source, fake)
        
        # Generator loss
        loss, loss_dict = self.loss_fn.generator_loss(
            fake_pred=fake_pred,
            fake=fake,
            real=target
        )
        
        return loss, loss_dict
    
    def compute_discriminator_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute discriminator loss.
        
        Args:
            source: Source input image
            target: Target real image
            fake: Generated fake image (detached)
            
        Returns:
            Tuple of (loss, loss_dict)
        """
        # Real prediction
        real_pred = self.discriminate(source, target)
        
        # Fake prediction (detached)
        fake_pred = self.discriminate(source, fake.detach())
        
        # Discriminator loss
        loss, loss_dict = self.loss_fn.discriminator_loss(real_pred, fake_pred)
        
        return loss, loss_dict
    
    def training_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            source: Source input image
            target: Target real image
            
        Returns:
            Dictionary of losses
        """
        # Generate fake
        fake = self.generator(source)
        
        # Generator loss
        loss_G, loss_dict_G = self.compute_generator_loss(source, target, fake)
        
        # Discriminator loss
        loss_D, loss_dict_D = self.compute_discriminator_loss(source, target, fake)
        
        # Combine
        loss_dict = {
            'fake': fake,
            **loss_dict_G,
            **loss_dict_D
        }
        
        return loss_dict
    
    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate images (inference mode)."""
        self.generator.eval()
        return self.generator(x)


class STABLEGANSmall(nn.Module):
    """
    Smaller STABLE-GAN for faster training.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        gen_base_channels: int = 32,
        disc_base_channels: int = 32,
        num_residual_blocks: int = 6,
        lambda_adv: float = 1.0,
        lambda_spatial: float = 5.0,
        lambda_quantitative: float = 2.0,
        lambda_l1: float = 10.0
    ):
        super().__init__()
        
        # Generator
        self.generator = STABLEGeneratorSmall(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=gen_base_channels,
            num_residual_blocks=num_residual_blocks
        )
        
        # Discriminator
        self.discriminator = ConditionalDiscriminatorSmall(
            in_channels=out_channels,
            condition_channels=in_channels,
            base_channels=disc_base_channels
        )
        
        # Loss
        self.loss_fn = STABLELoss(
            lambda_adv=lambda_adv,
            lambda_spatial=lambda_spatial,
            lambda_quantitative=lambda_quantitative,
            lambda_l1=lambda_l1,
            lambda_cycle=0.0,
            lambda_identity=0.0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)
    
    def discriminate(
        self,
        image: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        return self.discriminator(image, condition)
    
    def compute_generator_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_pred = self.discriminate(fake, source)
        
        loss, loss_dict = self.loss_fn.generator_loss(
            fake_pred=fake_pred,
            fake=fake,
            real=target
        )
        
        return loss, loss_dict
    
    def compute_discriminator_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        fake: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_pred = self.discriminate(target, source)
        fake_pred = self.discriminate(fake.detach(), source)
        
        loss, loss_dict = self.loss_fn.discriminator_loss(real_pred, fake_pred)
        
        return loss, loss_dict
    
    def training_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        fake = self.generator(source)
        
        loss_G, loss_dict_G = self.compute_generator_loss(source, target, fake)
        loss_D, loss_dict_D = self.compute_discriminator_loss(source, target, fake)
        
        return {
            'fake': fake,
            **loss_dict_G,
            **loss_dict_D
        }
    
    @torch.no_grad()
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        return self.generator(x)


class CycleSTABLEGAN(nn.Module):
    """
    Cycle-consistent STABLE-GAN for unpaired translation.
    Uses two generators (G: A->B, F: B->A) and two discriminators.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 32,
        num_residual_blocks: int = 6
    ):
        super().__init__()
        
        # Generators
        self.G = STABLEGeneratorSmall(in_channels, out_channels, base_channels, num_residual_blocks)  # A -> B
        self.F = STABLEGeneratorSmall(out_channels, in_channels, base_channels, num_residual_blocks)  # B -> A
        
        # Discriminators
        self.D_B = ConditionalDiscriminatorSmall(out_channels, in_channels, base_channels)  # Real B vs Fake B
        self.D_A = ConditionalDiscriminatorSmall(in_channels, out_channels, base_channels)  # Real A vs Fake A
        
        # Losses
        self.gan_loss = GANLoss('lsgan')
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()
        
        # Weights
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.5
    
    def forward(self, x: torch.Tensor, direction: str = 'AtoB') -> torch.Tensor:
        if direction == 'AtoB':
            return self.G(x)
        else:
            return self.F(x)
    
    def training_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for cycle GAN.
        
        Returns dict with all losses and generated images.
        """
        # Generate
        fake_B = self.G(real_A)
        fake_A = self.F(real_B)
        
        # Cycle
        rec_A = self.F(fake_B)
        rec_B = self.G(fake_A)
        
        # Identity
        idt_A = self.F(real_A)
        idt_B = self.G(real_B)
        
        # Generator losses
        loss_G_A = self.gan_loss(self.D_B(fake_B, real_A), True)
        loss_G_B = self.gan_loss(self.D_A(fake_A, real_B), True)
        
        loss_cycle_A = self.cycle_loss(rec_A, real_A)
        loss_cycle_B = self.cycle_loss(rec_B, real_B)
        
        loss_idt_A = self.identity_loss(idt_A, real_A)
        loss_idt_B = self.identity_loss(idt_B, real_B)
        
        loss_G = (
            loss_G_A + loss_G_B +
            self.lambda_cycle * (loss_cycle_A + loss_cycle_B) +
            self.lambda_identity * (loss_idt_A + loss_idt_B)
        )
        
        # Discriminator losses
        loss_D_B_real = self.gan_loss(self.D_B(real_B, real_A), True)
        loss_D_B_fake = self.gan_loss(self.D_B(fake_B.detach(), real_A), False)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
        
        loss_D_A_real = self.gan_loss(self.D_A(real_A, real_B), True)
        loss_D_A_fake = self.gan_loss(self.D_A(fake_A.detach(), real_B), False)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
        
        loss_D = loss_D_A + loss_D_B
        
        return {
            'fake_A': fake_A,
            'fake_B': fake_B,
            'rec_A': rec_A,
            'rec_B': rec_B,
            'loss_G': loss_G,
            'loss_D': loss_D,
            'loss_cycle': loss_cycle_A + loss_cycle_B,
            'loss_identity': loss_idt_A + loss_idt_B
        }


def create_stable_gan(
    model_type: str = 'small',
    **kwargs
) -> nn.Module:
    """
    Factory function to create STABLE-GAN models.
    
    Args:
        model_type: 'standard', 'small', or 'cycle'
        **kwargs: Additional arguments
        
    Returns:
        STABLE-GAN model
    """
    if model_type == 'standard':
        config = STABLEGANConfig(**kwargs)
        return STABLEGAN(config)
    elif model_type == 'small':
        return STABLEGANSmall(**kwargs)
    elif model_type == 'cycle':
        return CycleSTABLEGAN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
