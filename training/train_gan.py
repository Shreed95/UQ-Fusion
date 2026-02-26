# training/train_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json
import time
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gan import (
    STABLEGenerator,
    STABLEGeneratorSmall,
    PatchGANDiscriminator,
    PatchGANDiscriminatorSmall,
    STABLEGANLoss,
    GeneratorConfig,
    DiscriminatorConfig
)


@dataclass
class GANTrainingConfig:
    """Configuration for STABLE-GAN training."""
    # Model
    model_type: str = 'small'  # 'standard' or 'small'
    in_channels: int = 4
    out_channels: int = 4
    base_channels_g: int = 32
    base_channels_d: int = 32
    num_residual_blocks: int = 6
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    
    # Loss weights
    adv_weight: float = 1.0
    l1_weight: float = 10.0
    spatial_weight: float = 5.0
    quantitative_weight: float = 2.0
    identity_weight: float = 0.5
    
    # Training strategy
    n_critic: int = 1  # Train discriminator n times per generator update
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints/gan'
    log_dir: str = './outputs/logs/gan'
    
    # Validation
    val_every: int = 1
    
    # Device
    device: str = 'mps'


class ImagePool:
    """
    Image buffer for storing previously generated images.
    Used to reduce model oscillation during training.
    """
    
    def __init__(self, pool_size: int = 50):
        self.pool_size = pool_size
        self.images = []
    
    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = image.unsqueeze(0)
            
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, self.pool_size, (1,)).item()
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return torch.cat(return_images, dim=0)


class GANTrainer:
    """
    Trainer class for STABLE-GAN.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: GANTrainingConfig
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device selection (CUDA / MPS / CPU)
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif config.device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2)
        )
        
        # Schedulers
        self.scheduler_G = optim.lr_scheduler.LinearLR(
            self.optimizer_G,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.epochs
        )
        self.scheduler_D = optim.lr_scheduler.LinearLR(
            self.optimizer_D,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.epochs
        )
        
        # Loss
        self.criterion = STABLEGANLoss(
            adv_weight=config.adv_weight,
            l1_weight=config.l1_weight,
            spatial_weight=config.spatial_weight,
            quantitative_weight=config.quantitative_weight,
            identity_weight=config.identity_weight
        )
        
        # Image pool
        self.fake_pool = ImagePool(50)
        
        # Logging
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir / f"gan_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.history = {
            'train_g_loss': [],
            'train_d_loss': [],
            'val_g_loss': [],
            'val_d_loss': []
        }
    
    def train_step(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step."""
        source = source.to(self.device)
        target = target.to(self.device)
        
        # ====================
        # Train Discriminator
        # ====================
        self.optimizer_D.zero_grad()
        
        # Generate fake images
        with torch.no_grad():
            fake = self.generator(source)
        
        # Use image pool
        fake_pooled = self.fake_pool.query(fake.detach())
        
        # Real
        real_input = torch.cat([source, target], dim=1)
        disc_real = self.discriminator(real_input)
        
        # Fake
        fake_input = torch.cat([source, fake_pooled], dim=1)
        disc_fake = self.discriminator(fake_input)
        
        # Discriminator loss
        d_losses = self.criterion.discriminator_loss(disc_real, disc_fake)
        d_loss = d_losses['total_d_loss']
        
        d_loss.backward()
        self.optimizer_D.step()
        
        # ====================
        # Train Generator
        # ====================
        self.optimizer_G.zero_grad()
        
        # Generate
        fake = self.generator(source)
        
        # Discriminator output for fake
        fake_input = torch.cat([source, fake], dim=1)
        disc_fake = self.discriminator(fake_input)
        
        # Generator loss
        g_losses = self.criterion.generator_loss(
            pred=fake,
            target=target,
            source=source,
            disc_fake=disc_fake,
            compute_identity=True
        )
        g_loss = g_losses['total_g_loss']
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # Combine losses
        losses = {}
        losses.update({f'd_{k}': v.item() if torch.is_tensor(v) else v for k, v in d_losses.items()})
        losses.update({f'g_{k}': v.item() if torch.is_tensor(v) else v for k, v in g_losses.items()})
        
        return losses
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            # For image-to-image, source and target are the same (self-supervised)
            images = batch['image']
            
            losses = self.train_step(images, images)
            
            total_g_loss += losses['g_total_g_loss']
            total_d_loss += losses['d_total_d_loss']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f"{losses['g_total_g_loss']:.4f}",
                'D_loss': f"{losses['d_total_d_loss']:.4f}"
            })
            
            # Logging
            if self.global_step % 10 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            self.global_step += 1
        
        return {
            'g_loss': total_g_loss / num_batches,
            'd_loss': total_d_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.generator.eval()
        self.discriminator.eval()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            
            # Generate
            fake = self.generator(images)
            
            # Discriminator outputs
            real_input = torch.cat([images, images], dim=1)
            fake_input = torch.cat([images, fake], dim=1)
            
            disc_real = self.discriminator(real_input)
            disc_fake = self.discriminator(fake_input)
            
            # Losses
            d_losses = self.criterion.discriminator_loss(disc_real, disc_fake)
            g_losses = self.criterion.generator_loss(
                pred=fake,
                target=images,
                source=images,
                disc_fake=disc_fake,
                compute_identity=False
            )
            
            total_g_loss += g_losses['total_g_loss'].item()
            total_d_loss += d_losses['total_d_loss'].item()
            num_batches += 1
        
        return {
            'g_loss': total_g_loss / num_batches,
            'd_loss': total_d_loss / num_batches
        }
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample images."""
        self.generator.eval()
        
        batch = next(iter(self.val_loader))
        source = batch['image'][:num_samples].to(self.device)
        
        generated = self.generator(source)
        
        return source, generated
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'history': self.history
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        if (self.epoch + 1) % self.config.save_every == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.epoch + 1}.pth')
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        for self.epoch in range(self.epoch, self.config.epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Log
            self.history['train_g_loss'].append(train_metrics['g_loss'])
            self.history['train_d_loss'].append(train_metrics['d_loss'])
            
            self.writer.add_scalar('epoch/train_g_loss', train_metrics['g_loss'], self.epoch)
            self.writer.add_scalar('epoch/train_d_loss', train_metrics['d_loss'], self.epoch)
            self.writer.add_scalar('epoch/lr_G', self.optimizer_G.param_groups[0]['lr'], self.epoch)
            
            # Validate
            if (self.epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate()
                
                self.history['val_g_loss'].append(val_metrics['g_loss'])
                self.history['val_d_loss'].append(val_metrics['d_loss'])
                
                self.writer.add_scalar('epoch/val_g_loss', val_metrics['g_loss'], self.epoch)
                self.writer.add_scalar('epoch/val_d_loss', val_metrics['d_loss'], self.epoch)
                
                is_best = val_metrics['g_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['g_loss']
                
                print(f"\nEpoch {self.epoch + 1}/{self.config.epochs}")
                print(f"  Train G Loss: {train_metrics['g_loss']:.6f}, D Loss: {train_metrics['d_loss']:.6f}")
                print(f"  Val G Loss: {val_metrics['g_loss']:.6f}, D Loss: {val_metrics['d_loss']:.6f}")
                if is_best:
                    print("  *** New best model! ***")
                
                # Generate samples
                if (self.epoch + 1) % 5 == 0:
                    try:
                        source, generated = self.generate_samples(4)
                        for i in range(min(4, source.shape[0])):
                            self.writer.add_image(f'samples/source_{i}', source[i, 0:1].clamp(0, 1), self.epoch)
                            self.writer.add_image(f'samples/generated_{i}', generated[i, 0:1].clamp(0, 1), self.epoch)
                    except Exception as e:
                        print(f"  Warning: Could not generate samples: {e}")
                
                self.save_checkpoint(is_best)
            
            # Update schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()
        
        # Save history
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")


def create_gan_models(config: GANTrainingConfig) -> Tuple[nn.Module, nn.Module]:
    """Create generator and discriminator."""
    if config.model_type == 'small':
        generator = STABLEGeneratorSmall(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels_g,
            num_residual_blocks=config.num_residual_blocks
        )
        discriminator = PatchGANDiscriminatorSmall(
            in_channels=config.in_channels * 2,
            base_channels=config.base_channels_d,
            use_spectral_norm=True
        )
    else:
        gen_config = GeneratorConfig(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels_g,
            num_residual_blocks=config.num_residual_blocks
        )
        generator = STABLEGenerator(gen_config)
        
        disc_config = DiscriminatorConfig(
            in_channels=config.in_channels * 2,
            base_channels=config.base_channels_d,
            use_spectral_norm=True
        )
        discriminator = PatchGANDiscriminator(disc_config)
    
    return generator, discriminator


def train_gan(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[GANTrainingConfig] = None,
    resume_from: Optional[str] = None
) -> GANTrainer:
    """Convenience function to train GAN."""
    if config is None:
        config = GANTrainingConfig()
    
    generator, discriminator = create_gan_models(config)
    
    trainer = GANTrainer(generator, discriminator, train_loader, val_loader, config)
    
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
    
    trainer.train()
    
    return trainer
