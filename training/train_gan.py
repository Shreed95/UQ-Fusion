# training/train_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json
import time
import gc
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
    model_type: str = 'small'
    in_channels: int = 4
    out_channels: int = 4
    base_channels_g: int = 32
    base_channels_d: int = 32
    num_residual_blocks: int = 6
    epochs: int = 100
    batch_size: int = 8
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    adv_weight: float = 1.0
    l1_weight: float = 10.0
    spatial_weight: float = 5.0
    quantitative_weight: float = 2.0
    identity_weight: float = 0.5
    n_critic: int = 1
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints/gan'
    log_dir: str = './outputs/logs/gan'
    val_every: int = 1
    device: str = 'mps'


class ImagePool:
    """Image buffer to reduce GAN oscillation."""
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
                self.images.append(image.detach().cpu())
                return_images.append(image)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, self.pool_size, (1,)).item()
                    tmp = self.images[idx].clone().to(image.device)
                    self.images[idx] = image.detach().cpu()
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, dim=0)


class GANTrainer:
    """Trainer for STABLE-GAN, optimized for MPS."""

    def __init__(self, generator, discriminator, train_loader, val_loader, config: GANTrainingConfig):
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device
        if config.device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif config.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.is_mps = (self.device.type == 'mps')

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        # Optimizers
        self.optimizer_G = optim.Adam(generator.parameters(), lr=config.lr_g, betas=(config.beta1, config.beta2))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(config.beta1, config.beta2))

        # Schedulers
        self.scheduler_G = optim.lr_scheduler.LinearLR(self.optimizer_G, start_factor=1.0, end_factor=0.1, total_iters=config.epochs)
        self.scheduler_D = optim.lr_scheduler.LinearLR(self.optimizer_D, start_factor=1.0, end_factor=0.1, total_iters=config.epochs)

        # Loss
        self.criterion = STABLEGANLoss(
            adv_weight=config.adv_weight, l1_weight=config.l1_weight,
            spatial_weight=config.spatial_weight, quantitative_weight=config.quantitative_weight,
            identity_weight=config.identity_weight)

        # Image pool (store on CPU to save GPU memory)
        self.fake_pool = ImagePool(50)

        # Dirs
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_g_loss': [], 'train_d_loss': [], 'val_g_loss': [], 'val_d_loss': []}

    def _flush(self):
        gc.collect()
        if self.is_mps:
            torch.mps.synchronize()
            torch.mps.empty_cache()

    def train_step(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        source = source.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        # === Train Discriminator ===
        self.optimizer_D.zero_grad(set_to_none=True)

        with torch.no_grad():
            fake = self.generator(source)

        fake_pooled = self.fake_pool.query(fake.detach())

        real_input = torch.cat([source, target], dim=1)
        disc_real = self.discriminator(real_input)
        fake_input = torch.cat([source, fake_pooled], dim=1)
        disc_fake = self.discriminator(fake_input)

        d_losses = self.criterion.discriminator_loss(disc_real, disc_fake)
        d_loss = d_losses['total_d_loss']
        d_loss.backward()
        self.optimizer_D.step()

        d_loss_val = d_loss.item()
        del fake, fake_pooled, real_input, fake_input, disc_real, disc_fake, d_loss

        # === Train Generator ===
        self.optimizer_G.zero_grad(set_to_none=True)

        fake = self.generator(source)
        fake_input = torch.cat([source, fake], dim=1)
        disc_fake = self.discriminator(fake_input)

        g_losses = self.criterion.generator_loss(
            pred=fake, target=target, source=source,
            disc_fake=disc_fake, compute_identity=True)
        g_loss = g_losses['total_g_loss']
        g_loss.backward()
        self.optimizer_G.step()

        g_loss_val = g_loss.item()
        del fake, fake_input, disc_fake, g_loss

        # Build result dict with Python floats
        losses = {}
        losses.update({f'd_{k}': v.item() if torch.is_tensor(v) else v for k, v in d_losses.items()})
        losses.update({f'g_{k}': v.item() if torch.is_tensor(v) else v for k, v in g_losses.items()})
        del d_losses, g_losses, source, target

        return losses

    def train_epoch(self) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        tot_g = tot_d = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}", leave=False)
        for batch in pbar:
            images = batch['image']
            losses = self.train_step(images, images)

            tot_g += losses['g_total_g_loss']
            tot_d += losses['d_total_d_loss']
            n += 1
            self.global_step += 1

            if n % 20 == 0:
                pbar.set_postfix(G=f"{losses['g_total_g_loss']:.4f}", D=f"{losses['d_total_d_loss']:.4f}")

            if self.is_mps and n % 30 == 0:
                torch.mps.empty_cache()

        self._flush()
        return {'g_loss': tot_g / n, 'd_loss': tot_d / n}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.generator.eval()
        self.discriminator.eval()
        tot_g = tot_d = 0.0
        n = 0

        for batch in self.val_loader:
            images = batch['image'].to(self.device, non_blocking=True)
            fake = self.generator(images)

            real_input = torch.cat([images, images], dim=1)
            fake_input = torch.cat([images, fake], dim=1)
            disc_real = self.discriminator(real_input)
            disc_fake = self.discriminator(fake_input)

            d_losses = self.criterion.discriminator_loss(disc_real, disc_fake)
            g_losses = self.criterion.generator_loss(
                pred=fake, target=images, source=images,
                disc_fake=disc_fake, compute_identity=False)

            tot_g += g_losses['total_g_loss'].item()
            tot_d += d_losses['total_d_loss'].item()
            n += 1
            del images, fake, real_input, fake_input, disc_real, disc_fake, d_losses, g_losses

        self._flush()
        return {'g_loss': tot_g / n, 'd_loss': tot_d / n}

    def save_checkpoint(self, is_best=False):
        ckpt = {
            'epoch': self.epoch, 'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config), 'history': self.history,
        }
        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if (self.epoch + 1) % self.config.save_every == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{self.epoch + 1}.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt['generator_state_dict'])
        self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(ckpt['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(ckpt['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(ckpt['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(ckpt['scheduler_D_state_dict'])
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
        self.history = ckpt.get('history', self.history)
        print(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Generator: {sum(p.numel() for p in self.generator.parameters()):,} params")
        print(f"Discriminator: {sum(p.numel() for p in self.discriminator.parameters()):,} params")

        for self.epoch in range(self.epoch, self.config.epochs):
            t0 = time.time()
            tm = self.train_epoch()

            self.history['train_g_loss'].append(tm['g_loss'])
            self.history['train_d_loss'].append(tm['d_loss'])

            if (self.epoch + 1) % self.config.val_every == 0:
                vm = self.validate()
                self.history['val_g_loss'].append(vm['g_loss'])
                self.history['val_d_loss'].append(vm['d_loss'])

                is_best = vm['g_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = vm['g_loss']

                dt = time.time() - t0
                print(f"Epoch {self.epoch+1}/{self.config.epochs} ({dt:.0f}s) | "
                      f"G {tm['g_loss']:.4f}/{vm['g_loss']:.4f} | "
                      f"D {tm['d_loss']:.4f}/{vm['d_loss']:.4f}"
                      f"{' *** BEST ***' if is_best else ''}")
                self.save_checkpoint(is_best)

            self.scheduler_G.step()
            self.scheduler_D.step()
            self._flush()

        with open(self.ckpt_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete! Best val G loss: {self.best_val_loss:.6f}")


def create_gan_models(config: GANTrainingConfig) -> Tuple[nn.Module, nn.Module]:
    """Create generator and discriminator."""
    if config.model_type == 'small':
        generator = STABLEGeneratorSmall(
            in_channels=config.in_channels, out_channels=config.out_channels,
            base_channels=config.base_channels_g, num_residual_blocks=config.num_residual_blocks)
        discriminator = PatchGANDiscriminatorSmall(
            in_channels=config.in_channels * 2, base_channels=config.base_channels_d,
            use_spectral_norm=True)
    else:
        gen_config = GeneratorConfig(
            in_channels=config.in_channels, out_channels=config.out_channels,
            base_channels=config.base_channels_g, num_residual_blocks=config.num_residual_blocks)
        generator = STABLEGenerator(gen_config)
        disc_config = DiscriminatorConfig(
            in_channels=config.in_channels * 2, base_channels=config.base_channels_d,
            use_spectral_norm=True)
        discriminator = PatchGANDiscriminator(disc_config)
    return generator, discriminator


def train_gan(train_loader, val_loader, config=None, resume_from=None):
    """Convenience function to train GAN."""
    if config is None:
        config = GANTrainingConfig()
    generator, discriminator = create_gan_models(config)
    trainer = GANTrainer(generator, discriminator, train_loader, val_loader, config)
    if resume_from:
        trainer.load_checkpoint(resume_from)
    trainer.train()
    return trainer