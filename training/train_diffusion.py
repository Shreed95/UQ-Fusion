# training/train_diffusion.py

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

from models.diffusion import (
    LatentDiffusionModel,
    LatentDiffusionModelSmall,
    LatentDiffusionConfig,
    DDIMSampler
)
from models.vae import VAE, VAEConfig


@dataclass
class DiffusionTrainingConfig:
    """Configuration for Diffusion training."""
    model_type: str = 'small'
    latent_channels: int = 4
    base_channels: int = 64
    num_timesteps: int = 1000
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    optimizer: str = 'adamw'
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints/diffusion'
    log_dir: str = './outputs/logs/diffusion'
    vae_checkpoint: str = './outputs/checkpoints/vae/best.pth'
    vae_type: str = 'standard'
    use_ema: bool = True
    ema_decay: float = 0.9999
    val_every: int = 1
    num_val_samples: int = 4
    device: str = 'mps'
    mixed_precision: bool = False  # Only useful for CUDA


class EMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


class DiffusionTrainer:
    """Trainer for Latent Diffusion Model, optimized for MPS."""

    def __init__(self, model, vae, train_loader, val_loader, config: DiffusionTrainingConfig):
        self.model = model
        self.vae = vae
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

        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)

        # Freeze VAE
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.model.set_vae(self.vae)

        # Optimizer (only U-Net params)
        self.optimizer = optim.AdamW(
            self.model.unet.parameters(), lr=config.learning_rate,
            betas=config.betas, weight_decay=config.weight_decay)

        # Scheduler
        if config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs - config.warmup_epochs,
                eta_min=config.learning_rate * 0.01)
        else:
            self.scheduler = None

        # EMA on U-Net
        self.ema = EMA(self.model.unet, config.ema_decay) if config.use_ema else None

        # Dirs
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}

    def _warmup_lr(self):
        if self.epoch < self.config.warmup_epochs:
            factor = (self.epoch + 1) / self.config.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.config.learning_rate * factor

    def _flush(self):
        gc.collect()
        if self.is_mps:
            torch.mps.synchronize()
            torch.mps.empty_cache()

    @torch.no_grad()
    def _encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.vae.eval()
        return self.vae.get_latent(images, deterministic=True)

    def train_epoch(self) -> Dict[str, float]:
        self.model.unet.train()
        tot_loss = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)

            # Encode to latent (no grad)
            with torch.no_grad():
                latents = self._encode_batch(images)

            # Forward + loss
            loss_dict = self.model.get_loss(latents, condition=latents)
            loss = loss_dict['loss']
            lv = loss.item()

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), self.config.grad_clip)
            self.optimizer.step()

            if self.ema:
                self.ema.update(self.model.unet)

            # Free tensors
            del images, latents, loss, loss_dict

            tot_loss += lv
            n += 1
            self.global_step += 1

            if n % 20 == 0:
                pbar.set_postfix(loss=f"{lv:.6f}")

            if self.is_mps and n % 40 == 0:
                torch.mps.empty_cache()

        self._flush()
        return {'loss': tot_loss / n}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.unet.eval()
        if self.ema:
            self.ema.apply(self.model.unet)

        tot_loss = 0.0
        n = 0
        for batch in self.val_loader:
            images = batch['image'].to(self.device, non_blocking=True)
            latents = self._encode_batch(images)
            loss_dict = self.model.get_loss(latents, condition=latents)
            tot_loss += loss_dict['loss'].item()
            n += 1
            del images, latents, loss_dict

        if self.ema:
            self.ema.restore(self.model.unet)
        self._flush()
        return {'loss': tot_loss / n}

    def save_checkpoint(self, is_best=False):
        ckpt = {
            'epoch': self.epoch, 'global_step': self.global_step,
            'model_state_dict': self.model.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config), 'history': self.history,
        }
        if self.ema:
            ckpt['ema_shadow'] = self.ema.shadow
        if self.scheduler:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(ckpt, self.ckpt_dir / 'latest.pth')
        if (self.epoch + 1) % self.config.save_every == 0:
            torch.save(ckpt, self.ckpt_dir / f'epoch_{self.epoch + 1}.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.unet.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
        self.history = ckpt.get('history', self.history)
        if self.scheduler and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if self.ema and 'ema_shadow' in ckpt:
            self.ema.shadow = ckpt['ema_shadow']
        print(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self):
        print(f"Starting training on {self.device}")
        print(f"U-Net parameters: {sum(p.numel() for p in self.model.unet.parameters()):,}")

        for self.epoch in range(self.epoch, self.config.epochs):
            t0 = time.time()
            self._warmup_lr()
            tm = self.train_epoch()
            self.history['train_loss'].append(tm['loss'])

            if (self.epoch + 1) % self.config.val_every == 0:
                vm = self.validate()
                self.history['val_loss'].append(vm['loss'])

                is_best = vm['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = vm['loss']

                dt = time.time() - t0
                print(f"Epoch {self.epoch + 1}/{self.config.epochs} ({dt:.0f}s) | "
                      f"Train {tm['loss']:.6f} | Val {vm['loss']:.6f}"
                      f"{' *** BEST ***' if is_best else ''}")
                self.save_checkpoint(is_best)

            if self.scheduler and self.epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            self._flush()

        with open(self.ckpt_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.6f}")


def load_vae(checkpoint_path: str, model_type: str = 'standard', device: str = 'mps') -> nn.Module:
    """Load pre-trained VAE (compatible with new unified VAE architecture)."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt.get('config', {})

    vae = VAE(VAEConfig(
        in_channels=4, out_channels=4,
        latent_channels=cfg.get('latent_channels', 4),
        base_channels=cfg.get('base_channels', 64),
    ))
    vae.load_state_dict(ckpt['model_state_dict'])
    vae.eval()
    return vae


def train_diffusion(train_loader, val_loader, config=None, resume_from=None):
    """Convenience function to train diffusion model."""
    if config is None:
        config = DiffusionTrainingConfig()

    vae = load_vae(config.vae_checkpoint, config.vae_type, config.device)

    if config.model_type == 'small':
        model = LatentDiffusionModelSmall(
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            num_timesteps=config.num_timesteps)
    else:
        ldm_config = LatentDiffusionConfig(
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            num_timesteps=config.num_timesteps)
        model = LatentDiffusionModel(ldm_config)

    trainer = DiffusionTrainer(model, vae, train_loader, val_loader, config)
    if resume_from:
        trainer.load_checkpoint(resume_from)
    trainer.train()
    return trainer