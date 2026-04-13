# training/train_vae.py

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

from models.vae import VAE, VAEConfig, CombinedVAELoss


@dataclass
class TrainingConfig:
    """Configuration for VAE training."""
    model_type: str = 'standard'
    latent_channels: int = 4
    base_channels: int = 64
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    recon_weight: float = 1.0
    kl_weight: float = 0.0001
    ssim_weight: float = 0.0
    l1_weight: float = 1.0
    kl_warmup_epochs: int = 10
    optimizer: str = 'adamw'
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints/vae'
    log_dir: str = './outputs/logs/vae'
    use_ema: bool = True
    ema_decay: float = 0.9999
    val_every: int = 1
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


class VAETrainer:
    """VAE Trainer optimized for MPS (Apple Silicon)."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device setup
        if config.device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif config.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.is_mps = (self.device.type == 'mps')
        self.model = self.model.to(self.device)

        # Loss
        self.loss_fn = CombinedVAELoss(
            recon_weight=config.recon_weight, kl_weight=config.kl_weight,
            ssim_weight=config.ssim_weight, l1_weight=config.l1_weight)

        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      betas=config.betas, weight_decay=config.weight_decay)

        # Scheduler
        if config.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs - config.warmup_epochs,
                eta_min=config.learning_rate * 0.01)
        else:
            self.scheduler = None

        # EMA
        self.ema = EMA(self.model, config.ema_decay) if config.use_ema else None

        # Dirs
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'train_recon': [],
                        'val_recon': [], 'train_kl': [], 'val_kl': []}

    def _kl_weight(self) -> float:
        if self.epoch < self.config.kl_warmup_epochs:
            return self.config.kl_weight * (self.epoch + 1) / self.config.kl_warmup_epochs
        return self.config.kl_weight

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

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.loss_fn.kl_weight = self._kl_weight()
        tot_loss = tot_recon = tot_kl = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)

            recon, mean, log_var = self.model(images)
            loss, ld = self.loss_fn(recon, images, mean, log_var)

            lv, rv, kv = loss.item(), ld['recon_loss'].item(), ld['kl_loss'].item()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            if self.ema:
                self.ema.update(self.model)

            # Free everything
            del recon, mean, log_var, loss, ld, images

            tot_loss += lv; tot_recon += rv; tot_kl += kv; n += 1
            self.global_step += 1

            if n % 20 == 0:
                pbar.set_postfix(loss=f"{lv:.4f}", recon=f"{rv:.4f}", kl=f"{kv:.6f}")

            # Periodic MPS flush
            if self.is_mps and n % 40 == 0:
                torch.mps.empty_cache()

        self._flush()
        return {'loss': tot_loss/n, 'recon_loss': tot_recon/n, 'kl_loss': tot_kl/n}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        if self.ema:
            self.ema.apply(self.model)

        tot_loss = tot_recon = tot_kl = 0.0
        n = 0
        for batch in self.val_loader:
            images = batch['image'].to(self.device, non_blocking=True)
            recon, mean, log_var = self.model(images, deterministic=True)
            loss, ld = self.loss_fn(recon, images, mean, log_var)
            tot_loss += loss.item(); tot_recon += ld['recon_loss'].item()
            tot_kl += ld['kl_loss'].item(); n += 1
            del recon, mean, log_var, loss, ld, images

        if self.ema:
            self.ema.restore(self.model)
        self._flush()
        return {'loss': tot_loss/n, 'recon_loss': tot_recon/n, 'kl_loss': tot_kl/n}

    def save_checkpoint(self, is_best: bool = False):
        ckpt = {
            'epoch': self.epoch, 'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
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
            torch.save(ckpt, self.ckpt_dir / f'epoch_{self.epoch+1}.pth')
        if is_best:
            torch.save(ckpt, self.ckpt_dir / 'best.pth')
            if self.ema:
                torch.save({'model_state_dict': {k: v.clone() for k, v in self.ema.shadow.items()}},
                           self.ckpt_dir / 'best_ema.pth')

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
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
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {params:,}")

        for self.epoch in range(self.epoch, self.config.epochs):
            t0 = time.time()
            self._warmup_lr()
            tm = self.train_epoch()

            self.history['train_loss'].append(tm['loss'])
            self.history['train_recon'].append(tm['recon_loss'])
            self.history['train_kl'].append(tm['kl_loss'])

            if (self.epoch + 1) % self.config.val_every == 0:
                vm = self.validate()
                self.history['val_loss'].append(vm['loss'])
                self.history['val_recon'].append(vm['recon_loss'])
                self.history['val_kl'].append(vm['kl_loss'])

                is_best = vm['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = vm['loss']

                dt = time.time() - t0
                print(f"Epoch {self.epoch+1}/{self.config.epochs} ({dt:.0f}s) | "
                      f"Train {tm['loss']:.4f} | Val {vm['loss']:.4f}"
                      f"{' *** BEST ***' if is_best else ''}")
                self.save_checkpoint(is_best)

            if self.scheduler and self.epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            self._flush()

        with open(self.ckpt_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")