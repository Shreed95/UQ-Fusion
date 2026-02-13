# training/train_diffusion.py

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

from models.diffusion import (
    LatentDiffusionModel,
    LatentDiffusionModelSmall,
    LatentDiffusionConfig,
    DDIMSampler
)
from models.vae import VAE, VAESmall


@dataclass
class DiffusionTrainingConfig:
    """Configuration for Diffusion training."""
    # Model
    model_type: str = 'small'  # 'standard' or 'small'
    latent_channels: int = 4
    base_channels: int = 64
    num_timesteps: int = 1000
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    
    # Optimizer
    optimizer: str = 'adamw'
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints/diffusion'
    log_dir: str = './outputs/logs/diffusion'
    
    # VAE
    vae_checkpoint: str = './outputs/checkpoints/vae/best.pth'
    vae_type: str = 'small'
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Validation
    val_every: int = 1
    num_val_samples: int = 4
    
    # Device
    device: str = 'cuda'
    mixed_precision: bool = True


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class DiffusionTrainer:
    """
    Trainer class for Latent Diffusion Model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DiffusionTrainingConfig
    ):
        self.model = model
        self.vae = vae
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)
        
        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Set VAE in diffusion model
        self.model.set_vae(self.vae)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # EMA
        self.ema = EMA(self.model.unet, config.ema_decay) if config.use_ema else None
        
        # Logging
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir / f"diffusion_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        # Only optimize U-Net parameters (VAE is frozen)
        params = self.model.unet.parameters()
        
        if self.config.optimizer == 'adam':
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        elif self.config.scheduler == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _warmup_lr(self):
        if self.epoch < self.config.warmup_epochs:
            warmup_factor = (self.epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor
    
    @torch.no_grad()
    def _encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using frozen VAE."""
        self.vae.eval()
        latents = self.vae.get_latent(images, deterministic=True)
        return latents
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.unet.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            
            # Encode to latent space
            with torch.no_grad():
                latents = self._encode_batch(images)
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss_dict = self.model.get_loss(latents, condition=latents)
                    loss = loss_dict['loss']
            else:
                loss_dict = self.model.get_loss(latents, condition=latents)
                loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            # Accumulate metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
            # Logging
            self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)
            self.global_step += 1
        
        metrics = {
            'loss': total_loss / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.unet.eval()
        
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            
            # Encode to latent
            latents = self._encode_batch(images)
            
            # Compute loss
            loss_dict = self.model.get_loss(latents, condition=latents)
            
            total_loss += loss_dict['loss'].item()
            num_batches += 1
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        metrics = {
            'loss': total_loss / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample images for visualization."""
        self.model.unet.eval()
        
        if self.ema is not None:
            self.ema.apply()
        
        # Get source images from validation set
        batch = next(iter(self.val_loader))
        source_images = batch['image'][:num_samples].to(self.device)
        
        # Encode source
        source_latents = self._encode_batch(source_images)
        
        # Generate using image-to-image
        generated_latents = self.model.sample(
            batch_size=num_samples,
            condition=source_latents,
            latent_shape=(60, 60),
            num_inference_steps=50,
            show_progress=False
        )
        
        # Decode
        generated_images = self.model.decode(generated_latents)
        
        if self.ema is not None:
            self.ema.restore()
        
        return source_images, generated_images
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'history': self.history
        }
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save periodic
        if (self.epoch + 1) % self.config.save_every == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.epoch + 1}.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.unet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"U-Net parameters: {sum(p.numel() for p in self.model.unet.parameters()):,}")
        
        for self.epoch in range(self.epoch, self.config.epochs):
            # Learning rate warmup
            self._warmup_lr()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log training metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.writer.add_scalar('train/loss', train_metrics['loss'], self.epoch)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.epoch)
            
            # Validate
            if (self.epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate()
                
                self.history['val_loss'].append(val_metrics['loss'])
                self.writer.add_scalar('val/loss', val_metrics['loss'], self.epoch)
                
                # Check for best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                print(f"\nEpoch {self.epoch + 1}/{self.config.epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.6f}")
                print(f"  Val Loss: {val_metrics['loss']:.6f}")
                if is_best:
                    print("  *** New best model! ***")
                
                # Generate samples
                if (self.epoch + 1) % 5 == 0:
                    try:
                        source, generated = self.generate_samples(self.config.num_val_samples)
                        for i in range(min(4, source.shape[0])):
                            self.writer.add_image(
                                f'samples/source_{i}',
                                source[i, 0:1].clamp(0, 1),
                                self.epoch
                            )
                            self.writer.add_image(
                                f'samples/generated_{i}',
                                generated[i, 0:1].clamp(0, 1),
                                self.epoch
                            )
                    except Exception as e:
                        print(f"  Warning: Could not generate samples: {e}")
                
                # Save checkpoint
                self.save_checkpoint(is_best)
            
            # Update scheduler
            if self.scheduler is not None and self.epoch >= self.config.warmup_epochs:
                self.scheduler.step()
        
        # Save final history
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")


def load_vae(checkpoint_path: str, model_type: str = 'small', device: str = 'cuda') -> nn.Module:
    """Load pre-trained VAE."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'small':
        vae = VAESmall(in_channels=4, out_channels=4, latent_channels=4)
    else:
        from models.vae import VAEConfig
        config = VAEConfig()
        vae = VAE(config)
    
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    return vae


def train_diffusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[DiffusionTrainingConfig] = None,
    resume_from: Optional[str] = None
) -> DiffusionTrainer:
    """
    Convenience function to train diffusion model.
    """
    if config is None:
        config = DiffusionTrainingConfig()
    
    # Load VAE
    print(f"Loading VAE from {config.vae_checkpoint}")
    vae = load_vae(config.vae_checkpoint, config.vae_type, config.device)
    
    # Create diffusion model
    if config.model_type == 'small':
        model = LatentDiffusionModelSmall(
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            num_timesteps=config.num_timesteps
        )
    else:
        ldm_config = LatentDiffusionConfig(
            latent_channels=config.latent_channels,
            base_channels=config.base_channels,
            num_timesteps=config.num_timesteps
        )
        model = LatentDiffusionModel(ldm_config)
    
    # Create trainer
    trainer = DiffusionTrainer(model, vae, train_loader, val_loader, config)
    
    # Resume if specified
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
    
    # Train
    trainer.train()
    
    return trainer
