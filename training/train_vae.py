# training/train_vae.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple, Callable
from pathlib import Path
from tqdm import tqdm
import json
import time
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import VAE, VAESmall, VAEConfig, VAELoss, CombinedVAELoss


@dataclass
class TrainingConfig:
    """Configuration for VAE training."""
    # Model
    model_type: str = 'standard'  # 'standard' or 'small'
    latent_channels: int = 4
    base_channels: int = 64
    
    # Training
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    
    # Loss weights
    recon_weight: float = 1.0
    kl_weight: float = 0.0001
    kl_warmup_epochs: int = 10
    
    # Optimizer
    optimizer: str = 'adamw'  # 'adam', 'adamw'
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Scheduler
    scheduler: str = 'cosine'  # 'cosine', 'step', 'none'
    warmup_epochs: int = 5
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints'
    log_dir: str = './outputs/logs'
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    
    # Validation
    val_every: int = 1
    
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
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )
    
    def apply(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class VAETrainer:
    """
    Trainer class for VAE model.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig
    ):
        """
        Initialize trainer.
        
        Args:
            model: VAE model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Loss function
        self.loss_fn = CombinedVAELoss(
            recon_weight=config.recon_weight,
            kl_weight=config.kl_weight
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # EMA
        self.ema = EMA(self.model, config.ema_decay) if config.use_ema else None
        
        # Logging
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir / f"vae_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': [],
            'train_kl': [],
            'val_kl': []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        params = self.model.parameters()
        
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
        """Create learning rate scheduler."""
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
    
    def _get_kl_weight(self) -> float:
        """Get KL weight with warmup."""
        if self.epoch < self.config.kl_warmup_epochs:
            return self.config.kl_weight * (self.epoch + 1) / self.config.kl_warmup_epochs
        return self.config.kl_weight
    
    def _warmup_lr(self):
        """Apply learning rate warmup."""
        if self.epoch < self.config.warmup_epochs:
            warmup_factor = (self.epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        # Update KL weight
        kl_weight = self._get_kl_weight()
        self.loss_fn.kl_weight = kl_weight
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    recon, mean, log_var = self.model(images)
                    loss, loss_dict = self.loss_fn(recon, images, mean, log_var)
            else:
                recon, mean, log_var = self.model(images)
                loss, loss_dict = self.loss_fn(recon, images, mean, log_var)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            # Update EMA
            if self.ema is not None:
                self.ema.update()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon += loss_dict['recon_loss'].item()
            total_kl += loss_dict['kl_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['recon_loss'].item():.4f}",
                'kl': f"{loss_dict['kl_loss'].item():.6f}"
            })
            
            # Logging
            self.writer.add_scalar('train/loss_step', loss.item(), self.global_step)
            self.global_step += 1
        
        # Compute epoch averages
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            
            recon, mean, log_var = self.model(images, deterministic=True)
            loss, loss_dict = self.loss_fn(recon, images, mean, log_var)
            
            total_loss += loss.item()
            total_recon += loss_dict['recon_loss'].item()
            total_kl += loss_dict['kl_loss'].item()
            num_batches += 1
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon / num_batches,
            'kl_loss': total_kl / num_batches
        }
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
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
            
            # Also save EMA weights separately
            if self.ema is not None:
                ema_state = {k: v.clone() for k, v in self.ema.shadow.items()}
                torch.save({'model_state_dict': ema_state}, self.checkpoint_dir / 'best_ema.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for self.epoch in range(self.epoch, self.config.epochs):
            # Learning rate warmup
            self._warmup_lr()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Log training metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon_loss'])
            self.history['train_kl'].append(train_metrics['kl_loss'])
            
            self.writer.add_scalar('train/loss', train_metrics['loss'], self.epoch)
            self.writer.add_scalar('train/recon_loss', train_metrics['recon_loss'], self.epoch)
            self.writer.add_scalar('train/kl_loss', train_metrics['kl_loss'], self.epoch)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.epoch)
            self.writer.add_scalar('train/kl_weight', self._get_kl_weight(), self.epoch)
            
            # Validate
            if (self.epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate()
                
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_recon'].append(val_metrics['recon_loss'])
                self.history['val_kl'].append(val_metrics['kl_loss'])
                
                self.writer.add_scalar('val/loss', val_metrics['loss'], self.epoch)
                self.writer.add_scalar('val/recon_loss', val_metrics['recon_loss'], self.epoch)
                self.writer.add_scalar('val/kl_loss', val_metrics['kl_loss'], self.epoch)
                
                # Check for best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                print(f"\nEpoch {self.epoch + 1}/{self.config.epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f} (Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.6f})")
                print(f"  Val Loss: {val_metrics['loss']:.4f} (Recon: {val_metrics['recon_loss']:.4f}, KL: {val_metrics['kl_loss']:.6f})")
                if is_best:
                    print("  *** New best model! ***")
                
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
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    @torch.no_grad()
    def visualize_reconstructions(self, num_samples: int = 4):
        """Generate reconstruction visualizations."""
        self.model.eval()
        
        if self.ema is not None:
            self.ema.apply()
        
        # Get a batch
        batch = next(iter(self.val_loader))
        images = batch['image'][:num_samples].to(self.device)
        
        # Reconstruct
        recons = self.model.reconstruct(images)
        
        # Log to tensorboard
        for i in range(min(num_samples, images.shape[0])):
            # Original (first channel)
            self.writer.add_image(
                f'reconstruction/original_{i}',
                images[i, 0:1],
                self.epoch
            )
            # Reconstruction
            self.writer.add_image(
                f'reconstruction/recon_{i}',
                recons[i, 0:1],
                self.epoch
            )
        
        if self.ema is not None:
            self.ema.restore()


def train_vae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    resume_from: Optional[str] = None
) -> VAETrainer:
    """
    Convenience function to train VAE.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained VAETrainer
    """
    if config is None:
        config = TrainingConfig()
    
    # Create model
    vae_config = VAEConfig(
        latent_channels=config.latent_channels,
        base_channels=config.base_channels,
        kl_weight=config.kl_weight
    )
    
    if config.model_type == 'small':
        model = VAESmall(
            latent_channels=config.latent_channels
        )
    else:
        model = VAE(vae_config)
    
    # Create trainer
    trainer = VAETrainer(model, train_loader, val_loader, config)
    
    # Resume if specified
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
    
    # Train
    trainer.train()
    
    return trainer
