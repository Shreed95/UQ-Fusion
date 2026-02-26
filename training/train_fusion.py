# training/train_fusion.py

"""
Training module for learnable fusion networks.
"""

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

from models.fusion import (
    LearnableFusionNetwork,
    LearnableFusionConfig,
    CompositeFusionLoss,
    FusionQualityMetrics,
    create_learnable_fusion
)
from models.uncertainty import UncertaintyAwareDualBranch, UncertaintyWrapperConfig


@dataclass
class FusionTrainingConfig:
    """Configuration for fusion training."""
    # Model
    fusion_method: str = 'simple'  # 'simple', 'attention', 'unet', 'hybrid'
    hidden_channels: int = 64
    num_layers: int = 3
    
    # Training
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss weights
    reconstruction_weight: float = 1.0
    ssim_weight: float = 1.0
    perceptual_weight: float = 0.0
    regularization_weight: float = 0.01
    consistency_weight: float = 0.1
    
    # Uncertainty settings
    num_mc_samples: int = 5
    diffusion_steps: int = 50
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints/fusion'
    log_dir: str = './outputs/logs/fusion'
    
    # Device
    device: str = 'cuda'


class FusionTrainer:
    """Trainer for learnable fusion networks."""
    
    def __init__(
        self,
        fusion_network: nn.Module,
        dual_branch_model: UncertaintyAwareDualBranch,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: FusionTrainingConfig
    ):
        self.fusion_network = fusion_network
        self.dual_branch_model = dual_branch_model
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

        self.fusion_network = self.fusion_network.to(self.device)
        self.dual_branch_model = self.dual_branch_model.to(self.device)

        # Freeze dual branch model
        for param in self.dual_branch_model.parameters():
            param.requires_grad = False
        self.dual_branch_model.eval()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.fusion_network.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
        
        # Loss
        self.criterion = CompositeFusionLoss(
            reconstruction_weight=config.reconstruction_weight,
            ssim_weight=config.ssim_weight,
            perceptual_weight=config.perceptual_weight,
            regularization_weight=config.regularization_weight,
            consistency_weight=config.consistency_weight
        )
        
        # Logging
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir / f"fusion_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # State
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_step(self, images: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        images = images.to(self.device)
        
        # Get dual branch outputs (with uncertainty)
        with torch.no_grad():
            fusion_inputs = self.dual_branch_model.get_fusion_inputs(
                images,
                diffusion_steps=self.config.diffusion_steps,
                diffusion_strength=0.8
            )
        
        I_diff = fusion_inputs['I_diff']
        I_gan = fusion_inputs['I_gan']
        U_diff = fusion_inputs['U_diff']
        U_gan = fusion_inputs['U_gan']
        
        # Forward through fusion network
        self.optimizer.zero_grad()
        
        fusion_result = self.fusion_network(I_diff, I_gan, U_diff, U_gan)
        
        # Compute loss (target is original image)
        losses = self.criterion(
            fused=fusion_result['fused'],
            target=images,
            alpha=fusion_result['alpha'],
            beta=fusion_result['beta'],
            U_diff=U_diff,
            U_gan=U_gan
        )
        
        # Backward
        losses['total_loss'].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.fusion_network.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            images = batch['image']
            
            losses = self.train_step(images)
            
            total_loss += losses['total_loss']
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{losses['total_loss']:.4f}"})
            
            if self.global_step % 10 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            self.global_step += 1
        
        return {'loss': total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.fusion_network.eval()
        
        total_loss = 0.0
        total_psnr = 0.0
        total_psnr_improvement = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            
            # Get dual branch outputs
            fusion_inputs = self.dual_branch_model.get_fusion_inputs(
                images,
                diffusion_steps=self.config.diffusion_steps,
                diffusion_strength=0.8
            )
            
            I_diff = fusion_inputs['I_diff']
            I_gan = fusion_inputs['I_gan']
            U_diff = fusion_inputs['U_diff']
            U_gan = fusion_inputs['U_gan']
            
            # Fusion
            fusion_result = self.fusion_network(I_diff, I_gan, U_diff, U_gan)
            
            # Loss
            losses = self.criterion(
                fused=fusion_result['fused'],
                target=images,
                alpha=fusion_result['alpha'],
                beta=fusion_result['beta'],
                U_diff=U_diff,
                U_gan=U_gan
            )
            
            # Metrics
            metrics = FusionQualityMetrics.compute_all(
                fusion_result['fused'], images, I_diff, I_gan
            )
            
            total_loss += losses['total_loss'].item()
            total_psnr += metrics['psnr']
            total_psnr_improvement += metrics['psnr_improvement']
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'psnr': total_psnr / num_batches,
            'psnr_improvement': total_psnr_improvement / num_batches
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.fusion_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        
        self.fusion_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Fusion network parameters: {sum(p.numel() for p in self.fusion_network.parameters()):,}")
        
        for self.epoch in range(self.epoch, self.config.epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], self.epoch)
            
            # Validate
            val_metrics = self.validate()
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], self.epoch)
            self.writer.add_scalar('epoch/val_psnr', val_metrics['psnr'], self.epoch)
            self.writer.add_scalar('epoch/psnr_improvement', val_metrics['psnr_improvement'], self.epoch)
            
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            print(f"\nEpoch {self.epoch + 1}/{self.config.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"  PSNR Improvement: {val_metrics['psnr_improvement']:+.2f} dB")
            if is_best:
                print("  *** New best model! ***")
            
            self.save_checkpoint(is_best)
            self.scheduler.step()
        
        # Save history
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")


def create_fusion_trainer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    diffusion_checkpoint: str,
    gan_checkpoint: str,
    vae_checkpoint: str,
    config: Optional[FusionTrainingConfig] = None
) -> FusionTrainer:
    """Create fusion trainer with loaded models."""
    if config is None:
        config = FusionTrainingConfig()
    
    # Proper device resolution
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load dual branch model
    from models.vae import VAESmall
    from models.diffusion import LatentDiffusionModelSmall
    from models.gan import STABLEGeneratorSmall
    
    # VAE
    vae = VAESmall(in_channels=4, out_channels=4, latent_channels=4)
    vae_ckpt = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae = vae.to(device)
    
    # Diffusion
    diffusion = LatentDiffusionModelSmall(latent_channels=4, base_channels=64, num_timesteps=1000)
    diff_ckpt = torch.load(diffusion_checkpoint, map_location=device)
    diffusion.unet.load_state_dict(diff_ckpt['model_state_dict'])
    diffusion.set_vae(vae)
    diffusion = diffusion.to(device)
    
    # GAN
    gan_ckpt = torch.load(gan_checkpoint, map_location=device)
    gan_config = gan_ckpt.get('config', {})
    generator = STABLEGeneratorSmall(
        in_channels=gan_config.get('in_channels', 4),
        out_channels=gan_config.get('out_channels', 4),
        base_channels=gan_config.get('base_channels_g', 32),
        num_residual_blocks=gan_config.get('num_residual_blocks', 6)
    )
    generator.load_state_dict(gan_ckpt['generator_state_dict'])
    generator = generator.to(device)
    
    # Create dual branch
    unc_config = UncertaintyWrapperConfig(
        num_mc_samples=config.num_mc_samples,
        normalize_uncertainty=True
    )
    dual_branch = UncertaintyAwareDualBranch(diffusion, generator, unc_config)
    dual_branch = dual_branch.to(device)
    
    # Create fusion network
    fusion_network = create_learnable_fusion(
        method=config.fusion_method,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers
    )
    
    return FusionTrainer(fusion_network, dual_branch, train_loader, val_loader, config)
