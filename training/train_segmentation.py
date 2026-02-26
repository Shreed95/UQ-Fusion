# training/train_segmentation.py

"""
Training module for tumor segmentation.

Supports:
- Training on original dataset (baseline)
- Training on augmented dataset (original + synthetic)
- Training on synthetic-only dataset (ablation)

FIXED: Properly loads segmentation from synthetic npz files.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
import json
import time
import numpy as np
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.segmentation import (
    SegmentationUNet,
    SegmentationUNetSmall,
    create_segmentation_model,
    BraTSLoss,
    CombinedSegmentationLoss,
    SegmentationMetrics,
    DiceScore
)


@dataclass
class SegmentationTrainingConfig:
    """Configuration for segmentation training."""
    # Model
    model_type: str = 'small'      # 'small', 'standard', 'large'
    in_channels: int = 4
    num_classes: int = 4
    
    # Training
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = 'cosine'   # 'cosine', 'step', 'plateau'
    
    # Loss
    dice_weight: float = 1.0
    ce_weight: float = 0.5
    
    # Data
    use_augmentation: bool = True
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints/segmentation'
    log_dir: str = './outputs/logs/segmentation'
    
    # Device
    device: str = 'cuda'


class SegmentationTrainer:
    """Trainer for segmentation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: SegmentationTrainingConfig,
        experiment_name: str = 'baseline'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_name = experiment_name
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == 'mps' else 'cpu')
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        if config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        elif config.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )
        elif config.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5
            )
        
        # Loss
        self.criterion = BraTSLoss(
            dice_weight=config.dice_weight,
            ce_weight=config.ce_weight
        )
        
        # Metrics
        self.metrics = SegmentationMetrics(num_classes=config.num_classes)
        self.dice_scorer = DiceScore()
        
        # Logging
        self.checkpoint_dir = Path(config.checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # State
        self.epoch = 0
        self.global_step = 0
        self.best_dice = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice_wt': [],
            'val_dice_tc': [],
            'val_dice_et': []
        }
    
    def train_step(
        self,
        images: torch.Tensor,
        masks: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step."""
        images = images.to(self.device)
        masks = masks.to(self.device)
        
        self.optimizer.zero_grad()
        
        logits = self.model(images)
        losses = self.criterion(logits, masks)
        
        losses['total_loss'].backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [Train]")
        
        for batch in pbar:
            images = batch['image']
            masks = batch['segmentation']
            
            losses = self.train_step(images, masks)
            
            total_loss += losses['total_loss']
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'dice': f"{losses['dice_loss']:.4f}"
            })
            
            if self.global_step % 10 == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)
            
            self.global_step += 1
        
        return {'loss': total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        all_dice_wt = []
        all_dice_tc = []
        all_dice_et = []
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc=f"Epoch {self.epoch + 1} [Val]"):
            images = batch['image'].to(self.device)
            masks = batch['segmentation'].to(self.device)
            
            logits = self.model(images)
            losses = self.criterion(logits, masks)
            
            total_loss += losses['total_loss'].item()
            
            # Compute Dice scores
            pred = torch.argmax(logits, dim=1)
            
            for i in range(pred.shape[0]):
                dice_scores = self.dice_scorer.compute_brats_regions(pred[i], masks[i])
                all_dice_wt.append(dice_scores['dice_wt'])
                all_dice_tc.append(dice_scores['dice_tc'])
                all_dice_et.append(dice_scores['dice_et'])
            
            num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'dice_wt': np.mean(all_dice_wt),
            'dice_tc': np.mean(all_dice_tc),
            'dice_et': np.mean(all_dice_et),
            'dice_mean': np.mean([np.mean(all_dice_wt), np.mean(all_dice_tc), np.mean(all_dice_et)])
        }
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_dice': self.best_dice,
            'config': asdict(self.config),
            'history': self.history,
            'experiment_name': self.experiment_name
        }
        
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        if (self.epoch + 1) % self.config.save_every == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.epoch + 1}.pth')
        
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   
        if checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_dice = checkpoint['best_dice']
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for self.epoch in range(self.epoch, self.config.epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], self.epoch)
            
            # Validate
            val_metrics = self.validate()
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_dice_wt'].append(val_metrics['dice_wt'])
            self.history['val_dice_tc'].append(val_metrics['dice_tc'])
            self.history['val_dice_et'].append(val_metrics['dice_et'])
            
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], self.epoch)
            self.writer.add_scalar('epoch/val_dice_wt', val_metrics['dice_wt'], self.epoch)
            self.writer.add_scalar('epoch/val_dice_tc', val_metrics['dice_tc'], self.epoch)
            self.writer.add_scalar('epoch/val_dice_et', val_metrics['dice_et'], self.epoch)
            self.writer.add_scalar('epoch/val_dice_mean', val_metrics['dice_mean'], self.epoch)
            
            is_best = val_metrics['dice_mean'] > self.best_dice
            if is_best:
                self.best_dice = val_metrics['dice_mean']
            
            print(f"\nEpoch {self.epoch + 1}/{self.config.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val Dice WT: {val_metrics['dice_wt']:.4f}")
            print(f"  Val Dice TC: {val_metrics['dice_tc']:.4f}")
            print(f"  Val Dice ET: {val_metrics['dice_et']:.4f}")
            print(f"  Val Dice Mean: {val_metrics['dice_mean']:.4f}")
            if is_best:
                print("  *** New best model! ***")
            
            self.save_checkpoint(is_best)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['dice_mean'])
            else:
                self.scheduler.step()
        
        # Save final history
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        
        print("\nTraining complete!")
        print(f"Best validation Dice: {self.best_dice:.4f}")
        
        return self.history


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset that combines original and synthetic data.
    
    FIXED: Properly loads segmentation from synthetic npz files.
    """
    
    def __init__(
        self,
        original_dataset: torch.utils.data.Dataset,
        synthetic_dir: Path,
        synthetic_ratio: float = 1.0
    ):
        self.original_dataset = original_dataset
        self.synthetic_dir = Path(synthetic_dir)
        self.synthetic_ratio = synthetic_ratio
        
        # Load synthetic data
        self.synthetic_files = sorted(self.synthetic_dir.glob('synthetic_*.npz'))
        
        # Determine how many synthetic samples to use
        num_synthetic = int(len(self.original_dataset) * synthetic_ratio)
        num_synthetic = min(num_synthetic, len(self.synthetic_files))
        self.synthetic_files = self.synthetic_files[:num_synthetic]
        
        # Check if synthetic files have segmentation
        if len(self.synthetic_files) > 0:
            test_data = np.load(self.synthetic_files[0], allow_pickle=True)
            if 'segmentation' not in test_data.files:
                print("\n" + "!" * 60)
                print("WARNING: Synthetic files do not contain segmentation masks!")
                print("Please regenerate the dataset using the fixed generate_dataset.py")
                print("!" * 60 + "\n")
        
        print(f"Augmented dataset: {len(self.original_dataset)} original + {len(self.synthetic_files)} synthetic")
    
    def __len__(self):
        return len(self.original_dataset) + len(self.synthetic_files)
    
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            sample = self.original_dataset[idx]
            # Ensure consistent types
            return {
                'image': sample['image'].float(),
                'segmentation': sample['segmentation'].long()
            }
        else:
            # Load synthetic sample
            syn_idx = idx - len(self.original_dataset)
            data = np.load(self.synthetic_files[syn_idx], allow_pickle=True)
            
            # Handle image shape: could be (1, 4, H, W) or (4, H, W)
            image = data['image']
            if image.ndim == 4:
                image = image.squeeze(0)
            image = torch.from_numpy(image).float()
            
            # Load segmentation from npz file (FIXED!)
            if 'segmentation' in data.files:
                seg = data['segmentation']
                if seg.ndim == 3:
                    seg = seg.squeeze(0)
                seg = torch.from_numpy(seg).long()
            else:
                # Fallback: use original dataset (NOT IDEAL - will cause issues!)
                print(f"Warning: No segmentation in {self.synthetic_files[syn_idx]}, using fallback")
                orig_idx = syn_idx % len(self.original_dataset)
                orig_sample = self.original_dataset[orig_idx]
                seg = orig_sample['segmentation'].long()
            
            return {
                'image': image,
                'segmentation': seg
            }


class SyntheticOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset for synthetic-only training (ablation study).
    
    FIXED: Properly loads segmentation from synthetic npz files.
    """
    
    def __init__(self, synthetic_dir: Path, original_dataset=None):
        self.synthetic_dir = Path(synthetic_dir)
        self.synthetic_files = sorted(self.synthetic_dir.glob('synthetic_*.npz'))
        self.original_dataset = original_dataset
        
        # Check if synthetic files have segmentation
        if len(self.synthetic_files) > 0:
            test_data = np.load(self.synthetic_files[0], allow_pickle=True)
            if 'segmentation' not in test_data.files:
                print("\n" + "!" * 60)
                print("WARNING: Synthetic files do not contain segmentation masks!")
                print("Please regenerate the dataset using the fixed generate_dataset.py")
                print("!" * 60 + "\n")
        
        print(f"Synthetic dataset: {len(self.synthetic_files)} samples")
    
    def __len__(self):
        return len(self.synthetic_files)
    
    def __getitem__(self, idx):
        data = np.load(self.synthetic_files[idx], allow_pickle=True)
        
        # Handle image shape
        image = data['image']
        if image.ndim == 4:
            image = image.squeeze(0)
        image = torch.from_numpy(image).float()
        
        # Load segmentation from npz file (FIXED!)
        if 'segmentation' in data.files:
            seg = data['segmentation']
            if seg.ndim == 3:
                seg = seg.squeeze(0)
            seg = torch.from_numpy(seg).long()
        elif self.original_dataset is not None:
            # Fallback (NOT IDEAL)
            print(f"Warning: No segmentation in {self.synthetic_files[idx]}, using fallback")
            orig_idx = idx % len(self.original_dataset)
            orig_sample = self.original_dataset[orig_idx]
            seg = orig_sample['segmentation'].long()
        else:
            # Last resort: zeros (will cause training issues!)
            print(f"Warning: No segmentation available for {self.synthetic_files[idx]}")
            seg = torch.zeros(image.shape[1:], dtype=torch.long)
        
        return {
            'image': image,
            'segmentation': seg
        }


def create_segmentation_trainer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[SegmentationTrainingConfig] = None,
    experiment_name: str = 'baseline'
) -> SegmentationTrainer:
    """Create segmentation trainer."""
    if config is None:
        config = SegmentationTrainingConfig()
    
    model = create_segmentation_model(
        model_type=config.model_type,
        in_channels=config.in_channels,
        num_classes=config.num_classes
    )
    
    return SegmentationTrainer(model, train_loader, val_loader, config, experiment_name)
