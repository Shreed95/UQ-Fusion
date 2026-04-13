# training/train_segmentation.py

"""
Training module for tumor segmentation.
UPDATED: Joint image+mask augmentation for both original and synthetic data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
import json
import time
import numpy as np
import random
import gc
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
    model_type: str = 'small'
    in_channels: int = 4
    num_classes: int = 4
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler: str = 'cosine'
    dice_weight: float = 1.0
    ce_weight: float = 0.5
    use_augmentation: bool = True
    save_every: int = 10
    checkpoint_dir: str = './outputs/checkpoints/segmentation'
    log_dir: str = './outputs/logs/segmentation'
    device: str = 'mps'


class SegmentationAugmentor:
    """
    Joint augmentation for image + segmentation mask.
    
    CRITICAL: The SAME geometric transform must be applied to both image and mask,
    otherwise the mask won't align with the image (the pairing bug all over again).
    Mask uses nearest-neighbor interpolation to preserve label values.
    """

    def __init__(self, p_flip_h=0.5, p_flip_v=0.3, p_rotate=0.5, p_intensity=0.5,
                 rotation_range=15.0, brightness_range=0.1, noise_std=0.02):
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.p_rotate = p_rotate
        self.p_intensity = p_intensity
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.noise_std = noise_std

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        """
        Apply joint augmentation.
        
        Args:
            image: (C, H, W) float tensor
            mask: (H, W) long tensor
            
        Returns:
            augmented image, augmented mask
        """
        # Horizontal flip
        if random.random() < self.p_flip_h:
            image = torch.flip(image, [2])  # flip W dimension
            mask = torch.flip(mask, [1])    # flip W dimension

        # Vertical flip
        if random.random() < self.p_flip_v:
            image = torch.flip(image, [1])  # flip H dimension
            mask = torch.flip(mask, [0])    # flip H dimension

        # Random rotation
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            # Use grid_sample for smooth rotation
            image = self._rotate_tensor(image.unsqueeze(0), angle, mode='bilinear').squeeze(0)
            mask_4d = mask.unsqueeze(0).unsqueeze(0).float()
            mask = self._rotate_tensor(mask_4d, angle, mode='nearest').squeeze(0).squeeze(0).long()

        # Intensity augmentation (image only, NOT mask)
        if random.random() < self.p_intensity:
            # Brightness shift
            shift = random.uniform(-self.brightness_range, self.brightness_range)
            image = (image + shift).clamp(0, 1)

            # Multiplicative contrast
            factor = random.uniform(0.9, 1.1)
            image = (image * factor).clamp(0, 1)

        # Gaussian noise (image only)
        if random.random() < 0.3:
            noise = torch.randn_like(image) * self.noise_std
            image = (image + noise).clamp(0, 1)

        return image, mask

    @staticmethod
    def _rotate_tensor(x, angle_deg, mode='bilinear'):
        """Rotate a 4D tensor by angle_deg using affine grid."""
        angle_rad = angle_deg * np.pi / 180.0
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)
        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)
        pad_mode = 'zeros' if mode == 'nearest' else 'zeros'
        return torch.nn.functional.grid_sample(x, grid, mode=mode, padding_mode=pad_mode, align_corners=False)


class SegmentationTrainer:
    """Trainer for segmentation models, MPS-optimized."""

    def __init__(self, model, train_loader, val_loader, config, experiment_name='baseline'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_name = experiment_name

        # Device
        if config.device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif config.device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.is_mps = (self.device.type == 'mps')
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Scheduler
        if config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        elif config.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        elif config.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)

        # Loss & metrics
        self.criterion = BraTSLoss(dice_weight=config.dice_weight, ce_weight=config.ce_weight)
        self.metrics = SegmentationMetrics(num_classes=config.num_classes)
        self.dice_scorer = DiceScore()

        # Dirs
        self.checkpoint_dir = Path(config.checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_dice = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_dice_wt': [], 'val_dice_tc': [], 'val_dice_et': []}

    def _flush(self):
        gc.collect()
        if self.is_mps:
            torch.mps.synchronize()
            torch.mps.empty_cache()

    def train_step(self, images, masks):
        images = images.to(self.device)
        masks = masks.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        losses = self.criterion(logits, masks)
        losses['total_loss'].backward()
        self.optimizer.step()
        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]", leave=False)
        for batch in pbar:
            images = batch['image']
            masks = batch['segmentation']
            losses = self.train_step(images, masks)
            total_loss += losses['total_loss']
            n += 1
            self.global_step += 1

            if n % 20 == 0:
                pbar.set_postfix(loss=f"{losses['total_loss']:.4f}")

            if self.is_mps and n % 30 == 0:
                torch.mps.empty_cache()

        self._flush()
        return {'loss': total_loss / n}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_wt, all_tc, all_et = [], [], []
        n = 0

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            masks = batch['segmentation'].to(self.device)
            logits = self.model(images)
            losses = self.criterion(logits, masks)
            total_loss += losses['total_loss'].item()

            pred = torch.argmax(logits, dim=1)
            for i in range(pred.shape[0]):
                dice = self.dice_scorer.compute_brats_regions(pred[i], masks[i])
                all_wt.append(dice['dice_wt'])
                all_tc.append(dice['dice_tc'])
                all_et.append(dice['dice_et'])
            n += 1

        self._flush()
        return {
            'loss': total_loss / n,
            'dice_wt': np.mean(all_wt),
            'dice_tc': np.mean(all_tc),
            'dice_et': np.mean(all_et),
            'dice_mean': np.mean([np.mean(all_wt), np.mean(all_tc), np.mean(all_et)])
        }

    def save_checkpoint(self, is_best=False):
        ckpt = {
            'epoch': self.epoch, 'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_dice': self.best_dice, 'config': asdict(self.config),
            'history': self.history, 'experiment_name': self.experiment_name
        }
        torch.save(ckpt, self.checkpoint_dir / 'latest.pth')
        if (self.epoch + 1) % self.config.save_every == 0:
            torch.save(ckpt, self.checkpoint_dir / f'epoch_{self.epoch+1}.pth')
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best.pth')

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_dice = ckpt['best_dice']
        self.history = ckpt.get('history', self.history)
        print(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self):
        print(f"\nStarting: {self.experiment_name} on {self.device}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} params")
        print(f"Train: {len(self.train_loader.dataset)} | Val: {len(self.val_loader.dataset)}")

        for self.epoch in range(self.epoch, self.config.epochs):
            t0 = time.time()
            tm = self.train_epoch()
            vm = self.validate()

            self.history['train_loss'].append(tm['loss'])
            self.history['val_loss'].append(vm['loss'])
            self.history['val_dice_wt'].append(vm['dice_wt'])
            self.history['val_dice_tc'].append(vm['dice_tc'])
            self.history['val_dice_et'].append(vm['dice_et'])

            is_best = vm['dice_mean'] > self.best_dice
            if is_best:
                self.best_dice = vm['dice_mean']

            dt = time.time() - t0
            print(f"Epoch {self.epoch+1}/{self.config.epochs} ({dt:.0f}s) | "
                  f"Loss {tm['loss']:.4f}/{vm['loss']:.4f} | "
                  f"Dice WT:{vm['dice_wt']:.4f} TC:{vm['dice_tc']:.4f} ET:{vm['dice_et']:.4f} "
                  f"Mean:{vm['dice_mean']:.4f}{' *** BEST ***' if is_best else ''}")

            self.save_checkpoint(is_best)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(vm['dice_mean'])
            else:
                self.scheduler.step()

            self._flush()

        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete! Best Dice: {self.best_dice:.4f}")
        return self.history


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset combining original + synthetic data with joint augmentation.
    FIXED: Loads segmentation from npz. Applies same transform to image+mask.
    """

    def __init__(self, original_dataset, synthetic_dir, synthetic_ratio=1.0, augmentor=None):
        self.original_dataset = original_dataset
        self.synthetic_dir = Path(synthetic_dir)
        self.synthetic_ratio = synthetic_ratio
        self.augmentor = augmentor

        self.synthetic_files = sorted(self.synthetic_dir.glob('synthetic_*.npz'))
        num_synthetic = int(len(self.original_dataset) * synthetic_ratio)
        num_synthetic = min(num_synthetic, len(self.synthetic_files))
        self.synthetic_files = self.synthetic_files[:num_synthetic]

        if len(self.synthetic_files) > 0:
            test_data = np.load(self.synthetic_files[0], allow_pickle=True)
            if 'segmentation' not in test_data.files:
                print("!" * 60)
                print("WARNING: Synthetic files missing segmentation masks!")
                print("!" * 60)

        print(f"Augmented dataset: {len(self.original_dataset)} original + "
              f"{len(self.synthetic_files)} synthetic (augmentor={'ON' if augmentor else 'OFF'})")

    def __len__(self):
        return len(self.original_dataset) + len(self.synthetic_files)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            sample = self.original_dataset[idx]
            image = sample['image'].float()
            seg = sample['segmentation'].long()
        else:
            syn_idx = idx - len(self.original_dataset)
            data = np.load(self.synthetic_files[syn_idx], allow_pickle=True)
            image = data['image']
            if image.ndim == 4:
                image = image.squeeze(0)
            image = torch.from_numpy(image).float()

            if 'segmentation' in data.files:
                seg = data['segmentation']
                if seg.ndim == 3:
                    seg = seg.squeeze(0)
                seg = torch.from_numpy(seg).long()
            else:
                orig_idx = syn_idx % len(self.original_dataset)
                orig_sample = self.original_dataset[orig_idx]
                seg = orig_sample['segmentation'].long()

        # Apply joint augmentation
        if self.augmentor is not None:
            image, seg = self.augmentor(image, seg)

        return {'image': image, 'segmentation': seg}


class SyntheticOnlyDataset(torch.utils.data.Dataset):
    """Synthetic-only dataset with optional augmentation."""

    def __init__(self, synthetic_dir, original_dataset=None, augmentor=None):
        self.synthetic_dir = Path(synthetic_dir)
        self.synthetic_files = sorted(self.synthetic_dir.glob('synthetic_*.npz'))
        self.original_dataset = original_dataset
        self.augmentor = augmentor
        print(f"Synthetic dataset: {len(self.synthetic_files)} samples")

    def __len__(self):
        return len(self.synthetic_files)

    def __getitem__(self, idx):
        data = np.load(self.synthetic_files[idx], allow_pickle=True)
        image = data['image']
        if image.ndim == 4:
            image = image.squeeze(0)
        image = torch.from_numpy(image).float()

        if 'segmentation' in data.files:
            seg = data['segmentation']
            if seg.ndim == 3:
                seg = seg.squeeze(0)
            seg = torch.from_numpy(seg).long()
        elif self.original_dataset is not None:
            orig_idx = idx % len(self.original_dataset)
            seg = self.original_dataset[orig_idx]['segmentation'].long()
        else:
            seg = torch.zeros(image.shape[1:], dtype=torch.long)

        if self.augmentor is not None:
            image, seg = self.augmentor(image, seg)

        return {'image': image, 'segmentation': seg}


def create_segmentation_trainer(train_loader, val_loader, config=None, experiment_name='baseline'):
    if config is None:
        config = SegmentationTrainingConfig()
    model = create_segmentation_model(model_type=config.model_type,
                                       in_channels=config.in_channels, num_classes=config.num_classes)
    return SegmentationTrainer(model, train_loader, val_loader, config, experiment_name)