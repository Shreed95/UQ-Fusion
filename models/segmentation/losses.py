# models/segmentation/losses.py

"""
Loss functions for brain tumor segmentation.

Includes:
- Dice Loss
- Cross-Entropy Loss
- Combined Dice + CE Loss
- Focal Loss
- Tversky Loss

Note: BraTS uses labels {0, 1, 2, 4} - we remap 4 to 3 for compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


def remap_brats_labels(targets: torch.Tensor) -> torch.Tensor:
    """
    Remap BraTS labels from {0, 1, 2, 4} to {0, 1, 2, 3}.
    
    BraTS convention:
    - 0: Background
    - 1: NCR/NET (Necrotic and Non-Enhancing Tumor)
    - 2: ED (Peritumoral Edema)
    - 4: ET (Enhancing Tumor) -> remapped to 3
    """
    remapped = targets.clone()
    remapped[targets == 4] = 3
    return remapped


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        reduction: str = 'mean',
        ignore_index: Optional[int] = None
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Ground truth labels (B, H, W) or (B, C, H, W)
            
        Returns:
            Dice loss value
        """
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot if needed
        if targets.dim() == 3:
            # Remap BraTS labels
            targets_remapped = remap_brats_labels(targets)
            targets_one_hot = F.one_hot(targets_remapped.long(), num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()
        
        # Softmax predictions
        probs = F.softmax(logits, dim=1)
        
        # Flatten
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)
        
        # Compute Dice per class
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Ignore background (class 0) optionally
        if self.ignore_index is not None:
            mask = torch.ones(num_classes, device=dice.device)
            mask[self.ignore_index] = 0
            dice = dice * mask.unsqueeze(0)
            dice = dice.sum(dim=1) / mask.sum()
        else:
            dice = dice.mean(dim=1)
        
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss with class weighting.
    
    Weights classes by inverse of their frequency to handle imbalance.
    """
    
    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized Dice Loss."""
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot
        if targets.dim() == 3:
            # Remap BraTS labels
            targets_remapped = remap_brats_labels(targets)
            targets_one_hot = F.one_hot(targets_remapped.long(), num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()
        
        probs = F.softmax(logits, dim=1)
        
        # Flatten
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)
        
        # Compute weights (inverse of class frequency)
        weights = 1.0 / (targets_flat.sum(dim=2) ** 2 + self.smooth)
        
        # Weighted Dice
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        weighted_intersection = (weights * intersection).sum(dim=1)
        weighted_union = (weights * union).sum(dim=1)
        
        dice = (2.0 * weighted_intersection + self.smooth) / (weighted_union + self.smooth)
        
        loss = 1 - dice
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Focal Loss."""
        num_classes = logits.shape[1]
        
        # Remap BraTS labels
        targets_remapped = remap_brats_labels(targets)
        
        ce_loss = F.cross_entropy(logits, targets_remapped.long(), reduction='none')
        
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets_remapped.long(), num_classes).permute(0, 3, 1, 2)
        p_t = (probs * targets_one_hot).sum(dim=1)
        
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=logits.device)[targets_remapped.long()]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation.
    
    TI = TP / (TP + α*FP + β*FN)
    
    α = β = 0.5 reduces to Dice
    α = β = 1.0 gives Jaccard/IoU
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1e-5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Tversky Loss."""
        num_classes = logits.shape[1]
        
        if targets.dim() == 3:
            # Remap BraTS labels
            targets_remapped = remap_brats_labels(targets)
            targets_one_hot = F.one_hot(targets_remapped.long(), num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()
        
        probs = F.softmax(logits, dim=1)
        
        # Flatten
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)
        
        # TP, FP, FN
        tp = (probs_flat * targets_flat).sum(dim=2)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=2)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=2)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        loss = 1 - tversky.mean(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation.
    
    Combines Dice Loss and Cross-Entropy Loss.
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weights: Optional[List[float]] = None,
        smooth: float = 1e-5
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        self.dice_loss = DiceLoss(smooth=smooth)
        
        if class_weights is not None:
            weight = torch.tensor(class_weights)
            self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.
        
        Returns:
            Dictionary with individual and total losses
        """
        # Remap for CE loss
        targets_remapped = remap_brats_labels(targets)
        
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets_remapped.long())
        
        total = self.dice_weight * dice + self.ce_weight * ce
        
        return {
            'dice_loss': dice,
            'ce_loss': ce,
            'total_loss': total
        }


class BraTSLoss(nn.Module):
    """
    Loss function specifically designed for BraTS segmentation.
    
    Handles the 3 tumor regions: NCR/NET, ED, ET
    with appropriate class weighting.
    
    Note: Automatically remaps BraTS labels {0,1,2,4} to {0,1,2,3}
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 0.5,
        focal_weight: float = 0.0
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
        # Class weights: Background, NCR/NET, ED, ET
        # Higher weights for smaller/harder classes
        self.register_buffer('class_weights', torch.tensor([0.1, 1.0, 0.5, 1.5]))
        
        self.dice_loss = GeneralizedDiceLoss()
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss(gamma=2.0)
        else:
            self.focal_loss = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> dict:
        """Compute BraTS-specific loss."""
        # Remap BraTS labels for CE
        targets_remapped = remap_brats_labels(targets)
        
        # Dice loss (handles remapping internally)
        dice = self.dice_loss(logits, targets)
        
        # CE loss with class weights
        ce = F.cross_entropy(
            logits, 
            targets_remapped.long(),
            weight=self.class_weights.to(logits.device)
        )
        
        total = self.dice_weight * dice + self.ce_weight * ce
        
        losses = {
            'dice_loss': dice,
            'ce_loss': ce
        }
        
        if self.focal_loss is not None:
            focal = self.focal_loss(logits, targets)
            total = total + self.focal_weight * focal
            losses['focal_loss'] = focal
        
        losses['total_loss'] = total
        
        return losses
