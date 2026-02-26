# models/segmentation/metrics.py

"""
Segmentation evaluation metrics.

Includes:
- Dice Score (DSC)
- Hausdorff Distance (HD95)
- Sensitivity / Specificity
- IoU / Jaccard Index
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import distance_transform_edt
from dataclasses import dataclass


@dataclass
class BraTSRegions:
    """BraTS tumor region definitions."""
    BACKGROUND = 0
    NCR_NET = 1      # Necrotic and Non-Enhancing Tumor
    ED = 2           # Peritumoral Edema
    ET = 4           # Enhancing Tumor (Note: BraTS uses label 4, not 3)
    
    # Composite regions for evaluation
    # Whole Tumor (WT) = NCR/NET + ED + ET
    # Tumor Core (TC) = NCR/NET + ET
    # Enhancing Tumor (ET) = ET only


class DiceScore:
    """Compute Dice Similarity Coefficient."""
    
    def __init__(self, smooth: float = 1e-5):
        self.smooth = smooth
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int = 4
    ) -> Dict[str, float]:
        """
        Compute Dice score per class.
        
        Args:
            pred: Predictions (B, H, W) - class indices
            target: Ground truth (B, H, W) - class indices
            num_classes: Number of classes
            
        Returns:
            Dictionary with per-class and mean Dice scores
        """
        dice_scores = {}
        
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores[f'dice_class_{c}'] = dice.item()
        
        # Mean Dice (excluding background)
        foreground_dice = [dice_scores[f'dice_class_{c}'] for c in range(1, num_classes)]
        dice_scores['dice_mean'] = np.mean(foreground_dice)
        
        return dice_scores
    
    def compute_brats_regions(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute BraTS region-wise Dice scores.
        
        Regions:
        - Whole Tumor (WT): Labels 1, 2, 4
        - Tumor Core (TC): Labels 1, 4
        - Enhancing Tumor (ET): Label 4
        """
        # Convert label 4 to label 3 if needed (BraTS convention)
        pred = pred.clone()
        target = target.clone()
        
        # Handle both conventions (label 4 or label 3 for ET)
        if (target == 4).any():
            target[target == 4] = 3
        if (pred == 4).any():
            pred[pred == 4] = 3
        
        scores = {}
        
        # Whole Tumor (WT): all tumor labels (1, 2, 3)
        pred_wt = (pred >= 1).float()
        target_wt = (target >= 1).float()
        wt_dice = self._dice(pred_wt, target_wt)
        scores['dice_wt'] = wt_dice
        
        # Tumor Core (TC): labels 1 and 3 (NCR/NET + ET)
        pred_tc = ((pred == 1) | (pred == 3)).float()
        target_tc = ((target == 1) | (target == 3)).float()
        tc_dice = self._dice(pred_tc, target_tc)
        scores['dice_tc'] = tc_dice
        
        # Enhancing Tumor (ET): label 3 only
        pred_et = (pred == 3).float()
        target_et = (target == 3).float()
        et_dice = self._dice(pred_et, target_et)
        scores['dice_et'] = et_dice
        
        # Mean of three regions
        scores['dice_mean_brats'] = (wt_dice + tc_dice + et_dice) / 3.0
        
        return scores
    
    def _dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return ((2.0 * intersection + self.smooth) / (union + self.smooth)).item()


class HausdorffDistance:
    """
    Compute Hausdorff Distance (HD95).
    
    HD95 uses the 95th percentile to reduce sensitivity to outliers.
    """
    
    def __init__(self, percentile: float = 95.0):
        self.percentile = percentile
    
    def __call__(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> float:
        """
        Compute HD95 between two binary masks.
        
        Args:
            pred: Predicted binary mask (H, W)
            target: Ground truth binary mask (H, W)
            
        Returns:
            HD95 distance in pixels
        """
        if pred.sum() == 0 or target.sum() == 0:
            return float('inf')
        
        # Distance transforms
        pred_boundary = self._get_boundary(pred)
        target_boundary = self._get_boundary(target)
        
        if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
            return float('inf')
        
        # Distance from pred boundary to target
        target_dist = distance_transform_edt(~target.astype(bool))
        pred_to_target = target_dist[pred_boundary > 0]
        
        # Distance from target boundary to pred
        pred_dist = distance_transform_edt(~pred.astype(bool))
        target_to_pred = pred_dist[target_boundary > 0]
        
        # Combine and get percentile
        all_distances = np.concatenate([pred_to_target, target_to_pred])
        hd = np.percentile(all_distances, self.percentile)
        
        return float(hd)
    
    def _get_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary pixels from mask."""
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        boundary = mask.astype(float) - eroded.astype(float)
        return boundary
    
    def compute_brats_regions(
        self,
        pred: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, float]:
        """Compute HD95 for BraTS regions."""
        # Handle label convention
        pred = pred.copy()
        target = target.copy()
        if (target == 4).any():
            target[target == 4] = 3
        if (pred == 4).any():
            pred[pred == 4] = 3
        
        scores = {}
        
        # Whole Tumor
        pred_wt = (pred >= 1).astype(np.uint8)
        target_wt = (target >= 1).astype(np.uint8)
        scores['hd95_wt'] = self(pred_wt, target_wt)
        
        # Tumor Core
        pred_tc = ((pred == 1) | (pred == 3)).astype(np.uint8)
        target_tc = ((target == 1) | (target == 3)).astype(np.uint8)
        scores['hd95_tc'] = self(pred_tc, target_tc)
        
        # Enhancing Tumor
        pred_et = (pred == 3).astype(np.uint8)
        target_et = (target == 3).astype(np.uint8)
        scores['hd95_et'] = self(pred_et, target_et)
        
        return scores


class IoUScore:
    """Intersection over Union (Jaccard Index)."""
    
    def __init__(self, smooth: float = 1e-5):
        self.smooth = smooth
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int = 4
    ) -> Dict[str, float]:
        """Compute IoU per class."""
        iou_scores = {}
        
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_scores[f'iou_class_{c}'] = iou.item()
        
        # Mean IoU (excluding background)
        foreground_iou = [iou_scores[f'iou_class_{c}'] for c in range(1, num_classes)]
        iou_scores['iou_mean'] = np.mean(foreground_iou)
        
        return iou_scores


class SensitivitySpecificity:
    """Compute Sensitivity (Recall) and Specificity."""
    
    def __init__(self, smooth: float = 1e-5):
        self.smooth = smooth
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int = 4
    ) -> Dict[str, float]:
        """Compute sensitivity and specificity per class."""
        metrics = {}
        
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            tp = (pred_c * target_c).sum()
            fn = ((1 - pred_c) * target_c).sum()
            fp = (pred_c * (1 - target_c)).sum()
            tn = ((1 - pred_c) * (1 - target_c)).sum()
            
            sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
            specificity = (tn + self.smooth) / (tn + fp + self.smooth)
            
            metrics[f'sensitivity_class_{c}'] = sensitivity.item()
            metrics[f'specificity_class_{c}'] = specificity.item()
        
        return metrics


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator.
    
    Computes all relevant metrics for BraTS evaluation.
    """
    
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.dice = DiceScore()
        self.hd95 = HausdorffDistance()
        self.iou = IoUScore()
        self.sens_spec = SensitivitySpecificity()
    
    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        compute_hd: bool = True
    ) -> Dict[str, float]:
        """
        Compute all segmentation metrics.
        
        Args:
            pred: Predictions (B, H, W) or (H, W)
            target: Ground truth (B, H, W) or (H, W)
            compute_hd: Whether to compute Hausdorff distance (slow)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Ensure 2D for processing
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        if target.dim() == 3:
            target = target.squeeze(0)
        
        # Dice scores
        dice_scores = self.dice.compute_brats_regions(pred, target)
        metrics.update(dice_scores)
        
        # IoU
        iou_scores = self.iou(pred, target, self.num_classes)
        metrics.update(iou_scores)
        
        # Sensitivity/Specificity
        sens_spec = self.sens_spec(pred, target, self.num_classes)
        metrics.update(sens_spec)
        
        # Hausdorff distance (optional, slow)
        if compute_hd:
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            hd_scores = self.hd95.compute_brats_regions(pred_np, target_np)
            metrics.update(hd_scores)
        
        return metrics
    
    def compute_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        compute_hd: bool = False
    ) -> List[Dict[str, float]]:
        """Compute metrics for each sample in batch."""
        batch_metrics = []
        
        for i in range(pred.shape[0]):
            metrics = self.compute_all(pred[i], target[i], compute_hd)
            batch_metrics.append(metrics)
        
        return batch_metrics
    
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate metrics across samples."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list if not np.isinf(m[key])]
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
        
        return aggregated


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """Convenience function for quick Dice computation."""
    dice = DiceScore()
    scores = dice.compute_brats_regions(pred, target)
    return scores['dice_mean_brats']


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """Convenience function for all metrics."""
    metrics = SegmentationMetrics()
    return metrics.compute_all(pred, target, compute_hd=False)
