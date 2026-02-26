# validation/statistical_validator.py

"""
Statistical Validation System.

Provides adaptive threshold computation and statistical validation
for synthetic medical images based on original dataset statistics.
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
from tqdm import tqdm

from .metrics import MetricsCalculator, MetricsConfig


@dataclass
class ValidationThresholds:
    """Thresholds for image quality validation."""
    psnr_min: float = 25.0      # Minimum PSNR in dB
    ssim_min: float = 0.80      # Minimum SSIM
    lpips_max: float = 0.30     # Maximum LPIPS
    fid_max: float = 50.0       # Maximum FID
    nrmse_max: float = 0.15     # Maximum NRMSE
    mae_max: float = 0.10       # Maximum MAE
    
    # Uncertainty thresholds
    uncertainty_max: float = 0.50  # Maximum total uncertainty


@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold computation."""
    percentile_low: float = 5.0   # Percentile for lower bound
    percentile_high: float = 95.0  # Percentile for upper bound
    margin: float = 0.1            # Safety margin (percentage)
    update_rate: float = 0.1       # Running update rate


class BaselineStatistics:
    """
    Computes and stores baseline statistics from original dataset.
    
    Used to derive adaptive thresholds for validation.
    """
    
    def __init__(self):
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []
        self.mae_values = []
        self.nrmse_values = []
        
        self.statistics = {}
        self.computed = False
    
    def add_sample(
        self,
        psnr: float,
        ssim: float,
        lpips: float = None,
        mae: float = None,
        nrmse: float = None
    ):
        """Add a sample's metrics."""
        self.psnr_values.append(psnr)
        self.ssim_values.append(ssim)
        if lpips is not None:
            self.lpips_values.append(lpips)
        if mae is not None:
            self.mae_values.append(mae)
        if nrmse is not None:
            self.nrmse_values.append(nrmse)
    
    def compute_statistics(self, config: Optional[AdaptiveThresholdConfig] = None):
        """Compute statistics from collected samples."""
        if config is None:
            config = AdaptiveThresholdConfig()
        
        self.statistics = {}
        
        for name, values in [
            ('psnr', self.psnr_values),
            ('ssim', self.ssim_values),
            ('lpips', self.lpips_values),
            ('mae', self.mae_values),
            ('nrmse', self.nrmse_values)
        ]:
            if len(values) > 0:
                arr = np.array(values)
                self.statistics[name] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr)),
                    'p5': float(np.percentile(arr, config.percentile_low)),
                    'p95': float(np.percentile(arr, config.percentile_high))
                }
        
        self.computed = True
        return self.statistics
    
    def get_adaptive_thresholds(
        self,
        config: Optional[AdaptiveThresholdConfig] = None
    ) -> ValidationThresholds:
        """Derive adaptive thresholds from statistics."""
        if not self.computed:
            self.compute_statistics(config)
        
        if config is None:
            config = AdaptiveThresholdConfig()
        
        margin = config.margin
        
        thresholds = ValidationThresholds()
        
        # PSNR: use 5th percentile as minimum (higher is better)
        if 'psnr' in self.statistics:
            thresholds.psnr_min = self.statistics['psnr']['p5'] * (1 - margin)
        
        # SSIM: use 5th percentile as minimum (higher is better)
        if 'ssim' in self.statistics:
            thresholds.ssim_min = self.statistics['ssim']['p5'] * (1 - margin)
        
        # LPIPS: use 95th percentile as maximum (lower is better)
        if 'lpips' in self.statistics:
            thresholds.lpips_max = self.statistics['lpips']['p95'] * (1 + margin)
        
        # MAE: use 95th percentile as maximum (lower is better)
        if 'mae' in self.statistics:
            thresholds.mae_max = self.statistics['mae']['p95'] * (1 + margin)
        
        # NRMSE: use 95th percentile as maximum (lower is better)
        if 'nrmse' in self.statistics:
            thresholds.nrmse_max = self.statistics['nrmse']['p95'] * (1 + margin)
        
        return thresholds
    
    def save(self, path: str):
        """Save statistics to file."""
        data = {
            'statistics': self.statistics,
            'n_samples': len(self.psnr_values)
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load statistics from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.statistics = data['statistics']
        self.computed = True


class StatisticalValidator:
    """
    Statistical validation system for synthetic medical images.
    
    Features:
    - Adaptive thresholds based on original dataset
    - Per-metric validation with reasons
    - Region-aware validation (tumor vs healthy)
    - Running acceptance rate tracking
    """
    
    def __init__(
        self,
        thresholds: Optional[ValidationThresholds] = None,
        metrics_config: Optional[MetricsConfig] = None
    ):
        if thresholds is None:
            thresholds = ValidationThresholds()
        
        self.thresholds = thresholds
        self.metrics = MetricsCalculator(metrics_config)
        
        # Tracking
        self.total_validated = 0
        self.total_accepted = 0
        self.rejection_reasons = {}
    
    def validate_single(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Validate a single generated image.
        
        Args:
            generated: Generated image (1, C, H, W)
            reference: Reference image (1, C, H, W)
            uncertainty: Uncertainty map (1, 1, H, W)
            
        Returns:
            Validation result dictionary
        """
        # Compute metrics
        psnr = self.metrics.psnr(generated, reference).item()
        ssim = self.metrics.ssim(generated, reference).item()
        mae = self.metrics.mae(generated, reference).item()
        nrmse = self.metrics.nrmse(generated, reference).item()
        
        # Validation checks
        checks = {
            'psnr': psnr >= self.thresholds.psnr_min,
            'ssim': ssim >= self.thresholds.ssim_min,
            'mae': mae <= self.thresholds.mae_max,
            'nrmse': nrmse <= self.thresholds.nrmse_max
        }
        
        # Uncertainty check
        if uncertainty is not None:
            mean_uncertainty = uncertainty.mean().item()
            checks['uncertainty'] = mean_uncertainty <= self.thresholds.uncertainty_max
        else:
            mean_uncertainty = None
        
        # Overall decision
        is_accepted = all(checks.values())
        
        # Collect rejection reasons
        rejection_reasons = [k for k, v in checks.items() if not v]
        
        # Update tracking
        self.total_validated += 1
        if is_accepted:
            self.total_accepted += 1
        else:
            for reason in rejection_reasons:
                self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        
        return {
            'accepted': is_accepted,
            'metrics': {
                'psnr': psnr,
                'ssim': ssim,
                'mae': mae,
                'nrmse': nrmse,
                'uncertainty': mean_uncertainty
            },
            'checks': checks,
            'rejection_reasons': rejection_reasons
        }
    
    def validate_batch(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        Validate a batch of images.
        
        Args:
            generated: Generated images (B, C, H, W)
            reference: Reference images (B, C, H, W)
            uncertainty: Uncertainty maps (B, 1, H, W)
            
        Returns:
            List of validation results
        """
        results = []
        
        for i in range(generated.shape[0]):
            unc = uncertainty[i:i+1] if uncertainty is not None else None
            result = self.validate_single(
                generated[i:i+1],
                reference[i:i+1],
                unc
            )
            results.append(result)
        
        return results
    
    def get_acceptance_rate(self) -> float:
        """Get current acceptance rate."""
        if self.total_validated == 0:
            return 0.0
        return self.total_accepted / self.total_validated
    
    def get_rejection_summary(self) -> Dict[str, int]:
        """Get summary of rejection reasons."""
        return self.rejection_reasons.copy()
    
    def reset_tracking(self):
        """Reset tracking counters."""
        self.total_validated = 0
        self.total_accepted = 0
        self.rejection_reasons = {}
    
    def get_statistics(self) -> Dict:
        """Get validation statistics."""
        return {
            'total_validated': self.total_validated,
            'total_accepted': self.total_accepted,
            'acceptance_rate': self.get_acceptance_rate(),
            'rejection_reasons': self.rejection_reasons,
            'thresholds': {
                'psnr_min': self.thresholds.psnr_min,
                'ssim_min': self.thresholds.ssim_min,
                'mae_max': self.thresholds.mae_max,
                'nrmse_max': self.thresholds.nrmse_max,
                'uncertainty_max': self.thresholds.uncertainty_max
            }
        }


class RegionAwareValidator(StatisticalValidator):
    """
    Region-aware validation with different thresholds for different regions.
    
    Uses segmentation masks to apply stricter validation in tumor regions.
    """
    
    def __init__(
        self,
        thresholds: Optional[ValidationThresholds] = None,
        tumor_thresholds: Optional[ValidationThresholds] = None,
        metrics_config: Optional[MetricsConfig] = None
    ):
        super().__init__(thresholds, metrics_config)
        
        # Stricter thresholds for tumor regions
        if tumor_thresholds is None:
            tumor_thresholds = ValidationThresholds(
                psnr_min=28.0,
                ssim_min=0.85,
                mae_max=0.08
            )
        self.tumor_thresholds = tumor_thresholds
    
    def validate_with_regions(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        segmentation: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Validate with region-aware thresholds.
        
        Args:
            generated: Generated image (1, C, H, W)
            reference: Reference image (1, C, H, W)
            segmentation: Segmentation mask (1, 1, H, W)
            uncertainty: Uncertainty map (1, 1, H, W)
            
        Returns:
            Validation result with regional metrics
        """
        # Overall validation
        overall_result = self.validate_single(generated, reference, uncertainty)
        
        # Create masks
        tumor_mask = (segmentation > 0).float()
        healthy_mask = 1 - tumor_mask
        
        # Compute regional metrics
        if tumor_mask.sum() > 0:
            # Tumor region metrics
            tumor_gen = generated * tumor_mask
            tumor_ref = reference * tumor_mask
            
            tumor_mae = torch.abs(tumor_gen - tumor_ref).sum() / (tumor_mask.sum() * generated.shape[1] + 1e-8)
            tumor_mse = ((tumor_gen - tumor_ref) ** 2).sum() / (tumor_mask.sum() * generated.shape[1] + 1e-8)
            
            tumor_metrics = {
                'tumor_mae': tumor_mae.item(),
                'tumor_mse': tumor_mse.item()
            }
            
            # Validate tumor region
            tumor_accepted = tumor_mae.item() <= self.tumor_thresholds.mae_max
        else:
            tumor_metrics = {'tumor_mae': None, 'tumor_mse': None}
            tumor_accepted = True
        
        overall_result['regional_metrics'] = tumor_metrics
        overall_result['tumor_accepted'] = tumor_accepted
        overall_result['accepted'] = overall_result['accepted'] and tumor_accepted
        
        return overall_result


def compute_baseline_statistics(
    dataloader,
    metrics_calculator: MetricsCalculator,
    num_samples: int = None
) -> BaselineStatistics:
    """
    Compute baseline statistics from a dataloader.
    
    Uses augmented versions of images to establish quality baselines.
    """
    baseline = BaselineStatistics()
    
    samples_processed = 0
    
    for batch in tqdm(dataloader, desc="Computing baseline"):
        if num_samples and samples_processed >= num_samples:
            break
        
        images = batch['image']
        
        # Add small noise to create "augmented" version for baseline
        noise = torch.randn_like(images) * 0.02
        augmented = torch.clamp(images + noise, 0, 1)
        
        # Compute metrics
        psnr_values = metrics_calculator.psnr(augmented, images)
        ssim_values = metrics_calculator.ssim(augmented, images)
        mae_values = metrics_calculator.mae(augmented, images)
        nrmse_values = metrics_calculator.nrmse(augmented, images)
        
        for i in range(images.shape[0]):
            baseline.add_sample(
                psnr=psnr_values[i].item(),
                ssim=ssim_values[i].item(),
                mae=mae_values[i].item(),
                nrmse=nrmse_values[i].item()
            )
        
        samples_processed += images.shape[0]
    
    baseline.compute_statistics()
    return baseline
