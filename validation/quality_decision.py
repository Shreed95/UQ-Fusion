# validation/quality_decision.py

"""
Quality Decision Engine.

Computes composite quality scores and makes accept/reject decisions
for synthetic medical images based on multiple criteria.

Composite Score:
Q = w₁·PSNR_norm + w₂·SSIM + w₃·(1-LPIPS_norm) + w₄·(1-MAE_norm) + w₅·(1-U_total)

Acceptance Criterion: Q > threshold (default 0.70)
"""

import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import csv
from datetime import datetime


@dataclass
class QualityWeights:
    """Weights for composite quality score."""
    psnr: float = 0.20        # PSNR weight
    ssim: float = 0.25        # SSIM weight (higher for structural importance)
    lpips: float = 0.15       # LPIPS weight
    mae: float = 0.15         # MAE weight
    nrmse: float = 0.05       # NRMSE weight
    uncertainty: float = 0.20  # Uncertainty penalty weight
    
    def __post_init__(self):
        # Normalize weights to sum to 1
        total = self.psnr + self.ssim + self.lpips + self.mae + self.nrmse + self.uncertainty
        if abs(total - 1.0) > 1e-6:
            self.psnr /= total
            self.ssim /= total
            self.lpips /= total
            self.mae /= total
            self.nrmse /= total
            self.uncertainty /= total


@dataclass
class QualityDecisionConfig:
    """Configuration for quality decision engine."""
    acceptance_threshold: float = 0.70
    weights: QualityWeights = field(default_factory=QualityWeights)
    
    # Normalization ranges (for converting metrics to [0, 1])
    psnr_range: Tuple[float, float] = (15.0, 40.0)  # dB
    lpips_range: Tuple[float, float] = (0.0, 0.5)
    mae_range: Tuple[float, float] = (0.0, 0.2)
    nrmse_range: Tuple[float, float] = (0.0, 0.3)
    
    # Logging
    log_all_decisions: bool = True
    log_path: Optional[str] = None


class QualityScoreCalculator:
    """Calculates composite quality scores."""
    
    def __init__(self, config: Optional[QualityDecisionConfig] = None):
        if config is None:
            config = QualityDecisionConfig()
        self.config = config
        self.weights = config.weights
    
    def normalize_metric(
        self,
        value: float,
        range_min: float,
        range_max: float,
        higher_is_better: bool = True
    ) -> float:
        """Normalize metric to [0, 1]."""
        value = np.clip(value, range_min, range_max)
        normalized = (value - range_min) / (range_max - range_min)
        
        if not higher_is_better:
            normalized = 1 - normalized
        
        return normalized
    
    def compute_score(
        self,
        metrics: Dict[str, float],
        uncertainty: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute composite quality score.
        
        Args:
            metrics: Dictionary with psnr, ssim, lpips, mae, nrmse
            uncertainty: Total uncertainty value (0-1)
            
        Returns:
            Dictionary with score components and total
        """
        components = {}
        
        # PSNR (higher is better)
        if 'psnr' in metrics:
            psnr_norm = self.normalize_metric(
                metrics['psnr'],
                self.config.psnr_range[0],
                self.config.psnr_range[1],
                higher_is_better=True
            )
            components['psnr_score'] = psnr_norm * self.weights.psnr
        else:
            components['psnr_score'] = 0.0
        
        # SSIM (higher is better, already in [0, 1])
        if 'ssim' in metrics:
            ssim_norm = np.clip(metrics['ssim'], 0, 1)
            components['ssim_score'] = ssim_norm * self.weights.ssim
        else:
            components['ssim_score'] = 0.0
        
        # LPIPS (lower is better)
        if 'lpips' in metrics:
            lpips_norm = self.normalize_metric(
                metrics['lpips'],
                self.config.lpips_range[0],
                self.config.lpips_range[1],
                higher_is_better=False
            )
            components['lpips_score'] = lpips_norm * self.weights.lpips
        else:
            components['lpips_score'] = 0.0
        
        # MAE (lower is better)
        if 'mae' in metrics:
            mae_norm = self.normalize_metric(
                metrics['mae'],
                self.config.mae_range[0],
                self.config.mae_range[1],
                higher_is_better=False
            )
            components['mae_score'] = mae_norm * self.weights.mae
        else:
            components['mae_score'] = 0.0
        
        # NRMSE (lower is better)
        if 'nrmse' in metrics:
            nrmse_norm = self.normalize_metric(
                metrics['nrmse'],
                self.config.nrmse_range[0],
                self.config.nrmse_range[1],
                higher_is_better=False
            )
            components['nrmse_score'] = nrmse_norm * self.weights.nrmse
        else:
            components['nrmse_score'] = 0.0
        
        # Uncertainty penalty (lower is better)
        if uncertainty is not None:
            uncertainty_norm = 1 - np.clip(uncertainty, 0, 1)
            components['uncertainty_score'] = uncertainty_norm * self.weights.uncertainty
        else:
            components['uncertainty_score'] = 0.0
        
        # Total score
        total = sum(components.values())
        components['total_score'] = total
        
        return components


class QualityDecisionEngine:
    """
    Quality decision engine for accept/reject decisions.
    
    Features:
    - Composite quality scoring
    - Configurable thresholds and weights
    - Detailed decision logging
    - Statistics tracking
    """
    
    def __init__(self, config: Optional[QualityDecisionConfig] = None):
        if config is None:
            config = QualityDecisionConfig()
        
        self.config = config
        self.score_calculator = QualityScoreCalculator(config)
        
        # Tracking
        self.decisions = []
        self.accepted_count = 0
        self.rejected_count = 0
        
        # Setup logging
        if config.log_path:
            self.log_path = Path(config.log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_log_file()
        else:
            self.log_path = None
    
    def _init_log_file(self):
        """Initialize CSV log file."""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'image_id', 'psnr', 'ssim', 'lpips', 'mae', 'nrmse',
                'uncertainty', 'total_score', 'accepted', 'rejection_reason'
            ])
    
    def make_decision(
        self,
        metrics: Dict[str, float],
        uncertainty: Optional[float] = None,
        image_id: Optional[str] = None
    ) -> Dict:
        """
        Make accept/reject decision for an image.
        
        Args:
            metrics: Quality metrics dictionary
            uncertainty: Uncertainty value
            image_id: Optional image identifier
            
        Returns:
            Decision dictionary
        """
        # Compute score
        score_components = self.score_calculator.compute_score(metrics, uncertainty)
        total_score = score_components['total_score']
        
        # Make decision
        accepted = total_score >= self.config.acceptance_threshold
        
        # Determine rejection reason if rejected
        rejection_reason = None
        if not accepted:
            # Find the weakest component
            component_scores = {
                k: v for k, v in score_components.items()
                if k != 'total_score'
            }
            weakest = min(component_scores, key=lambda k: component_scores[k])
            rejection_reason = f"Low {weakest.replace('_score', '')}"
        
        # Create decision record
        decision = {
            'image_id': image_id,
            'metrics': metrics.copy(),
            'uncertainty': uncertainty,
            'score_components': score_components,
            'total_score': total_score,
            'threshold': self.config.acceptance_threshold,
            'accepted': accepted,
            'rejection_reason': rejection_reason,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update tracking
        if accepted:
            self.accepted_count += 1
        else:
            self.rejected_count += 1
        
        if self.config.log_all_decisions:
            self.decisions.append(decision)
        
        # Log to file
        if self.log_path:
            self._log_decision(decision)
        
        return decision
    
    def _log_decision(self, decision: Dict):
        """Log decision to CSV file."""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                decision['timestamp'],
                decision['image_id'],
                decision['metrics'].get('psnr', ''),
                decision['metrics'].get('ssim', ''),
                decision['metrics'].get('lpips', ''),
                decision['metrics'].get('mae', ''),
                decision['metrics'].get('nrmse', ''),
                decision['uncertainty'],
                decision['total_score'],
                decision['accepted'],
                decision['rejection_reason']
            ])
    
    def batch_decisions(
        self,
        metrics_list: List[Dict[str, float]],
        uncertainties: Optional[List[float]] = None,
        image_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Make decisions for a batch of images."""
        results = []
        
        for i, metrics in enumerate(metrics_list):
            uncertainty = uncertainties[i] if uncertainties else None
            image_id = image_ids[i] if image_ids else f"image_{i}"
            
            result = self.make_decision(metrics, uncertainty, image_id)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get decision statistics."""
        total = self.accepted_count + self.rejected_count
        
        stats = {
            'total_processed': total,
            'accepted': self.accepted_count,
            'rejected': self.rejected_count,
            'acceptance_rate': self.accepted_count / total if total > 0 else 0.0,
            'threshold': self.config.acceptance_threshold
        }
        
        # Score distribution
        if self.decisions:
            scores = [d['total_score'] for d in self.decisions]
            stats['score_mean'] = float(np.mean(scores))
            stats['score_std'] = float(np.std(scores))
            stats['score_min'] = float(np.min(scores))
            stats['score_max'] = float(np.max(scores))
        
        return stats
    
    def get_rejection_analysis(self) -> Dict[str, int]:
        """Analyze rejection reasons."""
        reasons = {}
        for decision in self.decisions:
            if not decision['accepted'] and decision['rejection_reason']:
                reason = decision['rejection_reason']
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons
    
    def reset(self):
        """Reset tracking."""
        self.decisions = []
        self.accepted_count = 0
        self.rejected_count = 0
    
    def save_report(self, path: str):
        """Save detailed report."""
        report = {
            'config': {
                'threshold': self.config.acceptance_threshold,
                'weights': {
                    'psnr': self.config.weights.psnr,
                    'ssim': self.config.weights.ssim,
                    'lpips': self.config.weights.lpips,
                    'mae': self.config.weights.mae,
                    'nrmse': self.config.weights.nrmse,
                    'uncertainty': self.config.weights.uncertainty
                }
            },
            'statistics': self.get_statistics(),
            'rejection_analysis': self.get_rejection_analysis(),
            'decisions': self.decisions if len(self.decisions) <= 1000 else self.decisions[:1000]
        }
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


class DatasetExpansionValidator:
    """
    Complete validation pipeline for dataset expansion.
    
    Integrates metrics computation, quality scoring, and decision making.
    """
    
    def __init__(
        self,
        decision_config: Optional[QualityDecisionConfig] = None,
        output_dir: Optional[str] = None
    ):
        self.decision_engine = QualityDecisionEngine(decision_config)
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.accepted_dir = self.output_dir / 'accepted'
            self.rejected_dir = self.output_dir / 'rejected'
            self.accepted_dir.mkdir(exist_ok=True)
            self.rejected_dir.mkdir(exist_ok=True)
        else:
            self.output_dir = None
    
    def validate_and_save(
        self,
        generated: torch.Tensor,
        reference: torch.Tensor,
        metrics: Dict[str, float],
        uncertainty: Optional[float] = None,
        image_id: str = None,
        save_images: bool = True
    ) -> Dict:
        """
        Validate image and optionally save to appropriate directory.
        
        Args:
            generated: Generated image tensor
            reference: Reference image tensor
            metrics: Pre-computed metrics
            uncertainty: Uncertainty value
            image_id: Image identifier
            save_images: Whether to save images
            
        Returns:
            Decision dictionary
        """
        decision = self.decision_engine.make_decision(metrics, uncertainty, image_id)
        
        if save_images and self.output_dir:
            save_dir = self.accepted_dir if decision['accepted'] else self.rejected_dir
            
            # Save as numpy
            gen_path = save_dir / f"{image_id}_generated.npy"
            ref_path = save_dir / f"{image_id}_reference.npy"
            
            np.save(gen_path, generated.cpu().numpy())
            np.save(ref_path, reference.cpu().numpy())
            
            # Save metadata
            meta_path = save_dir / f"{image_id}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'uncertainty': uncertainty,
                    'score': decision['total_score'],
                    'accepted': decision['accepted']
                }, f, indent=2)
        
        return decision
    
    def get_expansion_summary(self) -> Dict:
        """Get summary of dataset expansion."""
        stats = self.decision_engine.get_statistics()
        
        summary = {
            'total_generated': stats['total_processed'],
            'accepted_for_expansion': stats['accepted'],
            'rejected': stats['rejected'],
            'acceptance_rate': stats['acceptance_rate'],
            'expansion_factor': stats['accepted'] / max(1, stats['total_processed'] - stats['accepted'])
        }
        
        if 'score_mean' in stats:
            summary['quality_summary'] = {
                'mean_score': stats['score_mean'],
                'std_score': stats['score_std'],
                'min_score': stats['score_min'],
                'max_score': stats['score_max']
            }
        
        return summary
    
    def finalize(self, report_path: Optional[str] = None):
        """Finalize validation and save report."""
        summary = self.get_expansion_summary()
        
        if report_path:
            self.decision_engine.save_report(report_path)
        elif self.output_dir:
            self.decision_engine.save_report(self.output_dir / 'validation_report.json')
        
        return summary


def create_quality_decision_engine(
    acceptance_threshold: float = 0.70,
    **weight_kwargs
) -> QualityDecisionEngine:
    """
    Factory function to create quality decision engine.
    
    Args:
        acceptance_threshold: Minimum score for acceptance
        **weight_kwargs: Weight parameters
        
    Returns:
        QualityDecisionEngine instance
    """
    weights = QualityWeights(**weight_kwargs)
    config = QualityDecisionConfig(
        acceptance_threshold=acceptance_threshold,
        weights=weights
    )
    return QualityDecisionEngine(config)
