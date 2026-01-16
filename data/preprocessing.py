# data/preprocessing.py

import numpy as np
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class NormalizationParams:
    """Store normalization parameters for denormalization."""
    mean: np.ndarray
    std: np.ndarray
    min_val: float
    max_val: float
    clip_low: float
    clip_high: float


class IntensityNormalizer:
    """
    Intensity normalization for medical images.
    Handles Z-score normalization within brain mask with percentile clipping.
    """
    
    def __init__(
        self,
        clip_percentile_low: float = 1.0,
        clip_percentile_high: float = 99.0,
        target_range: Tuple[float, float] = (0, 1),
        per_channel: bool = True
    ):
        """
        Initialize intensity normalizer.
        
        Args:
            clip_percentile_low: Lower percentile for clipping (default: 1.0)
            clip_percentile_high: Upper percentile for clipping (default: 99.0)
            target_range: Target range for final scaling (default: [0, 1])
            per_channel: Whether to normalize each channel independently
        """
        self.clip_percentile_low = clip_percentile_low
        self.clip_percentile_high = clip_percentile_high
        self.target_range = target_range
        self.per_channel = per_channel
        
    def compute_brain_mask(self, volume: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Compute brain mask by excluding background (zero-valued voxels).
        
        Args:
            volume: Input volume (H, W, D) or (C, H, W, D)
            threshold: Threshold for background (default: 0.0)
            
        Returns:
            Binary brain mask
        """
        if volume.ndim == 4:
            # For multi-channel, use any channel above threshold
            mask = np.any(volume > threshold, axis=0)
        else:
            mask = volume > threshold
            
        return mask.astype(np.float32)
    
    def zscore_normalize(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, NormalizationParams]:
        """
        Apply Z-score normalization within brain region.
        
        Args:
            volume: Input volume (C, H, W, D) or (H, W, D)
            mask: Optional brain mask
            
        Returns:
            Tuple of (normalized_volume, normalization_params)
        """
        if mask is None:
            mask = self.compute_brain_mask(volume)
        
        volume = volume.astype(np.float32)
        normalized = np.zeros_like(volume)
        
        if volume.ndim == 4 and self.per_channel:
            # Normalize each channel independently
            means = []
            stds = []
            
            for c in range(volume.shape[0]):
                channel = volume[c]
                brain_values = channel[mask > 0]
                
                if len(brain_values) > 0:
                    mean = np.mean(brain_values)
                    std = np.std(brain_values)
                    std = std if std > 1e-8 else 1.0
                else:
                    mean = 0.0
                    std = 1.0
                
                means.append(mean)
                stds.append(std)
                
                normalized[c] = (channel - mean) / std
                normalized[c] = normalized[c] * mask  # Zero out background
                
            params = NormalizationParams(
                mean=np.array(means),
                std=np.array(stds),
                min_val=0.0,
                max_val=1.0,
                clip_low=self.clip_percentile_low,
                clip_high=self.clip_percentile_high
            )
        else:
            # Normalize entire volume together
            if volume.ndim == 4:
                brain_values = volume[:, mask > 0].flatten()
            else:
                brain_values = volume[mask > 0]
            
            mean = np.mean(brain_values)
            std = np.std(brain_values)
            std = std if std > 1e-8 else 1.0
            
            normalized = (volume - mean) / std
            if volume.ndim == 4:
                normalized = normalized * mask[np.newaxis, ...]
            else:
                normalized = normalized * mask
            
            params = NormalizationParams(
                mean=np.array([mean]),
                std=np.array([std]),
                min_val=0.0,
                max_val=1.0,
                clip_low=self.clip_percentile_low,
                clip_high=self.clip_percentile_high
            )
            
        return normalized, params
    
    def percentile_clip(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Clip intensities at specified percentiles.
        
        Args:
            volume: Input volume (C, H, W, D) or (H, W, D)
            mask: Optional brain mask
            
        Returns:
            Clipped volume
        """
        if mask is None:
            mask = self.compute_brain_mask(volume)
            
        volume = volume.astype(np.float32)
        clipped = volume.copy()
        
        if volume.ndim == 4:
            for c in range(volume.shape[0]):
                channel = volume[c]
                brain_values = channel[mask > 0]
                
                if len(brain_values) > 0:
                    p_low = np.percentile(brain_values, self.clip_percentile_low)
                    p_high = np.percentile(brain_values, self.clip_percentile_high)
                    clipped[c] = np.clip(channel, p_low, p_high)
        else:
            brain_values = volume[mask > 0]
            if len(brain_values) > 0:
                p_low = np.percentile(brain_values, self.clip_percentile_low)
                p_high = np.percentile(brain_values, self.clip_percentile_high)
                clipped = np.clip(volume, p_low, p_high)
                
        return clipped
    
    def scale_to_range(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Scale volume to target range.
        
        Args:
            volume: Input volume
            mask: Optional brain mask
            
        Returns:
            Scaled volume
        """
        if mask is None:
            mask = self.compute_brain_mask(volume)
            
        volume = volume.astype(np.float32)
        scaled = np.zeros_like(volume)
        
        target_min, target_max = self.target_range
        
        if volume.ndim == 4:
            for c in range(volume.shape[0]):
                channel = volume[c]
                brain_values = channel[mask > 0]
                
                if len(brain_values) > 0:
                    v_min = np.min(brain_values)
                    v_max = np.max(brain_values)
                    
                    if v_max - v_min > 1e-8:
                        scaled[c] = (channel - v_min) / (v_max - v_min)
                        scaled[c] = scaled[c] * (target_max - target_min) + target_min
                    else:
                        scaled[c] = np.zeros_like(channel)
                        
                scaled[c] = scaled[c] * mask
        else:
            brain_values = volume[mask > 0]
            if len(brain_values) > 0:
                v_min = np.min(brain_values)
                v_max = np.max(brain_values)
                
                if v_max - v_min > 1e-8:
                    scaled = (volume - v_min) / (v_max - v_min)
                    scaled = scaled * (target_max - target_min) + target_min
                    
            scaled = scaled * mask
            
        return scaled
    
    def normalize(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        return_params: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
        """
        Apply full normalization pipeline: Z-score → Percentile clip → Scale to range.
        
        Args:
            volume: Input volume (C, H, W, D) or (H, W, D)
            mask: Optional brain mask
            return_params: Whether to return normalization parameters
            
        Returns:
            Normalized volume, optionally with parameters
        """
        if mask is None:
            mask = self.compute_brain_mask(volume)
        
        # Step 1: Z-score normalization
        normalized, params = self.zscore_normalize(volume, mask)
        
        # Step 2: Percentile clipping
        normalized = self.percentile_clip(normalized, mask)
        
        # Step 3: Scale to target range
        normalized = self.scale_to_range(normalized, mask)
        
        if return_params:
            return normalized, params
        return normalized
    
    def denormalize(
        self,
        volume: np.ndarray,
        params: NormalizationParams
    ) -> np.ndarray:
        """
        Reverse normalization using stored parameters.
        
        Args:
            volume: Normalized volume
            params: Normalization parameters
            
        Returns:
            Denormalized volume
        """
        # Reverse scale to range
        target_min, target_max = self.target_range
        volume = (volume - target_min) / (target_max - target_min)
        
        # Reverse z-score (approximate)
        if volume.ndim == 4:
            for c in range(volume.shape[0]):
                volume[c] = volume[c] * params.std[c] + params.mean[c]
        else:
            volume = volume * params.std[0] + params.mean[0]
            
        return volume


class VolumePreprocessor:
    """
    Complete preprocessing pipeline for BraTS volumes.
    Combines loading, normalization, and quality checks.
    """
    
    def __init__(
        self,
        clip_percentile_low: float = 1.0,
        clip_percentile_high: float = 99.0,
        target_range: Tuple[float, float] = (0, 1)
    ):
        """
        Initialize volume preprocessor.
        
        Args:
            clip_percentile_low: Lower percentile for clipping
            clip_percentile_high: Upper percentile for clipping
            target_range: Target intensity range
        """
        self.normalizer = IntensityNormalizer(
            clip_percentile_low=clip_percentile_low,
            clip_percentile_high=clip_percentile_high,
            target_range=target_range
        )
        
    def preprocess_volume(
        self,
        volume: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        return_mask: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Preprocess a single volume.
        
        Args:
            volume: Input volume (C, H, W, D) - 4 modalities stacked
            segmentation: Optional segmentation mask
            return_mask: Whether to return brain mask
            
        Returns:
            Preprocessed volume, optionally with brain mask
        """
        # Compute brain mask from all modalities
        brain_mask = self.normalizer.compute_brain_mask(volume)
        
        # Normalize
        normalized = self.normalizer.normalize(volume, brain_mask)
        
        if return_mask:
            return normalized, brain_mask
        return normalized
    
    def compute_statistics(self, volume: np.ndarray) -> Dict:
        """
        Compute statistics for a volume (for logging/debugging).
        
        Args:
            volume: Input volume
            
        Returns:
            Dictionary of statistics
        """
        mask = self.normalizer.compute_brain_mask(volume)
        
        stats = {
            'shape': volume.shape,
            'dtype': str(volume.dtype),
            'brain_fraction': float(np.mean(mask)),
        }
        
        if volume.ndim == 4:
            for c in range(volume.shape[0]):
                brain_values = volume[c][mask > 0]
                stats[f'channel_{c}'] = {
                    'min': float(np.min(brain_values)),
                    'max': float(np.max(brain_values)),
                    'mean': float(np.mean(brain_values)),
                    'std': float(np.std(brain_values))
                }
        
        return stats
    
    def save_params(self, params: NormalizationParams, path: Union[str, Path]):
        """Save normalization parameters to JSON."""
        params_dict = {
            'mean': params.mean.tolist(),
            'std': params.std.tolist(),
            'min_val': params.min_val,
            'max_val': params.max_val,
            'clip_low': params.clip_low,
            'clip_high': params.clip_high
        }
        
        with open(path, 'w') as f:
            json.dump(params_dict, f, indent=2)
    
    def load_params(self, path: Union[str, Path]) -> NormalizationParams:
        """Load normalization parameters from JSON."""
        with open(path, 'r') as f:
            params_dict = json.load(f)
            
        return NormalizationParams(
            mean=np.array(params_dict['mean']),
            std=np.array(params_dict['std']),
            min_val=params_dict['min_val'],
            max_val=params_dict['max_val'],
            clip_low=params_dict['clip_low'],
            clip_high=params_dict['clip_high']
        )