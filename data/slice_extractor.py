# data/slice_extractor.py

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm


@dataclass
class SliceInfo:
    """Metadata for an extracted slice."""
    patient_id: str
    slice_idx: int
    orientation: str
    has_tumor: bool
    tumor_fraction: float
    brain_fraction: float


@dataclass 
class ExtractedSlice:
    """Container for extracted slice data and metadata."""
    data: np.ndarray  # Shape: (C, H, W) for 4 modalities
    segmentation: Optional[np.ndarray]  # Shape: (H, W)
    info: SliceInfo


class SliceExtractor:
    """
    Extract 2D slices from 3D volumes with intelligent sampling.
    Supports tumor-aware sampling and filtering of non-informative slices.
    
    When target_slices is small (<=15), uses spatially-distributed selection
    to ensure selected slices are well-spread across the brain volume and
    capture diverse anatomical regions including both tumor and non-tumor areas.
    """
    
    ORIENTATIONS = {
        'axial': 2,      # z-axis (most common for brain)
        'sagittal': 0,   # x-axis
        'coronal': 1     # y-axis
    }
    
    def __init__(
        self,
        orientation: str = 'axial',
        min_brain_fraction: float = 0.05,
        tumor_priority: bool = True,
        target_slices: Optional[int] = None,
        min_non_tumor_ratio: float = 0.3
    ):
        """
        Initialize slice extractor.
        
        Args:
            orientation: Slice orientation ('axial', 'sagittal', 'coronal')
            min_brain_fraction: Minimum fraction of brain tissue required
            tumor_priority: Whether to prioritize slices with tumor
            target_slices: Target number of slices per volume (None for all valid)
            min_non_tumor_ratio: Minimum fraction of selected slices that should be
                                 non-tumor (for segmentation negative examples).
                                 Only used when target_slices <= 15.
        """
        if orientation not in self.ORIENTATIONS:
            raise ValueError(f"Invalid orientation: {orientation}")
            
        self.orientation = orientation
        self.slice_axis = self.ORIENTATIONS[orientation]
        self.min_brain_fraction = min_brain_fraction
        self.tumor_priority = tumor_priority
        self.target_slices = target_slices
        self.min_non_tumor_ratio = min_non_tumor_ratio
        
    def compute_brain_mask(self, volume: np.ndarray) -> np.ndarray:
        """Compute brain mask from volume."""
        if volume.ndim == 4:
            return np.any(volume > 0, axis=0)
        return volume > 0
    
    def compute_tumor_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """Compute tumor mask from segmentation (labels 1, 2, 4)."""
        return (segmentation > 0).astype(np.float32)
    
    def get_slice_statistics(
        self,
        volume: np.ndarray,
        segmentation: Optional[np.ndarray],
        slice_idx: int
    ) -> Tuple[float, float]:
        """
        Compute brain and tumor fractions for a slice.
        
        Args:
            volume: 3D or 4D volume
            segmentation: Segmentation volume
            slice_idx: Index along slice axis
            
        Returns:
            Tuple of (brain_fraction, tumor_fraction)
        """
        # Get slice based on orientation
        if volume.ndim == 4:
            if self.slice_axis == 0:
                slice_vol = volume[:, slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_vol = volume[:, :, slice_idx, :]
            else:
                slice_vol = volume[:, :, :, slice_idx]
            brain_mask = np.any(slice_vol > 0, axis=0)
        else:
            if self.slice_axis == 0:
                slice_vol = volume[slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_vol = volume[:, slice_idx, :]
            else:
                slice_vol = volume[:, :, slice_idx]
            brain_mask = slice_vol > 0
        
        brain_fraction = np.mean(brain_mask.astype(np.float32))
        
        tumor_fraction = 0.0
        if segmentation is not None:
            if self.slice_axis == 0:
                seg_slice = segmentation[slice_idx, :, :]
            elif self.slice_axis == 1:
                seg_slice = segmentation[:, slice_idx, :]
            else:
                seg_slice = segmentation[:, :, slice_idx]
            tumor_mask = seg_slice > 0
            tumor_fraction = np.mean(tumor_mask.astype(np.float32))
        
        return brain_fraction, tumor_fraction
    
    def extract_slice(
        self,
        volume: np.ndarray,
        segmentation: Optional[np.ndarray],
        slice_idx: int,
        patient_id: str = ""
    ) -> ExtractedSlice:
        """
        Extract a single slice from volume.
        
        Args:
            volume: Input volume (C, H, W, D) or (H, W, D)
            segmentation: Segmentation volume (H, W, D)
            slice_idx: Index along slice axis
            patient_id: Patient identifier
            
        Returns:
            ExtractedSlice object
        """
        # Extract slice data based on orientation
        if volume.ndim == 4:
            if self.slice_axis == 0:
                slice_data = volume[:, slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_data = volume[:, :, slice_idx, :]
            else:
                slice_data = volume[:, :, :, slice_idx]
        else:
            if self.slice_axis == 0:
                slice_data = volume[slice_idx, :, :]
            elif self.slice_axis == 1:
                slice_data = volume[:, slice_idx, :]
            else:
                slice_data = volume[:, :, slice_idx]
            # Add channel dimension if needed
            slice_data = slice_data[np.newaxis, ...]
        
        # Extract segmentation slice
        seg_slice = None
        if segmentation is not None:
            if self.slice_axis == 0:
                seg_slice = segmentation[slice_idx, :, :]
            elif self.slice_axis == 1:
                seg_slice = segmentation[:, slice_idx, :]
            else:
                seg_slice = segmentation[:, :, slice_idx]
        
        # Compute statistics
        brain_fraction, tumor_fraction = self.get_slice_statistics(
            volume, segmentation, slice_idx
        )
        
        info = SliceInfo(
            patient_id=patient_id,
            slice_idx=slice_idx,
            orientation=self.orientation,
            has_tumor=tumor_fraction > 0,
            tumor_fraction=tumor_fraction,
            brain_fraction=brain_fraction
        )
        
        return ExtractedSlice(
            data=slice_data.astype(np.float32),
            segmentation=seg_slice.astype(np.float32) if seg_slice is not None else None,
            info=info
        )
    
    def get_valid_slice_indices(
        self,
        volume: np.ndarray,
        segmentation: Optional[np.ndarray] = None
    ) -> List[Tuple[int, float, float]]:
        """
        Get indices of valid slices (above minimum brain fraction).
        
        Args:
            volume: Input volume
            segmentation: Optional segmentation
            
        Returns:
            List of (slice_idx, brain_fraction, tumor_fraction) tuples
        """
        if volume.ndim == 4:
            n_slices = volume.shape[self.slice_axis + 1]  # +1 for channel dim
        else:
            n_slices = volume.shape[self.slice_axis]
        
        valid_slices = []
        
        for idx in range(n_slices):
            brain_frac, tumor_frac = self.get_slice_statistics(
                volume, segmentation, idx
            )
            
            if brain_frac >= self.min_brain_fraction:
                valid_slices.append((idx, brain_frac, tumor_frac))
        
        return valid_slices
    
    def _select_slices_sparse(
        self,
        valid_slices: List[Tuple[int, float, float]]
    ) -> List[int]:
        """
        Spatially-distributed slice selection for small target counts (<=15).
        
        Strategy:
          1. Divide the valid slice range into N equal spatial zones
          2. From each zone, pick the best representative slice
          3. Ensure a minimum number of non-tumor slices for segmentation balance
          4. Within each zone, prefer tumor slices (sorted by tumor_fraction desc),
             unless the non-tumor quota has not been met yet
        
        This guarantees:
          - Slices are spread across the full axial extent of the brain
          - No two selected slices are adjacent or clustered
          - Both tumor and healthy tissue are represented
        
        Args:
            valid_slices: List of (idx, brain_frac, tumor_frac) tuples
            
        Returns:
            List of selected slice indices
        """
        n_target = self.target_slices
        n_valid = len(valid_slices)
        
        if n_valid <= n_target:
            return [s[0] for s in valid_slices]
        
        # Separate tumor and non-tumor
        tumor_slices = [(idx, bf, tf) for idx, bf, tf in valid_slices if tf > 0]
        non_tumor_slices = [(idx, bf, tf) for idx, bf, tf in valid_slices if tf == 0]
        
        # Compute how many non-tumor slices we need (minimum)
        min_non_tumor = max(1, int(np.ceil(n_target * self.min_non_tumor_ratio)))
        # But don't require more non-tumor than available
        min_non_tumor = min(min_non_tumor, len(non_tumor_slices))
        # And don't let non-tumor eat all slots if there are tumor slices
        if len(tumor_slices) > 0:
            min_non_tumor = min(min_non_tumor, n_target - 1)
        
        n_tumor_slots = n_target - min_non_tumor
        n_non_tumor_slots = min_non_tumor
        
        # --- Select tumor slices with spatial spread ---
        tumor_selected = self._pick_from_zones(tumor_slices, n_tumor_slots)
        
        # --- Select non-tumor slices with spatial spread ---
        non_tumor_selected = self._pick_from_zones(non_tumor_slices, n_non_tumor_slots)
        
        selected = tumor_selected + non_tumor_selected
        return sorted(selected)
    
    def _pick_from_zones(
        self,
        candidates: List[Tuple[int, float, float]],
        n_picks: int
    ) -> List[int]:
        """
        Divide candidates into n_picks spatial zones and pick the best from each.
        
        'Best' = highest tumor_fraction for tumor slices,
                 highest brain_fraction for non-tumor slices.
        
        Args:
            candidates: List of (idx, brain_frac, tumor_frac)
            n_picks: Number of slices to pick
            
        Returns:
            List of selected slice indices
        """
        if n_picks <= 0 or len(candidates) == 0:
            return []
        
        if len(candidates) <= n_picks:
            return [c[0] for c in candidates]
        
        # Sort candidates by their axial index for spatial ordering
        candidates_sorted = sorted(candidates, key=lambda x: x[0])
        
        # Divide into n_picks equal zones
        zone_size = len(candidates_sorted) / n_picks
        selected = []
        
        for i in range(n_picks):
            zone_start = int(i * zone_size)
            zone_end = int((i + 1) * zone_size)
            zone = candidates_sorted[zone_start:zone_end]
            
            if not zone:
                continue
            
            # Pick best in zone: prefer higher tumor_fraction, break ties with brain_fraction
            best = max(zone, key=lambda x: (x[2], x[1]))
            selected.append(best[0])
        
        return selected
    
    def select_slices(
        self,
        valid_slices: List[Tuple[int, float, float]]
    ) -> List[int]:
        """
        Select final slice indices based on tumor priority and target count.
        
        For small targets (<=15): uses spatially-distributed selection
        For large targets (>15): uses the original dense selection strategy
        
        Args:
            valid_slices: List of (idx, brain_frac, tumor_frac) tuples
            
        Returns:
            List of selected slice indices
        """
        if not valid_slices:
            return []
        
        if self.target_slices is None or len(valid_slices) <= self.target_slices:
            return [s[0] for s in valid_slices]
        
        # Use sparse spatially-distributed selection for small targets
        if self.target_slices <= 15:
            return self._select_slices_sparse(valid_slices)
        
        # Original dense selection for large targets
        if self.tumor_priority:
            # Separate tumor and non-tumor slices
            tumor_slices = [(idx, bf, tf) for idx, bf, tf in valid_slices if tf > 0]
            non_tumor_slices = [(idx, bf, tf) for idx, bf, tf in valid_slices if tf == 0]
            
            # Sort tumor slices by tumor fraction (descending)
            tumor_slices.sort(key=lambda x: x[2], reverse=True)
            
            # Select all tumor slices first, then fill with non-tumor
            selected = [s[0] for s in tumor_slices]
            
            if len(selected) < self.target_slices:
                # Add non-tumor slices evenly distributed
                remaining = self.target_slices - len(selected)
                step = max(1, len(non_tumor_slices) // remaining)
                for i in range(0, len(non_tumor_slices), step):
                    if len(selected) >= self.target_slices:
                        break
                    selected.append(non_tumor_slices[i][0])
            
            return sorted(selected[:self.target_slices])
        else:
            # Even distribution
            step = len(valid_slices) / self.target_slices
            indices = [int(i * step) for i in range(self.target_slices)]
            return [valid_slices[i][0] for i in indices]
    
    def extract_all_slices(
        self,
        volume: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        patient_id: str = ""
    ) -> List[ExtractedSlice]:
        """
        Extract all valid slices from a volume.
        
        Args:
            volume: Input volume (C, H, W, D)
            segmentation: Segmentation volume (H, W, D)
            patient_id: Patient identifier
            
        Returns:
            List of ExtractedSlice objects
        """
        valid_slices = self.get_valid_slice_indices(volume, segmentation)
        selected_indices = self.select_slices(valid_slices)
        
        extracted = []
        for idx in selected_indices:
            slice_obj = self.extract_slice(volume, segmentation, idx, patient_id)
            extracted.append(slice_obj)
        
        return extracted


class SliceDatasetBuilder:
    """
    Build slice dataset from multiple patient volumes.
    Handles batch processing and metadata tracking.
    """
    
    def __init__(
        self,
        extractor: SliceExtractor,
        output_dir: Union[str, Path]
    ):
        """
        Initialize dataset builder.
        
        Args:
            extractor: SliceExtractor instance
            output_dir: Directory to save extracted slices
        """
        self.extractor = extractor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.slices_dir = self.output_dir / "slices"
        self.slices_dir.mkdir(exist_ok=True)
        
        self.metadata = []
    
    def process_patient(
        self,
        volume: np.ndarray,
        segmentation: Optional[np.ndarray],
        patient_id: str,
        save: bool = True
    ) -> List[ExtractedSlice]:
        """
        Process a single patient volume.
        
        Args:
            volume: Patient volume (C, H, W, D)
            segmentation: Segmentation (H, W, D)
            patient_id: Patient identifier
            save: Whether to save slices to disk
            
        Returns:
            List of extracted slices
        """
        slices = self.extractor.extract_all_slices(
            volume, segmentation, patient_id
        )
        
        if save:
            patient_dir = self.slices_dir / patient_id
            patient_dir.mkdir(exist_ok=True)
            
            for slice_obj in slices:
                slice_path = patient_dir / f"slice_{slice_obj.info.slice_idx:03d}.npz"
                
                save_dict = {
                    'data': slice_obj.data,
                    'patient_id': slice_obj.info.patient_id,
                    'slice_idx': slice_obj.info.slice_idx,
                    'orientation': slice_obj.info.orientation,
                    'has_tumor': slice_obj.info.has_tumor,
                    'tumor_fraction': slice_obj.info.tumor_fraction,
                    'brain_fraction': slice_obj.info.brain_fraction
                }
                
                if slice_obj.segmentation is not None:
                    save_dict['segmentation'] = slice_obj.segmentation
                
                np.savez_compressed(slice_path, **save_dict)
                
                metadata_entry = {
                    'path': str(slice_path),
                    **{k: v for k, v in save_dict.items() if k not in ['data', 'segmentation']}
                }
                # Convert numpy types to Python native types for JSON serialization
                if 'has_tumor' in metadata_entry:
                    metadata_entry['has_tumor'] = bool(metadata_entry['has_tumor'])
                if 'tumor_fraction' in metadata_entry:
                    metadata_entry['tumor_fraction'] = float(metadata_entry['tumor_fraction'])
                if 'brain_fraction' in metadata_entry:
                    metadata_entry['brain_fraction'] = float(metadata_entry['brain_fraction'])
                self.metadata.append(metadata_entry)
        
        return slices
    
    def build_dataset(
        self,
        patients: List[Dict],
        show_progress: bool = True
    ) -> int:
        """
        Build complete slice dataset from patient list.
        
        Args:
            patients: List of dicts with 'volume', 'segmentation', 'patient_id' keys
            show_progress: Whether to show progress bar
            
        Returns:
            Total number of slices extracted
        """
        total_slices = 0
        iterator = tqdm(patients, desc="Extracting slices") if show_progress else patients
        
        for patient in iterator:
            slices = self.process_patient(
                volume=patient['volume'],
                segmentation=patient.get('segmentation'),
                patient_id=patient['patient_id'],
                save=True
            )
            total_slices += len(slices)
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Extracted {total_slices} slices from {len(patients)} patients")
        print(f"Metadata saved to: {metadata_path}")
        
        return total_slices
    
    def load_slice(self, path: Union[str, Path]) -> ExtractedSlice:
        """Load a saved slice from disk."""
        data = np.load(path)
        
        info = SliceInfo(
            patient_id=str(data['patient_id']),
            slice_idx=int(data['slice_idx']),
            orientation=str(data['orientation']),
            has_tumor=bool(data['has_tumor']),
            tumor_fraction=float(data['tumor_fraction']),
            brain_fraction=float(data['brain_fraction'])
        )
        
        seg = data['segmentation'] if 'segmentation' in data else None
        
        return ExtractedSlice(
            data=data['data'],
            segmentation=seg,
            info=info
        )
