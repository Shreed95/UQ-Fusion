# data/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List, Union, Callable
from pathlib import Path
import json
from dataclasses import dataclass
import random

from .nifti_loader import NIfTILoader, CachedNIfTILoader, PatientVolume
from .preprocessing import VolumePreprocessor, IntensityNormalizer
from .slice_extractor import SliceExtractor, ExtractedSlice, SliceDatasetBuilder
from .augmentation import MedicalImageAugmentor, AugmentationConfig


@dataclass
class DatasetConfig:
    """Configuration for dataset."""
    data_dir: str
    cache_dir: Optional[str] = None
    slices_dir: Optional[str] = None
    
    # Preprocessing
    clip_percentile_low: float = 1.0
    clip_percentile_high: float = 99.0
    target_range: Tuple[float, float] = (0, 1)
    
    # Slice extraction
    orientation: str = 'axial'
    min_brain_fraction: float = 0.05
    tumor_priority: bool = True
    target_slices: Optional[int] = 100
    
    # Augmentation
    augmentation_enabled: bool = True
    geometric_prob: float = 0.5
    intensity_prob: float = 0.5
    
    # DataLoader
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True


class BraTSSliceDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2D slices.
    Loads pre-extracted slices from disk with optional augmentation.
    """
    
    def __init__(
        self,
        slices_dir: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        augmentor: Optional[MedicalImageAugmentor] = None,
        return_segmentation: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            slices_dir: Directory containing extracted slices
            metadata_file: Path to metadata JSON file
            augmentor: Optional augmentation pipeline
            return_segmentation: Whether to return segmentation masks
            transform: Additional custom transform
        """
        self.slices_dir = Path(slices_dir)
        self.augmentor = augmentor
        self.return_segmentation = return_segmentation
        self.transform = transform
        
        # Load metadata
        if metadata_file is None:
            metadata_file = self.slices_dir.parent / "metadata.json"
        
        if Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            self.slice_paths = [m['path'] for m in self.metadata]
        else:
            # Discover slices from directory
            self.slice_paths = self._discover_slices()
            self.metadata = None
        
        print(f"Dataset initialized with {len(self.slice_paths)} slices")
    
    def _discover_slices(self) -> List[str]:
        """Discover all slice files in directory."""
        slice_paths = []
        
        for patient_dir in sorted(self.slices_dir.iterdir()):
            if patient_dir.is_dir():
                for slice_file in sorted(patient_dir.glob("*.npz")):
                    slice_paths.append(str(slice_file))
        
        return slice_paths
    
    def __len__(self) -> int:
        return len(self.slice_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single slice.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with 'image' and optionally 'segmentation' tensors
        """
        # Load slice
        slice_data = np.load(self.slice_paths[idx])
        image = slice_data['data'].astype(np.float32)
        
        segmentation = None
        if self.return_segmentation and 'segmentation' in slice_data:
            segmentation = slice_data['segmentation'].astype(np.float32)
        
        # Apply augmentation
        if self.augmentor is not None:
            image, segmentation = self.augmentor(image, segmentation)
        
        # Apply custom transform
        if self.transform is not None:
            image = self.transform(image)
            if segmentation is not None:
                segmentation = self.transform(segmentation)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        
        result = {
            'image': image_tensor,
            'patient_id': str(slice_data.get('patient_id', '')),
            'slice_idx': int(slice_data.get('slice_idx', idx))
        }
        
        if segmentation is not None:
            result['segmentation'] = torch.from_numpy(segmentation).float()
        
        return result


class BraTSVolumeDataset(Dataset):
    """
    PyTorch Dataset that loads full 3D volumes and extracts slices on-the-fly.
    More memory efficient for large datasets.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: DatasetConfig,
        patient_ids: Optional[List[str]] = None,
        augmentor: Optional[MedicalImageAugmentor] = None,
        return_segmentation: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to BraTS data directory
            config: Dataset configuration
            patient_ids: List of patient IDs to include (None for all)
            augmentor: Optional augmentation pipeline
            return_segmentation: Whether to return segmentation masks
        """
        self.config = config
        self.augmentor = augmentor
        self.return_segmentation = return_segmentation
        
        # Initialize loader
        if config.cache_dir:
            self.loader = CachedNIfTILoader(data_dir, config.cache_dir)
        else:
            self.loader = NIfTILoader(data_dir)
        
        # Filter patients if specified
        if patient_ids is not None:
            self.patient_ids = patient_ids
        else:
            self.patient_ids = self.loader.get_patient_ids()
        
        # Initialize preprocessor
        self.preprocessor = VolumePreprocessor(
            clip_percentile_low=config.clip_percentile_low,
            clip_percentile_high=config.clip_percentile_high,
            target_range=config.target_range
        )
        
        # Initialize slice extractor
        self.slice_extractor = SliceExtractor(
            orientation=config.orientation,
            min_brain_fraction=config.min_brain_fraction,
            tumor_priority=config.tumor_priority,
            target_slices=config.target_slices
        )
        
        # Build slice index
        self._build_slice_index()
    
    def _build_slice_index(self):
        """Build index mapping dataset indices to (patient_id, slice_idx)."""
        self.slice_index = []
        
        for patient_id in self.patient_ids:
            # Get number of valid slices for this patient
            # For efficiency, we estimate based on target_slices
            n_slices = self.config.target_slices or 100
            
            for slice_idx in range(n_slices):
                self.slice_index.append((patient_id, slice_idx))
        
        print(f"Dataset initialized with ~{len(self.slice_index)} slices from {len(self.patient_ids)} patients")
    
    def __len__(self) -> int:
        return len(self.slice_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single slice."""
        patient_id, relative_slice_idx = self.slice_index[idx]
        
        # Load and preprocess volume
        patient = self.loader.load_patient_by_id(patient_id)
        volume = self.loader.get_stacked_modalities(patient)  # (4, H, W, D)
        
        # Preprocess
        volume = self.preprocessor.preprocess_volume(volume, patient.seg)
        
        # Extract slices
        slices = self.slice_extractor.extract_all_slices(
            volume, patient.seg, patient_id
        )
        
        # Get the slice at relative index
        if relative_slice_idx >= len(slices):
            relative_slice_idx = relative_slice_idx % len(slices)
        
        slice_obj = slices[relative_slice_idx]
        image = slice_obj.data
        segmentation = slice_obj.segmentation
        
        # Apply augmentation
        if self.augmentor is not None:
            image, segmentation = self.augmentor(image, segmentation)
        
        # Convert to tensors
        result = {
            'image': torch.from_numpy(image).float(),
            'patient_id': patient_id,
            'slice_idx': slice_obj.info.slice_idx
        }
        
        if self.return_segmentation and segmentation is not None:
            result['segmentation'] = torch.from_numpy(segmentation).float()
        
        return result


class BraTSDataModule:
    """
    Data module that handles dataset creation, splitting, and dataloader setup.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize data module.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        
        # Initialize augmentor
        if config.augmentation_enabled:
            aug_config = AugmentationConfig()
            self.train_augmentor = MedicalImageAugmentor(
                config=aug_config,
                geometric_prob=config.geometric_prob,
                intensity_prob=config.intensity_prob
            )
        else:
            self.train_augmentor = None
        
        self.val_augmentor = None  # No augmentation for validation
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self, force_rebuild: bool = False):
        """
        Prepare data by extracting slices from volumes.
        Call this before setup() if slices haven't been extracted.
        """
        slices_dir = Path(self.config.slices_dir or "./data/slices")
        
        if slices_dir.exists() and not force_rebuild:
            print(f"Slices directory exists: {slices_dir}")
            return
        
        print("Extracting slices from volumes...")
        
        # Initialize components
        loader = NIfTILoader(self.config.data_dir)
        preprocessor = VolumePreprocessor(
            clip_percentile_low=self.config.clip_percentile_low,
            clip_percentile_high=self.config.clip_percentile_high,
            target_range=self.config.target_range
        )
        extractor = SliceExtractor(
            orientation=self.config.orientation,
            min_brain_fraction=self.config.min_brain_fraction,
            tumor_priority=self.config.tumor_priority,
            target_slices=self.config.target_slices
        )
        builder = SliceDatasetBuilder(extractor, slices_dir.parent)
        
        # Process all patients
        patients = []
        for patient_id in loader.get_patient_ids():
            patient = loader.load_patient_by_id(patient_id)
            volume = loader.get_stacked_modalities(patient)
            volume = preprocessor.preprocess_volume(volume, patient.seg)
            
            patients.append({
                'volume': volume,
                'segmentation': patient.seg,
                'patient_id': patient_id
            })
        
        builder.build_dataset(patients)
    
    def setup(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
        max_train_slices: Optional[int] = None
    ):
        """
        Setup train/val/test splits.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            seed: Random seed for reproducibility
            max_train_slices: Optional cap for number of training slices
        """
        slices_dir = Path(self.config.slices_dir or "./data/slices")
        metadata_file = slices_dir.parent / "metadata.json"
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Get unique patient IDs
        patient_ids = list(set(m['patient_id'] for m in metadata))
        
        # Shuffle and split
        random.seed(seed)
        random.shuffle(patient_ids)
        
        n_train = int(len(patient_ids) * train_ratio)
        n_val = int(len(patient_ids) * val_ratio)
        
        train_patients = set(patient_ids[:n_train])
        val_patients = set(patient_ids[n_train:n_train + n_val])
        test_patients = set(patient_ids[n_train + n_val:])
        
        # Split metadata
        train_metadata = [m for m in metadata if m['patient_id'] in train_patients]
        val_metadata = [m for m in metadata if m['patient_id'] in val_patients]
        test_metadata = [m for m in metadata if m['patient_id'] in test_patients]

        # Optionally cap training slices (deterministic, tumor-aware)
        if max_train_slices is not None and len(train_metadata) > max_train_slices:
            rng = random.Random(seed)
            tumor = [m for m in train_metadata if m.get("has_tumor", False)]
            non_tumor = [m for m in train_metadata if not m.get("has_tumor", False)]

            rng.shuffle(tumor)
            rng.shuffle(non_tumor)

            kept = tumor[:max_train_slices]
            remaining = max_train_slices - len(kept)
            if remaining > 0:
                kept += non_tumor[:remaining]

            train_metadata = kept
        
        # Save split metadata
        splits_dir = slices_dir.parent / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        for split_name, split_meta in [('train', train_metadata), 
                                        ('val', val_metadata), 
                                        ('test', test_metadata)]:
            split_file = splits_dir / f"{split_name}_metadata.json"
            with open(split_file, 'w') as f:
                json.dump(split_meta, f)
        
        # Create datasets
        self.train_dataset = BraTSSliceDataset(
            slices_dir=slices_dir,
            metadata_file=splits_dir / "train_metadata.json",
            augmentor=self.train_augmentor,
            return_segmentation=True
        )
        
        self.val_dataset = BraTSSliceDataset(
            slices_dir=slices_dir,
            metadata_file=splits_dir / "val_metadata.json",
            augmentor=None,  # No augmentation for validation
            return_segmentation=True
        )
        
        self.test_dataset = BraTSSliceDataset(
            slices_dir=slices_dir,
            metadata_file=splits_dir / "test_metadata.json",
            augmentor=None,
            return_segmentation=True
        )
        
        print(f"Dataset splits: Train={len(self.train_dataset)}, "
              f"Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    slices_dir: Optional[str] = None,
    augmentation: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create train/val/test dataloaders.
    
    Args:
        data_dir: Path to BraTS data
        batch_size: Batch size
        num_workers: Number of data loading workers
        slices_dir: Directory for extracted slices
        augmentation: Whether to enable augmentation
        train_ratio: Training data fraction
        val_ratio: Validation data fraction
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = DatasetConfig(
        data_dir=data_dir,
        slices_dir=slices_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_enabled=augmentation
    )
    
    data_module = BraTSDataModule(config)
    data_module.prepare_data()
    data_module.setup(train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader()
    )