# data/nifti_loader.py

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import h5py
from tqdm import tqdm


@dataclass
class PatientVolume:
    """Data class to hold patient volume data and metadata."""
    patient_id: str
    t1: np.ndarray
    t1ce: np.ndarray
    t2: np.ndarray
    flair: np.ndarray
    seg: Optional[np.ndarray]
    affine: np.ndarray
    header: dict
    spacing: Tuple[float, float, float]
    

class NIfTILoader:
    """
    NIfTI file loader for BraTS 2020 dataset.
    Handles loading of multi-modal MRI volumes with metadata preservation.
    """
    
    MODALITY_SUFFIXES = {
        't1': '_t1.nii',
        't1ce': '_t1ce.nii',
        't2': '_t2.nii',
        'flair': '_flair.nii',
        'seg': '_seg.nii'
    }
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize NIfTI loader.
        
        Args:
            data_dir: Path to BraTS2020_TrainingData directory
        """
        self.data_dir = Path(data_dir)
        self.patient_dirs = self._discover_patients()
        
    def _discover_patients(self) -> List[Path]:
        """Discover all patient directories in the dataset."""
        patient_dirs = []
        
        for item in sorted(self.data_dir.iterdir()):
            if item.is_dir() and item.name.startswith('BraTS'):
                # Verify all modality files exist
                if self._verify_patient_files(item):
                    patient_dirs.append(item)
                    
        print(f"Discovered {len(patient_dirs)} valid patient directories")
        return patient_dirs
    
    def _verify_patient_files(self, patient_dir: Path) -> bool:
        """Verify all required modality files exist for a patient."""
        patient_id = patient_dir.name
        
        for modality, suffix in self.MODALITY_SUFFIXES.items():
            file_path = patient_dir / f"{patient_id}{suffix}"
            if not file_path.exists():
                print(f"Warning: Missing {modality} for {patient_id}")
                return False
        return True
    
    def load_nifti_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load a single NIfTI file.
        
        Args:
            file_path: Path to .nii.gz file
            
        Returns:
            Tuple of (volume_data, affine_matrix, header_dict)
        """
        nii = nib.load(str(file_path))
        volume = nii.get_fdata().astype(np.float32)
        affine = nii.affine
        header = dict(nii.header)
        
        return volume, affine, header
    
    def load_patient(self, patient_dir: Union[str, Path]) -> PatientVolume:
        """
        Load all modalities for a single patient.
        
        Args:
            patient_dir: Path to patient directory
            
        Returns:
            PatientVolume object containing all modalities and metadata
        """
        patient_dir = Path(patient_dir)
        patient_id = patient_dir.name
        
        volumes = {}
        affine = None
        header = None
        
        for modality, suffix in self.MODALITY_SUFFIXES.items():
            file_path = patient_dir / f"{patient_id}{suffix}"
            vol, aff, hdr = self.load_nifti_file(file_path)
            volumes[modality] = vol
            
            if affine is None:
                affine = aff
                header = hdr
        
        # Extract spacing from affine matrix
        spacing = tuple(np.abs(np.diag(affine)[:3]))
        
        return PatientVolume(
            patient_id=patient_id,
            t1=volumes['t1'],
            t1ce=volumes['t1ce'],
            t2=volumes['t2'],
            flair=volumes['flair'],
            seg=volumes.get('seg'),
            affine=affine,
            header=header,
            spacing=spacing
        )
    
    def load_patient_by_id(self, patient_id: str) -> PatientVolume:
        """Load patient by ID string."""
        patient_dir = self.data_dir / patient_id
        if not patient_dir.exists():
            raise ValueError(f"Patient directory not found: {patient_id}")
        return self.load_patient(patient_dir)
    
    def get_stacked_modalities(self, patient: PatientVolume) -> np.ndarray:
        """
        Stack all 4 modalities into a single array.
        
        Args:
            patient: PatientVolume object
            
        Returns:
            Array of shape (4, H, W, D) - (channels, height, width, depth)
        """
        stacked = np.stack([
            patient.t1,
            patient.t1ce,
            patient.t2,
            patient.flair
        ], axis=0)
        
        return stacked
    
    def load_all_patients(self, show_progress: bool = True) -> List[PatientVolume]:
        """
        Load all patients in the dataset.
        
        Args:
            show_progress: Whether to show progress bar
            
        Returns:
            List of PatientVolume objects
        """
        patients = []
        iterator = tqdm(self.patient_dirs, desc="Loading patients") if show_progress else self.patient_dirs
        
        for patient_dir in iterator:
            try:
                patient = self.load_patient(patient_dir)
                patients.append(patient)
            except Exception as e:
                print(f"Error loading {patient_dir.name}: {e}")
                
        return patients
    
    def get_patient_ids(self) -> List[str]:
        """Get list of all patient IDs."""
        return [p.name for p in self.patient_dirs]
    
    def __len__(self) -> int:
        return len(self.patient_dirs)
    
    def __getitem__(self, idx: int) -> PatientVolume:
        return self.load_patient(self.patient_dirs[idx])


class CachedNIfTILoader(NIfTILoader):
    """
    NIfTI loader with HDF5 caching for faster subsequent access.
    """
    
    def __init__(self, data_dir: Union[str, Path], cache_dir: Union[str, Path] = None):
        """
        Initialize cached NIfTI loader.
        
        Args:
            data_dir: Path to BraTS2020_TrainingData directory
            cache_dir: Path to store HDF5 cache files
        """
        super().__init__(data_dir)
        
        if cache_dir is None:
            cache_dir = Path(data_dir).parent / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "brats_cache.h5"
        
    def _get_cache_key(self, patient_id: str) -> str:
        """Generate cache key for a patient."""
        return patient_id
    
    def is_cached(self, patient_id: str) -> bool:
        """Check if patient data is cached."""
        if not self.cache_file.exists():
            return False
            
        with h5py.File(self.cache_file, 'r') as f:
            return patient_id in f
    
    def cache_patient(self, patient: PatientVolume):
        """Cache patient data to HDF5 file."""
        with h5py.File(self.cache_file, 'a') as f:
            if patient.patient_id in f:
                del f[patient.patient_id]
                
            grp = f.create_group(patient.patient_id)
            grp.create_dataset('t1', data=patient.t1, compression='gzip')
            grp.create_dataset('t1ce', data=patient.t1ce, compression='gzip')
            grp.create_dataset('t2', data=patient.t2, compression='gzip')
            grp.create_dataset('flair', data=patient.flair, compression='gzip')
            if patient.seg is not None:
                grp.create_dataset('seg', data=patient.seg, compression='gzip')
            grp.create_dataset('affine', data=patient.affine)
            grp.attrs['spacing'] = patient.spacing
    
    def load_from_cache(self, patient_id: str) -> PatientVolume:
        """Load patient data from cache."""
        with h5py.File(self.cache_file, 'r') as f:
            grp = f[patient_id]
            
            seg = grp['seg'][:] if 'seg' in grp else None
            
            return PatientVolume(
                patient_id=patient_id,
                t1=grp['t1'][:],
                t1ce=grp['t1ce'][:],
                t2=grp['t2'][:],
                flair=grp['flair'][:],
                seg=seg,
                affine=grp['affine'][:],
                header={},
                spacing=tuple(grp.attrs['spacing'])
            )
    
    def load_patient(self, patient_dir: Union[str, Path]) -> PatientVolume:
        """Load patient with caching support."""
        patient_dir = Path(patient_dir)
        patient_id = patient_dir.name
        
        if self.is_cached(patient_id):
            return self.load_from_cache(patient_id)
        
        # Load from NIfTI and cache
        patient = super().load_patient(patient_dir)
        self.cache_patient(patient)
        
        return patient
    
    def build_cache(self, show_progress: bool = True):
        """Pre-build cache for all patients."""
        print("Building HDF5 cache for all patients...")
        iterator = tqdm(self.patient_dirs, desc="Caching") if show_progress else self.patient_dirs
        
        for patient_dir in iterator:
            patient_id = patient_dir.name
            if not self.is_cached(patient_id):
                patient = super().load_patient(patient_dir)
                self.cache_patient(patient)
                
        print(f"Cache built at: {self.cache_file}")