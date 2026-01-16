# configs/paths.py

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# BraTS dataset paths
BRATS_RAW_DIR = Path("/path/to/BraTS2020_TrainingData")  # Update this path
BRATS_PROCESSED_DIR = DATA_ROOT / "processed"
BRATS_SLICES_DIR = DATA_ROOT / "slices"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
GENERATED_DIR = OUTPUTS_DIR / "generated_images"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Create directories if they don't exist
for dir_path in [BRATS_PROCESSED_DIR, BRATS_SLICES_DIR, CHECKPOINTS_DIR, 
                 GENERATED_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Modality file suffixes
MODALITIES = {
    't1': '_t1.nii.gz',
    't1ce': '_t1ce.nii.gz',
    't2': '_t2.nii.gz',
    'flair': '_flair.nii.gz',
    'seg': '_seg.nii.gz'
}

MODALITY_ORDER = ['t1', 't1ce', 't2', 'flair']