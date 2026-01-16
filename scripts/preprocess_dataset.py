# scripts/preprocess_dataset.py

"""
Script to preprocess BraTS 2020 dataset.
Extracts 2D slices from 3D volumes with normalization.

Usage:
    python scripts/preprocess_dataset.py --data_dir /path/to/BraTS2020_TrainingData --output_dir ./data
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from data import (
    NIfTILoader,
    CachedNIfTILoader,
    VolumePreprocessor,
    SliceExtractor,
    SliceDatasetBuilder,
    BraTSDataModule,
    DatasetConfig
)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess BraTS 2020 dataset')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BraTS2020_TrainingData directory')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for processed data')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory for HDF5 cache (optional)')
    
    # Preprocessing options
    parser.add_argument('--clip_low', type=float, default=1.0,
                        help='Lower percentile for clipping')
    parser.add_argument('--clip_high', type=float, default=99.0,
                        help='Upper percentile for clipping')
    parser.add_argument('--target_range', type=float, nargs=2, default=[0, 1],
                        help='Target intensity range')
    
    # Slice extraction options
    parser.add_argument('--orientation', type=str, default='axial',
                        choices=['axial', 'sagittal', 'coronal'],
                        help='Slice orientation')
    parser.add_argument('--min_brain_fraction', type=float, default=0.05,
                        help='Minimum brain fraction to keep slice')
    parser.add_argument('--target_slices', type=int, default=100,
                        help='Target number of slices per volume')
    parser.add_argument('--tumor_priority', action='store_true', default=True,
                        help='Prioritize slices with tumor')
    
    # Dataset split
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    slices_dir = output_dir / "slices"
    processed_dir = output_dir / "processed"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("BraTS 2020 Dataset Preprocessing")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Orientation: {args.orientation}")
    print(f"Target slices per volume: {args.target_slices}")
    print("=" * 60)
    
    # Initialize loader
    print("\n[1/4] Initializing data loader...")
    if args.cache_dir:
        loader = CachedNIfTILoader(args.data_dir, args.cache_dir)
        print("Building cache (this may take a while on first run)...")
        loader.build_cache()
    else:
        loader = NIfTILoader(args.data_dir)
    
    print(f"Found {len(loader)} patients")
    
    # Initialize preprocessor
    print("\n[2/4] Initializing preprocessor...")
    preprocessor = VolumePreprocessor(
        clip_percentile_low=args.clip_low,
        clip_percentile_high=args.clip_high,
        target_range=tuple(args.target_range)
    )
    
    # Initialize slice extractor
    print("\n[3/4] Initializing slice extractor...")
    extractor = SliceExtractor(
        orientation=args.orientation,
        min_brain_fraction=args.min_brain_fraction,
        tumor_priority=args.tumor_priority,
        target_slices=args.target_slices
    )
    
    # Build slice dataset
    print("\n[4/4] Extracting slices from volumes...")
    builder = SliceDatasetBuilder(extractor, output_dir)
    
    patients_data = []
    for i, patient_id in enumerate(tqdm(loader.get_patient_ids(), desc="Loading patients")):
        patient = loader.load_patient_by_id(patient_id)
        volume = loader.get_stacked_modalities(patient)
        
        # Preprocess
        volume_processed = preprocessor.preprocess_volume(volume, patient.seg)
        
        patients_data.append({
            'volume': volume_processed,
            'segmentation': patient.seg,
            'patient_id': patient_id
        })
    
    # Extract and save slices
    total_slices = builder.build_dataset(patients_data, show_progress=True)
    
    # Create dataset splits
    print("\n[5/5] Creating dataset splits...")
    config = DatasetConfig(
        data_dir=args.data_dir,
        slices_dir=str(slices_dir),
        clip_percentile_low=args.clip_low,
        clip_percentile_high=args.clip_high,
        target_range=tuple(args.target_range),
        orientation=args.orientation,
        min_brain_fraction=args.min_brain_fraction,
        tumor_priority=args.tumor_priority,
        target_slices=args.target_slices
    )
    
    data_module = BraTSDataModule(config)
    data_module.setup(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Total slices extracted: {total_slices}")
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")
    print(f"Test samples: {len(data_module.test_dataset)}")
    print(f"\nOutput saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()