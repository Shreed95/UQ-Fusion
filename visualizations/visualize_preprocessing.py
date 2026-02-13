"""
Visualize BraTS preprocessing step-by-step for a single 2D slice.

Shows:
1) Raw slice (one MRI modality)
2) Normalized slice (after your VolumePreprocessor)
3) Normalized slice with tumor mask overlaid

Usage example (from project root):
    python visualizations/visualize_preprocessing.py \
        --data_dir /path/to/BraTS2020_TrainingData \
        --patient_id BraTS20_Training_001 \
        --orientation axial \
        --slice_idx 80

If --patient_id is omitted, the first discovered patient is used.
If --slice_idx is omitted, a slice with the most tumor is chosen.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path so we can import the local data module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import NIfTILoader, VolumePreprocessor, SliceExtractor  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize BraTS preprocessing for a single slice."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to BraTS2020_TrainingData directory",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Patient ID (e.g. BraTS20_Training_001). "
             "If not provided, the first patient is used.",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="axial",
        choices=["axial", "sagittal", "coronal"],
        help="Slice orientation to visualize.",
    )
    parser.add_argument(
        "--slice_idx",
        type=int,
        default=None,
        help="Index of slice along the chosen orientation. "
             "If not provided, a slice with tumor is selected automatically.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="flair",
        choices=["t1", "t1ce", "t2", "flair"],
        help="Which modality to display in the images.",
    )
    parser.add_argument(
        "--clip_low",
        type=float,
        default=1.0,
        help="Lower percentile for clipping during normalization.",
    )
    parser.add_argument(
        "--clip_high",
        type=float,
        default=99.0,
        help="Upper percentile for clipping during normalization.",
    )
    parser.add_argument(
        "--target_range",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Target intensity range after normalization.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save the figure instead of just showing it.",
    )

    return parser.parse_args()


def pick_patient_id(loader: NIfTILoader, requested_id: str | None) -> str:
    """Return a valid patient ID, using requested one if provided."""
    if requested_id is not None:
        return requested_id

    ids = loader.get_patient_ids()
    if not ids:
        raise RuntimeError("No patients found in the given data_dir.")
    return ids[0]


def choose_slice_with_tumor(
    volume: np.ndarray,
    segmentation: np.ndarray,
    orientation: str,
) -> int:
    """
    Choose a slice index that has the most tumor (for visualization).

    Falls back to the middle slice if there is no tumor at all.
    """
    extractor = SliceExtractor(
        orientation=orientation,
        min_brain_fraction=0.0,
        tumor_priority=True,
        target_slices=None,
    )

    valid_slices = extractor.get_valid_slice_indices(volume, segmentation)
    if not valid_slices:
        # If for some reason nothing is valid, just use the middle slice.
        if volume.ndim == 4:
            axis_len = volume.shape[extractor.slice_axis + 1]
        else:
            axis_len = volume.shape[extractor.slice_axis]
        return axis_len // 2

    # valid_slices is list of (idx, brain_fraction, tumor_fraction)
    # pick the index with the largest tumor_fraction
    best = max(valid_slices, key=lambda t: t[2])
    return best[0]


def extract_single_slice(
    volume: np.ndarray,
    segmentation: np.ndarray | None,
    orientation: str,
    slice_idx: int,
    patient_id: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Use your project's SliceExtractor to extract a single 2D slice
    (data and segmentation) from a 3D/4D volume.
    """
    extractor = SliceExtractor(
        orientation=orientation,
        min_brain_fraction=0.0,
        tumor_priority=False,
        target_slices=None,
    )

    slice_obj = extractor.extract_slice(
        volume=volume,
        segmentation=segmentation,
        slice_idx=slice_idx,
        patient_id=patient_id,
    )

    return slice_obj.data, slice_obj.segmentation


def visualize_slice_triplet(
    raw_slice: np.ndarray,
    norm_slice: np.ndarray,
    tumor_mask: np.ndarray | None,
    modality_name: str,
    title_prefix: str,
    save_path: Path | None = None,
) -> None:
    """
    Plot:
      - raw slice
      - normalized slice
      - normalized slice with tumor overlay
    """
    # raw_slice and norm_slice are expected shape (C, H, W)
    # we show a single channel (one modality).
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax in axes:
        ax.axis("off")

    vmin_raw = np.percentile(raw_slice, 1)
    vmax_raw = np.percentile(raw_slice, 99)

    vmin_norm = np.percentile(norm_slice, 1)
    vmax_norm = np.percentile(norm_slice, 99)

    axes[0].imshow(raw_slice, cmap="gray", vmin=vmin_raw, vmax=vmax_raw)
    axes[0].set_title(f"Raw {modality_name}")

    axes[1].imshow(norm_slice, cmap="gray", vmin=vmin_norm, vmax=vmax_norm)
    axes[1].set_title(f"Normalized {modality_name}")

    axes[2].imshow(norm_slice, cmap="gray", vmin=vmin_norm, vmax=vmax_norm)
    if tumor_mask is not None:
        # Overlay tumor in red
        axes[2].imshow(
            np.ma.masked_where(tumor_mask == 0, tumor_mask),
            cmap="Reds",
            alpha=0.5,
        )
    axes[2].set_title(f"Normalized + Tumor mask")

    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    loader = NIfTILoader(data_dir)

    patient_id = pick_patient_id(loader, args.patient_id)
    print(f"Using patient_id: {patient_id}")

    # Load patient and stack modalities: shape (4, H, W, D)
    patient = loader.load_patient_by_id(patient_id)
    if patient.seg is None:
        raise RuntimeError("Segmentation mask is required for visualization.")

    volume_raw = loader.get_stacked_modalities(patient)

    # Preprocess (normalize) the volume
    preprocessor = VolumePreprocessor(
        clip_percentile_low=args.clip_low,
        clip_percentile_high=args.clip_high,
        target_range=tuple(args.target_range),
    )
    volume_norm = preprocessor.preprocess_volume(volume_raw, patient.seg)

    # Decide which slice index to use
    if args.slice_idx is not None:
        slice_idx = args.slice_idx
    else:
        slice_idx = choose_slice_with_tumor(
            volume=volume_raw,
            segmentation=patient.seg,
            orientation=args.orientation,
        )
        print(f"No slice_idx provided, using slice with most tumor: {slice_idx}")

    # Extract the same slice from raw and normalized volumes
    raw_slice_4c, seg_slice = extract_single_slice(
        volume=volume_raw,
        segmentation=patient.seg,
        orientation=args.orientation,
        slice_idx=slice_idx,
        patient_id=patient_id,
    )
    norm_slice_4c, _ = extract_single_slice(
        volume=volume_norm,
        segmentation=patient.seg,
        orientation=args.orientation,
        slice_idx=slice_idx,
        patient_id=patient_id,
    )

    # Map modality name to channel index (consistent with NIfTILoader.get_stacked_modalities)
    modality_to_idx = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
    c = modality_to_idx[args.modality]

    raw_slice = raw_slice_4c[c]  # (H, W)
    norm_slice = norm_slice_4c[c]  # (H, W)

    tumor_mask = None
    if seg_slice is not None:
        tumor_mask = (seg_slice > 0).astype(np.float32)

    title_prefix = (
        f"Patient: {patient_id} | Modality: {args.modality.upper()} | "
        f"Orientation: {args.orientation} | Slice index: {slice_idx}"
    )

    save_path = Path(args.save_path) if args.save_path is not None else None

    visualize_slice_triplet(
        raw_slice=raw_slice,
        norm_slice=norm_slice,
        tumor_mask=tumor_mask,
        modality_name=args.modality.upper(),
        title_prefix=title_prefix,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()

