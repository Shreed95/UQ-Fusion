# utils/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple
from pathlib import Path
import torch


def visualize_slice(
    image: Union[np.ndarray, torch.Tensor],
    segmentation: Optional[Union[np.ndarray, torch.Tensor]] = None,
    modality_names: List[str] = ['T1', 'T1ce', 'T2', 'FLAIR'],
    save_path: Optional[str] = None,
    title: str = "MRI Slice"
):
    """
    Visualize multi-modal MRI slice.
    
    Args:
        image: Input image (C, H, W) or (H, W)
        segmentation: Optional segmentation mask (H, W)
        modality_names: Names for each channel
        save_path: Path to save figure
        title: Figure title
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if segmentation is not None and isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    
    n_channels = image.shape[0]
    n_cols = n_channels + (1 if segmentation is not None else 0)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    
    for i in range(n_channels):
        axes[i].imshow(image[i], cmap='gray')
        if i < len(modality_names):
            axes[i].set_title(modality_names[i])
        axes[i].axis('off')
    
    if segmentation is not None:
        axes[-1].imshow(segmentation, cmap='jet', vmin=0, vmax=4)
        axes[-1].set_title('Segmentation')
        axes[-1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch(
    batch: dict,
    num_samples: int = 4,
    save_path: Optional[str] = None
):
    """
    Visualize a batch of samples.
    
    Args:
        batch: Dictionary with 'image' and optionally 'segmentation'
        num_samples: Number of samples to show
        save_path: Path to save figure
    """
    images = batch['image']
    segmentations = batch.get('segmentation')
    
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if segmentations is not None and isinstance(segmentations, torch.Tensor):
        segmentations = segmentations.cpu().numpy()
    
    num_samples = min(num_samples, len(images))
    n_channels = images.shape[1]
    
    fig, axes = plt.subplots(num_samples, n_channels + 1, figsize=(4 * (n_channels + 1), 4 * num_samples))
    
    modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    for i in range(num_samples):
        for j in range(n_channels):
            axes[i, j].imshow(images[i, j], cmap='gray')
            if i == 0:
                axes[i, j].set_title(modality_names[j] if j < len(modality_names) else f'Ch{j}')
            axes[i, j].axis('off')
        
        if segmentations is not None:
            axes[i, -1].imshow(segmentations[i], cmap='jet', vmin=0, vmax=4)
            if i == 0:
                axes[i, -1].set_title('Segmentation')
        axes[i, -1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_augmentation_comparison(
    original: np.ndarray,
    augmented: np.ndarray,
    channel: int = 0,
    save_path: Optional[str] = None
):
    """
    Compare original and augmented images.
    
    Args:
        original: Original image (C, H, W)
        augmented: Augmented image (C, H, W)
        channel: Channel to display
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original[channel], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(augmented[channel], cmap='gray')
    axes[1].set_title('Augmented')
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(original[channel] - augmented[channel])
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metric_name: str = "Loss",
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        metric_name: Name of the metric
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label=f'Training {metric_name}')
    plt.plot(epochs, val_losses, 'r-', label=f'Validation {metric_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_grid_visualization(
    images: List[np.ndarray],
    n_cols: int = 4,
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Create grid visualization of multiple images.
    
    Args:
        images: List of images to display
        n_cols: Number of columns
        titles: Optional titles for each image
        save_path: Path to save figure
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, ax in enumerate(axes):
        if i < n_images:
            img = images[i]
            if img.ndim == 3:
                img = img[0]  # Take first channel
            ax.imshow(img, cmap='gray')
            if titles and i < len(titles):
                ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()