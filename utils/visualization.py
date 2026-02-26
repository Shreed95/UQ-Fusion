# utils/visualization.py

"""
Visualization utilities for UQ-Fusion.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.gridspec as gridspec


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None,
    title: str = "Training History"
):
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics
    metric_keys = [k for k in history.keys() if 'loss' not in k.lower()]
    for key in metric_keys[:3]:
        axes[1].plot(history[key], label=key)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mri_slices(
    images: torch.Tensor,
    titles: List[str] = None,
    save_path: str = None,
    cmap: str = 'gray'
):
    """Plot MRI modalities."""
    if images.dim() == 4:
        images = images[0]
    
    num_channels = images.shape[0]
    
    if titles is None:
        titles = ['T1', 'T1ce', 'T2', 'FLAIR'][:num_channels]
    
    fig, axes = plt.subplots(1, num_channels, figsize=(4*num_channels, 4))
    
    if num_channels == 1:
        axes = [axes]
    
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.imshow(images[i].cpu().numpy(), cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_generation_comparison(
    source: torch.Tensor,
    diffusion_output: torch.Tensor,
    gan_output: torch.Tensor,
    fused_output: torch.Tensor,
    uncertainty_diff: torch.Tensor = None,
    uncertainty_gan: torch.Tensor = None,
    save_path: str = None
):
    """Plot generation comparison."""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    # Row 1: Images (using FLAIR channel)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(source[3].cpu().numpy(), cmap='gray')
    ax1.set_title('Source')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(diffusion_output[3].cpu().numpy(), cmap='gray')
    ax2.set_title('Diffusion')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(gan_output[3].cpu().numpy(), cmap='gray')
    ax3.set_title('GAN')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(fused_output[3].cpu().numpy(), cmap='gray')
    ax4.set_title('Fused')
    ax4.axis('off')
    
    # Row 2: Uncertainties
    if uncertainty_diff is not None:
        ax5 = fig.add_subplot(gs[1, 1])
        im = ax5.imshow(uncertainty_diff.cpu().numpy(), cmap='hot')
        ax5.set_title('Diffusion Uncertainty')
        ax5.axis('off')
        plt.colorbar(im, ax=ax5, fraction=0.046)
    
    if uncertainty_gan is not None:
        ax6 = fig.add_subplot(gs[1, 2])
        im = ax6.imshow(uncertainty_gan.cpu().numpy(), cmap='hot')
        ax6.set_title('GAN Uncertainty')
        ax6.axis('off')
        plt.colorbar(im, ax=ax6, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_segmentation_results(
    image: torch.Tensor,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    save_path: str = None
):
    """Plot segmentation results."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # FLAIR image
    axes[0].imshow(image[3].cpu().numpy(), cmap='gray')
    axes[0].set_title('FLAIR')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(ground_truth.cpu().numpy(), cmap='tab10', vmin=0, vmax=4)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(prediction.cpu().numpy(), cmap='tab10', vmin=0, vmax=4)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(image[3].cpu().numpy(), cmap='gray')
    pred_overlay = np.ma.masked_where(prediction.cpu().numpy() == 0, prediction.cpu().numpy())
    axes[3].imshow(pred_overlay, cmap='hot', alpha=0.5, vmin=1, vmax=4)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_quality_distribution(
    scores: List[float],
    threshold: float = 0.70,
    save_path: str = None
):
    """Plot quality score distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(scores, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
    axes[0].axvline(np.mean(scores), color='g', linestyle='-', label=f'Mean: {np.mean(scores):.3f}')
    axes[0].set_xlabel('Quality Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Quality Score Distribution')
    axes[0].legend()
    
    # Accepted vs Rejected
    accepted = sum(1 for s in scores if s >= threshold)
    rejected = len(scores) - accepted
    axes[1].bar(['Accepted', 'Rejected'], [accepted, rejected], color=['green', 'red'], alpha=0.7)
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Acceptance Rate: {accepted/len(scores)*100:.1f}%')
    
    for i, v in enumerate([accepted, rejected]):
        axes[1].text(i, v + 1, str(v), ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_comparison_bars(
    baseline_metrics: Dict[str, float],
    augmented_metrics: Dict[str, float],
    metric_names: List[str] = None,
    save_path: str = None
):
    """Plot comparison bar chart."""
    if metric_names is None:
        metric_names = list(baseline_metrics.keys())
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    baseline_values = [baseline_metrics.get(m, 0) for m in metric_names]
    augmented_values = [augmented_metrics.get(m, 0) for m in metric_names]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='steelblue')
    bars2 = ax.bar(x + width/2, augmented_values, width, label='Augmented', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Baseline vs Augmented Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
