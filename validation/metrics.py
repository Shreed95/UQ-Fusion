# validation/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from scipy import linalg


class PSNR:
    """Peak Signal-to-Noise Ratio metric."""
    
    def __init__(self, max_val: float = 1.0):
        self.max_val = max_val
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PSNR.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(pred, target, reduction='mean')
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse + 1e-8))
        return psnr


class SSIM:
    """Structural Similarity Index metric."""
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 4
    ):
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.window = self._create_window(window_size, channel, sigma)
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int, sigma: float) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            SSIM value (0 to 1, higher is better)
        """
        channel = pred.shape[1]
        
        if channel != self.channel:
            window = self._create_window(self.window_size, channel, self.sigma)
        else:
            window = self.window
        
        window = window.to(pred.device)
        
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


class FID:
    """
    Fréchet Inception Distance metric.
    Measures distance between feature distributions of real and generated images.
    """
    
    def __init__(self, feature_extractor: Optional[nn.Module] = None):
        """
        Initialize FID.
        
        Args:
            feature_extractor: Model to extract features. If None, uses simple CNN.
        """
        if feature_extractor is None:
            # Simple feature extractor for medical images
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(4, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        else:
            self.feature_extractor = feature_extractor
    
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from images."""
        self.feature_extractor.eval()
        self.feature_extractor.to(images.device)
        
        with torch.no_grad():
            features = self.feature_extractor(images)
        
        return features.cpu().numpy()
    
    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def __call__(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> float:
        """
        Compute FID between real and fake images.
        
        Args:
            real_images: Real images (B, C, H, W)
            fake_images: Generated images (B, C, H, W)
            
        Returns:
            FID score (lower is better)
        """
        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)
        
        mu1, sigma1 = self.compute_statistics(real_features)
        mu2, sigma2 = self.compute_statistics(fake_features)
        
        # Compute FID
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)


class MetricsCalculator:
    """Calculator for multiple image quality metrics."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.psnr = PSNR()
        self.ssim = SSIM()
    
    @torch.no_grad()
    def calculate(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate all metrics.
        
        Args:
            pred: Predicted images
            target: Target images
            
        Returns:
            Dictionary of metric values
        """
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        # Ensure values are in [0, 1]
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        metrics = {
            'psnr': self.psnr(pred, target).item(),
            'ssim': self.ssim(pred, target).item(),
            'mse': F.mse_loss(pred, target).item(),
            'mae': F.l1_loss(pred, target).item()
        }
        
        return metrics
    
    @torch.no_grad()
    def calculate_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate metrics for each sample in batch."""
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        batch_size = pred.shape[0]
        
        psnr_values = []
        ssim_values = []
        
        for i in range(batch_size):
            psnr_values.append(self.psnr(pred[i:i+1], target[i:i+1]))
            ssim_values.append(self.ssim(pred[i:i+1], target[i:i+1]))
        
        return {
            'psnr': torch.stack(psnr_values),
            'ssim': torch.stack(ssim_values)
        }


def compute_reconstruction_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    num_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute reconstruction metrics for a VAE model.
    
    Args:
        model: VAE model
        dataloader: Data loader
        device: Device to use
        num_batches: Number of batches to evaluate (None for all)
        
    Returns:
        Dictionary of average metrics
    """
    model.eval()
    model.to(device)
    
    calculator = MetricsCalculator(device)
    
    all_psnr = []
    all_ssim = []
    all_mse = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break
            
            images = batch['image'].to(device)
            recons = model.reconstruct(images)
            
            # Clamp to valid range
            recons = torch.clamp(recons, 0, 1)
            
            metrics = calculator.calculate(recons, images)
            
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
            all_mse.append(metrics['mse'])
    
    return {
        'psnr': np.mean(all_psnr),
        'ssim': np.mean(all_ssim),
        'mse': np.mean(all_mse),
        'psnr_std': np.std(all_psnr),
        'ssim_std': np.std(all_ssim)
    }
