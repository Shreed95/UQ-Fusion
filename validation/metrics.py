# validation/metrics.py

"""
Image Quality Metrics for Medical Image Synthesis.

Includes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Medical-specific metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
from scipy import linalg


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""
    device: str = 'cuda'
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5
    lpips_net: str = 'vgg'  # 'vgg' or 'alex'
    fid_batch_size: int = 32


class PSNRMetric:
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
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
            
        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(pred, target, reduction='none')
        mse = mse.view(mse.shape[0], -1).mean(dim=1)
        psnr = 10 * torch.log10(self.max_val ** 2 / (mse + 1e-8))
        return psnr
    
    def compute_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute batch statistics."""
        psnr = self(pred, target)
        return {
            'psnr_mean': psnr.mean().item(),
            'psnr_std': psnr.std().item(),
            'psnr_min': psnr.min().item(),
            'psnr_max': psnr.max().item()
        }


class SSIMMetric:
    """Structural Similarity Index Measure."""
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 4,
        size_average: bool = False
    ):
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.size_average = size_average
        
        # Create Gaussian window
        self.window = self._create_window(window_size, sigma, channel)
    
    def _create_window(
        self,
        window_size: int,
        sigma: float,
        channel: int
    ) -> torch.Tensor:
        """Create Gaussian window."""
        coords = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM.
        
        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
            
        Returns:
            SSIM value (B,) or scalar
        """
        channel = pred.shape[1]
        
        if channel != self.channel:
            self.window = self._create_window(self.window_size, self.sigma, channel)
            self.channel = channel
        
        window = self.window.to(pred.device)
        
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.view(ssim_map.shape[0], -1).mean(dim=1)
    
    def compute_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute batch statistics."""
        ssim = self(pred, target)
        return {
            'ssim_mean': ssim.mean().item(),
            'ssim_std': ssim.std().item(),
            'ssim_min': ssim.min().item(),
            'ssim_max': ssim.max().item()
        }


class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity."""
    
    def __init__(self, net: str = 'vgg', device: str = 'cuda'):
        self.device = device
        self.net_type = net
        
        # Load VGG for perceptual features
        self._load_network()
    
    def _load_network(self):
        """Load pretrained VGG network."""
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features
        
        self.layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
        
        self.slices = nn.ModuleList()
        prev = 0
        for layer in self.layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:layer]))
            prev = layer
        
        self.slices = self.slices.to(self.device)
        
        for param in self.slices.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device))
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert to RGB and normalize."""
        if x.shape[1] == 4:
            x = x[:, :3]  # Use first 3 channels
        elif x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = (x - self.mean) / self.std
        return x
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LPIPS.
        
        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
            
        Returns:
            LPIPS value (B,)
        """
        pred = self._preprocess(pred.to(self.device))
        target = self._preprocess(target.to(self.device))
        
        diffs = []
        pred_feat = pred
        target_feat = target
        
        for slice_net in self.slices:
            pred_feat = slice_net(pred_feat)
            target_feat = slice_net(target_feat)
            
            diff = (pred_feat - target_feat) ** 2
            diff = diff.mean(dim=[1, 2, 3])
            diffs.append(diff)
        
        lpips = torch.stack(diffs, dim=1).sum(dim=1)
        return lpips
    
    def compute_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute batch statistics."""
        lpips = self(pred, target)
        return {
            'lpips_mean': lpips.mean().item(),
            'lpips_std': lpips.std().item(),
            'lpips_min': lpips.min().item(),
            'lpips_max': lpips.max().item()
        }


class FIDMetric:
    """Fréchet Inception Distance."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._load_inception()
    
    def _load_inception(self):
        """Load Inception network for feature extraction."""
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        except:
            from torchvision.models import inception_v3
            self.inception = inception_v3(pretrained=True, transform_input=False)
        
        self.inception.fc = nn.Identity()
        self.inception = self.inception.to(self.device)
        self.inception.eval()
        
        for param in self.inception.parameters():
            param.requires_grad = False
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess for Inception."""
        if x.shape[1] == 4:
            x = x[:, :3]
        elif x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Resize to 299x299
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize
        x = (x - 0.5) / 0.5
        
        return x
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract Inception features."""
        images = self._preprocess(images.to(self.device))
        features = self.inception(images)
        return features.cpu().numpy()
    
    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def __call__(
        self,
        real_features: np.ndarray,
        fake_features: np.ndarray
    ) -> float:
        """
        Compute FID between two sets of features.
        
        Args:
            real_features: Features from real images (N, D)
            fake_features: Features from generated images (N, D)
            
        Returns:
            FID score
        """
        mu1, sigma1 = self.compute_statistics(real_features)
        mu2, sigma2 = self.compute_statistics(fake_features)
        
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)


class MAEMetric:
    """Mean Absolute Error metric."""
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MAE."""
        mae = torch.abs(pred - target).view(pred.shape[0], -1).mean(dim=1)
        return mae
    
    def compute_batch(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute batch statistics."""
        mae = self(pred, target)
        return {
            'mae_mean': mae.mean().item(),
            'mae_std': mae.std().item(),
            'mae_min': mae.min().item(),
            'mae_max': mae.max().item()
        }


class NRMSEMetric:
    """Normalized Root Mean Squared Error."""
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute NRMSE."""
        mse = F.mse_loss(pred, target, reduction='none')
        mse = mse.view(mse.shape[0], -1).mean(dim=1)
        rmse = torch.sqrt(mse)
        
        # Normalize by target range
        target_flat = target.view(target.shape[0], -1)
        target_range = target_flat.max(dim=1)[0] - target_flat.min(dim=1)[0]
        nrmse = rmse / (target_range + 1e-8)
        
        return nrmse


class MetricsCalculator:
    """
    Unified metrics calculator for medical image quality assessment.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        if config is None:
            config = MetricsConfig()
        
        self.config = config
        self.device = config.device
        
        # Initialize metrics
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric(
            window_size=config.ssim_window_size,
            sigma=config.ssim_sigma
        )
        self.mae = MAEMetric()
        self.nrmse = NRMSEMetric()
        
        # Lazy initialization for heavy metrics
        self._lpips = None
        self._fid = None
    
    @property
    def lpips(self):
        if self._lpips is None:
            self._lpips = LPIPSMetric(net=self.config.lpips_net, device=self.device)
        return self._lpips
    
    @property
    def fid(self):
        if self._fid is None:
            self._fid = FIDMetric(device=self.device)
        return self._fid
    
    def compute_all(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        include_lpips: bool = True,
        include_fid: bool = False,
        real_images: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            include_lpips: Whether to compute LPIPS
            include_fid: Whether to compute FID
            real_images: Real images for FID computation
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self.psnr.compute_batch(pred, target))
        metrics.update(self.ssim.compute_batch(pred, target))
        metrics.update(self.mae.compute_batch(pred, target))
        
        nrmse = self.nrmse(pred, target)
        metrics['nrmse_mean'] = nrmse.mean().item()
        metrics['nrmse_std'] = nrmse.std().item()
        
        # LPIPS
        if include_lpips:
            metrics.update(self.lpips.compute_batch(pred, target))
        
        # FID
        if include_fid and real_images is not None:
            real_features = self.fid.extract_features(real_images)
            fake_features = self.fid.extract_features(pred)
            metrics['fid'] = self.fid(real_features, fake_features)
        
        return metrics
    
    def compute_per_sample(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for each sample.
        
        Returns:
            List of metric dictionaries, one per sample
        """
        results = []
        
        psnr_values = self.psnr(pred, target)
        ssim_values = self.ssim(pred, target)
        mae_values = self.mae(pred, target)
        nrmse_values = self.nrmse(pred, target)
        
        for i in range(pred.shape[0]):
            results.append({
                'psnr': psnr_values[i].item(),
                'ssim': ssim_values[i].item(),
                'mae': mae_values[i].item(),
                'nrmse': nrmse_values[i].item()
            })
        
        return results


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convenience function for PSNR."""
    return PSNRMetric()(pred, target)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convenience function for SSIM."""
    return SSIMMetric()(pred, target)
