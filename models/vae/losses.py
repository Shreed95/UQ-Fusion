# models/vae/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import torchvision.models as models


class VAELoss(nn.Module):
    """
    Combined VAE loss with reconstruction, KL divergence, and optional perceptual loss.
    
    Total Loss = recon_weight * L_recon + kl_weight * L_KL + perceptual_weight * L_perceptual
    """
    
    def __init__(
        self,
        recon_loss_type: str = 'l1',  # 'l1', 'l2', 'mse'
        recon_weight: float = 1.0,
        kl_weight: float = 0.0001,
        perceptual_weight: float = 0.0,
        use_perceptual: bool = False
    ):
        """
        Initialize VAE Loss.
        
        Args:
            recon_loss_type: Type of reconstruction loss ('l1', 'l2', 'mse')
            recon_weight: Weight for reconstruction loss
            kl_weight: Weight for KL divergence (beta in beta-VAE)
            perceptual_weight: Weight for perceptual loss
            use_perceptual: Whether to use perceptual loss
        """
        super().__init__()
        
        self.recon_loss_type = recon_loss_type
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        
        # Perceptual loss (VGG-based)
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def reconstruction_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            recon: Reconstructed images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            Reconstruction loss
        """
        if self.recon_loss_type == 'l1':
            return F.l1_loss(recon, target, reduction='mean')
        elif self.recon_loss_type in ['l2', 'mse']:
            return F.mse_loss(recon, target, reduction='mean')
        else:
            raise ValueError(f"Unknown reconstruction loss type: {self.recon_loss_type}")
    
    def kl_divergence(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from standard normal.
        
        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        
        Args:
            mean: Mean of latent distribution (B, C, H, W)
            log_var: Log variance of latent distribution (B, C, H, W)
            
        Returns:
            KL divergence loss
        """
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl / mean.numel()  # Normalize by number of elements
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total VAE loss.
        
        Args:
            recon: Reconstructed images (B, C, H, W)
            target: Target images (B, C, H, W)
            mean: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss
        l_recon = self.reconstruction_loss(recon, target)
        
        # KL divergence
        l_kl = self.kl_divergence(mean, log_var)
        
        # Total loss
        total_loss = self.recon_weight * l_recon + self.kl_weight * l_kl
        
        loss_dict = {
            'loss': total_loss,
            'recon_loss': l_recon,
            'kl_loss': l_kl
        }
        
        # Optional perceptual loss
        if self.use_perceptual and self.perceptual_weight > 0:
            l_perceptual = self.perceptual_loss(recon, target)
            total_loss = total_loss + self.perceptual_weight * l_perceptual
            loss_dict['perceptual_loss'] = l_perceptual
            loss_dict['loss'] = total_loss
        
        return total_loss, loss_dict


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    Compares feature representations rather than pixel values.
    """
    
    def __init__(
        self,
        feature_layers: Tuple[int, ...] = (3, 8, 15, 22),
        feature_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    ):
        """
        Initialize Perceptual Loss.
        
        Args:
            feature_layers: VGG16 layer indices to extract features from
            feature_weights: Weights for each feature layer
        """
        super().__init__()
        
        self.feature_layers = feature_layers
        self.feature_weights = feature_weights
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True)
        
        # Extract features up to the last required layer
        self.features = nn.Sequential(*list(vgg.features)[:max(feature_layers) + 1])
        
        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Normalization for VGG (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG."""
        return (x - self.mean) / self.std
    
    def convert_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert multi-channel medical image to RGB for VGG.
        Takes first 3 channels or repeats single channel.
        """
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        elif x.shape[1] >= 3:
            return x[:, :3]
        else:
            # Pad with zeros or repeat
            return torch.cat([x, x[:, :1].repeat(1, 3 - x.shape[1], 1, 1)], dim=1)
    
    def extract_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Extract features from specified layers."""
        features = {}
        
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.feature_layers:
                features[idx] = x
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            Perceptual loss
        """
        # Convert to RGB
        pred_rgb = self.convert_to_rgb(pred)
        target_rgb = self.convert_to_rgb(target)
        
        # Normalize
        pred_norm = self.normalize(pred_rgb)
        target_norm = self.normalize(target_rgb)
        
        # Extract features
        pred_features = self.extract_features(pred_norm)
        target_features = self.extract_features(target_norm)
        
        # Compute weighted feature loss
        loss = 0.0
        for idx, weight in zip(self.feature_layers, self.feature_weights):
            loss = loss + weight * F.mse_loss(pred_features[idx], target_features[idx])
        
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    SSIM measures structural similarity between images.
    """
    
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 4,
        reduction: str = 'mean'
    ):
        """
        Initialize SSIM Loss.
        
        Args:
            window_size: Size of Gaussian window
            sigma: Standard deviation of Gaussian
            channel: Number of image channels
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.reduction = reduction
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel, sigma))
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian window."""
        x = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian window for multiple channels."""
        _1D_window = self._gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            SSIM loss
        """
        channel = pred.shape[1]
        
        if channel != self.channel:
            window = self._create_window(self.window_size, channel, self.sigma).to(pred.device)
        else:
            window = self.window
        
        # Compute means
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.reduction == 'mean':
            return 1 - ssim_map.mean()
        elif self.reduction == 'sum':
            return 1 - ssim_map.sum()
        else:
            return 1 - ssim_map


class CombinedVAELoss(nn.Module):
    """
    Combined VAE loss with multiple components for medical images.
    
    Total Loss = recon_weight * (L1 + SSIM) + kl_weight * KL
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.0001,
        ssim_weight: float = 0.5,
        l1_weight: float = 0.5
    ):
        super().__init__()
        
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        
        self.ssim_loss = SSIMLoss()
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss."""
        # L1 loss
        l1 = F.l1_loss(recon, target, reduction='mean')
        
        # SSIM loss
        ssim = self.ssim_loss(recon, target)
        
        # Combined reconstruction loss
        l_recon = self.l1_weight * l1 + self.ssim_weight * ssim
        
        # KL divergence
        kl = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = self.recon_weight * l_recon + self.kl_weight * kl
        
        loss_dict = {
            'loss': total_loss,
            'recon_loss': l_recon,
            'l1_loss': l1,
            'ssim_loss': ssim,
            'kl_loss': kl
        }
        
        return total_loss, loss_dict
