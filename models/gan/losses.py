# models/gan/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    Supports multiple GAN loss types.
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        """
        Args:
            loss_type: 'vanilla', 'lsgan', or 'wgan'
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None  # Uses direct values
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        pred: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """
        Compute adversarial loss.
        
        Args:
            pred: Discriminator prediction
            target_is_real: Whether target should be real
            
        Returns:
            Loss value
        """
        if self.loss_type == 'wgan':
            if target_is_real:
                return -pred.mean()
            else:
                return pred.mean()
        else:
            if target_is_real:
                target = torch.ones_like(pred)
            else:
                target = torch.zeros_like(pred)
            return self.criterion(pred, target)


class L1Loss(nn.Module):
    """L1 reconstruction loss."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class L2Loss(nn.Module):
    """L2 reconstruction loss."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.
    Measures structural similarity between images.
    """
    
    def __init__(self, window_size: int = 11, channel: int = 4):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.sigma = 1.5
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM."""
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * self.sigma ** 2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        
        return window_2d.expand(channel, 1, window_size, window_size).contiguous()
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        size_average: bool = True
    ) -> torch.Tensor:
        """Compute SSIM."""
        channel = img1.size(1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).
        
        Args:
            pred: Predicted image
            target: Target image
            
        Returns:
            SSIM loss (lower is better)
        """
        window = self.window.to(pred.device).type_as(pred)
        
        if pred.size(1) != self.channel:
            window = self._create_window(self.window_size, pred.size(1)).to(pred.device).type_as(pred)
        
        ssim = self._ssim(pred, target, window)
        return 1 - ssim


class GradientLoss(nn.Module):
    """
    Gradient matching loss for edge preservation.
    Compares image gradients between prediction and target.
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def _compute_gradient(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute image gradients using Sobel filters."""
        B, C, H, W = img.shape
        
        # Expand filters for all channels
        sobel_x = self.sobel_x.expand(C, 1, 3, 3).to(img.device)
        sobel_y = self.sobel_y.expand(C, 1, 3, 3).to(img.device)
        
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=C)
        
        return grad_x, grad_y
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss.
        
        Args:
            pred: Predicted image
            target: Target image
            
        Returns:
            Gradient loss
        """
        pred_grad_x, pred_grad_y = self._compute_gradient(pred)
        target_grad_x, target_grad_y = self._compute_gradient(target)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y


class SpatialPreservationLoss(nn.Module):
    """
    STABLE Spatial Preservation Loss.
    
    Combines SSIM and gradient losses to preserve spatial structure.
    """
    
    def __init__(
        self,
        ssim_weight: float = 1.0,
        gradient_weight: float = 1.0
    ):
        super().__init__()
        
        self.ssim_loss = SSIMLoss()
        self.gradient_loss = GradientLoss()
        
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spatial preservation loss.
        
        Args:
            pred: Predicted image
            target: Target image
            
        Returns:
            Dictionary with loss components
        """
        ssim = self.ssim_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        
        total = self.ssim_weight * ssim + self.gradient_weight * gradient
        
        return {
            'spatial_loss': total,
            'ssim_loss': ssim,
            'gradient_loss': gradient
        }


class QuantitativePreservationLoss(nn.Module):
    """
    STABLE Quantitative Preservation Loss.
    
    Preserves intensity distributions and regional consistency.
    """
    
    def __init__(
        self,
        histogram_weight: float = 1.0,
        regional_weight: float = 1.0,
        contrast_weight: float = 1.0
    ):
        super().__init__()
        
        self.histogram_weight = histogram_weight
        self.regional_weight = regional_weight
        self.contrast_weight = contrast_weight
    
    def _histogram_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_bins: int = 64
    ) -> torch.Tensor:
        """
        Compute histogram matching loss using differentiable approximation.
        """
        B, C, H, W = pred.shape
        
        # Compute mean and std for each channel
        pred_mean = pred.mean(dim=(2, 3))
        pred_std = pred.std(dim=(2, 3))
        target_mean = target.mean(dim=(2, 3))
        target_std = target.std(dim=(2, 3))
        
        mean_loss = F.l1_loss(pred_mean, target_mean)
        std_loss = F.l1_loss(pred_std, target_std)
        
        return mean_loss + std_loss
    
    def _regional_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        num_regions: int = 4
    ) -> torch.Tensor:
        """
        Compute regional consistency loss.
        Divides image into regions and compares mean intensities.
        """
        B, C, H, W = pred.shape
        
        # Divide into grid
        h_step = H // num_regions
        w_step = W // num_regions
        
        loss = 0
        for i in range(num_regions):
            for j in range(num_regions):
                pred_region = pred[:, :, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                target_region = target[:, :, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                
                pred_mean = pred_region.mean(dim=(2, 3))
                target_mean = target_region.mean(dim=(2, 3))
                
                loss = loss + F.l1_loss(pred_mean, target_mean)
        
        return loss / (num_regions ** 2)
    
    def _contrast_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrast preservation loss.
        Preserves relative contrast between different tissue types.
        """
        # Use local contrast (difference between adjacent pixels)
        pred_contrast_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_contrast_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        target_contrast_h = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_contrast_w = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        loss_h = F.l1_loss(pred_contrast_h, target_contrast_h)
        loss_w = F.l1_loss(pred_contrast_w, target_contrast_w)
        
        return loss_h + loss_w
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute quantitative preservation loss.
        
        Args:
            pred: Predicted image
            target: Target image
            
        Returns:
            Dictionary with loss components
        """
        histogram = self._histogram_loss(pred, target)
        regional = self._regional_loss(pred, target)
        contrast = self._contrast_loss(pred, target)
        
        total = (
            self.histogram_weight * histogram +
            self.regional_weight * regional +
            self.contrast_weight * contrast
        )
        
        return {
            'quantitative_loss': total,
            'histogram_loss': histogram,
            'regional_loss': regional,
            'contrast_loss': contrast
        }


class IdentityLoss(nn.Module):
    """
    Identity loss for preserving input characteristics.
    When source and target are the same, output should match input.
    """
    
    def forward(self, pred: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, source)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    Compares high-level features for texture preservation.
    """
    
    def __init__(self, layers: Tuple[int, ...] = (3, 8, 15, 22)):
        super().__init__()
        
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features
        
        self.layers = layers
        self.slices = nn.ModuleList()
        
        prev = 0
        for layer in layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:layer]))
            prev = layer
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _convert_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel medical image to RGB."""
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        elif x.size(1) == 3:
            return x
        else:
            # Use first 3 channels or average
            return x[:, :3]
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to ImageNet stats."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        """
        # Convert to RGB
        pred_rgb = self._convert_to_rgb(pred)
        target_rgb = self._convert_to_rgb(target)
        
        # Normalize
        pred_rgb = self._normalize(pred_rgb)
        target_rgb = self._normalize(target_rgb)
        
        loss = 0
        pred_feat = pred_rgb
        target_feat = target_rgb
        
        for slice_module in self.slices:
            pred_feat = slice_module(pred_feat)
            target_feat = slice_module(target_feat)
            loss = loss + F.l1_loss(pred_feat, target_feat)
        
        return loss


class STABLEGANLoss(nn.Module):
    """
    Complete STABLE-GAN loss combining all components.
    
    L_total = λ_adv * L_adversarial + 
              λ_l1 * L_l1 + 
              λ_spa * L_spatial + 
              λ_qua * L_quantitative + 
              λ_id * L_identity
    """
    
    def __init__(
        self,
        adv_weight: float = 1.0,
        l1_weight: float = 10.0,
        spatial_weight: float = 5.0,
        quantitative_weight: float = 2.0,
        identity_weight: float = 0.5,
        perceptual_weight: float = 0.0,  # Optional
        gan_type: str = 'lsgan'
    ):
        super().__init__()
        
        self.adv_weight = adv_weight
        self.l1_weight = l1_weight
        self.spatial_weight = spatial_weight
        self.quantitative_weight = quantitative_weight
        self.identity_weight = identity_weight
        self.perceptual_weight = perceptual_weight
        
        self.adversarial_loss = AdversarialLoss(gan_type)
        self.l1_loss = L1Loss()
        self.spatial_loss = SpatialPreservationLoss()
        self.quantitative_loss = QuantitativePreservationLoss()
        self.identity_loss = IdentityLoss()
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
    
    def generator_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        disc_fake: torch.Tensor,
        compute_identity: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute generator loss.
        
        Args:
            pred: Generated image
            target: Real target image
            source: Source image
            disc_fake: Discriminator output for fake image
            compute_identity: Whether to compute identity loss
            
        Returns:
            Dictionary with all loss components
        """
        losses = {}
        
        # Adversarial loss
        adv_loss = self.adversarial_loss(disc_fake, target_is_real=True)
        losses['adv_loss'] = adv_loss
        
        # L1 loss
        l1_loss = self.l1_loss(pred, target)
        losses['l1_loss'] = l1_loss
        
        # Spatial preservation loss
        spatial_dict = self.spatial_loss(pred, target)
        losses.update(spatial_dict)
        
        # Quantitative preservation loss
        quant_dict = self.quantitative_loss(pred, target)
        losses.update(quant_dict)
        
        # Identity loss
        if compute_identity:
            id_loss = self.identity_loss(pred, source)
            losses['identity_loss'] = id_loss
        else:
            losses['identity_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            perc_loss = self.perceptual_loss(pred, target)
            losses['perceptual_loss'] = perc_loss
        else:
            losses['perceptual_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Total generator loss
        total = (
            self.adv_weight * losses['adv_loss'] +
            self.l1_weight * losses['l1_loss'] +
            self.spatial_weight * losses['spatial_loss'] +
            self.quantitative_weight * losses['quantitative_loss'] +
            self.identity_weight * losses['identity_loss'] +
            self.perceptual_weight * losses['perceptual_loss']
        )
        losses['total_g_loss'] = total
        
        return losses
    
    def discriminator_loss(
        self,
        disc_real: torch.Tensor,
        disc_fake: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute discriminator loss.
        
        Args:
            disc_real: Discriminator output for real images
            disc_fake: Discriminator output for fake images
            
        Returns:
            Dictionary with loss components
        """
        real_loss = self.adversarial_loss(disc_real, target_is_real=True)
        fake_loss = self.adversarial_loss(disc_fake, target_is_real=False)
        
        total = (real_loss + fake_loss) * 0.5
        
        return {
            'total_d_loss': total,
            'd_real_loss': real_loss,
            'd_fake_loss': fake_loss
        }
