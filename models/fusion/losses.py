# models/fusion/losses.py

"""
Fusion-specific loss functions.

Loss functions for training learnable fusion networks and
evaluating fusion quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FusionReconstructionLoss(nn.Module):
    """
    Reconstruction loss for fusion.
    
    Measures how well the fused image matches the target.
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        fused: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(fused, target)


class WeightRegularizationLoss(nn.Module):
    """
    Regularization loss for fusion weights.
    
    Encourages smooth weight maps and prevents extreme values.
    """
    
    def __init__(
        self,
        smoothness_weight: float = 0.1,
        entropy_weight: float = 0.01
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.entropy_weight = entropy_weight
    
    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses.
        
        Args:
            alpha: Diffusion weight map (B, 1, H, W)
            beta: GAN weight map (B, 1, H, W)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Smoothness loss (total variation)
        tv_alpha = self._total_variation(alpha)
        tv_beta = self._total_variation(beta)
        losses['smoothness_loss'] = self.smoothness_weight * (tv_alpha + tv_beta)
        
        # Entropy regularization (prevent extreme weights)
        weights = torch.stack([alpha, beta], dim=-1)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
        losses['entropy_loss'] = -self.entropy_weight * entropy  # Maximize entropy
        
        losses['total_reg_loss'] = losses['smoothness_loss'] + losses['entropy_loss']
        
        return losses
    
    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation."""
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return tv_h + tv_w


class UncertaintyConsistencyLoss(nn.Module):
    """
    Loss to encourage weights to be consistent with uncertainty.
    
    Higher uncertainty should generally lead to lower weight.
    """
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss.
        
        When U_diff > U_gan, alpha should be < beta (with margin).
        """
        # Where diffusion is more uncertain, its weight should be lower
        diff_more_uncertain = U_diff > U_gan + self.margin
        expected_alpha_lower = diff_more_uncertain.float()
        
        # Penalize when alpha > beta but diffusion is more uncertain
        violation = F.relu(alpha - beta) * expected_alpha_lower
        
        # Where GAN is more uncertain, beta should be lower
        gan_more_uncertain = U_gan > U_diff + self.margin
        expected_beta_lower = gan_more_uncertain.float()
        
        violation = violation + F.relu(beta - alpha) * expected_beta_lower
        
        return violation.mean()


class PerceptualFusionLoss(nn.Module):
    """
    Perceptual loss for fusion quality.
    
    Uses VGG features to measure perceptual similarity.
    """
    
    def __init__(self, layers: tuple = (3, 8, 15)):
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
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert to RGB and normalize."""
        if x.shape[1] > 3:
            x = x[:, :3]
        elif x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def forward(
        self,
        fused: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss."""
        fused_rgb = self._preprocess(fused)
        target_rgb = self._preprocess(target)
        
        loss = 0
        fused_feat = fused_rgb
        target_feat = target_rgb
        
        for slice_module in self.slices:
            fused_feat = slice_module(fused_feat)
            target_feat = slice_module(target_feat)
            loss = loss + F.l1_loss(fused_feat, target_feat)
        
        return loss


class SSIMFusionLoss(nn.Module):
    """SSIM-based loss for structural preservation."""
    
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        
        # Create Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size).float() - window_size // 2
        kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        
        self.register_buffer('window', kernel_2d.view(1, 1, window_size, window_size))
    
    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM."""
        C, H, W = x.shape[1:]
        
        window = self.window.expand(C, 1, -1, -1).to(x.device)
        
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=C)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=C)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, window, padding=self.window_size // 2, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=self.window_size // 2, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=C) - mu_xy
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim.mean()
    
    def forward(
        self,
        fused: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM)."""
        return 1 - self._ssim(fused, target)


class CompositeFusionLoss(nn.Module):
    """
    Composite loss for training learnable fusion networks.
    
    Combines multiple loss terms with configurable weights.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        ssim_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        regularization_weight: float = 0.01,
        consistency_weight: float = 0.1,
        reconstruction_type: str = 'l1'
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.regularization_weight = regularization_weight
        self.consistency_weight = consistency_weight
        
        self.reconstruction_loss = FusionReconstructionLoss(reconstruction_type)
        self.ssim_loss = SSIMFusionLoss()
        self.regularization_loss = WeightRegularizationLoss()
        self.consistency_loss = UncertaintyConsistencyLoss()
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualFusionLoss()
        else:
            self.perceptual_loss = None
    
    def forward(
        self,
        fused: torch.Tensor,
        target: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        U_diff: Optional[torch.Tensor] = None,
        U_gan: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.
        
        Args:
            fused: Fused image
            target: Target image
            alpha: Diffusion weights
            beta: GAN weights
            U_diff: Diffusion uncertainty
            U_gan: GAN uncertainty
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Reconstruction loss
        losses['recon_loss'] = self.reconstruction_weight * self.reconstruction_loss(fused, target)
        
        # SSIM loss
        losses['ssim_loss'] = self.ssim_weight * self.ssim_loss(fused, target)
        
        # Perceptual loss
        if self.perceptual_loss is not None:
            losses['perceptual_loss'] = self.perceptual_weight * self.perceptual_loss(fused, target)
        else:
            losses['perceptual_loss'] = torch.tensor(0.0, device=fused.device)
        
        # Regularization loss
        reg_losses = self.regularization_loss(alpha, beta)
        losses['reg_loss'] = self.regularization_weight * reg_losses['total_reg_loss']
        
        # Consistency loss
        if U_diff is not None and U_gan is not None:
            losses['consistency_loss'] = self.consistency_weight * self.consistency_loss(
                alpha, beta, U_diff, U_gan
            )
        else:
            losses['consistency_loss'] = torch.tensor(0.0, device=fused.device)
        
        # Total loss
        losses['total_loss'] = (
            losses['recon_loss'] +
            losses['ssim_loss'] +
            losses['perceptual_loss'] +
            losses['reg_loss'] +
            losses['consistency_loss']
        )
        
        return losses


class FusionQualityMetrics:
    """
    Metrics for evaluating fusion quality.
    """
    
    @staticmethod
    def compute_all(
        fused: torch.Tensor,
        target: torch.Tensor,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all fusion quality metrics.
        
        Args:
            fused: Fused image
            target: Target image
            I_diff: Diffusion image
            I_gan: GAN image
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # PSNR
        mse = F.mse_loss(fused, target)
        metrics['psnr'] = (10 * torch.log10(1.0 / mse)).item()
        
        # Compare to individual branches
        mse_diff = F.mse_loss(I_diff, target)
        mse_gan = F.mse_loss(I_gan, target)
        metrics['psnr_diff'] = (10 * torch.log10(1.0 / mse_diff)).item()
        metrics['psnr_gan'] = (10 * torch.log10(1.0 / mse_gan)).item()
        
        # Improvement over best branch
        best_branch_psnr = max(metrics['psnr_diff'], metrics['psnr_gan'])
        metrics['psnr_improvement'] = metrics['psnr'] - best_branch_psnr
        
        # MSE
        metrics['mse'] = mse.item()
        metrics['mse_diff'] = mse_diff.item()
        metrics['mse_gan'] = mse_gan.item()
        
        return metrics
