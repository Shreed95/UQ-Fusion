# models/vae/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SSIMLoss(nn.Module):
    """SSIM loss. Only instantiated when ssim_weight > 0."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5, channel: int = 4):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel, sigma))

    def _create_window(self, ws: int, ch: int, sigma: float) -> torch.Tensor:
        x = torch.arange(ws).float() - ws // 2
        g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        g = g / g.sum()
        w2d = g.unsqueeze(1).mm(g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        return w2d.expand(ch, 1, ws, ws).contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ch = pred.shape[1]
        win = self._create_window(self.window_size, ch, 1.5).to(pred.device) if ch != self.channel else self.window.to(pred.device)
        pad = self.window_size // 2
        mu1 = F.conv2d(pred, win, padding=pad, groups=ch)
        mu2 = F.conv2d(target, win, padding=pad, groups=ch)
        s1 = F.conv2d(pred * pred, win, padding=pad, groups=ch) - mu1 * mu1
        s2 = F.conv2d(target * target, win, padding=pad, groups=ch) - mu2 * mu2
        s12 = F.conv2d(pred * target, win, padding=pad, groups=ch) - mu1 * mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        return 1 - ssim.mean()


class VAELoss(nn.Module):
    """Basic VAE loss: recon + KL."""
    def __init__(self, recon_loss_type='l1', recon_weight=1.0, kl_weight=0.0001,
                 perceptual_weight=0.0, use_perceptual=False):
        super().__init__()
        self.recon_loss_type = recon_loss_type
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def forward(self, recon, target, mean, log_var):
        if self.recon_loss_type == 'l1':
            l_recon = F.l1_loss(recon, target)
        else:
            l_recon = F.mse_loss(recon, target)
        l_kl = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        total = self.recon_weight * l_recon + self.kl_weight * l_kl
        return total, {'loss': total, 'recon_loss': l_recon, 'kl_loss': l_kl}


class CombinedVAELoss(nn.Module):
    """
    Combined VAE loss: L1 + optional SSIM + KL.
    SSIM is NOT computed when ssim_weight=0 (saves significant memory on MPS).
    """
    def __init__(self, recon_weight=1.0, kl_weight=0.0001, ssim_weight=0.0, l1_weight=1.0):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.ssim_loss = SSIMLoss() if ssim_weight > 0 else None

    def forward(self, recon, target, mean, log_var):
        l1 = F.l1_loss(recon, target)

        if self.ssim_loss is not None and self.ssim_weight > 0:
            ssim = self.ssim_loss(recon, target)
            l_recon = self.l1_weight * l1 + self.ssim_weight * ssim
        else:
            ssim = torch.zeros(1, device=recon.device)
            l_recon = self.l1_weight * l1

        kl = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        total = self.recon_weight * l_recon + self.kl_weight * kl

        return total, {
            'loss': total, 'recon_loss': l_recon,
            'l1_loss': l1, 'ssim_loss': ssim, 'kl_loss': kl
        }