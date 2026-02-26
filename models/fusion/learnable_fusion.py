# models/fusion/learnable_fusion.py

"""
Learnable Fusion Network.

A lightweight CNN that learns optimal fusion weights from:
- Generated images from both branches
- Uncertainty maps from both branches

Input: Concatenation of I_diff, I_gan, U_diff, U_gan
Output: Softmax weights (alpha, beta) for each spatial location
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class LearnableFusionConfig:
    """Configuration for learnable fusion network."""
    in_channels: int = 10          # 4 + 4 + 1 + 1 (images + uncertainties)
    hidden_channels: int = 64
    num_layers: int = 3
    use_attention: bool = True
    dropout_rate: float = 0.1
    output_activation: str = 'softmax'  # 'softmax' or 'sigmoid'


class ConvBlock(nn.Module):
    """Convolutional block with norm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module (SE block)."""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class LearnableFusionNetwork(nn.Module):
    """
    Learnable fusion network that predicts optimal fusion weights.
    
    Takes concatenated inputs and outputs per-pixel fusion weights.
    """
    
    def __init__(self, config: Optional[LearnableFusionConfig] = None):
        super().__init__()
        
        if config is None:
            config = LearnableFusionConfig()
        
        self.config = config
        
        # Initial projection
        self.input_proj = ConvBlock(
            config.in_channels,
            config.hidden_channels,
            kernel_size=3,
            dropout_rate=config.dropout_rate
        )
        
        # Processing layers
        layers = []
        for i in range(config.num_layers - 1):
            layers.append(ConvBlock(
                config.hidden_channels,
                config.hidden_channels,
                kernel_size=3,
                dropout_rate=config.dropout_rate
            ))
            
            if config.use_attention and i == config.num_layers // 2:
                layers.append(SpatialAttention(config.hidden_channels))
                layers.append(ChannelAttention(config.hidden_channels))
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_conv = nn.Conv2d(config.hidden_channels, 2, kernel_size=1)
        
        # Initialize output to predict equal weights
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict fusion weights.
        
        Args:
            I_diff: Diffusion image (B, 4, H, W)
            I_gan: GAN image (B, 4, H, W)
            U_diff: Diffusion uncertainty (B, 1, H, W)
            U_gan: GAN uncertainty (B, 1, H, W)
            
        Returns:
            Dictionary with fused image and weights
        """
        # Concatenate inputs
        x = torch.cat([I_diff, I_gan, U_diff, U_gan], dim=1)
        
        # Process
        x = self.input_proj(x)
        x = self.layers(x)
        
        # Output weights
        weights = self.output_conv(x)
        
        if self.config.output_activation == 'softmax':
            weights = F.softmax(weights, dim=1)
        else:
            weights = torch.sigmoid(weights)
            weights = weights / weights.sum(dim=1, keepdim=True)
        
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        
        # Expand weights for multi-channel fusion
        alpha_exp = alpha.expand(-1, I_diff.shape[1], -1, -1)
        beta_exp = beta.expand(-1, I_gan.shape[1], -1, -1)
        
        # Fuse
        I_fused = alpha_exp * I_diff + beta_exp * I_gan
        
        return {
            'fused': I_fused,
            'alpha': alpha,
            'beta': beta,
            'raw_weights': weights
        }


class AttentionFusionNetwork(nn.Module):
    """
    Attention-based fusion network using cross-attention
    between branches and self-attention within features.
    """
    
    def __init__(
        self,
        image_channels: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.hidden_dim = hidden_dim
        
        # Encoders for each branch
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(image_channels + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        
        self.gan_encoder = nn.Sequential(
            nn.Conv2d(image_channels + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1)
        )
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with attention."""
        B, C, H, W = I_diff.shape
        
        # Encode each branch with uncertainty
        diff_input = torch.cat([I_diff, U_diff], dim=1)
        gan_input = torch.cat([I_gan, U_gan], dim=1)
        
        diff_feat = self.diff_encoder(diff_input)  # (B, hidden, H, W)
        gan_feat = self.gan_encoder(gan_input)
        
        # Reshape for attention: (B, H*W, hidden)
        diff_flat = diff_feat.flatten(2).transpose(1, 2)
        gan_flat = gan_feat.flatten(2).transpose(1, 2)
        
        # Cross-attention: diff attends to gan
        attn_out, _ = self.cross_attention(diff_flat, gan_flat, gan_flat)
        attn_feat = attn_out.transpose(1, 2).view(B, self.hidden_dim, H, W)
        
        # Combine features
        combined = torch.cat([diff_feat, attn_feat], dim=1)
        
        # Predict weights
        weights = self.weight_predictor(combined)
        weights = F.softmax(weights, dim=1)
        
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        
        # Fuse
        alpha_exp = alpha.expand(-1, C, -1, -1)
        beta_exp = beta.expand(-1, C, -1, -1)
        I_fused = alpha_exp * I_diff + beta_exp * I_gan
        
        return {
            'fused': I_fused,
            'alpha': alpha,
            'beta': beta
        }


class UNetFusionNetwork(nn.Module):
    """
    U-Net style fusion network for multi-scale fusion.
    """
    
    def __init__(
        self,
        in_channels: int = 10,
        base_channels: int = 32
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2, base_channels)
        
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        
        # Output
        self.output = nn.Conv2d(base_channels, 2, kernel_size=1)
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Concatenate inputs
        x = torch.cat([I_diff, I_gan, U_diff, U_gan], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Output weights
        weights = F.softmax(self.output(d1), dim=1)
        
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        
        # Fuse
        alpha_exp = alpha.expand(-1, I_diff.shape[1], -1, -1)
        beta_exp = beta.expand(-1, I_gan.shape[1], -1, -1)
        I_fused = alpha_exp * I_diff + beta_exp * I_gan
        
        return {
            'fused': I_fused,
            'alpha': alpha,
            'beta': beta
        }


class HybridFusion(nn.Module):
    """
    Hybrid fusion combining uncertainty-guided and learnable approaches.
    
    Uses uncertainty as a prior and learns refinements.
    """
    
    def __init__(
        self,
        uncertainty_weight: float = 0.5,
        learned_weight: float = 0.5,
        config: Optional[LearnableFusionConfig] = None
    ):
        super().__init__()
        
        self.uncertainty_weight = uncertainty_weight
        self.learned_weight = learned_weight
        
        # Learnable network
        self.learned_fusion = LearnableFusionNetwork(config)
    
    def forward(
        self,
        I_diff: torch.Tensor,
        I_gan: torch.Tensor,
        U_diff: torch.Tensor,
        U_gan: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Hybrid fusion combining both approaches."""
        # Uncertainty-based weights
        eps = 1e-6
        W_diff = 1.0 / (U_diff + eps)
        W_gan = 1.0 / (U_gan + eps)
        W_total = W_diff + W_gan
        alpha_unc = W_diff / W_total
        beta_unc = W_gan / W_total
        
        # Learned weights
        learned_result = self.learned_fusion(I_diff, I_gan, U_diff, U_gan)
        alpha_learned = learned_result['alpha']
        beta_learned = learned_result['beta']
        
        # Combine
        alpha = self.uncertainty_weight * alpha_unc + self.learned_weight * alpha_learned
        beta = self.uncertainty_weight * beta_unc + self.learned_weight * beta_learned
        
        # Normalize
        total = alpha + beta
        alpha = alpha / total
        beta = beta / total
        
        # Fuse
        alpha_exp = alpha.expand(-1, I_diff.shape[1], -1, -1)
        beta_exp = beta.expand(-1, I_gan.shape[1], -1, -1)
        I_fused = alpha_exp * I_diff + beta_exp * I_gan
        
        return {
            'fused': I_fused,
            'alpha': alpha,
            'beta': beta,
            'alpha_uncertainty': alpha_unc,
            'alpha_learned': alpha_learned
        }


def create_learnable_fusion(
    method: str = 'simple',
    **kwargs
) -> nn.Module:
    """
    Factory function for learnable fusion networks.
    
    Args:
        method: 'simple', 'attention', 'unet', or 'hybrid'
        **kwargs: Additional arguments
        
    Returns:
        Learnable fusion network
    """
    if method == 'simple':
        config = LearnableFusionConfig(**kwargs)
        return LearnableFusionNetwork(config)
    elif method == 'attention':
        return AttentionFusionNetwork(**kwargs)
    elif method == 'unet':
        return UNetFusionNetwork(**kwargs)
    elif method == 'hybrid':
        return HybridFusion(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
