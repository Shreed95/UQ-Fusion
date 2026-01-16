# data/augmentation.py

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union, Callable
from dataclasses import dataclass
from scipy.ndimage import map_coordinates, gaussian_filter
import random


@dataclass
class AugmentationConfig:
    """Configuration for augmentation parameters."""
    # Geometric
    rotation_range: float = 15.0
    horizontal_flip: bool = True
    vertical_flip: bool = False
    elastic_alpha: float = 100.0
    elastic_sigma: float = 10.0
    scale_range: Tuple[float, float] = (0.85, 1.0)
    
    # Intensity
    brightness_range: float = 0.1
    contrast_range: float = 0.1
    noise_std_range: Tuple[float, float] = (0.01, 0.03)
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    bias_field_coeff: float = 0.5


class GeometricAugmentation:
    """Geometric augmentation transforms for 2D medical images."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def rotate(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        angle: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Rotate image by random angle within range.
        
        Args:
            image: Input image (C, H, W)
            segmentation: Optional segmentation mask (H, W)
            angle: Specific angle or None for random
            
        Returns:
            Rotated image and segmentation
        """
        if angle is None:
            angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
        
        # Convert to tensor for rotation
        image_tensor = torch.from_numpy(image).float()
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Create rotation matrix
        angle_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotation matrix
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Apply affine transformation
        grid = F.affine_grid(theta, image_tensor.unsqueeze(0).size(), align_corners=False)
        rotated = F.grid_sample(
            image_tensor.unsqueeze(0), grid, 
            mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze(0).numpy()
        
        # Rotate segmentation with nearest neighbor
        seg_rotated = None
        if segmentation is not None:
            seg_tensor = torch.from_numpy(segmentation).float().unsqueeze(0).unsqueeze(0)
            seg_rotated = F.grid_sample(
                seg_tensor, grid,
                mode='nearest', padding_mode='zeros', align_corners=False
            ).squeeze().numpy()
        
        return rotated, seg_rotated
    
    def flip_horizontal(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Flip image horizontally."""
        flipped = np.flip(image, axis=-1).copy()
        seg_flipped = np.flip(segmentation, axis=-1).copy() if segmentation is not None else None
        return flipped, seg_flipped
    
    def flip_vertical(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Flip image vertically."""
        flipped = np.flip(image, axis=-2).copy()
        seg_flipped = np.flip(segmentation, axis=-2).copy() if segmentation is not None else None
        return flipped, seg_flipped
    
    def elastic_deformation(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply elastic deformation.
        
        Args:
            image: Input image (C, H, W)
            segmentation: Optional segmentation (H, W)
            alpha: Deformation magnitude
            sigma: Gaussian filter sigma
            
        Returns:
            Deformed image and segmentation
        """
        if alpha is None:
            alpha = self.config.elastic_alpha
        if sigma is None:
            sigma = self.config.elastic_sigma
        
        if image.ndim == 3:
            shape = image.shape[1:]  # (H, W)
        else:
            shape = image.shape
        
        # Generate random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma
        ) * alpha
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), sigma
        ) * alpha
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices_x = np.clip(x + dx, 0, shape[1] - 1)
        indices_y = np.clip(y + dy, 0, shape[0] - 1)
        
        # Apply deformation to each channel
        if image.ndim == 3:
            deformed = np.zeros_like(image)
            for c in range(image.shape[0]):
                deformed[c] = map_coordinates(
                    image[c], [indices_y, indices_x], order=1, mode='reflect'
                )
        else:
            deformed = map_coordinates(
                image, [indices_y, indices_x], order=1, mode='reflect'
            )
        
        # Apply to segmentation with nearest neighbor
        seg_deformed = None
        if segmentation is not None:
            seg_deformed = map_coordinates(
                segmentation, [indices_y, indices_x], order=0, mode='reflect'
            )
        
        return deformed, seg_deformed
    
    def random_crop_resize(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
        scale: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Random crop and resize back to original size.
        
        Args:
            image: Input image (C, H, W)
            segmentation: Optional segmentation (H, W)
            scale: Crop scale (0-1) or None for random
            
        Returns:
            Cropped and resized image and segmentation
        """
        if scale is None:
            scale = random.uniform(*self.config.scale_range)
        
        if image.ndim == 3:
            _, h, w = image.shape
        else:
            h, w = image.shape
        
        # Calculate crop size
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        
        # Random crop position
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        # Crop
        if image.ndim == 3:
            cropped = image[:, top:top+crop_h, left:left+crop_w]
        else:
            cropped = image[top:top+crop_h, left:left+crop_w]
        
        # Resize back using torch
        cropped_tensor = torch.from_numpy(cropped).float()
        if cropped_tensor.dim() == 2:
            cropped_tensor = cropped_tensor.unsqueeze(0)
        
        resized = F.interpolate(
            cropped_tensor.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0).numpy()
        
        # Process segmentation
        seg_resized = None
        if segmentation is not None:
            seg_cropped = segmentation[top:top+crop_h, left:left+crop_w]
            seg_tensor = torch.from_numpy(seg_cropped).float().unsqueeze(0).unsqueeze(0)
            seg_resized = F.interpolate(
                seg_tensor, size=(h, w), mode='nearest'
            ).squeeze().numpy()
        
        return resized, seg_resized


class IntensityAugmentation:
    """Intensity augmentation transforms for medical images."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def adjust_brightness(
        self,
        image: np.ndarray,
        factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image (C, H, W)
            factor: Brightness adjustment factor or None for random
            
        Returns:
            Adjusted image
        """
        if factor is None:
            factor = random.uniform(
                -self.config.brightness_range, 
                self.config.brightness_range
            )
        
        adjusted = image + factor
        return np.clip(adjusted, 0, 1)
    
    def adjust_contrast(
        self,
        image: np.ndarray,
        factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image (C, H, W)
            factor: Contrast adjustment factor or None for random
            
        Returns:
            Adjusted image
        """
        if factor is None:
            factor = 1.0 + random.uniform(
                -self.config.contrast_range,
                self.config.contrast_range
            )
        
        # Compute mean per channel
        if image.ndim == 3:
            mean = image.mean(axis=(1, 2), keepdims=True)
        else:
            mean = image.mean()
        
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 1)
    
    def add_gaussian_noise(
        self,
        image: np.ndarray,
        std: Optional[float] = None
    ) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image (C, H, W)
            std: Noise standard deviation or None for random
            
        Returns:
            Noisy image
        """
        if std is None:
            std = random.uniform(*self.config.noise_std_range)
        
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = image + noise
        return np.clip(noisy, 0, 1)
    
    def gamma_correction(
        self,
        image: np.ndarray,
        gamma: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply gamma correction.
        
        Args:
            image: Input image (C, H, W), expected in [0, 1]
            gamma: Gamma value or None for random
            
        Returns:
            Gamma corrected image
        """
        if gamma is None:
            gamma = random.uniform(*self.config.gamma_range)
        
        # Avoid numerical issues with 0
        epsilon = 1e-7
        corrected = np.power(image + epsilon, gamma)
        return np.clip(corrected, 0, 1)
    
    def simulate_bias_field(
        self,
        image: np.ndarray,
        coefficients: Optional[float] = None
    ) -> np.ndarray:
        """
        Simulate MRI bias field artifact.
        
        Args:
            image: Input image (C, H, W)
            coefficients: Bias field strength or None for random
            
        Returns:
            Image with bias field
        """
        if coefficients is None:
            coefficients = random.uniform(0, self.config.bias_field_coeff)
        
        if image.ndim == 3:
            h, w = image.shape[1], image.shape[2]
        else:
            h, w = image.shape
        
        # Create smooth bias field using low-frequency polynomial
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Random polynomial coefficients
        a = random.uniform(-1, 1) * coefficients
        b = random.uniform(-1, 1) * coefficients
        c = random.uniform(-1, 1) * coefficients
        
        bias_field = 1 + a * xx + b * yy + c * xx * yy
        bias_field = gaussian_filter(bias_field, sigma=30)
        
        # Normalize bias field
        bias_field = bias_field / bias_field.mean()
        
        if image.ndim == 3:
            bias_field = bias_field[np.newaxis, ...]
        
        biased = image * bias_field
        return np.clip(biased, 0, 1)


class MedicalImageAugmentor:
    """
    Complete augmentation pipeline for medical images.
    Combines geometric and intensity augmentations with configurable probabilities.
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        geometric_prob: float = 0.5,
        intensity_prob: float = 0.5
    ):
        """
        Initialize augmentor.
        
        Args:
            config: Augmentation configuration
            geometric_prob: Probability of applying each geometric transform
            intensity_prob: Probability of applying each intensity transform
        """
        if config is None:
            config = AugmentationConfig()
        
        self.config = config
        self.geometric_prob = geometric_prob
        self.intensity_prob = intensity_prob
        
        self.geometric = GeometricAugmentation(config)
        self.intensity = IntensityAugmentation(config)
    
    def __call__(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random augmentations.
        
        Args:
            image: Input image (C, H, W)
            segmentation: Optional segmentation mask (H, W)
            
        Returns:
            Augmented image and segmentation
        """
        aug_image = image.copy()
        aug_seg = segmentation.copy() if segmentation is not None else None
        
        # Geometric augmentations
        if random.random() < self.geometric_prob:
            aug_image, aug_seg = self.geometric.rotate(aug_image, aug_seg)
        
        if self.config.horizontal_flip and random.random() < self.geometric_prob:
            aug_image, aug_seg = self.geometric.flip_horizontal(aug_image, aug_seg)
        
        if self.config.vertical_flip and random.random() < self.geometric_prob:
            aug_image, aug_seg = self.geometric.flip_vertical(aug_image, aug_seg)
        
        if random.random() < self.geometric_prob * 0.5:  # Less frequent
            aug_image, aug_seg = self.geometric.elastic_deformation(aug_image, aug_seg)
        
        if random.random() < self.geometric_prob:
            aug_image, aug_seg = self.geometric.random_crop_resize(aug_image, aug_seg)
        
        # Intensity augmentations (only on image, not segmentation)
        if random.random() < self.intensity_prob:
            aug_image = self.intensity.adjust_brightness(aug_image)
        
        if random.random() < self.intensity_prob:
            aug_image = self.intensity.adjust_contrast(aug_image)
        
        if random.random() < self.intensity_prob:
            aug_image = self.intensity.add_gaussian_noise(aug_image)
        
        if random.random() < self.intensity_prob:
            aug_image = self.intensity.gamma_correction(aug_image)
        
        if random.random() < self.intensity_prob * 0.3:  # Less frequent
            aug_image = self.intensity.simulate_bias_field(aug_image)
        
        return aug_image.astype(np.float32), aug_seg


class AugmentationPipeline:
    """
    Deterministic augmentation pipeline for reproducible experiments.
    Allows specifying exact sequence of transforms.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        if config is None:
            config = AugmentationConfig()
        
        self.config = config
        self.geometric = GeometricAugmentation(config)
        self.intensity = IntensityAugmentation(config)
        self.transforms: List[Callable] = []
    
    def add_rotation(self, angle: float):
        """Add rotation transform."""
        self.transforms.append(
            lambda img, seg: self.geometric.rotate(img, seg, angle)
        )
        return self
    
    def add_horizontal_flip(self):
        """Add horizontal flip."""
        self.transforms.append(self.geometric.flip_horizontal)
        return self
    
    def add_vertical_flip(self):
        """Add vertical flip."""
        self.transforms.append(self.geometric.flip_vertical)
        return self
    
    def add_elastic(self, alpha: float = None, sigma: float = None):
        """Add elastic deformation."""
        self.transforms.append(
            lambda img, seg: self.geometric.elastic_deformation(img, seg, alpha, sigma)
        )
        return self
    
    def add_brightness(self, factor: float):
        """Add brightness adjustment."""
        self.transforms.append(
            lambda img, seg: (self.intensity.adjust_brightness(img, factor), seg)
        )
        return self
    
    def add_contrast(self, factor: float):
        """Add contrast adjustment."""
        self.transforms.append(
            lambda img, seg: (self.intensity.adjust_contrast(img, factor), seg)
        )
        return self
    
    def add_noise(self, std: float):
        """Add Gaussian noise."""
        self.transforms.append(
            lambda img, seg: (self.intensity.add_gaussian_noise(img, std), seg)
        )
        return self
    
    def add_gamma(self, gamma: float):
        """Add gamma correction."""
        self.transforms.append(
            lambda img, seg: (self.intensity.gamma_correction(img, gamma), seg)
        )
        return self
    
    def __call__(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply all transforms in sequence."""
        aug_image = image.copy()
        aug_seg = segmentation.copy() if segmentation is not None else None
        
        for transform in self.transforms:
            aug_image, aug_seg = transform(aug_image, aug_seg)
        
        return aug_image.astype(np.float32), aug_seg
    
    def clear(self):
        """Clear all transforms."""
        self.transforms = []
        return self