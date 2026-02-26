# configs/__init__.py

"""
Configuration management for UQ-Fusion.

Provides utilities for loading, validating, and accessing configuration.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_device(config: Dict[str, Any] = None) -> torch.device:
    """
    Get compute device based on config and availability.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        torch.device
    """
    if config is None:
        device_str = "auto"
    else:
        device_str = config.get('hardware', {}).get('device', 'auto')
    
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_str)


def setup_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Setup and create all required directories.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of Path objects
    """
    paths_config = config.get('paths', {})
    
    paths = {}
    for key, value in paths_config.items():
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        paths[key] = path
    
    # Create subdirectories
    checkpoint_dir = paths.get('checkpoint_dir', Path('./outputs/checkpoints'))
    for subdir in ['vae', 'diffusion', 'gan', 'fusion', 'segmentation']:
        (checkpoint_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    evaluation_dir = paths.get('evaluation_dir', Path('./outputs/evaluation'))
    for subdir in ['vae', 'diffusion', 'gan', 'uncertainty', 'fusion', 'quality', 'segmentation', 'comparison']:
        (evaluation_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    expanded_dir = paths.get('expanded_dataset_dir', Path('./outputs/expanded_dataset'))
    (expanded_dir / 'accepted').mkdir(parents=True, exist_ok=True)
    (expanded_dir / 'rejected').mkdir(parents=True, exist_ok=True)
    
    return paths


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path('./data'))
    checkpoint_dir: Path = field(default_factory=lambda: Path('./outputs/checkpoints'))
    output_dir: Path = field(default_factory=lambda: Path('./outputs'))
    
    # Hardware
    device: str = 'auto'
    num_workers: int = 4
    
    # Training flags
    train_vae: bool = True
    train_diffusion: bool = True
    train_gan: bool = True
    train_fusion: bool = False
    train_segmentation: bool = True
    
    # Generation
    generate_dataset: bool = True
    expansion_factor: int = 2
    
    # Evaluation
    evaluate_all: bool = True
    compare_augmentation: bool = True
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PipelineConfig':
        """Create from config dictionary."""
        return cls(
            data_dir=Path(config.get('paths', {}).get('data_dir', './data')),
            checkpoint_dir=Path(config.get('paths', {}).get('checkpoint_dir', './outputs/checkpoints')),
            output_dir=Path(config.get('paths', {}).get('output_dir', './outputs')),
            device=config.get('hardware', {}).get('device', 'auto'),
            num_workers=config.get('hardware', {}).get('num_workers', 4)
        )


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError if invalid
    """
    required_sections = ['paths', 'preprocessing', 'vae', 'diffusion', 'gan', 
                         'uncertainty', 'fusion', 'validation', 'segmentation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate paths
    paths = config.get('paths', {})
    if 'data_dir' not in paths:
        raise ValueError("Missing required path: data_dir")
    
    # Validate numeric ranges
    vae = config.get('vae', {})
    if vae.get('epochs', 0) <= 0:
        raise ValueError("VAE epochs must be positive")
    
    validation = config.get('validation', {})
    threshold = validation.get('acceptance_threshold', 0)
    if not 0 < threshold <= 1:
        raise ValueError("Acceptance threshold must be between 0 and 1")
    
    return True


def get_phase_config(config: Dict[str, Any], phase: str) -> Dict[str, Any]:
    """
    Get configuration for a specific phase.
    
    Args:
        config: Full configuration
        phase: Phase name (vae, diffusion, gan, etc.)
        
    Returns:
        Phase-specific configuration
    """
    phase_config = config.get(phase, {})
    
    # Add common settings
    phase_config['device'] = get_device(config)
    phase_config['num_workers'] = config.get('hardware', {}).get('num_workers', 4)
    
    # Add paths
    paths = config.get('paths', {})
    phase_config['checkpoint_dir'] = paths.get('checkpoint_dir', './outputs/checkpoints')
    phase_config['log_dir'] = paths.get('log_dir', './outputs/logs')
    
    return phase_config


# Convenience function
def load_and_validate_config(config_path: str = None) -> Dict[str, Any]:
    """Load and validate configuration."""
    config = load_config(config_path)
    validate_config(config)
    return config
