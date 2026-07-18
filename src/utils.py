"""
Utility functions for the image classifier project
"""

import torch
import yaml
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Get the device to use for training/inference
    
    Returns:
        str: 'cuda' if GPU is available, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    else:
        logger.info("Using CPU device")
        return 'cpu'


def create_directories(directories: List[str]) -> None:
    """
    Create directories if they don't exist
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the config YAML file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the config file
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved config to {config_path}")


def get_model_size(model: torch.nn.Module) -> float:
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        float: Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return size_all_mb


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
