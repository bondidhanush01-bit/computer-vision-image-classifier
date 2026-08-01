"""
Utility functions for the image classifier project
"""

import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def draw_bounding_box(
    image: np.ndarray,
    bbox: Union[Tuple[float, float, float, float], np.ndarray],
    label: str = "",
    confidence: float = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box with an optional label on an image.

    Args:
        image: Image as a numpy array (H, W, C)
        bbox: Bounding box as (x1, y1, x2, y2)
        label: Text label to draw above the box
        confidence: Optional confidence score (kept for API compatibility;
            include it in `label` if you want it displayed)
        color: BGR/RGB color tuple for the box and label background
        thickness: Line thickness for the box

    Returns:
        np.ndarray: Image with the bounding box (and label) drawn on it
    """
    image = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

        # Background rectangle for the label so text stays readable
        cv2.rectangle(
            image,
            (x1, max(0, y1 - text_h - baseline - 4)),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 2, max(text_h, y1 - baseline - 2)),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an RGB numpy image array to disk.

    Args:
        image: Image as a numpy array (H, W, C) in RGB order
        output_path: Path to save the image to
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite expects BGR order
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(output_path), bgr_image)
    if not success:
        raise IOError(f"Failed to save image to {output_path}")
    logger.info(f"Image saved to {output_path}")


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
