"""
Utility functions for the Computer Vision project
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}


def load_image(image_path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Load and optionally resize image
    
    Args:
        image_path: Path to image file
        target_size: Tuple of (height, width) for resizing
        
    Returns:
        Image as numpy array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]))
    
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save image to file
    
    Args:
        image: Image as numpy array
        output_path: Path to save image
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    logger.info(f"Image saved to {output_path}")


def normalize_image(image: np.ndarray, mean: Tuple = None, std: Tuple = None) -> np.ndarray:
    """
    Normalize image using mean and standard deviation
    
    Args:
        image: Input image
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image
    """
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)
    
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(std)
    return image


def draw_bounding_box(image: np.ndarray, bbox: Tuple, label: str, confidence: float = None, 
                     color: Tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on image
    
    Args:
        image: Input image
        bbox: Bounding box as (x1, y1, x2, y2)
        label: Label text
        confidence: Confidence score (optional)
        color: RGB color tuple
        thickness: Line thickness
        
    Returns:
        Image with drawn bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    if confidence:
        text = f"{label}: {confidence:.2f}"
    else:
        text = label
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # Draw text background
    cv2.rectangle(image, (x1, y1 - text_size[1] - 5), 
                  (x1 + text_size[0] + 5, y1), color, -1)
    cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    return image


def get_device() -> str:
    """
    Get available device (cuda or cpu)
    
    Returns:
        Device string
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def create_directories(paths: list) -> None:
    """
    Create directories if they don't exist
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {path}")


def get_model_info(model) -> Dict[str, Any]:
    """
    Get information about a model
    
    Args:
        model: PyTorch or TensorFlow model
        
    Returns:
        Dictionary with model information
    """
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "framework": "PyTorch"
            }
    except ImportError:
        pass
    
    return {"error": "Unable to get model info"}


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
