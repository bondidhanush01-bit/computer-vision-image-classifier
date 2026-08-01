"""
Data preprocessing module for Computer Vision project
"""

import cv2
import numpy as np
from typing import Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handle image preprocessing and augmentation"""
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224), 
                 normalize: bool = True,
                 augment: bool = False):
        """
        Initialize preprocessor
        
        Args:
            input_size: Target image size (height, width)
            normalize: Whether to normalize image
            augment: Whether to apply augmentation
        """
        self.input_size = input_size
        self.normalize = normalize
        self.augment = augment
        
        self.transform = self._get_transform()
    
    def _get_transform(self):
        """Get albumentations transform pipeline"""
        transforms = [
            A.Resize(self.input_size[0], self.input_size[1]),
        ]
        
        if self.augment:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(p=0.2),
            ])
        
        if self.normalize:
            transforms.append(
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            )
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess single image
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        transformed = self.transform(image=image)
        return transformed['image']


class DataAugmentor:
    """Advanced data augmentation for training"""
    
    @staticmethod
    def apply_augmentation(image: np.ndarray, augment_config: dict = None) -> np.ndarray:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image
            augment_config: Augmentation configuration dictionary
            
        Returns:
            Augmented image
        """
        if augment_config is None:
            augment_config = {
                'random_flip': True,
                'random_rotation': 15,
                'color_jitter': True,
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
            }
        
        transforms = []
        
        if augment_config.get('random_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if augment_config.get('random_rotation', 0) > 0:
            transforms.append(A.Rotate(
                limit=augment_config['random_rotation'], 
                p=0.5
            ))
        
        if augment_config.get('color_jitter', False):
            transforms.append(A.ColorJitter(
                brightness=augment_config.get('brightness', 0.2),
                contrast=augment_config.get('contrast', 0.2),
                saturation=augment_config.get('saturation', 0.2),
                hue=augment_config.get('hue', 0.1),
                p=0.5
            ))
        
        transforms.extend([
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
        ])
        
        augmentor = A.Compose(transforms)
        return augmentor(image=image)['image']


class BatchPreprocessor:
    """Handle batch preprocessing"""
    
    def __init__(self, preprocessor: ImagePreprocessor):
        """Initialize batch preprocessor"""
        self.preprocessor = preprocessor
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess batch of images
        
        Args:
            images: List of images
            
        Returns:
            Stacked preprocessed images
        """
        processed = []
        for img in images:
            processed_img = self.preprocessor.preprocess(img)
            processed.append(processed_img)
        
        return np.stack(processed)


def normalize_coordinates(bbox: Tuple, image_shape: Tuple) -> Tuple:
    """
    Normalize bounding box coordinates to [0, 1]
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        image_shape: Image shape (height, width)
        
    Returns:
        Normalized coordinates
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    
    return (x1/w, y1/h, x2/w, y2/h)


def denormalize_coordinates(bbox: Tuple, image_shape: Tuple) -> Tuple:
    """
    Denormalize coordinates from [0, 1] to image dimensions
    
    Args:
        bbox: Normalized bounding box (x1, y1, x2, y2)
        image_shape: Image shape (height, width)
        
    Returns:
        Denormalized coordinates
    """
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    
    return (int(x1*w), int(y1*h), int(x2*w), int(y2*h))
