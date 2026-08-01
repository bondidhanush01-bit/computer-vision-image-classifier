"""
Unit tests for image classifier
"""

import pytest
import torch
import numpy as np
from src.inference import ImageClassifier


class TestImageClassifier:
    """Test cases for ImageClassifier"""
    
    @pytest.fixture
    def classifier(self):
        """Initialize classifier for testing"""
        return ImageClassifier(model_type='resnet18', num_classes=10)
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier is not None
        assert classifier.model_type == 'resnet18'
        assert classifier.num_classes == 10
    
    def test_model_loading(self, classifier):
        """Test model loading"""
        assert classifier.model is not None
        assert isinstance(classifier.model, torch.nn.Module)
    
    def test_class_names(self, classifier):
        """Test class names"""
        assert len(classifier.class_names) == 10
        assert classifier.class_names[0] == 'airplane'
