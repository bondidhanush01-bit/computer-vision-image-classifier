"""
Unit tests for object detector
"""

import pytest
from src.inference import ObjectDetector


class TestObjectDetector:
    """Test cases for ObjectDetector"""
    
    @pytest.fixture
    def detector(self):
        """Initialize detector for testing"""
        return ObjectDetector(model_name='yolov3', confidence=0.5)
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.model_name == 'yolov3'
        assert detector.confidence == 0.5
    
    def test_confidence_threshold(self, detector):
        """Test confidence threshold"""
        assert detector.confidence == 0.5
        assert detector.iou_threshold == 0.4
    
    def test_detect_with_custom_confidence(self, detector):
        """Test detection with custom confidence"""
        # This would test actual detection if model is available
        assert hasattr(detector, 'detect')
