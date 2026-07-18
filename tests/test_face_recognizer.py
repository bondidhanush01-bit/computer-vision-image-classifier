"""
Unit tests for face recognizer
"""

import pytest
from src.inference import FaceRecognizer


class TestFaceRecognizer:
    """Test cases for FaceRecognizer"""
    
    @pytest.fixture
    def recognizer(self):
        """Initialize recognizer for testing"""
        return FaceRecognizer(detection_confidence=0.7)
    
    def test_recognizer_initialization(self, recognizer):
        """Test recognizer initialization"""
        assert recognizer is not None
        assert recognizer.detection_confidence == 0.7
    
    def test_face_detection_method_exists(self, recognizer):
        """Test face detection method exists"""
        assert hasattr(recognizer, 'detect_faces')
    
    def test_face_recognition_method_exists(self, recognizer):
        """Test face recognition method exists"""
        assert hasattr(recognizer, 'recognize')
    
    def test_recognize_with_empty_faces(self, recognizer):
        """Test recognize method with empty faces list"""
        result = recognizer.recognize([])
        assert result == []
