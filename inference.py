"""
Inference module for Image Classification, Object Detection, and Face Recognition
"""

import cv2
import numpy as np
import torch
import torchvision.models as models
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_image_rgb(image_path: str) -> np.ndarray:
    """
    Load an image from disk and convert it to RGB, raising a clear error
    if the path doesn't exist or the file can't be read as an image.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"Could not read '{image_path}' as an image. "
            f"It may be corrupted or not a supported image format."
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ImageClassifier:
    """Image classification using CNN"""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = 'resnet50',
                 num_classes: int = 10, device: str = 'cuda'):
        """
        Initialize image classifier
        
        Args:
            model_path: Path to pretrained model weights
            model_type: Type of model architecture
            num_classes: Number of output classes
            device: Device to use (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        self.num_classes = num_classes
        
        # Load model
        self.model = self._load_model(model_type, num_classes)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model loaded from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Class names (for CIFAR-10)
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
    
    def _load_model(self, model_type: str, num_classes: int):
        """Load pretrained model"""
        if model_type == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'efficientnet':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_type == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def predict(self, image_path: str, top_k: int = 5) -> Dict:
        """
        Make prediction on image
        
        Args:
            image_path: Path to image
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidences
        """
        image = _load_image_rgb(image_path)
        image_tensor = self._preprocess_image(image)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            results.append({
                'class': self.class_names[idx.item()],
                'confidence': prob.item()
            })
        
        return {'predictions': results, 'top_prediction': results[0]}
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)


class ObjectDetector:
    """Object detection using YOLOv3"""
    
    def __init__(self, model_name: str = 'yolov3', confidence: float = 0.5, iou_threshold: float = 0.4):
        """
        Initialize object detector
        
        Args:
            model_name: Name of YOLO model
            confidence: Confidence threshold
            iou_threshold: IOU threshold for NMS
        """
        self.model_name = model_name
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(f'{model_name}.pt')
            logger.info(f"YOLO model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model = None
    
    def detect(self, image_path: str, confidence: Optional[float] = None) -> List[Dict]:
        """
        Detect objects in image
        
        Args:
            image_path: Path to image
            confidence: Confidence threshold (optional, uses default if not provided)
            
        Returns:
            List of detections with bounding boxes and labels
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        conf = confidence or self.confidence
        results = self.model.predict(image_path, conf=conf, iou=self.iou_threshold)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy(),
                    'confidence': float(box.conf),
                    'class': int(box.cls),
                    'class_name': result.names[int(box.cls)]
                })
        
        return detections
    
    def visualize_detections(self, image_path: str, detections: List[Dict], output_path: Optional[str] = None):
        """
        Visualize detections on image
        
        Args:
            image_path: Path to image
            detections: List of detections
            output_path: Path to save annotated image
        """
        from src.utils import draw_bounding_box, save_image
        
        image = _load_image_rgb(image_path)

        for det in detections:
            bbox = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            image = draw_bounding_box(image, bbox, label, det['confidence'])
        
        if output_path:
            save_image(image, output_path)
        
        return image


_HAAR_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)
_HAAR_CASCADE_PATH = Path(__file__).resolve().parent.parent / "models" / "haarcascade_frontalface_default.xml"


class FaceRecognizer:
    """Face detection and recognition"""
    
    def __init__(self, model_path: Optional[str] = None, detection_confidence: float = 0.7):
        """
        Initialize face recognizer
        
        Uses OpenCV's Haar Cascade classifier for face detection. This avoids
        mediapipe's Tasks API, which has an open upstream bug on Windows +
        Python 3.12+ (AttributeError: function 'free' not found -- see
        https://github.com/google-ai-edge/mediapipe/issues/6187) that breaks
        every Tasks API feature and currently has no available pip-installable
        fix for recent Python versions on Windows.
        
        Args:
            model_path: Path to a Haar Cascade XML file. Defaults to
                models/haarcascade_frontalface_default.xml, downloading it
                from OpenCV's GitHub repo on first use if not present. This
                avoids relying on cv2.data.haarcascades, which some
                opencv-python wheels (e.g. on very new Python versions) don't
                ship with the underlying XML files present.
            detection_confidence: Unused by Haar Cascades (kept for API
                compatibility with callers).
        """
        self.detection_confidence = detection_confidence
        self.face_detection = None
        
        try:
            cascade_file = Path(model_path) if model_path else _HAAR_CASCADE_PATH
            if not cascade_file.exists():
                import urllib.request
                cascade_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading Haar Cascade file to {cascade_file}")
                urllib.request.urlretrieve(_HAAR_CASCADE_URL, str(cascade_file))

            self._cascade = cv2.CascadeClassifier(str(cascade_file))
            if self._cascade.empty():
                raise IOError(f"Could not load cascade file: {cascade_file}")
            
            # Non-None marks detection as "loaded" for health checks
            self.face_detection = self._cascade
            logger.info(f"OpenCV Haar Cascade face detector initialized ({cascade_file})")
        except Exception as e:
            logger.error(f"Error initializing face detection: {e}")
            self._cascade = None
            self.face_detection = None
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in image
        
        Args:
            image_path: Path to image
            
        Returns:
            List of detected faces with bounding boxes
        """
        if self._cascade is None:
            logger.error("Face detection not initialized")
            return []
        
        image_rgb = _load_image_rgb(image_path)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        detections = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detections:
            faces.append({
                'bbox': (int(x), int(y), int(x + w), int(y + h)),
                # Haar Cascades don't produce a confidence score; fixed at 1.0
                'confidence': 1.0,
                'landmarks': None
            })
        
        return faces
    
    def recognize(self, faces: List[Dict]) -> List[Dict]:
        """
        Recognize faces (placeholder for face recognition logic)
        
        Args:
            faces: List of detected faces
            
        Returns:
            List of recognized faces with identities
        """
        # This is a placeholder - actual implementation would use 
        # face recognition models like FaceNet, VGGFace2, or ArcFace
        recognized = []
        for i, face in enumerate(faces):
            recognized.append({
                'face_id': i,
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'identity': f'Face_{i}',  # Placeholder
                'match_confidence': 0.95  # Placeholder
            })
        return recognized
