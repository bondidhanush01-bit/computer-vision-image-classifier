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
            model = models.resnet50(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_type == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_type == 'vgg16':
            model = models.vgg16(pretrained=True)
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
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
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
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for det in detections:
            bbox = det['bbox']
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            image = draw_bounding_box(image, bbox, label, det['confidence'])
        
        if output_path:
            save_image(image, output_path)
        
        return image


class FaceRecognizer:
    """Face detection and recognition"""
    
    def __init__(self, model_path: Optional[str] = None, detection_confidence: float = 0.7):
        """
        Initialize face recognizer
        
        Args:
            model_path: Path to pretrained face recognition model
            detection_confidence: Confidence threshold for face detection
        """
        self.detection_confidence = detection_confidence
        
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=detection_confidence
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=5,
                min_detection_confidence=detection_confidence
            )
            logger.info("MediaPipe face detection initialized")
        except Exception as e:
            logger.error(f"Error initializing face detection: {e}")
            self.face_detection = None
    
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in image
        
        Args:
            image_path: Path to image
            
        Returns:
            List of detected faces with bounding boxes
        """
        if self.face_detection is None:
            logger.error("Face detection not initialized")
            return []
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                faces.append({
                    'bbox': (xmin, ymin, xmin + width, ymin + height),
                    'confidence': detection.score[0],
                    'landmarks': detection.location_data.relative_keypoints
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
