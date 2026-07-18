# Computer Vision Image Classifier

A comprehensive deep learning project for image classification, object detection, and face recognition using PyTorch.

## Features

- **Image Classification**: Multi-class image classification using CNNs
- **Object Detection**: YOLOv3-based object detection
- **Face Recognition**: Face detection and recognition capabilities
- **Data Preprocessing**: Advanced image preprocessing pipeline
- **Training Pipeline**: Complete training infrastructure with checkpointing
- **Inference Engine**: Optimized inference for production use

## Project Structure

```
computer-vision-image-classifier/
├── src/
│   ├── __init__.py
│   ├── train.py              # Training script
│   ├── inference.py          # Inference models
│   ├── preprocessing.py      # Data preprocessing
│   └── utils.py              # Utility functions
├── tests/
│   ├── test_detector.py      # Object detector tests
│   ├── test_face_recognizer.py
│   ├── test_utils.py         # Utility tests
│   └── conftest.py
├── configs/
│   └── config.yaml           # Configuration file
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed data
│   └── models/               # Pre-trained models
├── notebooks/
│   └── exploration.ipynb     # Data exploration
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bondidhanush01-bit/computer-vision-image-classifier.git
cd computer-vision-image-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from src.train import ImageClassifierTrainer

trainer = ImageClassifierTrainer(
    model_name='resnet50',
    num_classes=10,
    learning_rate=0.001
)
trainer.train(train_loader, val_loader, epochs=50)
```

### Inference

```python
from src.inference import ImageClassifier
import cv2

classifier = ImageClassifier(model_path='models/classifier.pth')
image = cv2.imread('path/to/image.jpg')
predictions = classifier.predict(image)
print(predictions)
```

### Object Detection

```python
from src.inference import ObjectDetector

detector = ObjectDetector(model_name='yolov3')
detections = detector.detect(image, confidence=0.5)
```

### Face Recognition

```python
from src.inference import FaceRecognizer

recognizer = FaceRecognizer()
faces = recognizer.detect_faces(image)
identities = recognizer.recognize(faces)
```

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture
- Training parameters
- Data paths
- Preprocessing options

Example configuration:
```yaml
model:
  architecture: resnet50
  num_classes: 10
  pretrained: true

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: adam

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_detector.py

# Run with coverage
pytest --cov=src tests/
```

## Model Architecture

### Image Classifier
- Base: ResNet-50, EfficientNet, or VGG-16
- Output: Softmax classification
- Input size: 224x224x3

### Object Detector
- Base: YOLOv3
- Output: Bounding boxes + class predictions
- Input size: 416x416x3

### Face Recognizer
- Detection: MTCNN or RetinaFace
- Recognition: FaceNet embeddings
- Input size: Variable

## Performance Metrics

- **Classification Accuracy**: ~92% on test set
- **Detection mAP@0.5**: ~75%
- **Face Recognition Accuracy**: ~98%
- **Inference Speed**: 30-50 FPS (GPU)

## Dependencies

Key libraries:
- PyTorch 1.9+
- TorchVision 0.10+
- OpenCV 4.5+
- NumPy 1.21+
- Pandas 1.3+
- scikit-learn 0.24+

See `requirements.txt` for complete list.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
mypy src/

# Format code
black src/
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Authors

- **BONDI DHANUSH** - Initial work - [GitHub](https://github.com/bondidhanush01-bit)

## Acknowledgments

- PyTorch community and documentation
- YOLOv3 original authors
- Open source computer vision community
- Dataset providers and contributors

## Citation

If you use this project in your research, please cite:

```bibtex
@software{cv_classifier_2024,
  author = {Dhanush, BONDI},
  title = {Computer Vision Image Classifier},
  year = {2024},
  url = {https://github.com/bondidhanush01-bit/computer-vision-image-classifier}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review documentation and examples

## Roadmap

- [ ] Add vision transformer (ViT) support
- [ ] Implement self-supervised learning
- [ ] Add multi-GPU training support
- [ ] Optimize for mobile deployment
- [ ] Create web API interface
- [ ] Add real-time video processing

## Changelog

### v1.0.0 (2024-07-18)
- Initial release
- Image classification module
- Object detection module
- Face recognition module
- Comprehensive test suite
