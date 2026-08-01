"""
FastAPI application for the Computer Vision Image Classifier project.

Exposes three endpoints, matching the ones documented in the project README:
  - POST /classify  -> image classification
  - POST /detect    -> object detection
  - POST /recognize -> face detection + (placeholder) recognition

Run with:
    cd api
    pip install -r requirements.txt
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Make the project root (parent of api/) importable so `from src...` works
# regardless of the working directory the server is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import FaceRecognizer, ImageClassifier, ObjectDetector  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Computer Vision Image Classifier API",
    description="Image classification, object detection, and face recognition over REST",
    version="1.0.0",
)

# Load config (falls back to sane defaults if config.yaml is missing/unreadable)
try:
    _config = load_config(str(PROJECT_ROOT / "config.yaml"))
except Exception as exc:  # noqa: BLE001
    logger.warning(f"Could not load config.yaml, using defaults: {exc}")
    _config = {}

_device = get_device()

# Models are created lazily on first request so the API starts up fast and
# doesn't pay the cost of loading every model if you only ever use one endpoint.
_classifier: Optional[ImageClassifier] = None
_detector: Optional[ObjectDetector] = None
_face_recognizer: Optional[FaceRecognizer] = None

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/webp"}


def get_classifier() -> ImageClassifier:
    global _classifier
    if _classifier is None:
        model_cfg = _config.get("model", {})
        logger.info("Loading ImageClassifier...")
        _classifier = ImageClassifier(
            model_type=model_cfg.get("type", "resnet50"),
            num_classes=model_cfg.get("num_classes", 10),
            device=_device,
        )
    return _classifier


def get_detector() -> ObjectDetector:
    global _detector
    if _detector is None:
        det_cfg = _config.get("detection", {})
        logger.info("Loading ObjectDetector...")
        _detector = ObjectDetector(
            model_name=det_cfg.get("model", "yolov8n"),
            confidence=det_cfg.get("confidence_threshold", 0.5),
            iou_threshold=det_cfg.get("iou_threshold", 0.4),
        )
    return _detector


def get_face_recognizer() -> FaceRecognizer:
    global _face_recognizer
    if _face_recognizer is None:
        face_cfg = _config.get("face_recognition", {})
        logger.info("Loading FaceRecognizer...")
        _face_recognizer = FaceRecognizer(
            detection_confidence=face_cfg.get("detection_confidence", 0.7),
        )
    return _face_recognizer


def _save_upload_to_tempfile(file: UploadFile) -> Path:
    """Validate and persist an uploaded image to a temp file, returning its path."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
    finally:
        tmp.close()
        file.file.close()

    return Path(tmp.name)


@app.get("/")
def root():
    """Basic health/info endpoint."""
    return {
        "service": "Computer Vision Image Classifier API",
        "status": "ok",
        "device": _device,
        "endpoints": ["/classify", "/detect", "/recognize", "/health"],
    }


@app.get("/health")
def health():
    """Health check endpoint for uptime monitoring / container orchestration."""
    return {"status": "healthy"}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...), top_k: int = 5):
    """
    Classify an uploaded image.

    Args:
        file: Image file (jpg/png/etc.)
        top_k: Number of top predictions to return (default 5)
    """
    tmp_path = _save_upload_to_tempfile(file)
    try:
        classifier = get_classifier()
        result = classifier.predict(str(tmp_path), top_k=top_k)
        return JSONResponse(content=result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), confidence: Optional[float] = None):
    """
    Detect objects in an uploaded image.

    Args:
        file: Image file (jpg/png/etc.)
        confidence: Optional confidence threshold override
    """
    tmp_path = _save_upload_to_tempfile(file)
    try:
        detector = get_detector()
        detections = detector.detect(str(tmp_path), confidence=confidence)

        # Convert numpy types to plain Python so the response is JSON-serializable
        serializable = []
        for det in detections:
            bbox = det["bbox"]
            serializable.append({
                "bbox": [float(v) for v in bbox],
                "confidence": float(det["confidence"]),
                "class": int(det["class"]),
                "class_name": det["class_name"],
            })

        return JSONResponse(content={"detections": serializable, "count": len(serializable)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Object detection failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    """
    Detect and (placeholder) recognize faces in an uploaded image.

    Args:
        file: Image file (jpg/png/etc.)
    """
    tmp_path = _save_upload_to_tempfile(file)
    try:
        recognizer = get_face_recognizer()
        faces = recognizer.detect_faces(str(tmp_path))
        identities = recognizer.recognize(faces)

        serializable = []
        for identity in identities:
            serializable.append({
                "face_id": identity["face_id"],
                "bbox": [int(v) for v in identity["bbox"]],
                "confidence": float(identity["confidence"]),
                "identity": identity["identity"],
                "match_confidence": float(identity["match_confidence"]),
            })

        return JSONResponse(content={"faces": serializable, "count": len(serializable)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Face recognition failed")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)
