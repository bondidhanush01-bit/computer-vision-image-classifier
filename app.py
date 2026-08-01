"""
FastAPI app exposing the image classifier, object detector, and face recognizer
defined in src/inference.py.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.inference import FaceRecognizer, ImageClassifier, ObjectDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Computer Vision API",
    description="Image classification, object detection, and face recognition",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models are loaded once at startup and reused across requests, so we don't
# reload weights on every call.
classifier: Optional[ImageClassifier] = None
detector: Optional[ObjectDetector] = None
face_recognizer: Optional[FaceRecognizer] = None


@app.on_event("startup")
def load_models() -> None:
    global classifier, detector, face_recognizer
    logger.info("Loading models...")

    # NOTE: no model_path is passed here, so ImageClassifier uses ImageNet-
    # pretrained backbone weights with a freshly-initialized (untrained) final
    # layer for 10 classes. Predictions will be meaningless until you point
    # model_path at a checkpoint trained with src/train.py. This is enough to
    # get the API running end-to-end though.
    classifier = ImageClassifier(model_type="resnet50", num_classes=10, device="cpu")

    # These two can fail to fully initialize (e.g. missing yolov3 weights or
    # mediapipe not installed) without raising -- inference.py logs an error
    # and leaves .model / .face_detection as None in that case, which the
    # endpoints below check for.
    detector = ObjectDetector()
    face_recognizer = FaceRecognizer()

    logger.info("Models loaded.")


def _save_upload_to_temp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as f:
        shutil.copyfileobj(upload.file, f)
    return Path(tmp.name)


@app.get("/")
def root():
    return {"status": "ok", "message": "Computer Vision API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_loaded": classifier is not None,
        "detector_loaded": detector is not None and detector.model is not None,
        "face_recognizer_loaded": (
            face_recognizer is not None and face_recognizer.face_detection is not None
        ),
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...), top_k: int = 5):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded")

    image_path = _save_upload_to_temp(file)
    try:
        return classifier.predict(str(image_path), top_k=top_k)
    except Exception as e:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        image_path.unlink(missing_ok=True)


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), confidence: float = 0.5):
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Object detector not loaded")

    image_path = _save_upload_to_temp(file)
    try:
        detections = detector.detect(str(image_path), confidence=confidence)
        # bbox arrives as a numpy array; make it JSON-serializable
        for d in detections:
            d["bbox"] = [float(v) for v in d["bbox"]]
        return {"detections": detections, "count": len(detections)}
    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        image_path.unlink(missing_ok=True)


@app.post("/faces")
async def detect_faces(file: UploadFile = File(...)):
    if face_recognizer is None or face_recognizer.face_detection is None:
        raise HTTPException(status_code=503, detail="Face recognizer not loaded")

    image_path = _save_upload_to_temp(file)
    try:
        faces = face_recognizer.detect_faces(str(image_path))
        recognized = face_recognizer.recognize(faces)
        response = [
            {
                "face_id": r["face_id"],
                "bbox": list(r["bbox"]),
                "confidence": float(r["confidence"]),
                "identity": r["identity"],
            }
            for r in recognized
        ]
        return {"faces": response, "count": len(response)}
    except Exception as e:
        logger.exception("Face detection failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        image_path.unlink(missing_ok=True)
