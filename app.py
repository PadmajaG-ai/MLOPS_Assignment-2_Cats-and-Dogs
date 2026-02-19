"""
M2: Model Packaging & Containerization
FastAPI inference service with:
  - GET  /health        → health check
  - POST /predict       → accepts an image, returns class + probabilities
  - GET  /metrics       → request count and average latency (M5 monitoring)
"""

import os
import time
import logging
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────
# LOGGING SETUP (M5: request/response logging)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "models/cats_dogs_cnn.pt")
IMAGE_SIZE  = 224
CLASS_NAMES = ["cat", "dog"]   # must match train.py class_to_idx order

# ─────────────────────────────────────────────
# SIMPLE CNN  (must match train.py exactly)
# ─────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─────────────────────────────────────────────
# LOAD MODEL  (once at startup)
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = SimpleCNN().to(device)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at '{MODEL_PATH}'.\n"
        "Run train.py first or set the MODEL_PATH environment variable."
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
logger.info(f"Model loaded from {MODEL_PATH}")

# ─────────────────────────────────────────────
# INFERENCE TRANSFORM  (same as val/test in train.py)
# ─────────────────────────────────────────────
infer_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# IN-MEMORY METRICS  (M5 monitoring)
# ─────────────────────────────────────────────
metrics = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
}

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Binary image classification API — M2 of MLOps Assignment 2",
    version="1.0.0",
)


# ── Endpoint 1: Health Check ──────────────────
@app.get("/health")
def health():
    """Returns service status and model info."""
    return {
        "status":     "ok",
        "model_path": MODEL_PATH,
        "device":     str(device),
        "classes":    CLASS_NAMES,
    }


# ── Endpoint 2: Predict ───────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a JPEG/PNG image and return:
      - predicted class label (cat / dog)
      - confidence score
      - probabilities for both classes
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG/PNG accepted."
        )

    start_time = time.time()

    try:
        contents = await file.read()
        image    = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Preprocess
    tensor = infer_transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)                          # raw logits
        probs   = torch.softmax(outputs, dim=1)[0]      # probabilities

    pred_idx    = probs.argmax().item()
    pred_label  = CLASS_NAMES[pred_idx]
    confidence  = round(probs[pred_idx].item(), 4)
    prob_dict   = {cls: round(probs[i].item(), 4)
                   for i, cls in enumerate(CLASS_NAMES)}

    latency_ms  = round((time.time() - start_time) * 1000, 2)

    # Update in-memory metrics
    metrics["total_requests"]   += 1
    metrics["total_latency_ms"] += latency_ms

    # Log request (exclude image bytes — no sensitive data logged)
    logger.info(
        f"PREDICT | file={file.filename} | "
        f"prediction={pred_label} | confidence={confidence} | "
        f"latency={latency_ms}ms"
    )

    return JSONResponse({
        "filename":    file.filename,
        "prediction":  pred_label,
        "confidence":  confidence,
        "probabilities": prob_dict,
        "latency_ms":  latency_ms,
    })


# ── Endpoint 3: Metrics  (M5 monitoring) ─────
@app.get("/metrics")
def get_metrics():
    """Returns basic request count and average latency."""
    total   = metrics["total_requests"]
    avg_lat = round(metrics["total_latency_ms"] / total, 2) if total > 0 else 0.0
    return {
        "total_requests":    total,
        "avg_latency_ms":    avg_lat,
        "total_latency_ms":  round(metrics["total_latency_ms"], 2),
    }
