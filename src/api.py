"""
FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
    GET  /health   -> service health check
    POST /predict  -> accepts an image file, returns class + probability
    GET  /metrics  -> Prometheus-format metrics (request count, latency)

Usage:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""
import logging
import time
from pathlib import Path
from io import BytesIO

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import PlainTextResponse
from PIL import Image
from torchvision import transforms
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.model import SimpleCNN

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pt"
CLASS_NAMES = ["cat", "dog"]

LOG_DIR = Path(__file__).resolve().parent.parent / "monitoring" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging setup ---
# Logs request metadata only (filename, prediction, latency) — never the
# raw image bytes, which keeps this compliant with "excluding sensitive data".
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api.log"),
        logging.StreamHandler(),  # also print to console
    ],
)
logger = logging.getLogger("cats-dogs-api")

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "api_requests_total", "Total number of requests received", ["endpoint", "status"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made, by predicted class", ["predicted_class"]
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

app = FastAPI(title="Cats vs Dogs Classifier API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request's method, path, status code, and latency."""
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    endpoint = request.url.path
    REQUEST_COUNT.labels(endpoint=endpoint, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    logger.info(
        f"method={request.method} path={endpoint} "
        f"status={response.status_code} latency_ms={latency * 1000:.2f}"
    )
    return response


@app.get("/health")
def health():
    """Simple liveness check — used by Docker/K8s/monitoring to confirm the service is up."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Prometheus-format metrics endpoint — request counts and latency histograms."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image file, returns predicted class and probability."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(input_tensor).squeeze(1)
        probability = torch.sigmoid(logit).item()

    predicted_class = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
    confidence = probability if probability > 0.5 else 1 - probability

    # Log prediction metadata (filename + result), never the raw image bytes
    logger.info(
        f"prediction filename={file.filename} predicted_class={predicted_class} "
        f"confidence={confidence:.4f}"
    )
    PREDICTION_COUNT.labels(predicted_class=predicted_class).inc()

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            "cat": round(1 - probability, 4),
            "dog": round(probability, 4),
        },
    }