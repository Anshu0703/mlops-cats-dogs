"""
FastAPI inference service for the Cats vs Dogs classifier.

Endpoints:
    GET  /health   -> service health check
    POST /predict  -> accepts an image file, returns class + probability

Usage:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""
from pathlib import Path
from io import BytesIO

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms

from src.model import SimpleCNN

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pt"
CLASS_NAMES = ["cat", "dog"]

app = FastAPI(title="Cats vs Dogs Classifier API")

# Load model once at startup, not per-request — much faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Must match eval_transform from dataset.py exactly, or predictions will be wrong
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/health")
def health():
    """Simple liveness check — used by Docker/K8s/monitoring to confirm the service is up."""
    return {"status": "ok"}


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

    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        logit = model(input_tensor).squeeze(1)
        probability = torch.sigmoid(logit).item()  # prob of "dog"

    predicted_class = CLASS_NAMES[1] if probability > 0.5 else CLASS_NAMES[0]
    confidence = probability if probability > 0.5 else 1 - probability

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            "cat": round(1 - probability, 4),
            "dog": round(probability, 4),
        },
    }