"""
Unit tests for the FastAPI inference service.
"""
import sys
from pathlib import Path
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """Health check should return 200 and status ok."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_valid_response():
    """Predict should accept an image and return a class label with probabilities."""
    # Build a fake in-memory image, since we're testing the API contract,
    # not model accuracy
    image = Image.new("RGB", (224, 224), color="blue")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buffer, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["cat", "dog"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "cat" in data["probabilities"]
    assert "dog" in data["probabilities"]


def test_predict_endpoint_rejects_non_image_file():
    """Uploading a non-image file should return a 400 error."""
    buffer = BytesIO(b"this is just text, not an image")

    response = client.post(
        "/predict",
        files={"file": ("test.txt", buffer, "text/plain")},
    )

    assert response.status_code == 400