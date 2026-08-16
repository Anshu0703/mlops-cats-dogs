# Cats vs Dogs — End-to-End MLOps Pipeline

**Course:** MLOps (S1-25_AIMLCZG523) — Assignment 2
**GitHub Repository:** https://github.com/Anshu0703/mlops-cats-dogs

An end-to-end MLOps pipeline for a binary image classifier (Cats vs Dogs) built for a pet
adoption platform use case — covering model development, experiment tracking, packaging,
containerization, CI, CD, and monitoring using open-source tools.

## Overview

| Module | What it covers | Key tools |
|---|---|---|
| M1 | Data/code versioning, CNN model, experiment tracking | Git, DVC, PyTorch, MLflow |
| M2 | Inference API, environment spec, containerization | FastAPI, Docker |
| M3 | Automated testing, CI pipeline, image publishing | pytest, GitHub Actions, GHCR |
| M4 | Deployment target, CD pipeline, smoke tests | Docker Compose, GitHub Actions |
| M5 | Request logging, metrics, performance tracking | Python logging, Prometheus |

## Dataset

[Cats and Dogs Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
— pre-processed to 224x224 RGB images, split 80/10/10 into train/val/test, with live data
augmentation (random flips, rotation, color jitter) applied during training.

## Project Structure

```
cats-dogs-mlops/
├── data/                    # raw + processed data (DVC-tracked; not included in zip)
├── src/
│   ├── download_data.py     # pulls dataset from Kaggle
│   ├── preprocess.py        # resize/split/validate images
│   ├── dataset.py           # PyTorch Dataset + augmentation
│   ├── model.py             # SimpleCNN architecture
│   ├── train.py             # training loop + MLflow logging
│   └── api.py                # FastAPI inference service (+ logging/metrics)
├── tests/
│   ├── test_preprocess.py   # unit tests for data validation
│   └── test_api.py           # unit tests for the API
├── monitoring/
│   └── track_performance.py # post-deployment accuracy tracking script
├── models/                  # trained model weights + MLflow artifacts
├── scripts/
│   └── smoke_test.sh        # post-deploy health/prediction check
├── .github/workflows/ci.yml # CI/CD pipeline definition
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # inference service dependencies (pinned)
└── requirements-dev.txt      # test/dev-only dependencies
```

## Model

A simple 3-block CNN (`src/model.py`) trained on 224x224 RGB images, binary classification
via a single sigmoid output. Achieved **~83% test accuracy** after 5 epochs.

## Running Locally

### 1. Set up environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Get the data (requires a Kaggle API token at ~/.kaggle/access_token)
```bash
python src/download_data.py
python src/preprocess.py
```

### 3. Train the model (logs to MLflow)
```bash
python src/train.py --epochs 5
mlflow ui   # view experiment results at http://localhost:5000
```

### 4. Run the API locally
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 5. Test it
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@path/to/image.jpg"
curl http://localhost:8000/metrics
```

## Running with Docker

```bash
docker build -t cats-dogs-api .
docker run -d -p 8000:8000 --name cats-dogs-container cats-dogs-api
```

Or via Docker Compose:
```bash
docker compose up --build
```

## CI/CD Pipeline

On every push to `main`, GitHub Actions automatically:
1. Installs dependencies and runs the pytest suite
2. Builds the Docker image and pushes it to GitHub Container Registry (GHCR)
3. Pulls the published image, deploys it via Docker Compose, and runs smoke tests
   (`/health` + one `/predict` call) — failing the pipeline if either check fails

Workflow file: `.github/workflows/ci.yml`
Published image: `ghcr.io/anshu0703/mlops-cats-dogs`

## Monitoring

- **Request/response logging** — every API request is logged (method, path, status, latency)
  to `monitoring/logs/api.log`; predictions log the filename, class, and confidence — never
  the raw image bytes.
- **Metrics** — exposed in Prometheus format at `/metrics` (request counts, per-class
  prediction counts, latency histograms).
- **Post-deployment performance tracking** — `monitoring/track_performance.py` sends a batch
  of test-set images (with known true labels) to the live API and reports accuracy, overall
  and per-class. Sample report included at `monitoring/reports/`.

## Experiment Tracking

MLflow logs parameters, per-epoch metrics, loss curves, and confusion matrices for every
training run. Run `mlflow ui` from the project root to view them (requires `mlruns/` or
`mlflow.db`, generated locally when you run `src/train.py`).