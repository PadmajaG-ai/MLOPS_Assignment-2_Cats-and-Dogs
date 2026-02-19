# Cats vs Dogs MLOps Pipeline

> End-to-end MLOps pipeline for binary image classification — BITS Pilani MLOps Assignment 2 (S1-25_AIMLCZG523)

---

##  Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Milestones](#milestones)
- [Setup & Installation](#setup--installation)
- [M1: Model Training & Experiment Tracking](#m1-model-training--experiment-tracking)
- [M2: Model Packaging & Containerization](#m2-model-packaging--containerization)
- [M3: CI Pipeline](#m3-ci-pipeline)
- [M4: CD Pipeline & Deployment](#m4-cd-pipeline--deployment)
- [M5: Monitoring & Logging](#m5-monitoring--logging)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)

---

## Project Overview

This project implements a complete MLOps pipeline for a **Cats vs Dogs binary image classifier** built for a pet adoption platform. It covers the full lifecycle from data versioning and model training to containerized deployment with CI/CD automation.

| Component | Tool Used |
|---|---|
| Dataset | Kaggle — bhavikjikadara/dog-and-cat-classification-dataset |
| Model | Simple CNN (PyTorch) |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Code Versioning | Git + GitHub |
| Inference API | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Docker Compose |
| Container Registry | Docker Hub |

---

## Project Structure

```
cats-dogs-mlops/
├── train.py                        # M1: Model training script
├── app.py                          # M2: FastAPI inference service
├── Dockerfile                      # M2: Container definition
├── docker-compose.yml              # M4: Deployment manifest
├── requirements.txt                # Python dependencies (version pinned)
├── .gitignore                      # Excludes venv, data, mlruns
├── .dvcignore                      # DVC ignore rules
├── data/
│   ├── processed.dvc               # DVC pointer for processed dataset
│   └── processed/                  # Train/val/test splits (DVC tracked)
│       ├── train/
│       │   ├── cat/
│       │   └── dog/
│       ├── val/
│       │   ├── cat/
│       │   └── dog/
│       └── test/
│           ├── cat/
│           └── dog/
├── models/
│   └── cats_dogs_cnn.pt            # Trained model weights
├── artifacts/
│   ├── loss_curves.png             # Training/validation loss & accuracy
│   └── confusion_matrix.png        # Test set confusion matrix
├── mlruns/                         # MLflow experiment tracking data
├── tests/
│   └── test_pipeline.py            # M3: Unit tests (pytest)
└── .github/
    └── workflows/
        └── ci_cd.yml               # M3 + M4: GitHub Actions pipeline
```

---

## Milestones

| Milestone | Description | Marks |
|---|---|---|
| M1 | Model Development & Experiment Tracking | 10 |
| M2 | Model Packaging & Containerization | 10 |
| M3 | CI Pipeline for Build, Test & Image Creation | 10 |
| M4 | CD Pipeline & Deployment | 10 |
| M5 | Monitoring, Logs & Final Submission | 10 |

---

## Setup & Installation

### Prerequisites
- Python 3.11
- Docker Desktop
- Git
- Kaggle account with API credentials

### 1. Clone the repository
```bash
git clone https://github.com/PadmajaG-ai/MLOPS_Assignment-2_Cats-and-Dogs.git
cd MLOPS_Assignment-2_Cats-and-Dogs
```

### 2. Create and activate virtual environment
```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Mac/Linux
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## M1: Model Training & Experiment Tracking

### Dataset
- **Source**: [Kaggle — Dog and Cat Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Download method**: `kagglehub` (automatic, no manual download needed)
- **Split**: 80% train / 10% validation / 10% test
- **Cap**: 5,000 images per class (for CPU training speed)

### Preprocessing
- Resize to 224×224 RGB
- Force RGB conversion (handles grayscale and RGBA images)
- Data augmentation on train set: RandomHorizontalFlip, RandomRotation(10°), ColorJitter
- ImageNet normalization on all sets

### Model Architecture — SimpleCNN
```
Input (3 × 224 × 224)
  → Conv Block 1: Conv2d(3→32)  + BN + ReLU + MaxPool  → 112×112
  → Conv Block 2: Conv2d(32→64) + BN + ReLU + MaxPool  → 56×56
  → Conv Block 3: Conv2d(64→128)+ BN + ReLU + MaxPool  → 28×28
  → Conv Block 4: Conv2d(128→128)+BN + ReLU + MaxPool  → 14×14
  → Flatten → Dropout(0.5) → FC(25088→512) → ReLU → Dropout(0.3) → FC(512→2)
Output: [cat_logit, dog_logit]
```

### Run Training
```bash
python train.py
```

### View MLflow Experiment Dashboard
```bash
mlflow ui
# Open http://127.0.0.1:5000 in your browser
```

MLflow logs: hyperparameters, per-epoch train/val loss & accuracy, test metrics, confusion matrix, loss curves.

### Data Versioning with DVC
```bash
dvc init
dvc add data/processed
git add data/processed.dvc .dvcignore
git commit -m "Track processed data with DVC"
```

---

## M2: Model Packaging & Containerization

### Run API Locally (without Docker)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Build and Run with Docker
```bash
# Build image
docker build -t cats-dogs-classifier:v1 .

# Run container (mount model from local machine)
docker run -p 8000:8000 -v "${PWD}/models:/app/models" --name cats-dogs-classifier cats-dogs-classifier:v1
```

### Test the API
```bash
# Health check
curl http://localhost:8000/health

# Predict (Windows PowerShell)
curl.exe -X POST "http://localhost:8000/predict" -F "file=@path\to\image.jpg"

# Metrics
curl http://localhost:8000/metrics
```

### Interactive API Docs
Open http://localhost:8000/docs in your browser for the full Swagger UI.

---

## M3: CI Pipeline

The GitHub Actions pipeline (`.github/workflows/ci_cd.yml`) triggers on every push/PR to `main` and runs 3 jobs:

```
push to main
     │
     ▼
┌─────────────┐
│  Job 1      │  Run Unit Tests (pytest)
│  test       │  → 11 tests: 6 preprocessing + 5 inference
└──────┬──────┘
       │ (only if tests pass)
       ▼
┌─────────────────┐
│  Job 2          │  Build Docker image
│  build-and-push │  Push to Docker Hub as:
│                 │  username/cats-dogs-classifier:latest
└──────┬──────────┘
       │
       ▼
┌─────────────┐
│  Job 3      │  Deploy with Docker Compose
│  deploy     │  Run smoke tests (/health + /predict)
└─────────────┘
```

### Required GitHub Secrets
| Secret | Value |
|---|---|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub password or access token |

---

## M4: CD Pipeline & Deployment

### Deploy with Docker Compose
```bash
docker compose up -d
```

### Check running container
```bash
docker ps
docker logs cats-dogs-classifier -f
```

### Stop deployment
```bash
docker compose down
```

The `docker-compose.yml` mounts the local `models/` folder into the container so the model file doesn't need to be baked into the image.

---

## M5: Monitoring & Logging

### Request Logging
Every prediction request is logged automatically:
```
2026-02-19 11:44:20,123  INFO  PREDICT | file=Cat_1.JPEG | prediction=cat | confidence=0.9231 | latency=45.3ms
```

### Metrics Endpoint
```bash
curl http://localhost:8000/metrics
```
```json
{
  "total_requests": 10,
  "avg_latency_ms": 47.3,
  "total_latency_ms": 473.0
}
```

### Docker Logs
```bash
# Live logs
docker logs cats-dogs-classifier -f

# Last 50 lines
docker logs cats-dogs-classifier --tail 50
```

---

## API Reference

### GET `/health`
Returns service status.
```json
{
  "status": "ok",
  "model_path": "models/cats_dogs_cnn.pt",
  "device": "cpu",
  "classes": ["cat", "dog"]
}
```

### POST `/predict`
Accepts a JPEG/PNG image and returns classification result.

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "filename": "cat.jpg",
  "prediction": "cat",
  "confidence": 0.9231,
  "probabilities": {
    "cat": 0.9231,
    "dog": 0.0769
  },
  "latency_ms": 45.3
}
```

### GET `/metrics`
Returns basic monitoring metrics.
```json
{
  "total_requests": 10,
  "avg_latency_ms": 47.3,
  "total_latency_ms": 473.0
}
```

---

## Running Tests

```bash
# Run all tests with verbose output
pytest tests/test_pipeline.py -v

# Run a specific test
pytest tests/test_pipeline.py::test_preprocess_rgb_image -v
```

### Test Coverage
| Test | Description |
|---|---|
| `test_preprocess_rgb_image` | RGB image → tensor shape (1,3,224,224) |
| `test_preprocess_grayscale_image` | Grayscale → converted to 3-channel |
| `test_preprocess_rgba_image` | RGBA → alpha channel dropped |
| `test_preprocess_output_is_float_tensor` | Output dtype is float32 |
| `test_preprocess_normalization_range` | Values in expected range after normalization |
| `test_preprocess_custom_size` | Custom image size respected |
| `test_model_output_shape` | Model outputs (1,2) logits |
| `test_inference_returns_valid_class` | Prediction is 0 or 1 |
| `test_inference_probabilities_sum_to_one` | Softmax sums to 1.0 |
| `test_inference_probabilities_in_range` | Each probability in [0,1] |
| `test_model_batch_inference` | Handles batch of 4 images |
| `test_model_eval_mode_no_grad` | No errors in eval mode |
