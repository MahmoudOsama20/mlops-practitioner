# MLOps Practitioner

An end-to-end MLOps project for predicting **NYC Green Taxi trip duration**.

This repository is part of a larger **five-mini-project MLOps journey**.  
**Mini Project 1 — From Notebook to Production-Ready Service** is now complete.

The project starts with a reproducible machine-learning baseline and progressively turns it into a production-oriented ML system. The completed implementation covers data preparation, feature engineering, model training, model persistence, ONNX serialization, a FastAPI prediction service, structured logging, automated testing, code quality, containerization, Docker Compose, and Docker Hub publishing.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Mini Project 1 — From Notebook to Production](#mini-project-1--from-notebook-to-production)
3. [Project Progress](#project-progress)
4. [Dataset](#dataset)
5. [Baseline Model](#baseline-model)
6. [Baseline Results](#baseline-results)
7. [Project Structure](#project-structure)
8. [Requirements](#requirements)
9. [Environment Setup](#environment-setup)
10. [Configuration](#configuration)
11. [Architecture](#architecture)
12. [Data and Feature Pipeline](#data-and-feature-pipeline)
13. [Training Pipeline](#training-pipeline)
14. [Model Persistence](#model-persistence)
15. [Serialization and ONNX](#serialization-and-onnx)
16. [Production API](#production-api)
17. [Logging and Request Correlation](#logging-and-request-correlation)
18. [Testing](#testing)
19. [Code Quality and Pre-commit](#code-quality-and-pre-commit)
20. [Docker](#docker)
21. [Docker Compose](#docker-compose)
22. [Docker Hub](#docker-hub)
23. [Running the API](#running-the-api)
24. [Example Prediction](#example-prediction)
25. [Development Workflow](#development-workflow)
26. [Reproducibility](#reproducibility)
27. [Reports](#reports)
28. [Current Status](#current-status)
29. [Next Steps](#next-steps)
30. [Baseline Reference](#baseline-reference)

---

# Project Overview

The objective is to predict the duration of a NYC Green Taxi trip in minutes.

The current model uses two features:

- `PU_DO`: pickup and drop-off location pair
- `trip_distance`: trip distance

The target variable is:

```text
duration
```

Trip duration is calculated from:

```text
lpep_dropoff_datetime - lpep_pickup_datetime
```

and converted from seconds to minutes.

The project has progressed from a notebook-based baseline to a tested, serialized, containerized, and publicly published ML prediction service.

---

# Mini Project 1 — From Notebook to Production

**Status: Complete**

The first mini project transforms the initial ML notebook into a production-oriented machine-learning service.

### Completed capabilities

- Python package structure using `src/`
- Reproducible model training
- Configuration management with `pydantic-settings`
- Feature engineering and data cleaning
- Pickle model persistence
- ONNX model serialization
- Pickle/ONNX prediction parity testing
- `DurationPredictor` prediction interface
- FastAPI application
- Pydantic request/response validation
- `/health` endpoint
- `/metadata` endpoint
- `/predict` endpoint
- `/predict/batch` endpoint
- Startup model loading
- Structured JSON logging
- Request correlation IDs
- Prediction timing
- Automated pytest test suite
- 70% minimum coverage gate
- Ruff
- Black
- Pre-commit
- Multi-stage Docker build
- Docker Compose
- Non-root container execution
- Docker health check
- Read-only model volume in Compose
- Docker Hub publication
- Versioned Docker image
- `latest` Docker image tag

The final Docker image can be pulled and run without recreating the local Python environment.

---

# Project Progress

The project evolved from a local ML baseline into a containerized model-serving application.

```text
NYC Green Taxi Dataset
        |
        v
Data Preparation
        |
        v
Feature Engineering
        |
        v
LinearRegression Baseline
        |
        +--------------------+
        |                    |
        v                    v
     Pickle                ONNX
        |                    |
        +---------+----------+
                  |
                  v
          DurationPredictor
                  |
                  v
             FastAPI API
                  |
                  v
             Docker Image
                  |
                  v
              Docker Hub
```

The implemented modules are:

| Module | Focus | Status |
|---|---|---|
| Module 1 | Baseline model and project packaging | Complete |
| Module 4 | Serialization and ONNX | Complete |
| Module 5 | Production FastAPI API | Complete |
| Module 7 | Containerization and publishing | Complete |

**Modules 2, 3, and 6 are not part of the current implementation.**

---

# Dataset

The project uses the **NYC Green Taxi Trip Records** dataset.

The local dataset used during development is:

```text
datasets/green_tripdata_2026-01.csv
```

The dataset contains fields including:

- pickup timestamp
- drop-off timestamp
- pickup location ID
- drop-off location ID
- trip distance
- passenger count
- payment information
- fare information
- trip type
- other taxi-trip attributes

The baseline intentionally uses only the features required for the initial model.

---

# Baseline Model

The baseline model consists of:

```text
DictVectorizer + LinearRegression
```

## Input Features

```text
PU_DO
trip_distance
```

## Target

```text
duration
```

## Data Split

```text
Training:   80%
Validation: 20%
Random seed: 42
```

The categorical `PU_DO` feature is transformed using `DictVectorizer`.

The numerical `trip_distance` feature is included in the same vectorized input representation.

---

# Baseline Results

The baseline validation performance is:

| Metric | Result |
|---|---:|
| MAE | 4.850 minutes |
| RMSE | 8.270 minutes |

The MAE of approximately **4.850 minutes** is the main baseline reference for future model improvements.

The baseline metrics were established using the reproducible training and validation pipeline.

---

# Project Structure

The current repository structure is:

```text
mlops-practitioner/
│
├── .gitignore
├── .pre-commit-config.yaml
├── .dockerignore
├── pyproject.toml
├── README.md
│
├── datasets/
│   └── green_tripdata_2026-01.csv
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── models/
│   ├── baseline.pkl
│   └── model.onnx
│
├── notebooks/
│   └── 00-baseline.ipynb
│
├── reports/
│   ├── module-1.md
│   ├── module-4.md
│   ├── module-5.md
│   └── module-7.md
│
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── schemas.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
│
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_features.py
    ├── test_predict.py
    └── test_serialization.py
```

## Directory Responsibilities

### `datasets/`

Contains the local training dataset.

### `models/`

Contains serialized model artifacts:

```text
baseline.pkl
model.onnx
```

### `docker/`

Contains the Docker build and Docker Compose configuration.

### `notebooks/`

Contains the initial baseline experiment notebook.

### `reports/`

Contains detailed documentation for the implemented modules.

### `src/prodml/`

Contains the production-oriented Python package.

### `tests/`

Contains automated unit and API tests.

---

# Requirements

The project requires:

```text
Python >= 3.10
```

The current development environment uses:

```text
Python 3.12.7
```

## Runtime Dependencies

The project declares:

- scikit-learn
- pandas
- pyarrow
- FastAPI
- Uvicorn
- Pydantic
- pydantic-settings
- python-json-logger
- skl2onnx
- ONNX
- ONNX Runtime

## Development Dependencies

The development environment includes:

- pytest
- pytest-cov
- Ruff
- Black
- mypy
- httpx
- pre-commit

All dependencies are declared in:

```text
pyproject.toml
```

---

# Environment Setup

## 1. Clone the repository

```powershell
git clone <repository-url>
cd mlops-practitioner
```

Replace `<repository-url>` with the actual GitHub repository URL.

## 2. Create a virtual environment

```powershell
python -m venv .venv
```

## 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The terminal should then show:

```text
(.venv)
```

## 4. Install the project

```powershell
pip install -e ".[dev]"
```

## 5. Verify the installation

```powershell
python -c "import prodml; print(prodml.__file__)"
```

The result should point to the installed `prodml` package under:

```text
src\prodml\
```

---

# Configuration

Configuration is centralized in:

```text
src/prodml/config.py
```

The project uses `pydantic-settings` to load configuration values from environment variables.

The environment variable prefix is:

```text
PRODML_
```

## Environment Variables

A local `.env` file can contain:

```env
PRODML_DATA_PATH=datasets/green_tripdata_2026-01.csv
PRODML_MODEL_PATH=models/baseline.pkl
PRODML_REPORT_PATH=reports/module-1.md
PRODML_TEST_SIZE=0.2
PRODML_RANDOM_STATE=42
```

The `.env` file is local configuration and should not be committed.

## Configuration Reference

| Variable | Purpose | Value |
|---|---|---|
| `PRODML_DATA_PATH` | Training dataset | `datasets/green_tripdata_2026-01.csv` |
| `PRODML_MODEL_PATH` | Pickle model artifact | `models/baseline.pkl` |
| `PRODML_REPORT_PATH` | Baseline report | `reports/module-1.md` |
| `PRODML_TEST_SIZE` | Validation fraction | `0.2` |
| `PRODML_RANDOM_STATE` | Random seed | `42` |

Docker uses additional runtime configuration:

```text
PRODML_MODEL_PATH=/app/models/baseline.pkl
PRODML_ONNX_PATH=/app/models/model.onnx
```

---

# Architecture

The current system separates data loading, feature engineering, training, prediction, API serving, and containerization.

```text
                    ┌─────────────────────┐
                    │       Dataset       │
                    │ green_tripdata_...  │
                    └──────────┬──────────┘
                               │
                               v
                    ┌─────────────────────┐
                    │       data.py       │
                    │    load + split     │
                    └──────────┬──────────┘
                               │
                               v
                    ┌─────────────────────┐
                    │     features.py     │
                    │ duration + PU_DO +  │
                    │ cleaning + vectorize│
                    └──────────┬──────────┘
                               │
                               v
                    ┌─────────────────────┐
                    │       train.py      │
                    │ LinearRegression    │
                    │ evaluation + saving │
                    └──────────┬──────────┘
                               │
                         +-----+-----+
                         |           |
                         v           v
                    baseline.pkl  model.onnx
                         |           |
                         +-----+-----+
                               |
                               v
                    ┌─────────────────────┐
                    │      predict.py     │
                    │  DurationPredictor  │
                    └──────────┬──────────┘
                               │
                               v
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    │ /health             │
                    │ /metadata           │
                    │ /predict            │
                    │ /predict/batch      │
                    └──────────┬──────────┘
                               │
                               v
                    ┌─────────────────────┐
                    │    Docker Image     │
                    │      appuser        │
                    └──────────┬──────────┘
                               │
                               v
                           Docker Hub
```

---

# Data and Feature Pipeline

The training pipeline follows:

```text
Raw Dataset
     |
     v
Load Data
     |
     v
Convert Datetimes
     |
     v
Create Duration
     |
     v
Clean Invalid/Extreme Trips
     |
     v
Create PU_DO
     |
     v
Select Features
     |
     v
Train / Validation Split
     |
     v
DictVectorizer
     |
     v
Linear Regression
     |
     v
Evaluation
     |
     +------> MAE / RMSE
     |
     v
Save Model
```

## Duration

Trip duration is calculated from the pickup and drop-off timestamps:

```python
data["duration"] = (
    data["lpep_dropoff_datetime"]
    - data["lpep_pickup_datetime"]
).dt.total_seconds() / 60
```

## Data Cleaning

The feature pipeline removes invalid or extreme trips.

Duration is restricted to:

```text
1 <= duration <= 120
```

Trip distance is restricted to:

```text
0 < trip_distance <= 100
```

## PU_DO

`PU_DO` combines the pickup and drop-off location IDs.

Example:

```text
PULocationID = 74
DOLocationID = 42

PU_DO = "74_42"
```

## Final Baseline Features

```text
PU_DO
trip_distance
```

---

# Training Pipeline

Training is implemented in:

```text
src/prodml/train.py
```

The training pipeline:

1. Loads the dataset.
2. Performs feature engineering and cleaning.
3. Splits the data into training and validation sets.
4. Fits a `DictVectorizer`.
5. Trains a `LinearRegression` model.
6. Calculates MAE and RMSE.
7. Saves the model and vectorizer.
8. Writes the baseline report.

## Train the Model

Using the Python module:

```powershell
python -m prodml.train
```

The project also exposes:

```powershell
prodml-train
```

Both execute the training entry point defined in:

```text
src/prodml/train.py
```

---

# Model Persistence

The Pickle model artifact is:

```text
models/baseline.pkl
```

It contains:

```text
LinearRegression
DictVectorizer
```

Both the fitted model and fitted vectorizer are persisted because prediction must use the same feature transformation learned during training.

The prediction interface loads this artifact through:

```python
from prodml.predict import DurationPredictor

predictor = DurationPredictor().load()
```

---

# Serialization and ONNX

Serialization and ONNX conversion are documented in:

```text
reports/module-4.md
```

The project provides two model representations:

```text
models/baseline.pkl
models/model.onnx
```

## Pickle

The Pickle artifact contains:

- fitted `LinearRegression`
- fitted `DictVectorizer`

## ONNX

The ONNX representation is used to evaluate model serialization and runtime compatibility.

The exported model supports a dynamic batch dimension and the vectorized feature representation.

The ONNX artifact was validated against the Pickle model using a prediction parity test.

The parity test currently passes:

```text
tests/test_serialization.py::test_onnx_prediction_parity PASSED
```

---

# Production API

The FastAPI application is implemented in:

```text
src/prodml/api/main.py
```

The API provides four main endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/metadata` | Model/service metadata |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

## API Flow

```text
HTTP Request
     |
     v
Correlation Middleware
     |
     +----> X-Request-ID
     |
     v
FastAPI Endpoint
     |
     v
Pydantic Validation
     |
     v
DurationPredictor
     |
     v
Prediction
     |
     v
Structured JSON Response
```

## Startup Model Loading

The model is loaded during application startup rather than being loaded for every request.

Container startup logging confirmed:

```text
model_loaded
```

with:

```text
/app/models/baseline.pkl
```

This keeps model loading outside the individual prediction request path.

---

## Health Endpoint

Request:

```http
GET /health
```

Example:

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Metadata Endpoint

Request:

```http
GET /metadata
```

This endpoint exposes model/service metadata defined by the API configuration.

---

## Single Prediction

Request:

```http
POST /predict
```

Example payload:

```json
{
  "PU_DO": "74_42",
  "trip_distance": 2.5
}
```

Example successful response:

```json
{
  "prediction": 12.727868515169943,
  "model_version": "0.1.0",
  "correlation_id": "...",
  "latency_ms": 4.834
}
```

The response also includes the request/correlation ID through the:

```text
X-Request-ID
```

header.

---

## Batch Prediction

Request:

```http
POST /predict/batch
```

Example payload:

```json
[
  {
    "PU_DO": "74_42",
    "trip_distance": 2.5
  },
  {
    "PU_DO": "75_41",
    "trip_distance": 4.1
  }
]
```

The API validates the request and returns predictions for the supplied records.

---

## API Validation

The API uses Pydantic schemas to validate incoming data.

The test suite covers:

- valid predictions
- batch predictions
- negative trip distance
- trip distance outside the training range
- unexpected prediction errors
- response behavior

Invalid input is rejected through API validation rather than being silently passed to the model.

---

# Logging and Request Correlation

Structured logging is implemented in:

```text
src/prodml/logging_conf.py
```

The API uses correlation IDs to trace requests through the service.

A request receives an:

```text
X-Request-ID
```

header.

The same identifier is used in structured log messages such as:

```text
request_started
prediction_served
request_completed
```

Prediction timing is also logged.

Example container log events include:

```text
model_loaded
request_started
function_timing
prediction_served
request_completed
```

This provides an end-to-end request trace from HTTP request to prediction response.

---

# Testing

The project uses `pytest` for automated testing.

Current test files:

```text
tests/conftest.py
tests/test_api.py
tests/test_features.py
tests/test_predict.py
tests/test_serialization.py
```

The test suite covers:

- API health endpoint
- API metadata endpoint
- single prediction
- batch prediction
- invalid prediction input
- out-of-range prediction input
- API error handling
- duration cleaning
- `PU_DO` feature creation
- unseen categorical values
- missing categories
- model loading
- deterministic prediction
- batch prediction
- Pickle/ONNX prediction parity

## Run Tests

```powershell
pytest -v
```

## Run Tests with Coverage

```powershell
pytest -v --cov=src/prodml --cov-report=term-missing
```

## Current Test Result

The complete test suite currently reports:

```text
17 passed
```

Coverage:

```text
75.99%
```

The project enforces a minimum coverage threshold of:

```text
70%
```

The current result passes the coverage gate:

```text
Required test coverage of 70% reached.
Total coverage: 75.99%
```

---

# Code Quality and Pre-commit

The project uses:

- Ruff
- Black
- pre-commit

## Ruff

Run:

```powershell
ruff check src tests
```

## Black

Check formatting:

```powershell
black --check src tests
```

Format:

```powershell
black src tests
```

## Pre-commit

Install the hooks:

```powershell
pre-commit install
```

Run all hooks:

```powershell
pre-commit run --all-files
```

Current hooks include:

```text
end-of-file-fixer
ruff check
black
```

The final verification currently passes all hooks:

```text
fix end of files ........ Passed
ruff check .............. Passed
black ................... Passed
```

---

# Docker

The API is containerized using a multi-stage Docker build.

Docker files:

```text
docker/Dockerfile
.dockerignore
```

The final runtime image uses:

```text
python:3.11-slim
```

## Multi-stage Build

### Builder Stage

The builder:

1. Uses `python:3.11-slim`.
2. Copies `pyproject.toml`.
3. Copies `src/`.
4. Installs the project and dependencies into `/install`.

### Runtime Stage

The runtime:

1. Uses a clean `python:3.11-slim`.
2. Copies the installed application.
3. Copies the model artifacts.
4. Creates `appuser`.
5. Runs as the non-root user.
6. Exposes port `8000`.
7. Defines a health check.
8. Starts Uvicorn.

## Build the Image

```powershell
docker build -f docker/Dockerfile -t prodml-api:0.1.0 .
```

Tag the latest version:

```powershell
docker tag prodml-api:0.1.0 prodml-api:latest
```

---

# Docker Image Size

The project compared a single-stage build with the final multi-stage build.

| Build | Disk usage | Content size |
|---|---:|---:|
| Single-stage | 1.13 GB | 256 MB |
| Multi-stage | 1.11 GB | 251 MB |

Measured difference:

```text
Approximately 20 MB less disk usage
Approximately 5 MB less content size
```

The multi-stage image was therefore smaller in the measured Docker environment.

---

# Docker Ignore

The root `.dockerignore` prevents unnecessary files from being sent to the Docker build context.

Excluded content includes:

```text
.git
.venv
__pycache__
.pytest_cache
.ruff_cache
coverage files
notebooks
tests
datasets
.env
IDE configuration
```

The `models/` directory is intentionally included because the runtime image needs the trained model artifacts.

---

# Docker Runtime

The runtime image contains:

```text
/app/models/baseline.pkl
/app/models/model.onnx
```

The model paths are configured as:

```text
PRODML_MODEL_PATH=/app/models/baseline.pkl
PRODML_ONNX_PATH=/app/models/model.onnx
```

The container runs as:

```text
appuser
```

Port:

```text
8000
```

The Docker health check targets:

```text
http://localhost:8000/health
```

---

# Docker Compose

Docker Compose is configured in:

```text
docker/docker-compose.yml
```

The Compose configuration provides:

- API service
- port mapping `8000:8000`
- model path environment variables
- model version configuration
- training date configuration
- read-only model volume
- restart policy

The restart policy is:

```yaml
restart: unless-stopped
```

The model directory is mounted read-only:

```text
../models:/app/models:ro
```

## Verify the Model Files

```powershell
docker compose -f docker/docker-compose.yml exec api ls -lh /app/models
```

Expected files:

```text
baseline.pkl
model.onnx
```

## Verify Non-root Execution

```powershell
docker compose -f docker/docker-compose.yml exec api whoami
```

Expected:

```text
appuser
```

The container was verified to run under the non-root `appuser` account.

---

# Docker Hub

The final image was published to:

```text
mahmoudosama20/prodml-api
```

Available tags:

```text
mahmoudosama20/prodml-api:0.1.0
mahmoudosama20/prodml-api:latest
```

Docker Hub repository:

https://hub.docker.com/r/mahmoudosama20/prodml-api

## Pull the Published Image

```powershell
docker pull mahmoudosama20/prodml-api:0.1.0
```

The published image was successfully pulled and verified.

The versioned image digest was:

```text
sha256:29112bd77ba93ee0687361088b316a21b4dc303af1be92af5296b640a3b9e920
```

The `latest` tag was also successfully pulled.

---

# Running the API

The final Docker image can be run without cloning the repository.

```powershell
docker run --rm -p 8000:8000 mahmoudosama20/prodml-api:0.1.0
```

The API will be available at:

```text
http://127.0.0.1:8000
```

This is the main reproducibility goal of the containerization work: a user can pull the published image and run the prediction service without recreating the local development environment.

---

# Example Prediction

## Health Check

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Expected:

```text
HTTP/1.1 200 OK
```

```json
{
  "status": "ok"
}
```

## Prediction Request

PowerShell:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"PU_DO":"74_42","trip_distance":2.5}'
```

Example response:

```json
{
  "prediction": 12.727868515169943,
  "model_version": "0.1.0",
  "correlation_id": "f3643606-0f14-4380-8578-bb4cf2054acd",
  "latency_ms": 4.834
}
```

The exact correlation ID and latency will differ between requests.

The observed prediction for:

```text
PU_DO = 74_42
trip_distance = 2.5
```

was:

```text
12.727868515169943 minutes
```

---

# API Verification

The final container was tested end-to-end.

```text
Docker image
     |
     v
Container startup
     |
     v
Model loaded
     |
     v
GET /health
     |
     v
POST /predict
     |
     v
Structured logs
```

Observed results:

| Check | Result |
|---|---|
| Docker build | PASS |
| Container startup | PASS |
| Model loading | PASS |
| Non-root execution | PASS |
| `/health` | 200 OK |
| `/predict` | 200 OK |
| Structured logging | PASS |
| Docker Hub push | PASS |
| Docker Hub pull | PASS |

---

# Development Workflow

The recommended local workflow is:

```text
Activate environment
       |
       v
Modify code
       |
       v
Run pre-commit
       |
       v
Run tests
       |
       v
Train/evaluate when required
       |
       v
Build/test Docker image
       |
       v
Review Git status
       |
       v
Commit
       |
       v
Push
```

## Complete Local Verification

### Activate

```powershell
.\.venv\Scripts\Activate.ps1
```

### Run pre-commit

```powershell
pre-commit run --all-files
```

### Run tests

```powershell
pytest -v --cov=src/prodml --cov-report=term-missing
```

### Train

```powershell
python -m prodml.train
```

### Check Git

```powershell
git status
```

A clean final state should report:

```text
nothing to commit, working tree clean
```

---

# Reproducibility

The baseline uses:

```text
test_size = 0.2
random_state = 42
```

The project also uses a fixed feature definition:

```text
PU_DO
trip_distance
```

and fixed data-cleaning rules.

The baseline reference metrics are:

```text
MAE  = 4.850 minutes
RMSE = 8.270 minutes
```

The Docker image adds another level of reproducibility by packaging:

- application code
- Python runtime
- runtime dependencies
- model artifacts
- API configuration

The published image can be pulled using:

```powershell
docker pull mahmoudosama20/prodml-api:0.1.0
```

and run using:

```powershell
docker run --rm -p 8000:8000 mahmoudosama20/prodml-api:0.1.0
```

---

# Reports

Detailed module documentation is available under:

```text
reports/
```

Current reports:

### Module 1

```text
reports/module-1.md
```

Covers the baseline model, validation metrics, packaging, and initial production-oriented project structure.

### Module 4

```text
reports/module-4.md
```

Covers model serialization, Pickle, ONNX, prediction parity, and serialization comparison.

### Module 5

```text
reports/module-5.md
```

Covers the FastAPI production API, validation, endpoints, startup model loading, structured logging, and request correlation.

### Module 7

```text
reports/module-7.md
```

Covers Docker containerization, multi-stage builds, image-size comparison, Docker Compose, non-root execution, and Docker Hub publishing.

---

# Current Status

## Mini Project 1 — From Notebook to Production

**Status: Complete**

### Module 1 — Baseline Model

- [x] Dataset loaded
- [x] Datetime conversion
- [x] Trip duration calculation
- [x] Data cleaning
- [x] `PU_DO` feature engineering
- [x] `trip_distance` feature
- [x] Train/validation split
- [x] `DictVectorizer`
- [x] Linear Regression baseline
- [x] MAE evaluation
- [x] RMSE evaluation
- [x] Model persistence
- [x] Python package structure
- [x] Configuration management

### Module 4 — Serialization + ONNX

- [x] Pickle model artifact
- [x] ONNX model artifact
- [x] ONNX conversion
- [x] ONNX Runtime integration
- [x] Pickle/ONNX prediction parity test
- [x] Serialization documentation

### Module 5 — Production API

- [x] FastAPI application
- [x] Pydantic request/response schemas
- [x] `/health`
- [x] `/metadata`
- [x] `/predict`
- [x] `/predict/batch`
- [x] Startup model loading
- [x] Structured JSON logging
- [x] Request correlation IDs
- [x] Prediction timing
- [x] API integration tests
- [x] Error handling tests

### Module 7 — Containerization and Publishing

- [x] Multi-stage Dockerfile
- [x] Runtime image
- [x] `.dockerignore`
- [x] Docker Compose
- [x] Model artifacts in container
- [x] Read-only model volume
- [x] Non-root `appuser`
- [x] Docker health check
- [x] Single-stage vs multi-stage comparison
- [x] Docker image build
- [x] Semantic version tag
- [x] `latest` tag
- [x] Docker Hub publication
- [x] Docker Hub pull verification
- [x] Container health verification
- [x] Container prediction verification

### Quality Gate

- [x] Pre-commit hooks pass
- [x] Ruff passes
- [x] Black passes
- [x] 17 tests pass
- [x] Coverage >= 70%
- [x] Current coverage: 75.99%
- [x] Working tree verified clean

---

# Next Steps

The current implementation has completed the modules included in Mini Project 1.

Potential future MLOps extensions include:

- [ ] CI/CD pipeline
- [ ] Automated Docker image builds
- [ ] Automated testing in CI
- [ ] Model performance monitoring
- [ ] Data-quality monitoring
- [ ] API load testing
- [ ] Deployment to a cloud/container platform
- [ ] Model version management
- [ ] Model registry integration
- [ ] More advanced models and feature sets
- [ ] Automated retraining
- [ ] Production observability and metrics

These are future extensions rather than completed components of the current implementation.

---

# Baseline Reference

The baseline should remain the reference point for future model improvements.

```text
Model
LinearRegression + DictVectorizer

Features
PU_DO
trip_distance

Target
duration

Validation split
20%

Random state
42

MAE
4.850 minutes

RMSE
8.270 minutes
```

Any future model or engineering change should be evaluated against this reference to determine whether it provides a measurable improvement.

---

# Final Project Snapshot

```text
Project
NYC Green Taxi Duration Prediction

Mini Project
1 / 5 — From Notebook to Production

Status
Complete

Model
LinearRegression + DictVectorizer

Features
PU_DO + trip_distance

Serialization
Pickle + ONNX

API
FastAPI

Tests
17 passed

Coverage
75.99%

Coverage Gate
70%

Container
Multi-stage Docker

Runtime User
appuser

Published Image
mahmoudosama20/prodml-api:0.1.0

Latest Tag
mahmoudosama20/prodml-api:latest
```

The project currently provides a reproducible path from raw taxi-trip data to a tested, serialized, containerized, and publicly published ML prediction service.
