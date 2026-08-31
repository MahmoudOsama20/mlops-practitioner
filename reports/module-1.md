# Module 1 — Baseline Model and Productionization

## 1. Objective

Build a reproducible baseline model for predicting NYC Green Taxi trip
duration and establish a production-oriented foundation for future MLOps
modules.

The module covers:

- reproducible data preparation
- feature engineering
- baseline model training
- model evaluation
- model serialization
- ONNX export and parity validation
- inference benchmarking
- FastAPI serving
- structured logging and request tracing
- automated testing
- code-quality enforcement
- Docker packaging
- Docker Hub publication

---

## 2. Dataset

**NYC TLC Green Taxi Trip Records**

The target variable is trip duration in minutes.

Trip duration is calculated from:

```text
lpep_dropoff_datetime - lpep_pickup_datetime
```

and converted from seconds to minutes.

---

## 3. Data Preparation

The training pipeline performs the following steps:

1. Load the dataset.
2. Convert pickup and drop-off timestamps to datetime.
3. Calculate trip duration in minutes.
4. Remove invalid and extreme trips.
5. Create the pickup/drop-off location pair feature.
6. Select the model features.
7. Split the data into training and validation sets.
8. Fit a `DictVectorizer`.

### Cleaning Rules

```text
1 <= duration <= 120 minutes
0 < trip_distance <= 100 miles
```

### Train / Validation Split

```text
Training:     80%
Validation:   20%
Random state: 42
```

---

## 4. Features

The baseline uses two features:

- `PU_DO`
- `trip_distance`

### PU_DO

`PU_DO` combines pickup and drop-off location IDs.

Example:

```text
PULocationID = 74
DOLocationID = 42

PU_DO = "74_42"
```

### trip_distance

The trip distance in miles.

---

## 5. Baseline Model

The baseline model is:

```text
DictVectorizer + LinearRegression
```

The fitted `DictVectorizer` transforms the categorical `PU_DO` feature
and numerical `trip_distance` feature into the representation consumed by
the regression model.

---

## 6. Validation Results

| Metric | Result |
|---|---:|
| MAE | 4.850 minutes |
| RMSE | 8.270 minutes |

### Interpretation

The baseline achieves an MAE of approximately **4.85 minutes** on the
validation set.

These results establish the reference point against which future model
improvements can be evaluated.

---

## 7. Model Artifacts

The trained baseline is persisted as:

```text
models/baseline.pkl
```

The Pickle artifact contains:

- trained `LinearRegression`
- fitted `DictVectorizer`

An additional ONNX representation is exported to:

```text
models/model.onnx
```

### Pickle Security

Pickle is Python-specific and should only be loaded from a trusted source.
Untrusted Pickle files must not be deserialized because Pickle
deserialization can execute arbitrary code.

---

## 8. ONNX Export

The baseline was exported to ONNX for portable runtime inference.

The exported model reports:

```text
Input name:  input
Input shape: [None, 4910]
Output name: variable
```

The first dimension is dynamic, allowing different batch sizes.

The export is implemented in:

```text
src/prodml/export.py
```

---

## 9. ONNX Prediction Parity

A dedicated test validates that the ONNX model produces predictions
consistent with the reference model.

Test:

```text
tests/test_serialization.py
```

Result:

```text
PASSED
```

The complete test suite includes the serialization parity test.

---

## 10. Inference Benchmark

Local inference benchmarking produced the following results:

| Format | Mean Latency | P95 Latency |
|---|---:|---:|
| Pickle / scikit-learn | 0.084 ms | 0.101 ms |
| ONNX Runtime | 0.223 ms | 0.231 ms |

### Result

For this small baseline model and local workload, the Pickle/scikit-learn
implementation was faster than ONNX Runtime.

ONNX therefore provides a portability and runtime-interoperability option,
but it did not provide a latency improvement for this particular workload.

---

## 11. Serialization Comparison

| Format | Advantages | Limitations |
|---|---|---|
| Pickle | Simple Python serialization; preserves the fitted scikit-learn model and vectorizer | Python-specific and unsafe for untrusted deserialization |
| ONNX | Portable model representation and ONNX Runtime support | Requires an export step and compatible runtime |
| JSON | Human-readable and widely supported | Not appropriate for storing a trained ML model |
| Protobuf | Efficient structured binary serialization | Requires an explicit schema and is not itself a trained-model format |

For this project, Pickle is the reference artifact and ONNX is the
additional portable inference representation.

---

## 12. Prediction API

The production API is implemented with FastAPI.

Main application:

```text
src/prodml/api/main.py
```

Schemas:

```text
src/prodml/api/schemas.py
```

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/metadata` | Model metadata and artifact hash |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

### Health Check

```text
GET /health
```

Returns:

```json
{"status":"ok"}
```

### Single Prediction

Request:

```json
{
  "PU_DO": "74_42",
  "trip_distance": 2.5
}
```

Observed response:

```json
{
  "prediction": 12.727868515169943,
  "model_version": "0.1.0",
  "correlation_id": "f3643606-0f14-4380-8578-bb4cf2054acd",
  "latency_ms": 4.834
}
```

### Batch Prediction

Request:

```json
{
  "trips": [
    {
      "PU_DO": "74_42",
      "trip_distance": 2.5
    },
    {
      "PU_DO": "41_42",
      "trip_distance": 5.0
    }
  ]
}
```

Observed predictions:

```text
12.727868515169943
19.903236786100038
```

---

## 13. API Validation

The API validates request inputs with Pydantic and applies the training
range constraints.

### Negative Distance

Input:

```json
{
  "PU_DO": "74_42",
  "trip_distance": -5
}
```

Result:

```text
HTTP 422
```

### Distance Outside Training Range

Input:

```json
{
  "PU_DO": "74_42",
  "trip_distance": 101
}
```

Result:

```text
HTTP 422
```

The API therefore rejects values outside the supported inference range.

---

## 14. Structured Logging and Request Tracing

The service uses structured JSON logging.

Representative events include:

```text
model_loaded
request_started
prediction_served
request_completed
validation_rejection
function_timing
```

Each request receives a correlation ID.

The correlation ID is:

- generated for each request
- included in structured logs
- returned in the `X-Request-ID` response header
- included in prediction responses

Prediction latency is measured and exposed by the API.

---

## 15. Model Metadata

The `/metadata` endpoint exposes:

```json
{
  "model_version": "0.1.0",
  "training_date": "2026-08-30",
  "features": [
    "PU_DO",
    "trip_distance"
  ],
  "framework": "scikit-learn",
  "artifact_hash": "c362d9d8622bf5a528dc17c92783518a3c81a216fc048c5b8760ddd96db9218a"
}
```

This provides basic model identity and artifact traceability.

---

## 16. Automated Testing

The project uses `pytest` and `pytest-cov`.

The final suite contains **17 tests** covering:

- API health
- API metadata
- single prediction
- batch prediction
- negative-distance validation
- training-range validation
- unexpected API errors
- feature cleaning
- `PU_DO` creation
- unseen categorical values
- missing categorical values
- predictor loading
- prediction
- deterministic prediction
- batch prediction
- ONNX prediction parity

### Final Test Result

```text
17 passed
```

---

## 17. Test Coverage

Final coverage:

```text
Total coverage: 75.99%
Required coverage: 70%
```

The coverage requirement is therefore satisfied.

The main remaining uncovered code is concentrated in executable training
and export entry points, while the core API, feature engineering,
configuration, logging, prediction, and serialization behavior are
covered by tests.

---

## 18. Code Quality

The project uses:

- Ruff
- Black
- pre-commit

Final pre-commit result:

```text
fix end of files ........ Passed
ruff check .............. Passed
black ................... Passed
```

The configured quality checks pass across the repository.

---

## 19. Dockerization

The API is packaged as a multi-stage Docker image.

The runtime image contains:

- Python runtime
- installed application package
- runtime dependencies
- Pickle model
- ONNX model

The container exposes:

```text
8000
```

and runs Uvicorn on:

```text
0.0.0.0:8000
```

### Container User

The container runs as:

```text
appuser
```

rather than root.

This was verified with:

```text
docker exec <container_id> whoami
```

which returned:

```text
appuser
```

---

## 20. Docker Image Verification

The published image was successfully pulled from Docker Hub:

```text
mahmoudosama20/prodml-api:0.1.0
```

The image started successfully and loaded:

```text
/app/models/baseline.pkl
```

The container successfully served:

```text
GET /health
POST /predict
```

with successful responses.

---

## 21. Docker Hub

Published repository:

```text
mahmoudosama20/prodml-api
```

Published tags:

```text
0.1.0
latest
```

The release image can be pulled with:

```powershell
docker pull mahmoudosama20/prodml-api:0.1.0
```

Docker Hub repository:

https://hub.docker.com/r/mahmoudosama20/prodml-api

---

## 22. Docker Image Size

The final multi-stage image was approximately:

```text
Disk usage:    1.11 GB
Content size:  251 MB
```

A separate single-stage build was also produced during the module work:

```text
Disk usage:    1.13 GB
Content size:  256 MB
```

The multi-stage build therefore reduced the reported image content size
by approximately **5 MB** compared with the single-stage build.

The runtime image also separates build dependencies from the final runtime
environment.

---

## 23. Docker Compose

The repository includes:

```text
docker/docker-compose.yml
```

The Compose configuration was used to verify:

- API startup
- model artifact availability
- `/app/models` contents
- non-root execution

The mounted model directory contained:

```text
baseline.pkl
model.onnx
```

and the running service user was:

```text
appuser
```

---

## 24. Production-Oriented Architecture

The resulting system can be summarized as:

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Feature Engineering
   │
   ▼
Train / Validation Split
   │
   ▼
DictVectorizer
   │
   ▼
LinearRegression
   │
   ├──────────────► baseline.pkl
   │
   └──────────────► model.onnx
                         │
                         ▼
                    FastAPI API
                         │
               ┌─────────┼─────────┐
               ▼         ▼         ▼
            /health   /predict  /metadata
                         │
                         ▼
                  Structured Logs
                         │
                         ▼
                    Docker Image
                         │
                         ▼
                     Docker Hub
```

---

## 25. Reproducibility

The baseline is configured with:

```text
test_size = 0.2
random_state = 42
```

The training code, preprocessing rules, feature definitions, and model
configuration are stored in the repository rather than being dependent
on notebook-only state.

Reference metrics:

```text
MAE  = 4.850 minutes
RMSE = 8.270 minutes
```

---

## 26. Maturity Self-Assessment

### Current State

The project has progressed from a notebook-based baseline to a
production-oriented ML service.

Implemented capabilities include:

- reproducible training code
- packaged Python source
- feature engineering
- persisted model artifacts
- ONNX serialization
- serialization parity testing
- FastAPI serving
- Pydantic validation
- structured logging
- correlation IDs
- request latency measurement
- automated API and unit tests
- coverage enforcement
- Ruff
- Black
- pre-commit
- multi-stage Docker build
- non-root container execution
- Docker Compose
- Docker Hub publication

### Remaining Gaps

The project is not yet a complete production ML platform.

Future improvements include:

- CI/CD automation
- automated deployment
- experiment tracking
- model registry
- automated retraining
- data drift monitoring
- model performance monitoring
- production alerting
- cloud deployment
- load testing
- stronger artifact lifecycle management

Therefore, the current maturity can be described as:

> **Production-oriented baseline service, not yet a fully automated
> production ML platform.**

---

## 27. Definition of Done

The Module 1 deliverable is complete when:

- [x] Baseline model implemented
- [x] Baseline MAE recorded
- [x] Baseline RMSE recorded
- [x] Features documented
- [x] Reproducible train/validation split implemented
- [x] Pickle artifact generated
- [x] ONNX artifact generated
- [x] ONNX prediction parity tested
- [x] Inference benchmark completed
- [x] FastAPI API implemented
- [x] API input validation implemented
- [x] Metadata endpoint implemented
- [x] Structured logging implemented
- [x] Correlation IDs implemented
- [x] Automated tests implemented
- [x] Coverage requirement satisfied
- [x] Ruff passed
- [x] Black passed
- [x] pre-commit passed
- [x] Docker image built
- [x] Docker image runs successfully
- [x] Container runs as non-root user
- [x] Docker Compose verified
- [x] Docker image published to Docker Hub
- [x] README documentation completed
- [x] Module 1 report completed

---

## 28. Final Summary

Module 1 establishes a reproducible baseline for NYC Green Taxi trip
duration prediction using:

```text
DictVectorizer + LinearRegression
```

with:

```text
PU_DO
trip_distance
```

The baseline achieves:

```text
MAE  = 4.850 minutes
RMSE = 8.270 minutes
```

The project was then extended into a production-oriented inference
service with:

```text
Pickle + ONNX
FastAPI
Pydantic validation
Structured JSON logging
Correlation IDs
Automated testing
75.99% test coverage
Ruff
Black
pre-commit
Multi-stage Docker
Non-root execution
Docker Compose
Docker Hub publication
```

The published release image is:

```text
mahmoudosama20/prodml-api:0.1.0
```

This implementation provides the foundation for subsequent MLOps work,
including CI/CD, monitoring, model versioning, automated deployment,
drift detection, and model lifecycle management.
