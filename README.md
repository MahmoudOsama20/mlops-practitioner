# MLOps Practitioner

An end-to-end MLOps project for predicting NYC Green Taxi trip duration.

The project starts with a reproducible machine-learning baseline and progressively turns it into a production-oriented ML system. The current implementation covers data preparation, feature engineering, model training, model persistence, prediction, configuration management, testing, logging, code quality, and pre-commit automation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Baseline Model](#baseline-model)
4. [Baseline Results](#baseline-results)
5. [Project Structure](#project-structure)
6. [Requirements](#requirements)
7. [Environment Setup](#environment-setup)
8. [Configuration](#configuration)
9. [Architecture](#architecture)
10. [Data Pipeline](#data-pipeline)
11. [Feature Engineering](#feature-engineering)
12. [Training Pipeline](#training-pipeline)
13. [Model Persistence](#model-persistence)
14. [Prediction Interface](#prediction-interface)
15. [Logging and Timing](#logging-and-timing)
16. [Testing](#testing)
17. [Code Quality](#code-quality)
18. [Pre-commit](#pre-commit)
19. [Notebook](#notebook)
20. [Development Workflow](#development-workflow)
21. [Reproducibility](#reproducibility)
22. [Current Status](#current-status)
23. [Planned Work](#planned-work)

---

## Project Overview

The objective is to predict the duration of a NYC Green Taxi trip in minutes.

The initial baseline uses two features:

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

The current baseline is intentionally simple. It provides a reproducible reference point that future models and production changes can be compared against.

---

## Dataset

The project uses NYC Green Taxi trip records.

The dataset contains fields including:

- Pickup timestamp
- Drop-off timestamp
- Pickup location ID
- Drop-off location ID
- Trip distance
- Passenger count
- Payment type
- Fare information
- Trip type
- Other taxi-trip attributes

The current local dataset is:

```text
datasets/green_tripdata_2026-01.csv
```

The baseline does not use every available column. Only the features required for the initial model are used.

---

## Baseline Model

The baseline model consists of:

```text
DictVectorizer + LinearRegression
```

### Input Features

```text
PU_DO
trip_distance
```

### Target

```text
duration
```

### Data Split

```text
Training:   80%
Validation: 20%
Random seed: 42
```

The categorical `PU_DO` feature is transformed using `DictVectorizer`.

The numerical `trip_distance` feature is passed through the same vectorization pipeline.

---

## Baseline Results

The current validation performance is:

| Metric | Result |
|---|---:|
| MAE | 4.850 minutes |
| RMSE | 8.270 minutes |

### Interpretation

The baseline has an average absolute error of approximately:

```text
4.850 minutes
```

on the validation set.

The RMSE is:

```text
8.270 minutes
```

The baseline metrics serve as the reference point for future model improvements.

---

## Project Structure

```text
mlops-practitioner/
│
├── .env
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
│
├── datasets/
│   └── green_tripdata_2026-01.csv
│
├── models/
│   └── baseline.pkl
│
├── notebooks/
│   └── 00-baseline.ipynb
│
├── reports/
│   ├── module-1.md
│   └── README.md
│
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
│
└── tests/
    └── test_predict.py
```

### Directory Responsibilities

#### `datasets/`

Contains local datasets used by the project.

#### `models/`

Contains persisted model artifacts.

#### `notebooks/`

Contains exploratory and demonstration notebooks.

#### `reports/`

Contains experiment and module reports.

#### `src/prodml/`

Contains the production-oriented Python package.

#### `tests/`

Contains automated tests.

---

## Requirements

The project requires Python 3.10 or newer.

The development environment currently uses:

```text
Python 3.12.7
```

Main runtime dependencies include:

- pandas
- scikit-learn
- pyarrow
- FastAPI
- Uvicorn
- Pydantic
- pydantic-settings

Development dependencies include:

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

After obtaining the repository, move into its root directory:

```powershell
cd E:\MLOps\projects\mlops-practitioner
```

The exact path will depend on your local environment.

---

## 2. Create the virtual environment

```powershell
python -m venv .venv
```

This creates:

```text
.venv/
```

at the project root.

---

## 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The terminal should then show:

```text
(.venv)
```

---

## 4. Install the project

Install the package in editable mode together with development dependencies:

```powershell
pip install -e ".[dev]"
```

Editable installation allows changes inside `src/prodml/` to be reflected immediately without reinstalling the package.

---

## 5. Verify the installation

Run:

```powershell
python -c "import prodml; print(prodml.__file__)"
```

The result should point to:

```text
src\prodml\__init__.py
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

---

## `.env`

Create a `.env` file in the project root:

```env
PRODML_DATA_PATH=datasets/green_tripdata_2026-01.csv
PRODML_MODEL_PATH=models/baseline.pkl
PRODML_REPORT_PATH=reports/module-1.md
PRODML_TEST_SIZE=0.2
PRODML_RANDOM_STATE=42
```

The `.env` file is local configuration and should not be committed to Git.

The `.gitignore` contains:

```text
.env
```

---

## Configuration Values

| Variable | Purpose | Default |
|---|---|---|
| `PRODML_DATA_PATH` | Training dataset | `datasets/green_tripdata_2026-01.csv` |
| `PRODML_MODEL_PATH` | Saved model artifact | `models/baseline.pkl` |
| `PRODML_REPORT_PATH` | Baseline report | `reports/module-1.md` |
| `PRODML_TEST_SIZE` | Validation fraction | `0.2` |
| `PRODML_RANDOM_STATE` | Random seed | `42` |

---

## Verify Configuration

Run:

```powershell
python -c "from prodml.config import settings; print(settings.data_path); print(settings.model_path); print(settings.test_size)"
```

Expected output should contain the configured dataset path, model path, and:

```text
0.2
```

---

# Architecture

The current architecture separates data loading, feature engineering, training, prediction, and configuration.

```text
                    ┌──────────────────────┐
                    │       Dataset        │
                    │ green_tripdata_...   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       data.py        │
                    │  load + split data   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     features.py      │
                    │ duration + PU_DO +   │
                    │ cleaning + vectorize │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       train.py       │
                    │ LinearRegression     │
                    │ evaluation + saving  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    baseline.pkl      │
                    │ model + vectorizer   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      predict.py      │
                    │ DurationPredictor    │
                    └──────────────────────┘
```

---

# Data Pipeline

The training pipeline follows these steps:

```text
Raw Dataset
     │
     ▼
Load Data
     │
     ▼
Convert Datetimes
     │
     ▼
Create Duration
     │
     ▼
Clean Invalid/Extreme Trips
     │
     ▼
Create PU_DO
     │
     ▼
Select Features
     │
     ▼
Train / Validation Split
     │
     ▼
DictVectorizer
     │
     ▼
Linear Regression
     │
     ▼
Evaluation
     │
     ├──────────────► MAE / RMSE
     │
     ▼
Save Model
```

---

# Feature Engineering

Feature engineering is implemented in:

```text
src/prodml/features.py
```

## Duration

Duration is calculated using the pickup and drop-off timestamps:

```python
data["duration"] = (
    data["lpep_dropoff_datetime"]
    - data["lpep_pickup_datetime"]
).dt.total_seconds() / 60
```

---

## Data Cleaning

Trips are filtered to remove invalid or extreme values.

Duration must satisfy:

```text
1 <= duration <= 120
```

Trip distance must satisfy:

```text
0 < trip_distance <= 100
```

---

## PU_DO

`PU_DO` is an engineered categorical feature combining:

```text
PULocationID
DOLocationID
```

Example:

```text
PULocationID = 74
DOLocationID = 42

PU_DO = "74_42"
```

---

## Model Features

The baseline uses:

```python
features = [
    "PU_DO",
    "trip_distance",
]
```

---

# Training Pipeline

Training is implemented in:

```text
src/prodml/train.py
```

The pipeline:

1. Loads the dataset.
2. Performs feature engineering and cleaning.
3. Splits the data into training and validation sets.
4. Fits a `DictVectorizer`.
5. Trains a `LinearRegression` model.
6. Calculates MAE and RMSE.
7. Saves the model and vectorizer.
8. Writes the baseline report.

---

## Train from the Module

Run:

```powershell
python -m prodml.train
```

Expected output:

```text
Validation MAE: 4.850 minutes
Validation RMSE: 8.270 minutes
```

---

## Train Using the Console Script

The project also defines:

```text
prodml-train
```

Run:

```powershell
prodml-train
```

Both commands execute the same `main()` function in:

```text
src/prodml/train.py
```

---

# Model Persistence

The trained artifact is saved to:

```text
models/baseline.pkl
```

The artifact contains:

```python
{
    "model": model,
    "vectorizer": vectorizer,
}
```

Both the trained model and fitted `DictVectorizer` are saved because prediction must use the same feature transformation that was fitted during training.

The model is persisted using Python's `pickle` mechanism.

---

# Prediction Interface

Prediction is implemented in:

```text
src/prodml/predict.py
```

The main interface is:

```python
DurationPredictor
```

It provides:

```python
load()
predict_one()
predict_batch()
```

This separates prediction from model training and provides a stable interface for future model-serving components.

---

## Load the Model

```python
from prodml.predict import DurationPredictor

predictor = DurationPredictor().load()
```

---

## Single Prediction

```python
prediction = predictor.predict_one(
    {
        "PU_DO": "74_42",
        "trip_distance": 2.5,
    }
)

print(f"Predicted duration: {prediction:.2f} minutes")
```

---

## Batch Prediction

```python
predictions = predictor.predict_batch(
    [
        {
            "PU_DO": "74_42",
            "trip_distance": 2.5,
        },
        {
            "PU_DO": "75_41",
            "trip_distance": 4.1,
        },
    ]
)

print(predictions)
```

---

## Prediction Interface Design

The predictor follows this pattern:

```text
DurationPredictor
       │
       ├── load()
       │
       ├── predict_one()
       │
       └── predict_batch()
```

This interface is intentionally separated from the underlying model implementation so that future serving technologies can reuse it.

---

# Logging and Timing

Logging utilities are implemented in:

```text
src/prodml/logging_conf.py
```

The project contains a custom:

```python
@timed
```

decorator.

The decorator measures function execution time and logs the result.

Example:

```python
from prodml.logging_conf import configure_logging
from prodml.predict import DurationPredictor

configure_logging()

predictor = DurationPredictor().load()

prediction = predictor.predict_one(
    {
        "PU_DO": "74_42",
        "trip_distance": 2.5,
    }
)
```

The prediction methods are decorated with `@timed`.

---

# Testing

Tests are implemented using `pytest`.

Current tests are located in:

```text
tests/test_predict.py
```

The current test suite verifies:

- Model loading
- Single prediction
- Batch prediction

---

## Run Tests

Basic test run:

```powershell
pytest -v
```

Run with coverage:

```powershell
pytest -v --cov=src/prodml --cov-report=term-missing
```

The current prediction tests pass successfully.

Example:

```text
3 passed
```

The current coverage is approximately:

```text
41%
```

The lower overall coverage is expected at this stage because the initial tests focus on the prediction interface. Additional tests for data loading, feature engineering, and training can be added as the project grows.

---

# Code Quality

The project uses:

- Ruff
- Black
- pre-commit

---

## Ruff

Check the source and test code:

```powershell
ruff check src tests
```

Ruff is also configured to run automatically through pre-commit.

---

## Black

Check formatting:

```powershell
black --check src tests
```

Format the project:

```powershell
black src tests
```

The project targets Python 3.12 for Black formatting.

---

# Pre-commit

The project uses pre-commit to automatically enforce basic code quality checks.

Configuration:

```text
.pre-commit-config.yaml
```

Current hooks include:

```text
end-of-file-fixer
ruff
black
```

---

## Install Pre-commit

```powershell
pre-commit install
```

This installs the Git hook:

```text
.git/hooks/pre-commit
```

---

## Run All Hooks

```powershell
pre-commit run --all-files
```

Expected result:

```text
fix end of files ........ Passed
ruff check .............. Passed
black ................... Passed
```

---

# Notebook

The initial baseline experiment is documented in:

```text
notebooks/00-baseline.ipynb
```

The notebook was used to establish the initial model and evaluate its performance.

The reusable implementation has since been moved into the Python package under:

```text
src/prodml/
```

The long-term goal is for the notebook to focus on experimentation and demonstration rather than containing production implementation logic.

---

# Development Workflow

The recommended workflow is:

```text
1. Activate .venv
        ↓
2. Modify source code
        ↓
3. Run pre-commit
        ↓
4. Run tests
        ↓
5. Train/evaluate model
        ↓
6. Check metrics
        ↓
7. Review Git changes
        ↓
8. Commit
```

---

## Complete Development Check

### Activate environment

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

### Run training

```powershell
python -m prodml.train
```

### Run the console training command

```powershell
prodml-train
```

---

# Reproducibility

The baseline uses a fixed random seed:

```text
random_state = 42
```

The validation fraction is:

```text
test_size = 0.2
```

These values are controlled through `pydantic-settings`.

The same dataset, preprocessing rules, feature definitions, split configuration, and model configuration should reproduce the baseline metrics approximately.

Current reference metrics:

```text
MAE  = 4.850 minutes
RMSE = 8.270 minutes
```

---

# Git and Repository Hygiene

The following local files/directories are excluded from Git:

```text
.venv/
.env
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.coverage
```

The `.env` file is intentionally excluded because environment-specific configuration should remain local.

A future `.env.example` can be used to document expected configuration variables without committing the local environment file.

---

# Current Status

## Completed

- [x] NYC Green Taxi dataset loaded
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
- [x] Baseline model persistence
- [x] Baseline report
- [x] Python package structure
- [x] `pyproject.toml`
- [x] Editable package installation
- [x] Virtual environment
- [x] `pydantic-settings`
- [x] `.env` configuration
- [x] Type hints for public functions
- [x] `DurationPredictor`
- [x] `predict_one()`
- [x] `predict_batch()`
- [x] Custom `@timed` decorator
- [x] Ruff
- [x] Black
- [x] pre-commit
- [x] Automated prediction tests
- [x] Training through `python -m prodml.train`
- [x] Training through `prodml-train`

---

# Planned Work

The project will continue toward a production-oriented MLOps system.

Planned components include:

- [ ] Complete notebook refactoring
- [ ] Increase test coverage
- [ ] Add data and feature-engineering tests
- [ ] Add FastAPI application
- [ ] Add API schemas
- [ ] Add `/health` endpoint
- [ ] Add prediction endpoints
- [ ] Add API integration tests
- [ ] Containerize the service
- [ ] Add production model serving
- [ ] Add load testing
- [ ] Add monitoring
- [ ] Add deployment infrastructure
- [ ] Add model optimization
- [ ] Add CI/CD

---

# Baseline Reference

The baseline should be treated as the reference model for future work.

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

Future models and engineering changes should be evaluated against this reference to determine whether they provide a measurable improvement.

---

# License

Add the project's license information here when the repository license is finalized.
