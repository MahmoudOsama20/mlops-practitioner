import logging
import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import prepare_data, vectorize_features
from prodml.logging_conf import configure_logging

logger = logging.getLogger(__name__)


def train_model(
    X_train: csr_matrix,
    y_train: pd.Series,
) -> LinearRegression:
    """Train the baseline Linear Regression model."""

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_model(
    model: LinearRegression,
    X_val: csr_matrix,
    y_val: pd.Series,
) -> tuple[float, float]:
    """Evaluate the model using MAE and RMSE."""

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(
        y_val,
        y_pred,
    )

    rmse = root_mean_squared_error(
        y_val,
        y_pred,
    )

    return mae, rmse


def save_model(
    model: LinearRegression,
    vectorizer,
    path: Path,
) -> None:
    """Save the trained model and vectorizer."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "wb") as file:
        pickle.dump(
            {
                "model": model,
                "vectorizer": vectorizer,
            },
            file,
        )


def save_report(
    mae: float,
    rmse: float,
    path: Path,
) -> None:
    """Save the baseline evaluation report."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = f"""# Module 1 — Baseline Model

## Validation Results

- MAE: {mae:.3f} minutes
- RMSE: {rmse:.3f} minutes

## Features

- PU_DO
- trip_distance

## Model

LinearRegression + DictVectorizer

## Dataset

NYC TLC Green Taxi Trip Records
"""

    with open(path, "w", encoding="utf-8") as file:
        file.write(report)


def main() -> None:
    """Run the complete baseline training pipeline."""

    # Load data
    data = load_data(settings.data_path)

    # Feature engineering and cleaning
    data = prepare_data(data)

    # Train/validation split
    X_train, X_val, y_train, y_val = split_data(data)

    # Vectorize features
    X_train_vectorized, X_val_vectorized, vectorizer = vectorize_features(
        X_train,
        X_val,
    )

    # Train model
    model = train_model(
        X_train_vectorized,
        y_train,
    )

    # Evaluate model
    mae, rmse = evaluate_model(
        model,
        X_val_vectorized,
        y_val,
    )

    logger.info(
        "model_evaluation",
        extra={
            "mae_minutes": round(mae, 3),
            "rmse_minutes": round(rmse, 3),
        },
    )

    # Save model
    save_model(
        model,
        vectorizer,
        settings.model_path,
    )

    # Save report
    save_report(
        mae,
        rmse,
        settings.report_path,
    )


if __name__ == "__main__":
    configure_logging()
    main()
