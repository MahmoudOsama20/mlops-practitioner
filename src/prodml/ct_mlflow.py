from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost

MODEL_PATH = Path("pipelines/models/model.pkl")
METRICS_PATH = Path("pipelines/metrics.json")

EXPERIMENT_NAME = "prodml-duration-prediction"
MODEL_NAME = "ride-duration-predictor"


def main() -> None:
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    # Load evaluation metrics
    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

    # Load deployment artifact:
    # {
    #     "model": XGBRegressor,
    #     "vectorizer": DictVectorizer,
    # }
    with open(MODEL_PATH, "rb") as file:
        artifact = pickle.load(file)

    model = artifact["model"]

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="continuous-training-candidate",
    ) as run:
        mlflow.log_metrics(
            {
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r2": float(metrics["r2"]),
            }
        )

        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "learning_rate": model.learning_rate,
            }
        )

        mlflow.set_tags(
            {
                "training_type": "continuous_training",
                "model_name": MODEL_NAME,
                "framework": "xgboost",
            }
        )

        mlflow.xgboost.log_model(
            model,
            name="model",
        )

        run_id = run.info.run_id

    print(f"MLflow run ID: {run_id}")
    print(f"Candidate MAE: {float(metrics['mae']):.6f}")
    print(f"Model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
