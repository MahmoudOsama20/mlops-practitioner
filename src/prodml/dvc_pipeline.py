from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from prodml.features import prepare_data

RAW_DATA = Path("datasets/green_tripdata_2026-02.csv")
PREPARED_DATA = Path("pipelines/data/prepared.csv")
FEATURES_DATA = Path("pipelines/data/features.pkl")
MODEL_PATH = Path("pipelines/models/model.pkl")
METRICS_PATH = Path("pipelines/metrics.json")
PARAMS_PATH = Path("params.yaml")


def load_params() -> dict:
    with open(PARAMS_PATH, encoding="utf-8") as file:
        return yaml.safe_load(file)


def prepare() -> None:
    """Prepare and clean the raw dataset."""

    data = pd.read_csv(RAW_DATA)
    data = prepare_data(data)

    PREPARED_DATA.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PREPARED_DATA, index=False)

    print(f"Prepared dataset: {data.shape}")


def featurize() -> None:
    """Split and vectorize the prepared dataset."""

    params = load_params()["data"]

    data = pd.read_csv(PREPARED_DATA)

    X = data[["PU_DO", "trip_distance"]]
    y = data["duration"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    vectorizer = DictVectorizer()

    X_train_vectorized = vectorizer.fit_transform(X_train.to_dict(orient="records"))
    X_val_vectorized = vectorizer.transform(X_val.to_dict(orient="records"))

    FEATURES_DATA.parent.mkdir(parents=True, exist_ok=True)

    with open(FEATURES_DATA, "wb") as file:
        pickle.dump(
            {
                "X_train": X_train_vectorized,
                "X_val": X_val_vectorized,
                "y_train": y_train,
                "y_val": y_val,
                "vectorizer": vectorizer,
            },
            file,
        )

    print("Features created.")


def train() -> None:
    """Train a deterministic XGBoost model."""

    params = load_params()

    with open(FEATURES_DATA, "rb") as file:
        data = pickle.load(file)

    training_params = params["training"]
    data_params = params["data"]

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=training_params["n_estimators"],
        max_depth=training_params["max_depth"],
        learning_rate=training_params["learning_rate"],
        random_state=data_params["random_state"],
        tree_method="hist",
    )

    model.fit(
        data["X_train"],
        data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("Model trained.")


def evaluate() -> None:
    """Evaluate the trained model and save DVC metrics."""

    with open(FEATURES_DATA, "rb") as file:
        data = pickle.load(file)

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(data["X_val"])

    metrics = {
        "mae": float(mean_absolute_error(data["y_val"], predictions)),
        "rmse": float(
            mean_squared_error(
                data["y_val"],
                predictions,
            )
            ** 0.5
        ),
        "r2": float(r2_score(data["y_val"], predictions)),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    command = sys.argv[1]

    commands = {
        "prepare": prepare,
        "featurize": featurize,
        "train": train,
        "evaluate": evaluate,
    }

    if command not in commands:
        raise ValueError(f"Unknown command: {command}")

    commands[command]()
