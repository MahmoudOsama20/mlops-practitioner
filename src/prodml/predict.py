from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from prodml.config import settings
from prodml.logging_conf import timed


class DurationPredictor:
    """Predict taxi trip duration using the MLflow Production model."""

    def __init__(
        self,
        model_uri: str = "models:/ride-duration-predictor/Production",
    ) -> None:
        self.model_uri = model_uri
        self.model: Any | None = None
        self.vectorizer: DictVectorizer | None = None

    def load(self) -> "DurationPredictor":
        """Load the vectorizer and the MLflow Production model."""

        # The vectorizer is still stored in the original training artifact.
        import pickle

        with open(settings.model_path, "rb") as file:
            artifact = pickle.load(file)

        self.vectorizer = artifact["vectorizer"]

        # Load whichever model is currently in MLflow Production.
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.model = mlflow.pyfunc.load_model(self.model_uri)

        return self

    @timed
    def predict_one(
        self,
        features: dict[str, Any],
    ) -> float:
        """Predict duration for a single trip."""

        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        X = self.vectorizer.transform([features]).toarray().astype(np.float32)

        prediction = self.model.predict(X)

        return float(np.asarray(prediction).reshape(-1)[0])
        # return float(prediction[0])

    @timed
    def predict_batch(
        self,
        features: list[dict[str, Any]],
    ) -> list[float]:
        """Predict duration for multiple trips."""

        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        X = self.vectorizer.transform(features).toarray().astype(np.float32)

        predictions = self.model.predict(X)

        return np.asarray(predictions).reshape(-1).tolist()
        # return predictions.tolist()
