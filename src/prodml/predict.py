import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.config import settings
from prodml.logging_conf import timed


class DurationPredictor:
    """Predict taxi trip duration using a trained model."""

    def __init__(
        self,
        model_path: Path = settings.model_path,
    ) -> None:
        self.model_path = model_path
        self.model: LinearRegression | None = None
        self.vectorizer: DictVectorizer | None = None

    def load(self) -> "DurationPredictor":
        """Load the trained model and vectorizer."""

        with open(self.model_path, "rb") as file:
            artifact = pickle.load(file)

        self.model = artifact["model"]
        self.vectorizer = artifact["vectorizer"]

        return self

    @timed
    def predict_one(
        self,
        features: dict[str, Any],
    ) -> float:
        """Predict duration for a single trip."""

        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        X = self.vectorizer.transform([features])

        prediction = self.model.predict(X)

        return float(prediction[0])

    @timed
    def predict_batch(
        self,
        features: list[dict[str, Any]],
    ) -> list[float]:
        """Predict duration for multiple trips."""

        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        X = self.vectorizer.transform(features)

        predictions = self.model.predict(X)

        return predictions.tolist()
