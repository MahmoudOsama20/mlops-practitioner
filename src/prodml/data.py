from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import settings


def load_data(path: Path) -> pd.DataFrame:
    """Load the taxi trip dataset."""
    return pd.read_csv(path)


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and validation sets."""

    X = data[["PU_DO", "trip_distance"]]
    y = data["duration"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    return X_train, X_val, y_train, y_val
