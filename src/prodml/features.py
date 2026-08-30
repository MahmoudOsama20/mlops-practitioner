import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def create_duration(data: pd.DataFrame) -> pd.DataFrame:
    """Create trip duration in minutes."""

    data = data.copy()

    data["lpep_pickup_datetime"] = pd.to_datetime(data["lpep_pickup_datetime"])

    data["lpep_dropoff_datetime"] = pd.to_datetime(data["lpep_dropoff_datetime"])

    data["duration"] = (
        data["lpep_dropoff_datetime"] - data["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid and extreme trips."""

    data = data[(data["duration"] >= 1) & (data["duration"] <= 120)].copy()

    data = data[(data["trip_distance"] > 0) & (data["trip_distance"] <= 100)].copy()

    return data


def create_pu_do(data: pd.DataFrame) -> pd.DataFrame:
    """Create the pickup-dropoff pair feature."""

    data = data.copy()

    data["PU_DO"] = (
        data["PULocationID"].astype(str) + "_" + data["DOLocationID"].astype(str)
    )

    return data


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering and data cleaning."""

    data = create_duration(data)
    data = clean_data(data)
    data = create_pu_do(data)

    return data


def vectorize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple:
    """Vectorize training and validation features."""

    dv = DictVectorizer()

    train_dicts = X_train.to_dict(orient="records")
    val_dicts = X_val.to_dict(orient="records")

    X_train_vectorized = dv.fit_transform(train_dicts)
    X_val_vectorized = dv.transform(val_dicts)

    return X_train_vectorized, X_val_vectorized, dv
