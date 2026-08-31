import pandas as pd
import pytest

from prodml.features import (
    clean_data,
    create_pu_do,
    vectorize_features,
)


@pytest.mark.parametrize(
    ("trip_distance", "expected_rows"),
    [
        (0, 0),
        (2.5, 1),
    ],
)
def test_clean_data_trip_distance(
    trip_distance: float,
    expected_rows: int,
) -> None:
    """Clean data should remove zero-distance trips."""

    data = pd.DataFrame(
        {
            "duration": [10.0],
            "trip_distance": [trip_distance],
        }
    )

    result = clean_data(data)

    assert len(result) == expected_rows


def test_create_pu_do() -> None:
    """Pickup and dropoff IDs should form the PU_DO feature."""

    data = pd.DataFrame(
        {
            "PULocationID": [74],
            "DOLocationID": [42],
        }
    )

    result = create_pu_do(data)

    assert result["PU_DO"].iloc[0] == "74_42"


def test_vectorizer_handles_unseen_pu_do() -> None:
    """Unseen PU_DO categories should be ignored safely."""

    train = pd.DataFrame(
        {
            "PU_DO": ["74_42"],
            "trip_distance": [2.5],
        }
    )

    validation = pd.DataFrame(
        {
            "PU_DO": ["999_999"],
            "trip_distance": [3.0],
        }
    )

    X_train, X_val, vectorizer = vectorize_features(
        train,
        validation,
    )

    assert X_train.shape[1] == X_val.shape[1]
    assert X_train.shape[1] > 0
    assert vectorizer is not None


def test_vectorizer_handles_missing_category() -> None:
    """Missing PU_DO values should not crash vectorization."""

    train = pd.DataFrame(
        {
            "PU_DO": ["74_42"],
            "trip_distance": [2.5],
        }
    )

    validation = pd.DataFrame(
        {
            "PU_DO": [None],
            "trip_distance": [3.0],
        }
    )

    X_train, X_val, _ = vectorize_features(
        train,
        validation,
    )

    assert X_train.shape[1] == X_val.shape[1]
