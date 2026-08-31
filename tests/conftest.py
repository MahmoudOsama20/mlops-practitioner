import pickle

import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app


@pytest.fixture
def sample_features() -> dict:
    """Return a valid feature dictionary for prediction tests."""

    return {
        "PU_DO": "74_42",
        "trip_distance": 2.5,
    }


@pytest.fixture(scope="session")
def trained_model() -> dict:
    """Load the trained model artifact once for the test session."""

    with open("models/baseline.pkl", "rb") as file:
        return pickle.load(file)


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client with application lifespan enabled."""

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client
