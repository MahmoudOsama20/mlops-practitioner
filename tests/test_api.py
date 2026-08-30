from fastapi.testclient import TestClient

from prodml.api.main import app


def test_health() -> None:
    """Health endpoint should confirm the model is loaded."""

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers


def test_metadata() -> None:
    """Metadata endpoint should return model metadata."""

    with TestClient(app) as client:
        response = client.get("/metadata")

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "0.1.0"
    assert body["training_date"] == "2026-08-30"
    assert body["features"] == [
        "PU_DO",
        "trip_distance",
    ]
    assert body["framework"] == "scikit-learn"

    # SHA-256 produces a 64-character hexadecimal hash.
    assert len(body["artifact_hash"]) == 64
    assert all(character in "0123456789abcdef" for character in body["artifact_hash"])

    assert "X-Request-ID" in response.headers


def test_predict() -> None:
    """Predict endpoint should return a prediction."""

    payload = {
        "PU_DO": "74_42",
        "trip_distance": 2.5,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert body["prediction"] > 0
    assert body["model_version"] == "0.1.0"
    assert body["correlation_id"] == response.headers["X-Request-ID"]
    assert body["latency_ms"] >= 0


def test_predict_batch() -> None:
    """Batch prediction endpoint should return predictions."""

    payload = {
        "trips": [
            {
                "PU_DO": "74_42",
                "trip_distance": 2.5,
            },
            {
                "PU_DO": "41_42",
                "trip_distance": 5.0,
            },
        ]
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json=payload,
        )

    assert response.status_code == 200

    body = response.json()

    assert len(body["predictions"]) == 2
    assert all(prediction > 0 for prediction in body["predictions"])
    assert body["model_version"] == "0.1.0"
    assert body["correlation_id"] == response.headers["X-Request-ID"]
    assert body["latency_ms"] >= 0


def test_predict_negative_distance() -> None:
    """Negative distance should be rejected by Pydantic."""

    payload = {
        "PU_DO": "74_42",
        "trip_distance": -5,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422

    body = response.json()

    assert body["detail"][0]["type"] == "greater_than"
    assert "X-Request-ID" in response.headers


def test_predict_distance_outside_training_range() -> None:
    """Distances above the training range should be rejected."""

    payload = {
        "PU_DO": "74_42",
        "trip_distance": 101,
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 422

    assert response.json()["detail"] == ("trip_distance must be <= 100 miles")

    assert "X-Request-ID" in response.headers


def test_predict_unexpected_error(monkeypatch) -> None:
    """Unexpected prediction errors should return a clean 500."""

    payload = {
        "PU_DO": "74_42",
        "trip_distance": 2.5,
    }

    def failing_predict_one(features: dict) -> float:
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(
        "prodml.api.main.predictor.predict_one",
        failing_predict_one,
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/predict",
            json=payload,
        )

    assert response.status_code == 500

    assert response.json() == {"detail": "Internal server error"}

    assert "traceback" not in response.text.lower()
    assert "simulated model failure" not in response.text.lower()
    assert "X-Request-ID" in response.headers
