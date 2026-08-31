from prodml.predict import DurationPredictor


def test_predictor_loads() -> None:
    """The predictor should load the model and vectorizer."""

    predictor = DurationPredictor().load()

    assert predictor.model is not None
    assert predictor.vectorizer is not None


def test_predict_one(sample_features: dict) -> None:
    """A single prediction should be a float in a sane range."""

    predictor = DurationPredictor().load()

    prediction = predictor.predict_one(sample_features)

    assert isinstance(prediction, float)
    assert 0 < prediction <= 120


def test_predict_one_is_deterministic(sample_features: dict) -> None:
    """The same input should produce the same prediction."""

    predictor = DurationPredictor().load()

    prediction_one = predictor.predict_one(sample_features)
    prediction_two = predictor.predict_one(sample_features)

    assert prediction_one == prediction_two


def test_predict_batch() -> None:
    """Batch prediction should return one float per input."""

    predictor = DurationPredictor().load()

    predictions = predictor.predict_batch(
        [
            {
                "PU_DO": "74_42",
                "trip_distance": 2.5,
            },
            {
                "PU_DO": "75_41",
                "trip_distance": 4.1,
            },
        ]
    )

    assert len(predictions) == 2
    assert all(isinstance(value, float) for value in predictions)
    assert all(0 < value <= 120 for value in predictions)
