from prodml.predict import DurationPredictor


def test_predictor_loads() -> None:
    predictor = DurationPredictor()

    predictor.load()

    assert predictor.model is not None
    assert predictor.vectorizer is not None


def test_predict_one() -> None:
    predictor = DurationPredictor().load()

    prediction = predictor.predict_one(
        {
            "PU_DO": "74_42",
            "trip_distance": 2.5,
        }
    )

    assert isinstance(prediction, float)


def test_predict_batch() -> None:
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
