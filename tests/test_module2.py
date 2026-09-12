import matplotlib

matplotlib.use("Agg")

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression

from prodml import dvc_pipeline, export, registry, train, training


def mock_mlflow_run(run_id="test-run"):
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_run.__enter__.return_value = mock_run
    mock_run.__exit__.return_value = None
    return mock_run


# ---------------------------------------------------------------------------
# Metadata / helper functions
# ---------------------------------------------------------------------------


def test_get_git_commit():
    with patch(
        "prodml.training.subprocess.check_output",
        return_value="abc123\n",
    ):
        assert training.get_git_commit() == "abc123"


def test_get_git_commit_without_git():
    with patch(
        "prodml.training.subprocess.check_output",
        side_effect=FileNotFoundError,
    ):
        assert training.get_git_commit() == "unknown"


def test_get_git_commit_command_error():
    with patch(
        "prodml.training.subprocess.check_output",
        side_effect=__import__("subprocess").CalledProcessError(1, "git"),
    ):
        assert training.get_git_commit() == "unknown"


def test_get_author():
    with patch(
        "prodml.training.subprocess.check_output",
        return_value="MahmoudOsama20\n",
    ):
        assert training.get_author() == "MahmoudOsama20"


def test_get_author_empty():
    with patch(
        "prodml.training.subprocess.check_output",
        return_value="\n",
    ):
        assert training.get_author() == "unknown"


def test_get_author_without_git():
    with patch(
        "prodml.training.subprocess.check_output",
        side_effect=FileNotFoundError,
    ):
        assert training.get_author() == "unknown"


def test_get_data_hash(tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_bytes(b"hello world")

    result = training.get_data_hash(data_file)

    assert isinstance(result, str)
    assert len(result) == 64


def test_get_dvc_data_hash(tmp_path):
    data_file = tmp_path / "data.csv"
    Path(f"{data_file}.dvc").write_text(
        """
outs:
  - md5: abc123def456
    size: 100
""",
        encoding="utf-8",
    )

    assert training.get_dvc_data_hash(data_file) == "abc123def456"


def test_get_dvc_data_hash_missing(tmp_path):
    data_file = tmp_path / "missing.csv"
    assert training.get_dvc_data_hash(data_file) == "unknown"


def test_get_dvc_data_hash_empty_outs(tmp_path):
    data_file = tmp_path / "data.csv"
    Path(f"{data_file}.dvc").write_text(
        "outs: []\n",
        encoding="utf-8",
    )
    assert training.get_dvc_data_hash(data_file) == "unknown"


def test_common_tags():
    tags = training.common_tags(
        framework="TestFramework",
        git_commit="git123",
        data_version="data123",
        dvc_data_hash="dvc123",
        author="tester",
    )

    assert tags == {
        "git_commit": "git123",
        "data_version": "data123",
        "dvc_data_hash": "dvc123",
        "author": "tester",
        "framework": "TestFramework",
    }


def test_log_common_params():
    with patch("prodml.training.mlflow.log_params") as log_params:
        training.log_common_params(test_size=0.2, random_state=42)

    log_params.assert_called_once_with({"test_size": 0.2, "random_state": 42})


def test_setup_mlflow():
    with (
        patch("prodml.training.mlflow.set_tracking_uri") as set_uri,
        patch("prodml.training.mlflow.set_experiment") as set_experiment,
    ):
        training.setup_mlflow()

    set_uri.assert_called_once()
    set_experiment.assert_called_once_with(training.EXPERIMENT_NAME)


def test_create_requirements_file(tmp_path):
    path = tmp_path / "requirements.txt"

    with patch(
        "prodml.training.subprocess.check_output",
        return_value="numpy==1.0\npandas==2.0\n",
    ):
        training.create_requirements_file(path)

    assert path.read_text(encoding="utf-8") == "numpy==1.0\npandas==2.0\n"


# ---------------------------------------------------------------------------
# Evaluation / artifacts
# ---------------------------------------------------------------------------


def test_evaluate_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    metrics = training.evaluate_predictions(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(np.mean(np.abs(y_true - y_pred)))
    assert metrics["rmse"] > 0
    assert metrics["r2"] < 1


def test_evaluate_predictions_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    metrics = training.evaluate_predictions(y_true, y_true)

    assert metrics["mae"] == pytest.approx(0)
    assert metrics["rmse"] == pytest.approx(0)
    assert metrics["r2"] == pytest.approx(1)


def test_model_size_mb(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(b"x" * 1024)

    assert training.model_size_mb(path) > 0


def test_model_size_mb_with_torch_model(tmp_path):
    model = training.DurationMLP(input_size=3, hidden_size=8)
    assert model.network[-1].out_features == 1

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    assert training.model_size_mb(path) > 0


def test_duration_mlp_forward():
    model = training.DurationMLP(input_size=3, hidden_size=8)

    output = model(torch.randn(5, 3))

    assert output.shape == (5, 1)


def test_save_residual_plot(tmp_path):
    path = tmp_path / "residuals.png"

    training.save_residual_plot(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 1.9, 3.2]),
        path,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_save_feature_importance_plot(tmp_path):
    path = tmp_path / "importance.png"

    training.save_feature_importance_plot(
        ["a", "b", "c"],
        np.array([0.1, -0.5, 0.2]),
        path,
        "Test Importance",
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_log_evaluation_artifacts(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==1.0\n", encoding="utf-8")

    model = LinearRegression().fit(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([1.0, 2.0, 3.0]),
    )

    with (
        patch("prodml.training.mlflow.log_artifact") as log_artifact,
        patch("prodml.training.mlflow.log_metric") as log_metric,
    ):
        size = training.log_evaluation_artifacts(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 1.9, 3.2]),
            ["a", "b", "c"],
            np.array([0.1, 0.2, 0.3]),
            requirements,
            tmp_path / "artifacts",
            model,
            "Test Model",
        )

    assert size > 0
    assert log_artifact.call_count == 3
    log_metric.assert_called_once()


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------


def test_train_linear_regression(tmp_path):
    X_train = csr_matrix(
        np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
            ]
        )
    )
    X_val = csr_matrix(
        np.array(
            [
                [1.5, 2.5],
                [3.5, 4.5],
                [5.5, 6.5],
            ]
        )
    )
    y_train = pd.Series([3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
    y_val = pd.Series([4.0, 8.0, 12.0])

    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy==1.0\n", encoding="utf-8")

    with (
        patch(
            "prodml.training.mlflow.start_run",
            return_value=mock_mlflow_run("linear-test-run"),
        ),
        patch("prodml.training.mlflow.set_tags"),
        patch("prodml.training.mlflow.log_params"),
        patch("prodml.training.mlflow.log_metrics"),
        patch("prodml.training.mlflow.log_artifact"),
        patch("prodml.training.mlflow.sklearn.log_model"),
        patch("prodml.training.create_requirements_file"),
    ):
        result = training.train_linear_regression(
            X_train,
            X_val,
            y_train,
            y_val,
            ["feature_1", "feature_2"],
            requirements,
            tmp_path / "artifacts",
            0.2,
            42,
            "git123",
            "data123",
            "dvc123",
            "tester",
        )

    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert "train_duration_sec" in result


# ---------------------------------------------------------------------------
# XGBoost sweep
# ---------------------------------------------------------------------------


def test_train_xgboost_sweep_runs_ten_trials(tmp_path):
    X_train = csr_matrix(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))
    X_val = csr_matrix(np.array([[1.5, 2.5], [2.5, 3.5]]))
    y_train = pd.Series([3.0, 5.0, 7.0])
    y_val = pd.Series([4.0, 6.0])

    class FakeXGB:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.feature_importances_ = np.array([0.4, 0.6])

        def fit(self, *args, **kwargs):
            return self

        def predict(self, X):
            return np.array([4.0, 6.0])[: X.shape[0]]

    runs = [mock_mlflow_run(f"run-{i}") for i in range(11)]

    with (
        patch("prodml.training.mlflow.xgboost.autolog"),
        patch("prodml.training.XGBRegressor", FakeXGB),
        patch(
            "prodml.training.mlflow.start_run",
            side_effect=runs,
        ),
        patch("prodml.training.mlflow.set_tags"),
        patch("prodml.training.mlflow.log_params"),
        patch("prodml.training.mlflow.log_param"),
        patch("prodml.training.mlflow.log_metrics"),
        patch("prodml.training.mlflow.log_artifact"),
        patch("prodml.training.mlflow.xgboost.log_model"),
        patch("prodml.training.log_evaluation_artifacts"),
    ):
        results = training.train_xgboost_sweep(
            X_train,
            X_val,
            y_train,
            y_val,
            ["f1", "f2"],
            tmp_path / "requirements.txt",
            tmp_path / "artifacts",
            0.2,
            42,
            "git123",
            "data123",
            "dvc123",
            "tester",
        )

    assert len(results) == 10
    assert [r["trial"] for r in results] == list(range(1, 11))


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------


def test_train_mlp_with_mocked_mlflow(tmp_path):
    X_train = csr_matrix(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]))
    X_val = csr_matrix(np.array([[1.5, 2.5], [3.5, 4.5]]))
    y_train = pd.Series([3.0, 5.0, 7.0, 9.0])
    y_val = pd.Series([4.0, 8.0])

    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.6.0\n", encoding="utf-8")

    with (
        patch(
            "prodml.training.mlflow.start_run",
            return_value=mock_mlflow_run("mlp-test-run"),
        ),
        patch("prodml.training.mlflow.set_tags"),
        patch("prodml.training.mlflow.log_params"),
        patch("prodml.training.mlflow.log_metrics"),
        patch("prodml.training.mlflow.log_artifact"),
        patch("prodml.training.mlflow.pytorch.log_model"),
    ):
        result = training.train_mlp(
            X_train,
            X_val,
            y_train,
            y_val,
            ["f1", "f2"],
            requirements,
            tmp_path / "artifacts",
            0.2,
            42,
            "git123",
            "data123",
            "dvc123",
            "tester",
        )

    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert result["r2"] <= 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _registry_client(
    candidate_metric, production_metric=None, candidate_registered=True
):
    candidate = MagicMock()
    candidate.data.metrics = {"mae": candidate_metric}

    client = MagicMock()
    client.get_run.side_effect = [candidate]

    production_version = MagicMock()
    production_version.name = registry.MODEL_NAME
    production_version.current_stage = "Production"
    production_version.run_id = "production-run"
    production_version.version = "1"

    if production_metric is not None:
        production_run = MagicMock()
        production_run.data.metrics = {"mae": production_metric}
        client.get_run.side_effect = [candidate, production_run]

    client.search_model_versions.side_effect = [
        [production_version],
        (
            [
                MagicMock(
                    name=registry.MODEL_NAME,
                    current_stage="None",
                    run_id="candidate-run",
                    version="2",
                )
            ]
            if candidate_registered
            else []
        ),
    ]
    return client


def test_promote_if_better_rejects_worse_candidate():
    candidate = MagicMock()
    candidate.data.metrics = {"mae": 10.0}

    production_version = MagicMock()
    production_version.name = registry.MODEL_NAME
    production_version.current_stage = "Production"
    production_version.run_id = "production-run"
    production_version.version = "1"

    production_run = MagicMock()
    production_run.data.metrics = {"mae": 5.0}

    client = MagicMock()
    client.get_run.side_effect = [candidate, production_run]
    client.search_model_versions.return_value = [production_version]

    with (
        patch("prodml.registry.MlflowClient", return_value=client),
        patch("prodml.registry.mlflow.set_tracking_uri"),
    ):
        assert registry.promote_if_better("candidate-run") is False


def test_promote_if_better_promotes_better_candidate():
    candidate = MagicMock()
    candidate.data.metrics = {"mae": 4.0}

    production_version = MagicMock()
    production_version.name = registry.MODEL_NAME
    production_version.current_stage = "Production"
    production_version.run_id = "production-run"
    production_version.version = "1"

    candidate_version = MagicMock()
    candidate_version.name = registry.MODEL_NAME
    candidate_version.current_stage = "None"
    candidate_version.run_id = "candidate-run"
    candidate_version.version = "2"

    production_run = MagicMock()
    production_run.data.metrics = {"mae": 5.0}

    client = MagicMock()
    client.get_run.side_effect = [candidate, production_run]
    client.search_model_versions.side_effect = [
        [production_version],
        [candidate_version],
    ]

    with (
        patch("prodml.registry.MlflowClient", return_value=client),
        patch("prodml.registry.mlflow.set_tracking_uri"),
    ):
        assert registry.promote_if_better("candidate-run") is True

    assert client.transition_model_version_stage.call_count == 2


def test_promote_if_better_missing_candidate_metric():
    candidate = MagicMock()
    candidate.data.metrics = {}

    client = MagicMock()
    client.get_run.return_value = candidate

    with (
        patch("prodml.registry.MlflowClient", return_value=client),
        patch("prodml.registry.mlflow.set_tracking_uri"),
        pytest.raises(ValueError),
    ):
        registry.promote_if_better("candidate-run")


def test_promote_if_better_missing_production():
    candidate = MagicMock()
    candidate.data.metrics = {"mae": 4.0}

    client = MagicMock()
    client.get_run.return_value = candidate
    client.search_model_versions.return_value = []

    with (
        patch("prodml.registry.MlflowClient", return_value=client),
        patch("prodml.registry.mlflow.set_tracking_uri"),
        pytest.raises(RuntimeError),
    ):
        registry.promote_if_better("candidate-run")


def test_promote_if_better_missing_production_metric():
    candidate = MagicMock()
    candidate.data.metrics = {"mae": 4.0}

    production_version = MagicMock()
    production_version.name = registry.MODEL_NAME
    production_version.current_stage = "Production"
    production_version.run_id = "production-run"

    production_run = MagicMock()
    production_run.data.metrics = {}

    client = MagicMock()
    client.get_run.side_effect = [candidate, production_run]
    client.search_model_versions.return_value = [production_version]

    with (
        patch("prodml.registry.MlflowClient", return_value=client),
        patch("prodml.registry.mlflow.set_tracking_uri"),
        pytest.raises(ValueError),
    ):
        registry.promote_if_better("candidate-run")


def test_promote_if_better_r2_higher_is_better():
    candidate = MagicMock()
    candidate.data.metrics = {"r2": 0.9}

    production_version = MagicMock()
    production_version.name = registry.MODEL_NAME
    production_version.current_stage = "Production"
    production_version.run_id = "production-run"
    production_version.version = "1"

    candidate_version = MagicMock()
    candidate_version.name = registry.MODEL_NAME
    candidate_version.current_stage = "None"
    candidate_version.run_id = "candidate-run"
    candidate_version.version = "2"

    production_run = MagicMock()
    production_run.data.metrics = {"r2": 0.8}

    client = MagicMock()
    client.get_run.side_effect = [candidate, production_run]
    client.search_model_versions.side_effect = [
        [production_version],
        [candidate_version],
    ]

    with (
        patch("prodml.registry.MlflowClient", return_value=client),
        patch("prodml.registry.mlflow.set_tracking_uri"),
    ):
        assert registry.promote_if_better("candidate-run", metric="r2") is True


# ---------------------------------------------------------------------------
# DVC pipeline
# ---------------------------------------------------------------------------


def test_load_params(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        """
data:
  test_size: 0.2
  random_state: 42
training:
  n_estimators: 10
  max_depth: 2
  learning_rate: 0.1
""",
        encoding="utf-8",
    )

    with patch.object(dvc_pipeline, "PARAMS_PATH", params_file):
        params = dvc_pipeline.load_params()

    assert params["data"]["test_size"] == 0.2
    assert params["training"]["n_estimators"] == 10


def test_dvc_pipeline_prepare(tmp_path):
    raw_file = tmp_path / "input.csv"
    prepared_file = tmp_path / "prepared.csv"

    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.date_range("2026-01-01", periods=6, freq="h"),
            "lpep_dropoff_datetime": pd.date_range(
                "2026-01-01 00:10", periods=6, freq="h"
            ),
            "PULocationID": [1, 2, 3, 4, 5, 6],
            "DOLocationID": [6, 5, 4, 3, 2, 1],
            "trip_distance": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "fare_amount": [5, 6, 7, 8, 9, 10],
        }
    )
    df.to_csv(raw_file, index=False)

    with (
        patch.object(dvc_pipeline, "RAW_DATA", raw_file),
        patch.object(dvc_pipeline, "PREPARED_DATA", prepared_file),
    ):
        dvc_pipeline.prepare()

    assert prepared_file.exists()

    prepared = pd.read_csv(prepared_file)
    assert "duration" in prepared.columns
    assert "PU_DO" in prepared.columns


def test_dvc_pipeline_featurize(tmp_path):
    prepared_file = tmp_path / "prepared.csv"
    features_file = tmp_path / "features.pkl"

    pd.DataFrame(
        {
            "PU_DO": ["1_2", "2_3", "3_4", "4_5", "5_6"],
            "trip_distance": [1.0, 2.0, 3.0, 4.0, 5.0],
            "duration": [5.0, 10.0, 15.0, 20.0, 25.0],
        }
    ).to_csv(prepared_file, index=False)

    with (
        patch.object(dvc_pipeline, "PREPARED_DATA", prepared_file),
        patch.object(dvc_pipeline, "FEATURES_DATA", features_file),
        patch.object(
            dvc_pipeline,
            "load_params",
            return_value={
                "data": {
                    "test_size": 0.2,
                    "random_state": 42,
                },
                "training": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                },
            },
        ),
    ):
        dvc_pipeline.featurize()

    assert features_file.exists()

    with open(features_file, "rb") as file:
        features = pickle.load(file)

    assert "X_train" in features
    assert "X_val" in features
    assert "y_train" in features
    assert "y_val" in features


def test_dvc_pipeline_train_and_evaluate(tmp_path):
    features_file = tmp_path / "features.pkl"
    model_file = tmp_path / "model.pkl"
    metrics_file = tmp_path / "metrics.json"

    data = {
        "X_train": np.array(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]
        ),
        "X_val": np.array([[1.5, 2.5], [3.5, 4.5]]),
        "y_train": np.array([3.0, 5.0, 7.0, 9.0, 11.0]),
        "y_val": np.array([4.0, 8.0]),
    }

    with open(features_file, "wb") as file:
        pickle.dump(data, file)

    with (
        patch.object(dvc_pipeline, "FEATURES_DATA", features_file),
        patch.object(dvc_pipeline, "MODEL_PATH", model_file),
        patch.object(dvc_pipeline, "METRICS_PATH", metrics_file),
        patch.object(
            dvc_pipeline,
            "load_params",
            return_value={
                "data": {"test_size": 0.2, "random_state": 42},
                "training": {
                    "n_estimators": 5,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                },
            },
        ),
    ):
        dvc_pipeline.train()
        dvc_pipeline.evaluate()

    assert model_file.exists()
    assert metrics_file.exists()

    metrics = __import__("json").loads(metrics_file.read_text(encoding="utf-8"))
    assert {"mae", "rmse", "r2"} <= metrics.keys()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_to_onnx(tmp_path):
    model = LinearRegression().fit(
        np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        np.array([3.0, 5.0, 7.0]),
    )

    model_path = tmp_path / "model.pkl"
    onnx_path = tmp_path / "model.onnx"

    with open(model_path, "wb") as file:
        pickle.dump({"model": model}, file)

    export.export_to_onnx(model_path, onnx_path)

    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# train.py orchestration
# ---------------------------------------------------------------------------


def test_train_main_calls_run_experiments():
    df = pd.DataFrame(
        {
            "duration": [10.0, 20.0, 30.0, 40.0],
            "PU_DO": ["1_2", "2_3", "3_4", "4_5"],
            "trip_distance": [1.0, 2.0, 3.0, 4.0],
        }
    )

    X_train = np.array([[1.0, 2.0], [2.0, 3.0]])
    X_val = np.array([[3.0, 4.0], [4.0, 5.0]])
    y_train = np.array([10.0, 20.0])
    y_val = np.array([30.0, 40.0])

    with (
        patch("prodml.train.load_data", return_value=df),
        patch("prodml.train.prepare_data", return_value=df),
        patch(
            "prodml.train.split_data",
            return_value=(df.iloc[:2], df.iloc[2:], y_train, y_val),
        ),
        patch(
            "prodml.train.vectorize_features",
            return_value=(
                csr_matrix(X_train),
                csr_matrix(X_val),
                MagicMock(feature_names_=["f1", "f2"]),
            ),
        ),
        patch("prodml.train.run_experiments") as run_experiments,
    ):
        train.main()

    run_experiments.assert_called_once()

    kwargs = run_experiments.call_args.kwargs
    assert kwargs["feature_names"] == ["f1", "f2"]
    assert kwargs["test_size"] == train.settings.test_size
    assert kwargs["random_state"] == train.settings.random_state


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


def test_run_experiments_calls_all_families(tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("x\n1\n", encoding="utf-8")

    X_train = csr_matrix(np.array([[1.0, 2.0], [2.0, 3.0]]))
    X_val = csr_matrix(np.array([[1.5, 2.5]]))
    y_train = pd.Series([3.0, 5.0])
    y_val = pd.Series([4.0])

    linear_result = {"mae": 1.0, "rmse": 1.0, "r2": 0.5}
    xgb_result = [{"trial": 1, "mae": 0.9, "rmse": 1.0, "r2": 0.6}]
    mlp_result = {"mae": 0.8, "rmse": 0.9, "r2": 0.7}

    with (
        patch("prodml.training.setup_mlflow"),
        patch("prodml.training.get_git_commit", return_value="git123"),
        patch("prodml.training.get_author", return_value="tester"),
        patch("prodml.training.get_data_hash", return_value="data123"),
        patch("prodml.training.get_dvc_data_hash", return_value="dvc123"),
        patch("prodml.training.create_requirements_file"),
        patch(
            "prodml.training.train_linear_regression",
            return_value=linear_result,
        ) as linear,
        patch(
            "prodml.training.train_xgboost_sweep",
            return_value=xgb_result,
        ) as xgb,
        patch(
            "prodml.training.train_mlp",
            return_value=mlp_result,
        ) as mlp,
    ):
        training.run_experiments(
            X_train,
            X_val,
            y_train,
            y_val,
            ["f1", "f2"],
            data_path,
            0.2,
            42,
            tmp_path / "artifacts",
        )

    linear.assert_called_once()
    xgb.assert_called_once()
    mlp.assert_called_once()

    assert linear.call_args.args[-4:] == (
        "git123",
        "data123",
        "dvc123",
        "tester",
    )
