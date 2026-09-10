import hashlib
import pickle
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from torch import nn
from xgboost import XGBRegressor

from prodml.config import settings

EXPERIMENT_NAME = "prodml-duration-prediction"


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_author() -> str:
    """Return the configured Git author name."""
    try:
        author = subprocess.check_output(
            ["git", "config", "user.name"],
            text=True,
        ).strip()

        return author or "unknown"

    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_data_hash(path: Path) -> str:
    """Return a SHA-256 hash of the dataset file."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_dvc_data_hash(path: Path) -> str:
    """Return the DVC hash recorded for the dataset."""
    import yaml

    dvc_path = Path(f"{path}.dvc")

    if not dvc_path.exists():
        return "unknown"

    with open(dvc_path, encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    outs = metadata.get("outs", [])

    if not outs:
        return "unknown"

    return str(outs[0].get("md5", "unknown"))


def create_requirements_file(path: Path) -> None:
    """Capture the exact Python environment used for training."""
    requirements = subprocess.check_output(
        ["python", "-m", "pip", "freeze"],
        text=True,
    )

    path.write_text(
        requirements,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def setup_mlflow() -> None:
    """Configure the MLflow tracking server and experiment."""
    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri,
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME,
    )


def common_tags(
    *,
    framework: str,
    git_commit: str,
    data_version: str,
    dvc_data_hash: str,
    author: str,
) -> dict[str, str]:
    """Build common MLflow tags."""
    return {
        "git_commit": git_commit,
        "data_version": data_version,
        "dvc_data_hash": dvc_data_hash,
        "author": author,
        "framework": framework,
    }


def log_common_params(
    *,
    test_size: float,
    random_state: int,
) -> None:
    """Log parameters shared by all model families."""
    mlflow.log_params(
        {
            "test_size": test_size,
            "random_state": random_state,
        }
    )


# ---------------------------------------------------------------------------
# Evaluation / artifacts
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate the required regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def model_size_mb(model) -> float:
    """Estimate serialized model size in megabytes."""
    payload = pickle.dumps(model)
    return len(payload) / (1024**2)


def save_residual_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    """Create and save a residual plot."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.35, s=10)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted duration (minutes)")
    plt.ylabel("Residual (minutes)")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance_plot(
    feature_names: list[str],
    importances: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Create a top-feature importance plot."""
    importances = np.asarray(importances)

    n_features = min(20, len(feature_names))

    indices = np.argsort(np.abs(importances))[-n_features:]

    names = np.asarray(feature_names)[indices]
    values = importances[indices]

    plt.figure(figsize=(10, 7))
    plt.barh(names, values)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def log_evaluation_artifacts(
    y_val,
    y_pred,
    feature_names: list[str],
    importances: np.ndarray,
    requirements_path: Path,
    artifact_dir: Path,
    model,
    model_framework: str,
) -> float:
    """Log plots, requirements and model artifacts."""

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    residual_path = artifact_dir / "residual_plot.png"
    importance_path = artifact_dir / "feature_importance.png"

    save_residual_plot(
        y_val,
        y_pred,
        residual_path,
    )

    save_feature_importance_plot(
        feature_names,
        importances,
        importance_path,
        f"{model_framework} Feature Importance",
    )

    mlflow.log_artifact(str(residual_path))
    mlflow.log_artifact(str(importance_path))
    mlflow.log_artifact(str(requirements_path))

    size_mb = model_size_mb(model)

    mlflow.log_metric(
        "model_size_mb",
        size_mb,
    )

    return size_mb


# ---------------------------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------------------------


def train_linear_regression(
    X_train: csr_matrix,
    X_val: csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    feature_names: list[str],
    requirements_path: Path,
    artifact_dir: Path,
    test_size: float,
    random_state: int,
    git_commit: str,
    data_version: str,
    dvc_data_hash: str,
    author: str,
) -> dict[str, float]:
    """Train and track the Linear Regression baseline."""

    with mlflow.start_run(
        run_name="linear-regression",
    ):
        mlflow.set_tags(
            common_tags(
                framework="scikit-learn",
                git_commit=git_commit,
                data_version=data_version,
                dvc_data_hash=dvc_data_hash,
                author=author,
            )
        )

        mlflow.log_params(
            {
                "fit_intercept": True,
                "copy_X": True,
                "n_jobs": None,
            }
        )

        log_common_params(
            test_size=test_size,
            random_state=random_state,
        )

        model = LinearRegression()

        start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        train_duration = time.perf_counter() - start

        y_pred = model.predict(X_val)

        metrics = evaluate_predictions(
            y_val,
            y_pred,
        )

        metrics["train_duration_sec"] = train_duration

        mlflow.log_metrics(metrics)

        importances = model.coef_

        log_evaluation_artifacts(
            y_val,
            y_pred,
            feature_names,
            importances,
            requirements_path,
            artifact_dir,
            model,
            "Linear Regression",
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
        )

        return metrics


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def train_xgboost_sweep(
    X_train: csr_matrix,
    X_val: csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    feature_names: list[str],
    requirements_path: Path,
    artifact_dir: Path,
    test_size: float,
    random_state: int,
    git_commit: str,
    data_version: str,
    dvc_data_hash: str,
    author: str,
) -> list[dict[str, float]]:
    """Run a 10-trial nested XGBoost hyperparameter sweep."""

    # Required by the handbook: demonstrate framework autologging.
    mlflow.xgboost.autolog(
        log_models=True,
        silent=True,
    )

    results = []

    trials = [
        {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
        {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
        {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
        {
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ]

    # One parent run makes the sweep easy to identify in the UI.
    with mlflow.start_run(
        run_name="xgboost-sweep",
    ):
        mlflow.set_tags(
            common_tags(
                framework="xgboost",
                git_commit=git_commit,
                data_version=data_version,
                dvc_data_hash=dvc_data_hash,
                author=author,
            )
        )

        mlflow.log_params(
            {
                "sweep_trials": len(trials),
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        for trial_number, params in enumerate(
            trials,
            start=1,
        ):
            with mlflow.start_run(
                run_name=f"xgboost-trial-{trial_number:02d}",
                nested=True,
            ):
                mlflow.set_tags(
                    common_tags(
                        framework="xgboost",
                        git_commit=git_commit,
                        data_version=data_version,
                        dvc_data_hash=dvc_data_hash,
                        author=author,
                    )
                    | {
                        "sweep": "xgboost",
                        "trial": str(trial_number),
                    }
                )

                mlflow.log_params(params)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("random_state", random_state)

                model = XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=random_state,
                    tree_method="hist",
                    **params,
                )

                start = time.perf_counter()

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                train_duration = time.perf_counter() - start

                y_pred = model.predict(X_val)

                metrics = evaluate_predictions(
                    y_val,
                    y_pred,
                )

                metrics["train_duration_sec"] = train_duration

                mlflow.log_metrics(metrics)

                importances = model.feature_importances_

                log_evaluation_artifacts(
                    y_val,
                    y_pred,
                    feature_names,
                    importances,
                    requirements_path,
                    artifact_dir / f"xgboost_trial_{trial_number:02d}",
                    model,
                    "XGBoost",
                )

                # Explicit model logging in addition to autologging
                # makes the requirement unambiguous.
                mlflow.xgboost.log_model(
                    model,
                    name="model",
                )

                results.append(
                    {
                        "trial": trial_number,
                        **metrics,
                        **params,
                    }
                )

    return results


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------


class DurationMLP(nn.Module):
    """Small feed-forward neural network for duration prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def train_mlp(
    X_train: csr_matrix,
    X_val: csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    feature_names: list[str],
    requirements_path: Path,
    artifact_dir: Path,
    test_size: float,
    random_state: int,
    git_commit: str,
    data_version: str,
    dvc_data_hash: str,
    author: str,
) -> dict[str, float]:
    """Train and track a small PyTorch MLP."""

    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset is small enough to safely densify.
    X_train_dense = X_train.toarray().astype(
        np.float32,
        copy=False,
    )

    X_val_dense = X_val.toarray().astype(
        np.float32,
        copy=False,
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(dtype=np.float32),
        dtype=torch.float32,
    ).view(-1, 1)

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_dense),
        y_train_tensor,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
    )

    model = DurationMLP(
        input_size=X_train.shape[1],
        hidden_size=64,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    with mlflow.start_run(
        run_name="pytorch-mlp",
    ):
        mlflow.set_tags(
            common_tags(
                framework="pytorch",
                git_commit=git_commit,
                data_version=data_version,
                dvc_data_hash=dvc_data_hash,
                author=author,
            )
            | {
                "device": str(device),
            }
        )

        mlflow.log_params(
            {
                "hidden_size": 64,
                "second_hidden_size": 32,
                "learning_rate": 0.001,
                "batch_size": 256,
                "epochs": 20,
                "optimizer": "Adam",
                "loss": "MSELoss",
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        start = time.perf_counter()

        model.train()

        for _ in range(20):
            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                predictions = model(features)

                loss = criterion(
                    predictions,
                    targets,
                )

                loss.backward()
                optimizer.step()

        train_duration = time.perf_counter() - start

        model.eval()

        with torch.no_grad():
            validation_features = torch.from_numpy(X_val_dense).to(device)

            y_pred = model(validation_features).cpu().numpy().reshape(-1)

        metrics = evaluate_predictions(
            y_val,
            y_pred,
        )

        metrics["train_duration_sec"] = train_duration

        mlflow.log_metrics(metrics)

        # Mean absolute weight contribution of each input feature.
        first_layer = model.network[0]
        importances = first_layer.weight.detach().cpu().numpy()

        importances = np.mean(
            np.abs(importances),
            axis=0,
        )

        artifact_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        residual_path = artifact_dir / "mlp_residual_plot.png"
        importance_path = artifact_dir / "mlp_feature_importance.png"
        model_path = artifact_dir / "mlp_model.pt"

        save_residual_plot(
            y_val,
            y_pred,
            residual_path,
        )

        save_feature_importance_plot(
            feature_names,
            importances,
            importance_path,
            "PyTorch MLP Feature Importance",
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": X_train.shape[1],
                "hidden_size": 64,
            },
            model_path,
        )

        mlflow.log_artifact(str(residual_path))
        mlflow.log_artifact(str(importance_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(requirements_path))

        # input_example = X_train[:1].toarray().astype("float32")

        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )

        model_size = model_path.stat().st_size / (1024**2)

        mlflow.log_metric(
            "model_size_mb",
            model_size,
        )

        return metrics


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiments(
    X_train: csr_matrix,
    X_val: csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    feature_names: list[str],
    data_path: Path,
    test_size: float,
    random_state: int,
    artifact_dir: Path,
) -> None:
    """Run and track all required model families."""

    setup_mlflow()

    git_commit = get_git_commit()
    author = get_author()
    data_version = get_data_hash(data_path)
    dvc_data_hash = get_dvc_data_hash(data_path)

    requirements_path = artifact_dir / "requirements.txt"

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    create_requirements_file(
        requirements_path,
    )

    print("\n" + "=" * 60)
    print("MLflow Experiment")
    print("=" * 60)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Git commit: {git_commit}")
    print(f"Data hash:  {data_version}")
    print(f"Author:     {author}")
    print("=" * 60)

    print("\n[1/3] Training Linear Regression...")

    linear_metrics = train_linear_regression(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        requirements_path,
        artifact_dir,
        test_size,
        random_state,
        git_commit,
        data_version,
        dvc_data_hash,
        author,
    )

    print(
        f"Linear Regression → "
        f"MAE={linear_metrics['mae']:.3f}, "
        f"RMSE={linear_metrics['rmse']:.3f}, "
        f"R²={linear_metrics['r2']:.3f}"
    )

    print("\n[2/3] Running XGBoost sweep (10 trials)...")

    xgb_results = train_xgboost_sweep(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        requirements_path,
        artifact_dir,
        test_size,
        random_state,
        git_commit,
        data_version,
        dvc_data_hash,
        author,
    )

    best_xgb = min(
        xgb_results,
        key=lambda result: result["mae"],
    )

    print(
        f"Best XGBoost trial → "
        f"Trial={best_xgb['trial']}, "
        f"MAE={best_xgb['mae']:.3f}, "
        f"RMSE={best_xgb['rmse']:.3f}, "
        f"R²={best_xgb['r2']:.3f}"
    )

    print("\n[3/3] Training PyTorch MLP...")

    mlp_metrics = train_mlp(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        requirements_path,
        artifact_dir,
        test_size,
        random_state,
        git_commit,
        data_version,
        dvc_data_hash,
        author,
    )

    print(
        f"PyTorch MLP → "
        f"MAE={mlp_metrics['mae']:.3f}, "
        f"RMSE={mlp_metrics['rmse']:.3f}, "
        f"R²={mlp_metrics['r2']:.3f}"
    )

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("Open http://localhost:5000 to inspect the runs.")
    print("=" * 60)
