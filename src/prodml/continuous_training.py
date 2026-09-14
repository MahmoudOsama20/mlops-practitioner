from __future__ import annotations

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "ride-duration-predictor"


def get_production_version(client: MlflowClient):
    """Return the current Production model version."""
    versions = [
        version
        for version in client.search_model_versions()
        if version.name == MODEL_NAME and version.current_stage == "Production"
    ]

    return versions[0] if versions else None


def get_run_mae(client: MlflowClient, run_id: str) -> float:
    """Return MAE for an MLflow run."""
    run = client.get_run(run_id)
    mae = run.data.metrics.get("mae")

    if mae is None:
        raise ValueError(f"Run '{run_id}' does not contain an 'mae' metric.")

    return float(mae)


def gate_candidate(
    candidate_mae: float,
    production_mae: float | None,
    margin: float,
) -> bool:
    """Return True when candidate beats Production by the required margin."""

    if production_mae is None:
        print("No Production model found.")
        print("Gate passed: candidate can become the initial Staging model.")
        return True

    required_mae = production_mae * (1.0 - margin)

    print(f"Production MAE : {production_mae:.6f}")
    print(f"Candidate MAE  : {candidate_mae:.6f}")
    print(f"Required MAE   : {required_mae:.6f}")
    print(f"Margin         : {margin:.2%}")

    if candidate_mae < required_mae:
        print("Gate passed: candidate is sufficiently better.")
        return True

    print("Gate rejected: candidate is not sufficiently better.")
    return False


def register_candidate(
    client: MlflowClient,
    run_id: str,
) -> str:
    """Register the candidate model and return its version."""

    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    return str(result.version)


def promote_to_staging(
    client: MlflowClient,
    version: str,
) -> None:
    """Promote candidate to Staging."""

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous-training model gate.")

    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow run ID for the candidate model.",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=float(os.getenv("CT_PROMOTION_MARGIN", "0.01")),
        help="Required relative improvement in MAE.",
    )

    args = parser.parse_args()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    candidate_mae = get_run_mae(
        client,
        args.run_id,
    )

    production_version = get_production_version(client)

    production_mae = None

    if production_version is not None:
        production_mae = get_run_mae(
            client,
            production_version.run_id,
        )

    if not gate_candidate(
        candidate_mae=candidate_mae,
        production_mae=production_mae,
        margin=args.margin,
    ):
        print("CT result: candidate rejected.")
        return

    candidate_version = register_candidate(
        client,
        args.run_id,
    )

    promote_to_staging(
        client,
        candidate_version,
    )

    print(
        f"CT result: candidate promoted to Staging " f"as version {candidate_version}."
    )


if __name__ == "__main__":
    main()
