from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from prodml.config import settings

MODEL_NAME = "ride-duration-predictor"


def promote_if_better(
    candidate_run_id: str,
    metric: str = "mae",
) -> bool:
    """
    Promote a candidate model to Production only if it
    performs better than the current Production model.

    For MAE, lower is better.
    """

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    # Get candidate run
    candidate_run = client.get_run(candidate_run_id)
    candidate_metric = candidate_run.data.metrics.get(metric)

    if candidate_metric is None:
        raise ValueError(
            f"Candidate run '{candidate_run_id}' does not contain metric '{metric}'."
        )

    # Find current Production model
    production_versions = [
        version
        for version in client.search_model_versions()
        if version.name == MODEL_NAME and version.current_stage == "Production"
    ]

    if not production_versions:
        raise RuntimeError(f"No Production version found for model '{MODEL_NAME}'.")

    production_version = production_versions[0]

    # Get metrics of the current Production run
    production_run = client.get_run(production_version.run_id)
    production_metric = production_run.data.metrics.get(metric)

    if production_metric is None:
        raise ValueError(
            f"Production run '{production_version.run_id}' "
            f"does not contain metric '{metric}'."
        )

    # MAE: lower is better
    if metric.lower() in {"mae", "rmse", "loss"}:
        is_better = candidate_metric < production_metric
    else:
        # R2 and similar metrics: higher is better
        is_better = candidate_metric > production_metric

    if not is_better:
        print(
            f"Candidate rejected: {metric}={candidate_metric:.6f} "
            f"is not better than Production "
            f"{metric}={production_metric:.6f}."
        )
        return False

    # Find the registered version belonging to the candidate run
    candidate_versions = [
        version
        for version in client.search_model_versions()
        if version.name == MODEL_NAME and version.run_id == candidate_run_id
    ]

    if not candidate_versions:
        raise RuntimeError(
            f"No registered version found for candidate run " f"'{candidate_run_id}'."
        )

    candidate_version = candidate_versions[0]

    # Move current Production model out of Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=production_version.version,
        stage="Archived",
    )

    # Promote candidate
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=candidate_version.version,
        stage="Production",
    )

    print(
        f"Candidate promoted: version {candidate_version.version} "
        f"({metric}={candidate_metric:.6f})"
    )

    return True
