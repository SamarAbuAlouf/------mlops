"""
src/register_model.py
Logs the champion model, metrics, and preprocessor to MLflow
and registers it to the MLflow Model Registry with stage 'Production'.

Requirement 5:
- Track runs, parameters, metrics, and artifacts with MLflow
- Register the chosen model with version and stage
- Allows service to load model from registry
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    CONFIG,
    FEATURE_NAMES_PATH,
    MLFLOW_MODEL_NAME,
    MLFLOW_URI,
    MODEL_PATH,
    MODEL_THRESHOLD,
    MODEL_VERSION,
    PREPROCESSOR_PATH,
)
from src.logger import get_logger

logger = get_logger(__name__)


def register_champion_model():
    """Register the trained HistGradientBoosting model into MLflow."""
    logger.info("Setting MLflow Tracking URI: %s", MLFLOW_URI)
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    exp_name = CONFIG["mlflow"]["experiment_name"]
    logger.info("Ensuring experiment exists: %s", exp_name)
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        exp_id = client.create_experiment(exp_name)
    else:
        exp_id = experiment.experiment_id

    # Load results metrics
    results_path = Path("artifacts/reports/model_results.json")
    metrics = {}
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            raw_metrics = json.load(f)
            # Flatten numeric metrics
            for k, v in raw_metrics.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)

    # Load model and preprocessor
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"champion-{MODEL_VERSION}"
    ) as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run: %s", run_id)

        # Log parameters
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("version", MODEL_VERSION)
        mlflow.log_param("threshold", MODEL_THRESHOLD)
        if hasattr(model, "get_params"):
            params = model.get_params()
            for p_name, p_val in list(params.items())[:20]:
                mlflow.log_param(p_name, str(p_val))

        # Log metrics
        if metrics:
            logger.info("Logging %d metrics to MLflow", len(metrics))
            mlflow.log_metrics(metrics)

        # Log artifacts
        logger.info("Logging model artifacts to MLflow...")
        mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="preprocessor")
        mlflow.log_artifact(str(FEATURE_NAMES_PATH), artifact_path="metadata")
        if Path("artifacts/reports/model_results.md").exists():
            mlflow.log_artifact(
                "artifacts/reports/model_results.md", artifact_path="reports"
            )

        # Log model & register
        logger.info("Registering model as '%s'...", MLFLOW_MODEL_NAME)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        logger.info("Model registered: %s", model_info.model_uri)

        # Transition model to Production stage
        latest_versions = client.get_latest_versions(MLFLOW_MODEL_NAME)
        if latest_versions:
            latest_version = latest_versions[-1].version
            logger.info(
                "Transitioning model version %s to 'Production' stage", latest_version
            )
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Model successfully transitioned to Production!")

    return run_id


if __name__ == "__main__":
    register_champion_model()
