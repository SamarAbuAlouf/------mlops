"""
src/predict.py
Inference layer — the only place that calls model.predict().

Loads the model once (singleton pattern), applies the threshold from config,
and returns a structured result dict.

Requirement 2: Load saved fitted objects — never re-fit at inference.
Requirement 5: Service loads model from registry or artifact store.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config import MODEL_PATH, MODEL_THRESHOLD, MODEL_VERSION, MLFLOW_URI, MLFLOW_MODEL_NAME
from src.logger import PredictionLogger, get_logger

logger = get_logger(__name__)
_pred_logger = PredictionLogger()

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_model = None
_loaded_model_version = MODEL_VERSION


def load_model(path: Path = MODEL_PATH):
    """Load (and cache) the trained model from disk."""
    global _model, _loaded_model_version
    if _model is None:
        logger.info("Loading model from %s", path)
        _model = joblib.load(path)
        _loaded_model_version = MODEL_VERSION
        logger.info("Model loaded: %s v%s", type(_model).__name__, _loaded_model_version)
    return _model


def try_load_from_mlflow() -> bool:
    """
    Attempt to load the registered Production model from MLflow.
    Falls back to local artifact if MLflow is unreachable.
    Returns True if MLflow model was loaded.
    """
    global _model, _loaded_model_version
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        if versions:
            v = versions[0]
            logger.info(
                "Loading model from MLflow registry: %s v%s (run_id=%s)",
                MLFLOW_MODEL_NAME, v.version, v.run_id,
            )
            _model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")
            _loaded_model_version = f"mlflow-v{v.version}"
            return True
    except Exception as exc:
        logger.warning("MLflow unreachable (%s) — using local model artifact.", exc)
    return False


def get_model_version() -> str:
    return _loaded_model_version


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def predict(X: np.ndarray, order_id: str | None = None) -> dict[str, Any]:
    """
    Run inference on a pre-transformed feature array.

    Returns:
        {
          "prediction": 0 | 1,
          "label": "on_time" | "late",
          "probability": float,
          "model_version": str,
          "threshold": float,
        }
    """
    model = load_model()
    threshold = MODEL_THRESHOLD

    t0 = time.perf_counter()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Log first row (single prediction path)
    _pred_logger.log(
        order_id=order_id,
        prediction=int(pred[0]),
        probability=float(proba[0]),
        model_version=get_model_version(),
        latency_ms=latency_ms,
    )

    return {
        "prediction": int(pred[0]),
        "label": "late" if pred[0] == 1 else "on_time",
        "probability": float(round(proba[0], 6)),
        "model_version": get_model_version(),
        "threshold": threshold,
    }


def predict_batch(X: np.ndarray) -> list[dict[str, Any]]:
    """Run inference on a batch. Returns a list of result dicts."""
    model = load_model()
    threshold = MODEL_THRESHOLD

    t0 = time.perf_counter()
    probas = model.predict_proba(X)[:, 1]
    preds = (probas >= threshold).astype(int)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Batch predict: %d rows | late=%.1f%% | latency=%.1fms",
        len(preds),
        100 * preds.mean(),
        latency_ms,
    )

    results = []
    for pred, prob in zip(preds, probas):
        results.append(
            {
                "prediction": int(pred),
                "label": "late" if pred == 1 else "on_time",
                "probability": float(round(prob, 6)),
                "model_version": get_model_version(),
                "threshold": threshold,
            }
        )
    return results
