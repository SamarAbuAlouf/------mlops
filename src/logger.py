"""
src/logger.py
Centralised logging setup.

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("hello")

Requirement 3: Use the logging library, not print statements.
Log levels, log format, log to file and to console.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Load config once at module level
_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config" / "config.yaml"


def _load_log_config() -> dict:
    """Load logging section from config.yaml."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("logging", {})
    except Exception:
        return {}


_log_cfg = _load_log_config()

LOG_LEVEL = os.getenv("LOG_LEVEL", _log_cfg.get("level", "INFO")).upper()
LOG_FORMAT = _log_cfg.get(
    "format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
LOG_DIR = _ROOT / _log_cfg.get("log_dir", "logs")
APP_LOG_FILE = _ROOT / _log_cfg.get("app_log_file", "logs/app.log")
PREDICTION_LOG_FILE = _ROOT / _log_cfg.get(
    "prediction_log_file", "logs/predictions.jsonl"
)

# Create log directory
LOG_DIR.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Root logger configuration (done once)
stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(APP_LOG_FILE, encoding="utf-8")

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        stream_handler,
        file_handler,
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger inheriting the root configuration."""
    return logging.getLogger(name)


# Prediction logger — writes one JSON line per request
class PredictionLogger:
    """
    Appends structured JSON logs for every prediction request.

    Each line:
        {timestamp, order_id, prediction, probability, model_version, latency_ms, status}

    Storing prediction logs enables offline evaluation when real delivery
    dates arrive (Requirement 10 — monitoring).
    """

    def __init__(self, filepath: Path = PREDICTION_LOG_FILE) -> None:
        self._filepath = filepath
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("prediction")

    def log(
        self,
        *,
        order_id: str | None,
        prediction: int,
        probability: float,
        model_version: str,
        latency_ms: float,
        status: str = "ok",
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "prediction": prediction,
            "prediction_label": "late" if prediction == 1 else "on_time",
            "probability": round(probability, 6),
            "model_version": model_version,
            "latency_ms": round(latency_ms, 2),
            "status": status,
        }
        if extra:
            record.update(extra)
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._logger.info(
            "Prediction logged | order_id=%s | label=%s | prob=%.4f | latency=%.1fms",
            order_id,
            record["prediction_label"],
            probability,
            latency_ms,
        )
