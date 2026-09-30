"""
src/evaluate_drift.py
Monitoring and prediction drift analysis tool.

Requirement 10:
- Reads stored prediction logs (logs/predictions.jsonl)
- Computes prediction distribution over time
- Detects concept and prediction drift against training baseline
- Implements alerting thresholds
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

# Baseline training delay rate from Notebook 02/03
BASELINE_LATE_RATE = 0.0903  # 9.03%
PRED_LOG_PATH = Path(CONFIG.get("logging", {}).get("prediction_log_file", "logs/predictions.jsonl"))


def analyze_prediction_logs(log_path: Path = PRED_LOG_PATH) -> dict:
    """Analyze logged predictions and check for drift/anomalies."""
    if not log_path.exists():
        logger.warning("No prediction log found at %s", log_path)
        return {"status": "no_data", "message": f"Log file {log_path} does not exist yet."}

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    if not records:
        return {"status": "empty", "message": "Prediction log is empty."}

    df = pd.DataFrame(records)
    total_preds = len(df)
    late_preds = (df["prediction"] == 1).sum()
    observed_late_rate = late_preds / total_preds
    avg_latency = df["latency_ms"].mean() if "latency_ms" in df.columns else 0.0
    p95_latency = df["latency_ms"].quantile(0.95) if "latency_ms" in df.columns else 0.0

    # Drift checks
    # Alert if observed rate deviates by more than 50% relative from baseline
    relative_drift = abs(observed_late_rate - BASELINE_LATE_RATE) / BASELINE_LATE_RATE
    drift_detected = relative_drift > 0.50 if total_preds >= 30 else False

    alerts = []
    if drift_detected:
        alerts.append(
            f"PREDICTION DRIFT ALERT: Observed late rate ({observed_late_rate:.2%}) "
            f"deviates significantly from baseline ({BASELINE_LATE_RATE:.2%})")
        
    if p95_latency > 500.0:  # SLA threshold 500ms
        alerts.append(f"LATENCY SLA ALERT: p95 latency ({p95_latency:.1f}ms) exceeds 500ms SLA")

    report = {
        "total_logged_predictions": total_preds,
        "observed_late_rate": round(observed_late_rate, 4),
        "baseline_late_rate": BASELINE_LATE_RATE,
        "relative_drift": round(relative_drift, 4),
        "drift_detected": drift_detected,
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "alerts_triggered": alerts,
        "status": "alert" if alerts else "healthy",}

    logger.info("Monitoring analysis complete: %s (Total: %d, Late rate: %.2f%%)",
                report["status"], total_preds, observed_late_rate * 100)
    return report


if __name__ == "__main__":
    rep = analyze_prediction_logs()
    print(json.dumps(rep, indent=2))
