#  Production Monitoring & Alerting Strategy
**Project:** Olist E-Commerce Delivery Delay Prediction (MLOps Task 3)  
**System:** Real-Time Inference Service (`/predict`, `/predict/batch`)

---

## 1. Architecture Overview

```mermaid
flowchart LR
    Client[Client / App] -->|POST /predict| API[FastAPI Inference Service]
    API -->|Validation & Checks| Val[Great Expectations Validator]
    API -->|Log Prediction| Audit[(logs/predictions.jsonl)]
    API -->|Service Telemetry| Metrics[/metrics Endpoint]
    Audit -->|Batch Evaluation| Drift[src/evaluate_drift.py]
    Drift -->|Alert Notifications| Slack[Alerting: Slack / PagerDuty]
```

---

## 2. Telemetry & Metrics Tracked

| Metric | Source | Target SLA | Action on Violation |
|---|---|---|---|
| **Request Count** | `/metrics` | N/A | Track throughput & traffic bursts |
| **Error Rate (4xx / 5xx)** | `/metrics` | `< 1.0%` | Alert if error rate > 2.0% for 5 mins |
| **P95 Latency** | `/metrics` | `< 300 ms` | Scale replicas if p95 > 500 ms |
| **Prediction Drift** | `predictions.jsonl` | `Baseline ~ 9.03%` | Trigger data drift investigation if late rate `< 4%` or `> 18%` |
| **Data Validation Failures**| Application Logs | `< 5.0%` | Alert upstream data pipeline of schema or range drift |

---

## 3. Concrete Alerting Rules & Thresholds

###  Alert 1: Prediction Concept Drift (P1)
- **Condition:** Over any rolling window of 1,000 predictions, the predicted late rate deviates by more than ±50% from the training baseline (9.03%).
  - Range: `Predicted Late Rate < 4.5%` OR `Predicted Late Rate > 13.5%`.
- **Root Causes:** Logistics carrier disruptions, severe weather, black Friday volume spikes, or holiday season behavior.
- **Action:** Notify ML Engineer; compare feature distributions using Kolmogorov-Smirnov test; schedule model retraining if seasonal drift is confirmed.

###  Alert 2: Service Error Rate Spike (P0)
- **Condition:** HTTP 5xx error rate exceeds `2.0%` over a 5-minute rolling window.
- **Root Causes:** Out of memory, missing environment variables, DB connection exhaustion.
- **Action:** Page on-call engineer; check container logs; restart degraded pods.

###  Alert 3: Latency SLA Breach (P2)
- **Condition:** 95th percentile latency exceeds `500 ms` for 10 consecutive minutes.
- **Root Causes:** Heavy batch prediction load, resource throttling.
- **Action:** Autoscale inference container replicas; verify caching.

###  Alert 4: Ground Truth Delayed Feedback Evaluation (Weekly Job)
- **Mechanism:** As actual delivery dates arrive from the postal carrier (`order_delivered_customer_date`), the system matches the logged prediction in `logs/predictions.jsonl` by `order_id`.
- **Condition:** If rolling 30-day Test ROC-AUC falls below `0.65` (baseline test is `0.7226`), trigger automated model retraining pipeline.

---

## 4. Prediction Audit Log Schema

Every prediction is persisted to `logs/predictions.jsonl`:
```json
{
  "timestamp": "2026-09-30T13:00:03.635000+00:00",
  "order_id": "test-123",
  "prediction": 0,
  "prediction_label": "on_time",
  "probability": 0.333917,
  "model_version": "1.0.0",
  "latency_ms": 23.4,
  "status": "ok"
}
```
This enables zero-data-loss auditability and ground truth reconciliation.
