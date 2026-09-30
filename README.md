# 🚀 Olist E-Commerce Delivery Delay Prediction — End-to-End MLOps System
## From Relational Database to Production Inference Service (Tasks 1, 2 & 3)

[![CI/CD Pipeline](https://github.com/SamarAbuAlouf/MLOps-Olist-Delivery/actions/workflows/ci.yml/badge.svg)](https://github.com/SamarAbuAlouf/MLOps-Olist-Delivery/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI_0.111-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow_v2.14-0194E2.svg?logo=mlflow)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/Data_Version-DVC_3.67-945DD6.svg?logo=dvc)](https://dvc.org)
[![Docker](https://img.shields.io/badge/Container-Docker_Compose-2496ED.svg?logo=docker)](https://www.docker.com)
[![Pytest](https://img.shields.io/badge/Tests-21_Passed-brightgreen.svg?logo=pytest)](https://pytest.org)

An enterprise-grade, reproducible Machine Learning Operations (MLOps) system built on the Brazilian **Olist E-Commerce Dataset** (~100k customer orders). The system predicts at purchase checkout time whether an order will experience a shipping delay (`is_late = 1`) relative to the promised estimated delivery date (`order_estimated_delivery_date`).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Tier
        DB[(PostgreSQL Database)] -->|SQLAlchemy Extract & Join| D01[01_joined_orders.parquet]
        D01 -->|Target Definition| D02[02_labeled_orders.parquet]
        D02 -->|Temporal 70/15/15 Split| D03[03_train/val/test.parquet]
    end

    subgraph Notebooks & ML Research
        D03 -->|Train-Only EDA| NB4[04_exploratory_data_analysis.ipynb]
        D03 -->|Anti-Leakage Feature Eng.| NB5[05_feature_engineering.ipynb]
        NB5 -->|Fitted Preprocessor| P_ART[preprocessor.joblib]
        NB5 -->|Model Tuning & Selection| NB6[06_train_tune_evaluate.ipynb]
        NB6 -->|Champion Model| M_ART[final_model.joblib]
    end

    subgraph MLOps Production Tier
        P_ART & M_ART -->|MLflow Registration| REG[MLflow Model Registry: Production]
        D03 -->|Data & Pipeline Versioning| DVC_P[DVC: artifacts/data.dvc & dvc.yaml]
        REG -->|Model Loader| P_PIPE[src/predict.py]
        P_ART -->|Pure Transform| F_PIPE[src/features.py]
        F_PIPE & P_PIPE --> API[FastAPI Service: app/main.py]
        API -->|Request Validation| VAL[Great Expectations Validator: src/validation.py]
        API -->|JSONL Audit Trail| AUDIT[(logs/predictions.jsonl)]
        API -->|Prometheus Metrics| METRICS[/metrics]
        AUDIT -->|Drift & Health Checks| DRIFT[src/evaluate_drift.py]
    end

    subgraph Deployment & Orchestration
        API & DB & REG --> DOCKER[Docker Compose Multi-Container Stack]
        DOCKER --> CICD[GitHub Actions CI/CD Pipeline]
    end
```

---

## 📂 Repository Structure & Purpose of Each Folder

```text
├── app/                                    # 🌐 FastAPI Production Inference Application
│   ├── main.py                             # API entry point, lifespan, CORS, telemetry middleware, /metrics
│   ├── schemas.py                          # Pydantic v2 request & response schemas with OpenAPI examples
│   └── routes/
│       ├── health.py                       # GET /health — Liveness & readiness probes
│       ├── info.py                         # GET /info — Active model type, version, features
│       └── predict.py                      # POST /predict & /predict/batch with error handling
├── config/                                 # ⚙️ Central Configuration (Zero Hardcoding)
│   └── config.yaml                         # Central config: database, paths, threshold, validation rules
├── src/                                    # 🧠 Refactored Production Python Modules
│   ├── __init__.py                         # Package declaration
│   ├── config.py                           # Dynamic YAML loader resolving ${ENV:default} variables
│   ├── logger.py                           # Centralised logging (stdout + app.log + predictions.jsonl)
│   ├── data.py                             # Database extract, Haversine distance, table aggregations
│   ├── features.py                         # Pure feature engineering & ColumnTransformer loading
│   ├── predict.py                          # Model singleton, MLflow registry loader, thresholding
│   ├── validation.py                       # Great Expectations validator (reject / flag / default)
│   ├── register_model.py                   # Script registering champion model to MLflow Production stage
│   └── evaluate_drift.py                   # Prediction log analyzer for latency SLAs & drift detection
├── tests/                                  # 🧪 Comprehensive Automated Pytest Suite (21 tests)
│   ├── conftest.py                         # Pytest fixtures and mock payloads
│   ├── test_preprocessing.py               # Unit tests for Haversine distances & table aggregations
│   ├── test_features.py                    # Unit tests for feature ratios, regions, and transform shapes
│   ├── test_data.py                        # Data tests: schema checks, range violations, target leakage
│   ├── test_model.py                       # Model tests: probabilities in [0, 1], predict_proba methods
│   └── test_api.py                         # Integration tests for all routes + intentional bad payload tests
├── notebooks/                              # 📓 6 Self-Contained Research Notebooks (Task 2)
│   ├── 01_read_and_join_tables.ipynb       # Database extraction, aggregation, Haversine distance
│   ├── 02_create_labels.ipynb              # Target creation (is_late), class imbalance analysis
│   ├── 03_train_val_test_split.ipynb       # Chronological/temporal train/val/test splitting
│   ├── 04_exploratory_data_analysis.ipynb  # Train-only exploratory analysis & 8 charts
│   ├── 05_feature_engineering.ipynb        # Anti-leakage features & fitted ColumnTransformer
│   └── 06_train_tune_evaluate.ipynb         # Baselines vs Champion, threshold tuning, test evaluation
├── artifacts/                              # 📦 Project Artifacts & Data Lineage
│   ├── data/                               # Parquet datasets tracked by DVC (artifacts/data.dvc)
│   ├── models/                             # Serialized models (final_model.joblib, preprocessor.joblib)
│   ├── figures/                            # High-resolution EDA and evaluation plots
│   └── reports/                            # Findings summaries, model metrics, monitoring plan
├── .github/workflows/                      # 🤖 CI/CD Automation
│   └── ci.yml                              # Linting (Ruff/Black), pytest coverage, and Docker build
├── Dockerfile                              # 🐳 Lightweight production container for FastAPI service
├── docker-compose.yml                      # 🐳 Orchestration: PostgreSQL + MLflow + FastAPI service
├── dvc.yaml                                # 🗃️ Reproducible DVC pipeline stage definition
├── .pre-commit-config.yaml                 # 🛡️ Pre-commit hooks for format, yaml, and secrets safety
├── requirements.txt                        # 📌 Pinned runtime dependencies
├── requirements-dev.txt                    # 🛠️ Pinned development and testing dependencies
└── README.md                               # 📖 Full system documentation
```

---

## 🛠️ Tools Used & Why They Are Here

| Tool | Category | Why It Was Chosen |
|---|---|---|
| **FastAPI** | Serving | High performance async ASGI framework, native Pydantic validation, automatic OpenAPI / Swagger interactive documentation. |
| **MLflow** | MLOps Tracking | Standardized experiment tracking, parameter/metric logging, and centralized Model Registry with production stage gating. |
| **DVC (Data Version Control)** | Data Versioning | Manages large datasets outside of Git, versions parquet files with `.dvc` pointers, guarantees reproducible pipeline stages via `dvc.yaml`. |
| **Great Expectations Principles** | Data Validation | Pre-inference data checks (column existence, allowed values, expected numeric boundaries) to prevent silent model failures. |
| **Docker & Docker Compose** | Containerization | Ensures deterministic runtime environment across development, staging, and production with a single startup command. |
| **Pytest** | Automated Testing | Comprehensive unit, data, model, and integration testing runnable with a single command (`pytest`). |
| **GitHub Actions** | CI/CD | Continuous integration running linting, formatting, test suites, and container build on every commit. |
| **Uvicorn** | ASGI Web Server | Production-ready HTTP server powering the FastAPI inference engine. |

---

## ⚙️ Configuration & Environment Variables

All parameters and paths are defined in `config/config.yaml`. **No hardcoded paths or parameters exist in source files.**

Environment variables override configuration seamlessly:
- `DB_HOST`: Hostname of PostgreSQL database (default: `localhost`, or `db` inside Docker).
- `DB_PASSWORD`: Database password (e.g. set in `.env`, see `.env.example`).
- `MLFLOW_TRACKING_URI`: URL of MLflow server (default: `http://localhost:5000`).
- `LOG_LEVEL`: Log verbosity (default: `INFO`).

---

## 🚀 Quickstart: Running From Zero

### Option A: Complete Stack via Docker Compose (Recommended)
To bring up PostgreSQL, MLflow, and the FastAPI Inference API in one command:
```bash
docker compose up -d --build
```
- **API Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking UI:** [http://localhost:5000](http://localhost:5000)
- **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)
- **Service Telemetry:** [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

### Option B: Local Python Environment

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Register the Champion Model in MLflow:**
   ```bash
   python -m src.register_model
   ```

3. **Run the Test Suite (21 Tests):**
   ```bash
   pytest -v
   ```

4. **Verify DVC Data Lineage:**
   ```bash
   python -m dvc repro
   ```

5. **Start the FastAPI Inference Server:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 📡 API Usage & Sample Requests

### 1. Single Order Prediction (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "e481f51cbdc54678b7cc49136f2d6af7",
    "customer_state": "SP",
    "seller_state": "SP",
    "dominant_payment_type": "credit_card",
    "order_items_count": 1,
    "total_price": 29.99,
    "total_freight": 8.72,
    "total_weight_g": 500.0,
    "total_volume_cm3": 1976.0,
    "total_payment_value": 38.71,
    "max_payment_installments": 1,
    "distance_km": 18.5,
    "estimated_delivery_days": 15.5
  }'
```

**Response (`200 OK`):**
```json
{
  "order_id": "e481f51cbdc54678b7cc49136f2d6af7",
  "prediction": 0,
  "label": "on_time",
  "probability": 0.137628,
  "threshold": 0.5,
  "model_version": "1.0.0",
  "validation_passed": true,
  "warnings": []
}
```

---

### 2. Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "orders": [
      {
        "order_id": "order-001",
        "customer_state": "SP",
        "seller_state": "SP",
        "total_price": 50.0,
        "total_freight": 10.0,
        "total_payment_value": 60.0
      },
      {
        "order_id": "order-002",
        "customer_state": "AM",
        "seller_state": "SP",
        "total_price": 450.0,
        "total_freight": 95.0,
        "total_payment_value": 545.0,
        "distance_km": 2800.0
      }
    ]
  }'
```

---

## 🔍 Validation & Error Handling (Definition of Done #4)

### Breaking Something on Purpose:
1. **Missing Mandatory Fields:**
   If a client submits a payload without required fields (e.g. missing `total_price`), Pydantic validation rejects it immediately with **`422 Unprocessable Entity`**:
   ```json
   {
     "detail": [
       {
         "type": "missing",
         "loc": ["body", "total_price"],
         "msg": "Field required"
       }
     ]
   }
   ```

2. **Negative Values:**
   If `total_price: -25.0` is sent, Pydantic's `ge=0.0` rule triggers **`422 Unprocessable Entity`**:
   ```json
   {
     "detail": [
       {
         "type": "greater_than_equal",
         "loc": ["body", "total_price"],
         "msg": "Input should be greater than or equal to 0"
       }
     ]
   }
   ```

3. **Data Range Drift / Anomaly Flagging:**
   If `distance_km: 15000.0` (greater than Brazil's borders), the service handles it gracefully according to the configured policy without crashing, records the warning in the audit log, and informs the client:
   ```json
   {
     "validation_passed": false,
     "warnings": ["Column 'distance_km' out of expected range [0, 5000]: sample value 15000.0"]
   }
   ```

---

## 📊 Pipeline Reproducibility (Definition of Done #3)

The production inference pipeline (`src/features.py` + `src/predict.py`) loads the pre-fitted transformers without refitting and matches the Jupyter Notebook output **with 0.0 error**:

$$\max |\mathbf{X}_{\text{pipeline}} - \mathbf{X}_{\text{notebook}}| = 0.0$$

Verified via automated test:
```python
diff = np.abs(X_test_transformed - X_saved)
assert np.nanmax(diff) < 1e-6  # Max difference is exactly 0.0!
```

---

## 📈 Monitoring & Drift Detection (Requirement 10)

1. **Structured Prediction Audit Trail:**
   Every prediction request is logged to `logs/predictions.jsonl` with latency, prediction, probability, and timestamp.
2. **Real-Time Telemetry:**
   Access `GET /metrics` for real-time error rates, request counts, average latency, and p95 latency.
3. **Drift Analyzer:**
   Run `python -m src.evaluate_drift` to compute observed late rates vs. baseline (9.03%) and detect concept drift.
4. **Monitoring Plan:**
   Detailed alerting thresholds and SLOs documented in [artifacts/reports/monitoring_plan.md](artifacts/reports/monitoring_plan.md).

---

## 🏆 Model Performance Summary (Holdout Test Set)

| Metric | Champion Model (HistGradientBoosting) | Baseline (Balanced Logistic Regression) |
|---|---|---|
| **ROC-AUC** | **0.7226** | 0.6908 |
| **PR-AUC (Avg Precision)** | **0.1217** | 0.1042 |
| **Recall (Delayed Orders)**| **29.68%** | 58.10% |
| **Accuracy** | **82.41%** | 66.83% |
| **F1-Score** | **0.1825** | 0.1770 |

---

## ✅ Definition of Done Checklist

- [x] **1. Structured Repository:** Clean separation of `app/`, `config/`, `src/`, `tests/`, `notebooks/`, `artifacts/`, requirements files, and zero hardcoded paths.
- [x] **2. Modular Inference Pipeline:** Functions for data access, feature extraction, pre-fitted transformer loading, and threshold inference.
- [x] **3. Data & Model Lineage:** DVC versioning (`artifacts/data.dvc` & `dvc.yaml`), Great Expectations validation, and MLflow Model Registry tracking with `Production` stage.
- [x] **4. Automated Pytest Suite:** 21 passing unit, data, model, and API integration tests.
- [x] **5. Production FastAPI Service:** Endpoints for `/health`, `/info`, `/predict`, `/predict/batch`, and `/metrics`.
- [x] **6. Containerization:** Multi-service `docker-compose.yml` orchestrating PostgreSQL, MLflow, and FastAPI.
- [x] **7. Automated CI/CD:** GitHub Actions workflow executing linting, test suites with coverage, and Docker image build.
- [x] **8. Monitoring & Drift Logging:** JSONL prediction audit logs, telemetry metrics endpoint, and alerting strategy.