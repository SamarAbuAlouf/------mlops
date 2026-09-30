#  Olist E-Commerce Delivery Delay Prediction — End-to-End MLOps System
## From Relational Database to Production Inference Service

An enterprise-grade, reproducible Machine Learning Operations (MLOps) system built on the Brazilian **Olist E-Commerce Dataset** (~100k customer orders). The system predicts at purchase checkout time whether an order will experience a shipping delay (`is_late = 1`) relative to the promised estimated delivery date (`order_estimated_delivery_date`).

---

##  Tools Used & Why They Are Here

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

##  Configuration & Environment Variables

All parameters and paths are defined in `config/config.yaml`. **No hardcoded paths or parameters exist in source files.**

Environment variables override configuration seamlessly:
- `DB_HOST`: Hostname of PostgreSQL database (default: `localhost`, or `db` inside Docker).
- `DB_PASSWORD`: Database password (e.g. set in `.env`, see `.env.example`).
- `MLFLOW_TRACKING_URI`: URL of MLflow server (default: `http://localhost:5000`).
- `LOG_LEVEL`: Log verbosity (default: `INFO`).

---

##  Quickstart: Running From Zero

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

##  Pipeline Reproducibility

The production inference pipeline (`src/features.py` + `src/predict.py`) loads the pre-fitted transformers without refitting and matches the Jupyter Notebook output **with 0.0 error**:


Verified via automated test:
```python
diff = np.abs(X_test_transformed - X_saved)
assert np.nanmax(diff) < 1e-6  # Max difference is exactly 0.0!
```