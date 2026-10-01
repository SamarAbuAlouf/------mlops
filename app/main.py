"""
app/main.py
FastAPI application entry point.

Requirements 7, 8, 10:
- Exposes REST API with Swagger docs (/docs, /redoc)
- Implements Prometheus / monitoring metrics (/metrics)
- Integrates health, info, and predict routers
- Loads model at startup
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import health, info, predict
from src.config import CONFIG
from src.features import load_preprocessor
from src.logger import get_logger
from src.predict import load_model, try_load_from_mlflow

logger = get_logger("app.main")

# Service telemetry counters (in-memory monitoring)
_METRICS = {
    "total_requests": 0,
    "predictions_count": 0,
    "errors_count": 0,
    "latencies_ms": [],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown lifecycle management."""
    logger.info("Initializing Olist Delay Prediction Service...")
    # Attempt loading registered model from MLflow first, fallback to disk artifact
    loaded_from_mlflow = try_load_from_mlflow()
    if not loaded_from_mlflow:
        load_model()
    load_preprocessor()
    logger.info("Service warmup complete. Ready to receive inference traffic.")
    yield
    logger.info("Shutting down Olist Delay Prediction Service...")


app = FastAPI(
    title=CONFIG["api"]["title"],
    description=(
        "Production inference service for Olist E-Commerce Delivery Delay Prediction. "
        "Built according to MLOps best practices (madewithml.com / Qafza MLOps Task 3)."
    ),
    version=CONFIG["api"]["version"],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware tracking request duration and telemetry."""
    _METRICS["total_requests"] += 1
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        latency = (time.perf_counter() - t0) * 1000
        _METRICS["latencies_ms"].append(latency)
        # Keep window of last 1000 latencies
        if len(_METRICS["latencies_ms"]) > 1000:
            _METRICS["latencies_ms"].pop(0)
        return response
    except Exception as exc:
        _METRICS["errors_count"] += 1
        latency = (time.perf_counter() - t0) * 1000
        logger.exception(
            "Unhandled error on %s %s: %s", request.method, request.url.path, exc
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )


# Monitoring metrics route (Requirement 10)
@app.get("/metrics", tags=["Monitoring"], summary="Service performance metrics")
async def get_metrics():
    """
    Exposes service telemetry metrics:
    - total_requests
    - errors_count
    - avg_latency_ms
    - p95_latency_ms
    """
    lats = _METRICS["latencies_ms"]
    avg_lat = round(sum(lats) / len(lats), 2) if lats else 0.0
    p95_lat = (
        round(sorted(lats)[int(0.95 * len(lats))], 2) if len(lats) >= 20 else avg_lat
    )

    return {
        "total_requests": _METRICS["total_requests"],
        "errors_count": _METRICS["errors_count"],
        "error_rate": round(
            _METRICS["errors_count"] / max(1, _METRICS["total_requests"]), 4
        ),
        "avg_latency_ms": avg_lat,
        "p95_latency_ms": p95_lat,
        "active_version": CONFIG["project"]["version"],
    }


# Include sub-routers
app.include_router(health.router)
app.include_router(info.router)
app.include_router(predict.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": CONFIG["api"]["title"],
        "version": CONFIG["api"]["version"],
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
    }
