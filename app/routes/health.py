"""
app/routes/health.py
Health check endpoint.
"""
from datetime import datetime, timezone
from fastapi import APIRouter

from app.schemas import HealthResponse
from src.features import load_preprocessor
from src.predict import get_model_version, load_model

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Check service health")
async def health_check():
    """Verify that the service is alive and model artifacts are loaded in memory."""
    model_ok = load_model() is not None
    preproc_ok = load_preprocessor() is not None
    return HealthResponse(
        status="healthy" if (model_ok and preproc_ok) else "degraded",
        model_loaded=model_ok,
        preprocessor_loaded=preproc_ok,
        model_version=get_model_version(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
