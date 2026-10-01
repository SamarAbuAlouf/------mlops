"""
app/routes/info.py
Model metadata and feature schema endpoint.
"""

from fastapi import APIRouter

from app.schemas import ModelInfoResponse
from src.config import CONFIG, MODEL_THRESHOLD
from src.features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.predict import get_model_version, load_model

router = APIRouter(tags=["Model Info"])


@router.get(
    "/info", response_model=ModelInfoResponse, summary="Get model metadata and schema"
)
async def get_model_info():
    """Returns details about current active model, version, decision threshold, and feature expectations."""
    model = load_model()
    return ModelInfoResponse(
        project_name=CONFIG["project"]["name"],
        model_name=CONFIG["mlflow"]["model_name"],
        model_type=type(model).__name__,
        model_version=get_model_version(),
        threshold=MODEL_THRESHOLD,
        features_count=len(ALL_FEATURES),
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
