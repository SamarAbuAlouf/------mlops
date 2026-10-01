"""
app/routes/predict.py
Prediction endpoints for single order and batch orders.

Requirements 2, 3, 4, 7:
- Validates request payload
- Runs data validation
- Executes inference pipeline
- Returns prediction, probability, and model version
- Handles errors gracefully without crashing
"""

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.schemas import (
    BatchOrderInput,
    BatchPredictionResponse,
    OrderInput,
    PredictionResponse,
)

from src.features import transform
from src.logger import get_logger
from src.predict import get_model_version, predict, predict_batch
from src.validation import DataValidationError, default_validator

logger = get_logger(__name__)
router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict delivery delay for a single order",
    status_code=status.HTTP_200_OK,
)
async def predict_order(order: OrderInput):
    """
    Predict whether a customer order will be delivered late (`is_late`).
    Returns 0 (On Time) or 1 (Late) along with the probability score.
    """
    order_dict = order.model_dump()
    order_id = order_dict.get("order_id")

    try:
        # 1. Data validation (schema, ranges, categories)
        is_valid, issues, cleaned_df = default_validator.validate(order_dict)
    except DataValidationError as exc:
        logger.warning("Validation rejected order %s: %s", order_id, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "DataValidationError", "message": str(exc)},
        )

    try:
        # 2. Transform features using pre-fitted transformers
        X_trans = transform(cleaned_df)

        # 3. Model prediction
        pred_res = predict(X_trans[:1], order_id=order_id)

        return PredictionResponse(
            order_id=order_id,
            prediction=pred_res["prediction"],
            label=pred_res["label"],
            probability=pred_res["probability"],
            threshold=pred_res["threshold"],
            model_version=pred_res["model_version"],
            validation_passed=is_valid,
            warnings=issues,
        )

    except Exception as exc:
        logger.exception("Unexpected inference error for order %s: %s", order_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction for multiple orders",
    status_code=status.HTTP_200_OK,
)
async def predict_orders_batch(batch: BatchOrderInput):
    """
    Predict delivery delay for multiple orders in a single request.
    Efficiently transforms features and performs vectorised inference.
    """
    if not batch.orders:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order list cannot be empty",
        )

    orders_list = [o.model_dump() for o in batch.orders]
    df_raw = pd.DataFrame(orders_list)

    try:
        # Validate batch
        is_valid, issues, cleaned_df = default_validator.validate(df_raw)
    except DataValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "DataValidationError", "message": str(exc)},
        )

    try:
        X_trans = transform(cleaned_df)
        preds = predict_batch(X_trans)

        results = []
        for i, pred_res in enumerate(preds):
            o_id = orders_list[i].get("order_id")
            results.append(
                PredictionResponse(
                    order_id=o_id,
                    prediction=pred_res["prediction"],
                    label=pred_res["label"],
                    probability=pred_res["probability"],
                    threshold=pred_res["threshold"],
                    model_version=pred_res["model_version"],
                    validation_passed=is_valid,
                    warnings=issues if i == 0 else [],
                )
            )

        return BatchPredictionResponse(
            predictions=results,
            count=len(results),
            model_version=get_model_version(),
        )

    except Exception as exc:
        logger.exception("Batch prediction failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        )
