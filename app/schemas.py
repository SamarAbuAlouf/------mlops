"""
app/schemas.py
Pydantic schemas for FastAPI request and response validation.

Requirement 7:
- Validate request and response with schemas — reject bad payloads clearly
- Return prediction, probability, and model version
- Provide OpenAPI / Swagger examples
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    """Input payload for a single order prediction."""
    order_id: Optional[str] = Field(None, example="e481f51cbdc54678b7cc49136f2d6af7", description="Unique order identifier")
    customer_state: str = Field(..., example="SP", description="Customer 2-letter state code")
    seller_state: str = Field(..., example="SP", description="Seller 2-letter state code")
    dominant_payment_type: str = Field(default="credit_card", example="credit_card", description="Main payment method")
    order_items_count: int = Field(default=1, ge=1, le=100, example=1, description="Number of items in the order")
    total_price: float = Field(..., ge=0.0, example=29.99, description="Total order price (BRL)")
    total_freight: float = Field(..., ge=0.0, example=8.72, description="Total shipping freight cost (BRL)")
    total_weight_g: float = Field(default=500.0, ge=0.0, example=500.0, description="Total product weight in grams")
    total_volume_cm3: float = Field(default=1000.0, ge=0.0, example=1976.0, description="Total product volume in cubic centimeters")
    total_payment_value: float = Field(..., ge=0.0, example=38.71, description="Total payment value including freight (BRL)")
    max_payment_installments: int = Field(default=1, ge=1, le=36, example=1, description="Maximum installment count")
    distance_km: float = Field(default=50.0, ge=0.0, example=18.5, description="Geographic distance between customer and seller (km)")
    estimated_delivery_days: Optional[float] = Field(default=15.0, ge=0.0, example=15.5, description="Promised lead time in days")
    order_purchase_timestamp: Optional[str] = Field(default=None, example="2017-10-02 10:56:33", description="ISO timestamp of purchase")
    order_estimated_delivery_date: Optional[str] = Field(default=None, example="2017-10-18 00:00:00", description="ISO timestamp of estimated delivery")

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "estimated_delivery_days": 15.5,
                "order_purchase_timestamp": "2017-10-02 10:56:33",
                "order_estimated_delivery_date": "2017-10-18 00:00:00"}}}


class PredictionResponse(BaseModel):
    """Response payload for a single order prediction."""
    order_id: Optional[str] = Field(None, description="Order ID if provided")
    prediction: int = Field(..., description="0 = On Time, 1 = Late")
    label: str = Field(..., description="'on_time' or 'late'")
    probability: float = Field(..., description="Probability of late delivery")
    threshold: float = Field(..., description="Decision threshold applied")
    model_version: str = Field(..., description="Model version producing the prediction")
    validation_passed: bool = Field(True, description="Whether input passed data validation checks")
    warnings: List[str] = Field(default_factory=list, description="Any data validation or drift warnings")


class BatchOrderInput(BaseModel):
    """Batch input payload containing multiple orders."""
    orders: List[OrderInput] = Field(..., min_length=1, description="List of orders to evaluate")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response containing results for all orders."""
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    count: int = Field(..., description="Total orders evaluated")
    model_version: str = Field(..., description="Model version")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is in memory")
    preprocessor_loaded: bool = Field(..., description="Whether preprocessor is loaded")
    model_version: str = Field(..., description="Current loaded model version")
    timestamp: str = Field(..., description="UTC timestamp")


class ModelInfoResponse(BaseModel):
    """Metadata regarding current model and feature schema."""
    project_name: str
    model_name: str
    model_type: str
    model_version: str
    threshold: float
    features_count: int
    numerical_features: List[str]
    categorical_features: List[str]
