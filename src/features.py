"""
src/features.py
Feature engineering — mirrors Notebook 05 logic.

Responsibility: Given a raw order row (or DataFrame of rows), compute the
24 model features and apply the *already-fitted* preprocessor.

Key rule: The preprocessor is LOADED, never re-fitted here.
(Requirement 2: Load the saved fitted objects — never fit again at inference)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES_PATH, MODEL_THRESHOLD, PREPROCESSOR_PATH
from src.logger import get_logger

logger = get_logger(__name__)

# Brazilian macro-regions mapping (exact match with Notebook 05)
STATE_TO_REGION = {
    # Southeast
    'SP': 'Southeast', 'RJ': 'Southeast', 'MG': 'Southeast', 'ES': 'Southeast',
    # South
    'PR': 'South', 'SC': 'South', 'RS': 'South',
    # Northeast
    'BA': 'Northeast', 'PE': 'Northeast', 'CE': 'Northeast', 'MA': 'Northeast',
    'PB': 'Northeast', 'RN': 'Northeast', 'AL': 'Northeast', 'PI': 'Northeast', 'SE': 'Northeast',
    # North
    'AM': 'North', 'PA': 'North', 'RO': 'North', 'TO': 'North', 'AC': 'North', 'AP': 'North', 'RR': 'North',
    # Center-West
    'GO': 'Center-West', 'MT': 'Center-West', 'MS': 'Center-West', 'DF': 'Center-West'}

NUMERICAL_FEATURES = [
    "order_items_count", "total_price", "total_freight", "total_weight_g",
    "total_volume_cm3", "total_payment_value", "max_payment_installments",
    "distance_km", "estimated_delivery_days", "purchase_month",
    "purchase_dayofweek", "purchase_hour", "freight_ratio",
    "freight_per_item", "price_per_item", "density_g_cm3",]

CATEGORICAL_FEATURES = [
    "customer_state", "seller_state", "dominant_payment_type",
    "customer_region", "seller_region", "is_same_state",
    "is_inter_region", "purchase_is_weekend",]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# Feature engineering (pure transformation, no fitting)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered columns to an order DataFrame.
    Matches Notebook 05 logic exactly.
    """
    data = df.copy()

    # 1. Datetime conversions
    if "order_purchase_timestamp" in data.columns:
        data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"], errors="coerce")
    if "order_estimated_delivery_date" in data.columns:
        data["order_estimated_delivery_date"] = pd.to_datetime(data["order_estimated_delivery_date"], errors="coerce")

    # 2. Promised lead time in days (if not already provided)
    if "estimated_delivery_days" not in data.columns or data["estimated_delivery_days"].isna().all():
        if "order_estimated_delivery_date" in data.columns and "order_purchase_timestamp" in data.columns:
            data["estimated_delivery_days"] = (
                data["order_estimated_delivery_date"] - data["order_purchase_timestamp"]
            ).dt.total_seconds() / 86400.0

    # 3. Calendar & purchase time features
    if "order_purchase_timestamp" in data.columns:
        data["purchase_month"] = data["order_purchase_timestamp"].dt.month
        data["purchase_dayofweek"] = data["order_purchase_timestamp"].dt.dayofweek
        data["purchase_hour"] = data["order_purchase_timestamp"].dt.hour
        data["purchase_is_weekend"] = data["purchase_dayofweek"].isin([5, 6]).astype(int)

    # 4. Freight and pricing ratios (safe with Notebook 05 constants)
    if "total_freight" in data.columns and "total_price" in data.columns:
        data["freight_ratio"] = data["total_freight"] / (data["total_price"] + 1.0)
    if "total_freight" in data.columns and "order_items_count" in data.columns:
        data["freight_per_item"] = data["total_freight"] / (data["order_items_count"] + 1e-5)
    if "total_price" in data.columns and "order_items_count" in data.columns:
        data["price_per_item"] = data["total_price"] / (data["order_items_count"] + 1e-5)

    # 5. Density (weight / volume)
    if "total_weight_g" in data.columns and "total_volume_cm3" in data.columns:
        data["density_g_cm3"] = data["total_weight_g"] / (data["total_volume_cm3"] + 1.0)

    # 6. Macro-regions & cross-region flags
    if "customer_state" in data.columns:
        data["customer_region"] = data["customer_state"].map(STATE_TO_REGION).fillna("Other")
    if "seller_state" in data.columns:
        data["seller_region"] = data["seller_state"].map(STATE_TO_REGION).fillna("Other")

    if "customer_region" in data.columns and "seller_region" in data.columns:
        data["is_inter_region"] = (data["customer_region"] != data["seller_region"]).astype(int)
    if "customer_state" in data.columns and "seller_state" in data.columns:
        data["is_same_state"] = (data["customer_state"] == data["seller_state"]).astype(int)

    logger.debug("Feature engineering done: %d rows", len(data))
    return data


# Preprocessor loader (singleton)
_preprocessor = None
_feature_names: list[str] | None = None


def load_preprocessor(path: Path = PREPROCESSOR_PATH):
    """Load (and cache) the fitted ColumnTransformer from disk."""
    global _preprocessor
    if _preprocessor is None:
        logger.info("Loading preprocessor from %s", path)
        _preprocessor = joblib.load(path)
    return _preprocessor


def load_feature_names(path: Path = FEATURE_NAMES_PATH) -> list[str]:
    """Load feature name list saved by Notebook 05."""
    global _feature_names
    if _feature_names is None:
        logger.info("Loading feature names from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _feature_names = data if isinstance(data, list) else data.get("features", [])
    return _feature_names


# Transform a DataFrame to the model-ready numpy array
def transform(df: pd.DataFrame) -> np.ndarray:
    """
    Run feature engineering + preprocessor on a raw order DataFrame.
    Returns a 2-D numpy array ready for model.predict().

    Steps:
    1. Engineer features (ratios, regions, temporal)
    2. Select the 24 feature columns
    3. Apply fitted preprocessor (imputer + scaler + encoder) — never re-fitted
    """
    df_feat = engineer_features(df)

    # Ensure all expected columns exist (fill missing with NaN)
    for col in ALL_FEATURES:
        if col not in df_feat.columns:
            df_feat[col] = np.nan
            logger.warning("Column '%s' missing in input — filled with NaN", col)

    X = df_feat[ALL_FEATURES]
    preprocessor = load_preprocessor()
    X_transformed = preprocessor.transform(X)
    logger.debug("Transform complete: shape %s", X_transformed.shape)
    return X_transformed


# Single-order dict → transformed array (for API use)
def order_dict_to_array(order: dict[str, Any]) -> np.ndarray:
    """
    Convert a single order dict (from API request) to a transformed array.
    Missing optional fields default to NaN; the preprocessor handles them.
    """
    df = pd.DataFrame([order])
    return transform(df)
