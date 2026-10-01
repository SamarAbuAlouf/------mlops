"""
tests/conftest.py
Shared pytest fixtures for unit, data, and integration testing.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_raw_order_dict():
    """A valid, realistic single order payload for testing."""
    return {
        "order_id": "test-order-valid-001",
        "customer_state": "SP",
        "seller_state": "SP",
        "dominant_payment_type": "credit_card",
        "order_items_count": 1,
        "total_price": 50.0,
        "total_freight": 10.0,
        "total_weight_g": 600.0,
        "total_volume_cm3": 1500.0,
        "total_payment_value": 60.0,
        "max_payment_installments": 1,
        "distance_km": 30.0,
        "estimated_delivery_days": 14.0,
        "order_purchase_timestamp": "2018-01-15 12:00:00",
        "order_estimated_delivery_date": "2018-01-29 00:00:00",
    }


@pytest.fixture
def sample_raw_order_df(sample_raw_order_dict):
    """A pandas DataFrame with one valid order row."""
    return pd.DataFrame([sample_raw_order_dict])
