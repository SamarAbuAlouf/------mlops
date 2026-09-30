"""
tests/test_preprocessing.py
Unit tests for data access and preprocessing helpers in src/data.py.
"""
import pytest
import pandas as pd
from src.data import haversine_km, aggregate_order_items, aggregate_payments


def test_haversine_same_point():
    """Distance between identical coordinates must be 0.0 km."""
    dist = haversine_km(-23.5505, -46.6333, -23.5505, -46.6333)
    assert pytest.approx(dist, abs=1e-3) == 0.0


def test_haversine_known_distance():
    """Distance between Sao Paulo (-23.55, -46.63) and Rio (-22.90, -43.17) is ~360 km."""
    dist = haversine_km(-23.55, -46.63, -22.90, -43.17)
    assert 340.0 < dist < 380.0


def test_aggregate_order_items():
    """Verify aggregation of order items groups correctly by order_id."""
    df_items = pd.DataFrame([
        {
            "order_id": "ord_1",
            "order_item_id": 1,
            "price": 100.0,
            "freight_value": 15.0,
            "product_weight_g": 500.0,
            "product_volume_cm3": 1000.0,
            "seller_id": "sel_1",
            "product_id": "prod_1",
        },
        {
            "order_id": "ord_1",
            "order_item_id": 2,
            "price": 50.0,
            "freight_value": 10.0,
            "product_weight_g": 300.0,
            "product_volume_cm3": 800.0,
            "seller_id": "sel_1",
            "product_id": "prod_2",
        },
    ])
    agg = aggregate_order_items(df_items)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["order_items_count"] == 2
    assert row["total_price"] == 150.0
    assert row["total_freight"] == 25.0
    assert row["total_weight_g"] == 800.0


def test_aggregate_payments():
    """Verify payment aggregation computes total value and dominant type."""
    df_pay = pd.DataFrame([
        {"order_id": "ord_1", "payment_value": 70.0, "payment_installments": 2, "payment_type": "credit_card"},
        {"order_id": "ord_1", "payment_value": 30.0, "payment_installments": 1, "payment_type": "voucher"},
    ])
    agg = aggregate_payments(df_pay)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["total_payment_value"] == 100.0
    assert row["max_payment_installments"] == 2
    assert row["dominant_payment_type"] == "credit_card"
