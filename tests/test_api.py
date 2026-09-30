"""
tests/test_api.py
Integration tests for FastAPI endpoints end-to-end.

Requirement 6: Integration tests for the API routes end to end.
Definition of Done #4: Break something on purpose — bad data — and show the system catches it.
"""
from fastapi import status


def test_api_health_route(client):
    """GET /health must return 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_api_info_route(client):
    """GET /info must return model metadata and feature counts."""
    response = client.get("/info")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_type"] == "HistGradientBoostingClassifier"
    assert data["features_count"] == 24
    assert len(data["numerical_features"]) == 16
    assert len(data["categorical_features"]) == 8


def test_api_predict_single_order(client, sample_raw_order_dict):
    """POST /predict with a valid payload must return 200 and valid prediction."""
    response = client.post("/predict", json=sample_raw_order_dict)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["order_id"] == sample_raw_order_dict["order_id"]
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("on_time", "late")
    assert 0.0 <= data["probability"] <= 1.0
    assert "model_version" in data
    assert "threshold" in data


def test_api_predict_batch_orders(client, sample_raw_order_dict):
    """POST /predict/batch must process multiple orders."""
    order2 = sample_raw_order_dict.copy()
    order2["order_id"] = "test-order-002"
    order2["distance_km"] = 800.0

    batch_payload = {"orders": [sample_raw_order_dict, order2]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["order_id"] == sample_raw_order_dict["order_id"]
    assert data["predictions"][1]["order_id"] == "test-order-002"


def test_api_predict_rejects_missing_required_fields(client):
    """
    Definition of Done #4: Break something on purpose — bad data —
    and show the system catches it with HTTP 422 Unprocessable Entity.
    """
    bad_payload = {
        "order_id": "bad-order-missing-price",
        "customer_state": "SP",
        # Missing total_price and total_payment_value
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_api_predict_rejects_negative_price(client, sample_raw_order_dict):
    """
    Definition of Done #4: Break something on purpose — negative price violating ge=0.0 schema constraint.
    """
    bad_payload = sample_raw_order_dict.copy()
    bad_payload["total_price"] = -50.0  # Invalid negative price
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_api_metrics_route(client):
    """GET /metrics must expose telemetry metrics."""
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "total_requests" in data
    assert "errors_count" in data
    assert "avg_latency_ms" in data
