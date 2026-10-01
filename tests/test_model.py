"""
tests/test_model.py
Model tests: model loads, predicts valid shape and probability ranges, behaves on known inputs.

Requirement 6: Model tests: the model loads, predicts the right shape, behaves on known inputs.
"""

import numpy as np
from src.predict import load_model, predict, predict_batch


def test_model_loads_successfully():
    """Verify that the model artifact loads and has predict_proba method."""
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_predict_single_sample_structure():
    """Verify predict returns a dictionary with the required fields and valid probability range."""
    # Synthetic feature vector with 86 features
    X = np.zeros((1, 86))
    res = predict(X, order_id="test-mock-01")

    assert "prediction" in res
    assert res["prediction"] in (0, 1)
    assert "label" in res
    assert res["label"] in ("on_time", "late")
    assert "probability" in res
    assert 0.0 <= res["probability"] <= 1.0
    assert "model_version" in res
    assert "threshold" in res


def test_predict_batch_returns_list_of_results():
    """Verify predict_batch outputs a list of prediction results matching input rows."""
    X = np.zeros((5, 86))
    results = predict_batch(X)

    assert isinstance(results, list)
    assert len(results) == 5
    for r in results:
        assert 0.0 <= r["probability"] <= 1.0
        assert r["prediction"] in (0, 1)
