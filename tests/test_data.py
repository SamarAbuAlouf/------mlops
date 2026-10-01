"""
tests/test_data.py
Data tests: schema, ranges, null rates, and leakage checks.

Requirement 6: Data tests: schema, ranges, nulls, leakage checks.
"""

import pytest
from src.features import ALL_FEATURES
from src.validation import DataValidator, DataValidationError


def test_data_leakage_target_exclusion():
    """Verify that target columns (is_late, delivery_delay_days, order_delivered_customer_date)
    are strictly excluded from the model feature set to prevent target leakage."""
    leakage_columns = [
        "is_late",
        "delivery_delay_days",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
    ]
    for leak_col in leakage_columns:
        assert (
            leak_col not in ALL_FEATURES
        ), f"Target leakage detected! '{leak_col}' found in ALL_FEATURES"


def test_validation_detects_out_of_bounds_range():
    """Verify that DataValidator detects out-of-range numerical features."""
    validator = DataValidator(on_failure="reject")
    bad_payload = {"distance_km": 99999.0}  # Expected max 5000 km
    with pytest.raises(DataValidationError) as exc:
        validator.validate(bad_payload)
    assert "distance_km" in str(exc.value)


def test_validation_detects_invalid_category():
    """Verify that DataValidator detects invalid categorical values."""
    validator = DataValidator(on_failure="reject")
    bad_payload = {"dominant_payment_type": "crypto_bitcoin"}
    with pytest.raises(DataValidationError) as exc:
        validator.validate(bad_payload)
    assert "dominant_payment_type" in str(exc.value)


def test_validation_default_strategy_clips_values():
    """Verify that 'default' on_failure strategy repairs invalid inputs without error."""
    validator = DataValidator(on_failure="default")
    bad_payload = {"distance_km": 99999.0, "total_price": 50.0}
    is_valid, issues, cleaned_df = validator.validate(bad_payload)
    assert not is_valid
    assert len(issues) > 0
    # Clamped to upper bound of 5000.0
    assert cleaned_df["distance_km"].iloc[0] == 5000.0
