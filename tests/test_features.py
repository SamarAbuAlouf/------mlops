"""
tests/test_features.py
Unit tests for feature engineering and transformation pipeline in src/features.py.
"""

import numpy as np
from src.features import engineer_features, load_preprocessor, transform


def test_engineer_features_creates_required_columns(sample_raw_order_df):
    """Test that engineer_features generates macro-regions, ratios, and calendar features."""
    df_feat = engineer_features(sample_raw_order_df)

    assert "customer_region" in df_feat.columns
    assert "seller_region" in df_feat.columns
    assert "is_same_state" in df_feat.columns
    assert "is_inter_region" in df_feat.columns
    assert "freight_ratio" in df_feat.columns
    assert "density_g_cm3" in df_feat.columns
    assert "purchase_month" in df_feat.columns
    assert "purchase_dayofweek" in df_feat.columns
    assert "purchase_is_weekend" in df_feat.columns

    # Verify calculation values
    row = df_feat.iloc[0]
    assert row["customer_region"] == "Southeast"
    assert row["seller_region"] == "Southeast"
    assert row["is_same_state"] == 1
    assert row["is_inter_region"] == 0
    assert row["purchase_month"] == 1


def test_preprocessor_loading():
    """Verify that the preprocessor loads as a fitted scikit-learn ColumnTransformer."""
    preprocessor = load_preprocessor()
    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")


def test_transform_shape_and_type(sample_raw_order_df):
    """Verify that transform returns a 2D numpy array with correct number of encoded features."""
    X = transform(sample_raw_order_df)
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] == 86  # 16 numerical + one-hot encoded categories = 86 features
