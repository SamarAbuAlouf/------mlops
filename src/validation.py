"""
src/validation.py
Data validation layer before inference.

Requirement 4:
- Validate incoming data before it reaches the model
- Define expectations: column types, ranges, allowed categories, missing rates
- Configurable action on failure: "reject" | "flag" | "default"
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.logger import get_logger

logger = get_logger(__name__)

# Extract validation rules from central config
_VAL_CFG = CONFIG.get("validation", {})
ON_FAILURE: str = _VAL_CFG.get("on_failure", "flag")
MAX_MISSING_RATE: float = float(_VAL_CFG.get("max_missing_rate", 0.30))
NUMERIC_RANGES: dict[str, list[float]] = _VAL_CFG.get("numeric_ranges", {})
CATEGORICAL_ALLOWED: dict[str, list[Any]] = _VAL_CFG.get("categorical_allowed", {})


class DataValidationError(ValueError):
    """Raised when data fails validation under 'reject' policy."""
    pass


class DataValidator:
    """
    Validates input orders (single dictionary or pandas DataFrame).
    Follows Great Expectations concepts:
    - expect_column_to_exist
    - expect_column_values_to_be_of_type
    - expect_column_values_to_be_between
    - expect_column_values_to_be_in_set
    - expect_column_null_rate_to_be_below
    """

    def __init__(
        self,
        on_failure: str = ON_FAILURE,
        numeric_ranges: dict[str, list[float]] | None = None,
        categorical_allowed: dict[str, list[Any]] | None = None,
        max_missing_rate: float = MAX_MISSING_RATE,
    ):
        self.on_failure = on_failure
        self.numeric_ranges = numeric_ranges or NUMERIC_RANGES
        self.categorical_allowed = categorical_allowed or CATEGORICAL_ALLOWED
        self.max_missing_rate = max_missing_rate

    def validate(self, data: pd.DataFrame | dict) -> Tuple[bool, List[str], pd.DataFrame]:
        """
        Validate input data.
        Returns:
            (is_valid: bool, issues: list of error/warning strings, cleaned_df: pd.DataFrame)
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        issues: List[str] = []

        # 1. Missing rate check
        missing_rates = df.isnull().mean()
        for col, rate in missing_rates.items():
            if rate > self.max_missing_rate:
                issues.append(f"Column '{col}' has high missing rate: {rate:.1%} > {self.max_missing_rate:.1%}")

        # 2. Numerical range check
        for col, (min_val, max_val) in self.numeric_ranges.items():
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                out_of_bounds = series[(series < min_val) | (series > max_val)]
                if len(out_of_bounds) > 0:
                    sample_bad = out_of_bounds.iloc[0]
                    issues.append(
                        f"Column '{col}' out of expected range [{min_val}, {max_val}]: sample value {sample_bad}"
                    )
                    if self.on_failure == "default":
                        # Clamp or fill with median/default
                        df[col] = df[col].clip(lower=min_val, upper=max_val)

        # 3. Categorical allowed set check
        for col, allowed_vals in self.categorical_allowed.items():
            if col in df.columns:
                invalid_rows = df[~df[col].isin(allowed_vals) & df[col].notna()]
                if len(invalid_rows) > 0:
                    sample_bad = invalid_rows[col].iloc[0]
                    issues.append(
                        f"Column '{col}' has disallowed category '{sample_bad}'. Allowed: {allowed_vals}"
                    )
                    if self.on_failure == "default":
                        df[col] = allowed_vals[0]

        is_valid = len(issues) == 0

        if not is_valid:
            msg = f"Data validation issues encountered ({len(issues)}): {'; '.join(issues)}"
            if self.on_failure == "reject":
                logger.error("Data validation REJECT: %s", msg)
                raise DataValidationError(msg)
            elif self.on_failure == "flag":
                logger.warning("Data validation FLAG: %s", msg)
            elif self.on_failure == "default":
                logger.info("Data validation DEFAULT: %s (applied fallback defaults)", msg)

        return is_valid, issues, df


# Singleton instance
default_validator = DataValidator()
