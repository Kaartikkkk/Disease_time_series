"""Validation utilities for series, bounds, dates, and schema compliance."""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np


def validate_time_series_input(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "incidence_rate",
    min_length: int = 14,
) -> Tuple[bool, List[str]]:
    """Validate that a DataFrame is a well-formed temporal series.

    Returns:
        (is_valid, error_messages)
    """
    errors = []
    if df.empty:
        return False, ["DataFrame is empty."]

    if date_col not in df.columns:
        errors.append(f"Missing required date column: '{date_col}'")

    if target_col not in df.columns:
        errors.append(f"Missing required target column: '{target_col}'")

    if errors:
        return False, errors

    # Check date column conversion
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            pd.to_datetime(df[date_col])
        except Exception as e:
            errors.append(f"Date column '{date_col}' cannot be parsed into datetime: {e}")

    # Check chronological ordering
    if not df[date_col].is_monotonic_increasing:
        errors.append("Observations are not strictly chronologically sorted.")

    # Check target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        errors.append(f"Target column '{target_col}' must be numeric.")
    else:
        # Check finite values
        valid_targets = df[target_col].dropna()
        if len(valid_targets) < min_length:
            errors.append(
                f"Insufficient non-null observations: found {len(valid_targets)}, required >= {min_length}."
            )
        if not np.isfinite(valid_targets).all():
            errors.append(f"Target column '{target_col}' contains infinite or invalid values.")

    return len(errors) == 0, errors


def enforce_target_bounds(
    predictions: np.ndarray,
    lower_bound: float = 0.0,
    upper_bound: Optional[float] = None,
) -> np.ndarray:
    """Ensure forecasted values do not violate natural domain bounds (e.g., non-negative incidence)."""
    bounded = np.maximum(predictions, lower_bound)
    if upper_bound is not None:
        bounded = np.minimum(bounded, upper_bound)
    return bounded
