"""Common Forecasting Model Contract and Output Validation.

Defines the lightweight `ForecastContext` input container and standardized
output validation used by all classical, machine learning, and Milestone 3
forecasting adapters.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ForecastContext:
    """Validation-safe inputs passed to every forecasting model adapter."""

    train_df: pd.DataFrame
    horizon: int
    target_col: str = "documented_occurrence_count"
    exog_cols: Optional[List[str]] = None
    future_df: Optional[pd.DataFrame] = None
    config: Dict[str, Any] = field(default_factory=dict)


ModelAdapterFn = Callable[[ForecastContext], Dict[str, Any]]


def _to_1d_float_array(values: Any, field_name: str, expected_horizon: int) -> np.ndarray:
    """Convert array-like output to a finite 1D float array matching expected_horizon."""
    if values is None:
        raise ValueError(f"'{field_name}' must not be None.")
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must contain numeric values: {exc}") from exc

    if len(arr) != expected_horizon:
        raise ValueError(
            f"'{field_name}' length ({len(arr)}) does not match requested horizon ({expected_horizon})."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{field_name}' contains NaN or infinite values.")
    return arr


def validate_forecast_output(
    output: Dict[str, Any],
    expected_horizon: int,
    clip_non_negative: bool = True,
) -> Dict[str, Any]:
    """Validate and normalize a model adapter's standardized output dictionary.

    Contract:
        {
            "predictions": np.ndarray,
            "lower_bound": Optional[np.ndarray],
            "upper_bound": Optional[np.ndarray],
            "model_name": str,
            "metadata": dict,
        }
    """
    if not isinstance(output, dict):
        raise ValueError("Model output must be a dictionary.")
    if expected_horizon <= 0:
        raise ValueError(f"expected_horizon must be positive, got {expected_horizon}.")

    preds = _to_1d_float_array(output.get("predictions"), "predictions", expected_horizon)
    if clip_non_negative:
        preds = np.maximum(0.0, preds)

    raw_lb = output.get("lower_bound")
    raw_ub = output.get("upper_bound")

    if (raw_lb is None) != (raw_ub is None):
        raise ValueError("Both 'lower_bound' and 'upper_bound' must be provided together or both be None.")

    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None
    if raw_lb is not None and raw_ub is not None:
        lb = _to_1d_float_array(raw_lb, "lower_bound", expected_horizon)
        ub = _to_1d_float_array(raw_ub, "upper_bound", expected_horizon)
        if clip_non_negative:
            lb = np.maximum(0.0, lb)
            ub = np.maximum(0.0, ub)
        if np.any(lb > ub):
            raise ValueError("Invalid confidence interval: lower_bound exceeds upper_bound.")

    model_name = str(output.get("model_name") or "unknown")
    metadata = dict(output.get("metadata") or {})

    return {
        "predictions": preds,
        "lower_bound": lb,
        "upper_bound": ub,
        "model_name": model_name,
        "metadata": metadata,
    }


def build_failure_output(
    model_name: str,
    expected_horizon: int,
    error: Exception,
    model_family: str = "",
) -> Dict[str, Any]:
    """Build a structured failure output using NaN (never 0.0) for failed predictions."""
    return {
        "predictions": np.full(expected_horizon, np.nan, dtype=float),
        "lower_bound": None,
        "upper_bound": None,
        "model_name": model_name,
        "metadata": {
            "status": "failed",
            "error": str(error),
            "model_family": model_family,
        },
    }

