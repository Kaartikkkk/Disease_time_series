"""Backward-compatibility facade for src.models.base.
Canonical implementation is located in src.models.common.base.
"""
from src.models.common.base import (
    ForecastContext,
    ModelAdapterFn,
    validate_forecast_output,
    build_failure_output,
    _to_1d_float_array,
)

__all__ = [
    "ForecastContext",
    "ModelAdapterFn",
    "validate_forecast_output",
    "build_failure_output",
]
