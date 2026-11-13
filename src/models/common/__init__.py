"""Common Model Architecture Components.

Contains base forecast context specifications, output validation schemas, failure handling,
and model adapters that normalize heterogeneous forecasting routines into a uniform API.
"""

from src.models.common.base import (
    ForecastContext,
    ModelAdapterFn,
    build_failure_output,
    validate_forecast_output,
)
from src.models.common.adapters import (
    adapt_seasonal_naive,
    adapt_holt_winters,
    adapt_ar,
    adapt_ma,
    adapt_arima,
    adapt_sarima,
    adapt_arimax,
    adapt_sarimax,
    adapt_gradient_boosting,
    adapt_random_forest,
    adapt_xgboost,
    adapt_global_random_forest,
    adapt_global_xgboost,
    adapt_hurdle_xgboost,
    adapt_prophet,
    adapt_lstm,
    adapt_gru,
)

__all__ = [
    "ForecastContext",
    "ModelAdapterFn",
    "build_failure_output",
    "validate_forecast_output",
    "adapt_seasonal_naive",
    "adapt_holt_winters",
    "adapt_ar",
    "adapt_ma",
    "adapt_arima",
    "adapt_sarima",
    "adapt_arimax",
    "adapt_sarimax",
    "adapt_gradient_boosting",
    "adapt_random_forest",
    "adapt_xgboost",
    "adapt_global_random_forest",
    "adapt_global_xgboost",
    "adapt_hurdle_xgboost",
    "adapt_prophet",
    "adapt_lstm",
    "adapt_gru",
]

