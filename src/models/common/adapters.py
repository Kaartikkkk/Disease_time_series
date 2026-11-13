"""Common model adapters.

Normalizes all heterogeneous model signatures into a uniform
(ctx: ForecastContext) -> Dict[str, Any] interface.
"""
from src.models.adapters import *
from src.models.adapters import (
    _build_output,
    _extract_exog,
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
    forecast_global_xgboost,
    forecast_xgboost,
    forecast_gru,
    forecast_lstm,
    forecast_prophet,
    HAS_TORCH,
    HAS_PROPHET,
    HAS_XGBOOST,
)
