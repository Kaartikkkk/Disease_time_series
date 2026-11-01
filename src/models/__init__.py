"""Verdant Intelligence — Individual Time Series Forecasting Model Suite.

This package provides standalone, modular implementations for all 11 benchmark models:

1. Baseline & Smoothing:
   - Seasonal Naive (baseline): `from src.models.seasonal_naive import forecast_seasonal_naive`
   - Holt-Winters: `from src.models.holt_winters import forecast_holt_winters`

2. Box-Jenkins Classical Univariate:
   - AR(1): `from src.models.ar_model import forecast_ar`
   - MA(1): `from src.models.ma_model import forecast_ma`
   - ARIMA(1,1,1): `from src.models.arima_model import forecast_arima`
   - SARIMA(1,0,1)(1,0,1)[12]: `from src.models.sarima_model import forecast_sarima`

3. Exogenous Weather-Coupled Econometric:
   - ARIMAX: `from src.models.arimax_model import forecast_arimax`
   - SARIMAX: `from src.models.sarimax_model import forecast_sarimax_model`

4. Modern Machine Learning & Additive:
   - Meta Prophet: `from src.models.prophet_model import forecast_prophet, HAS_PROPHET`
   - Hist Gradient Boosting: `from src.models.gradient_boosting import forecast_gradient_boosting`
   - Random Forest: `from src.models.random_forest import forecast_random_forest`
   - ML Autoregressor Base: `from src.models.ml_autoregressor import forecast_ml_autoregressor`
"""

from src.models.ar_model import forecast_ar
from src.models.arima_model import forecast_arima
from src.models.arimax_model import forecast_arimax
from src.models.gradient_boosting import forecast_gradient_boosting
from src.models.holt_winters import forecast_holt_winters
from src.models.ma_model import forecast_ma
from src.models.ml_autoregressor import forecast_ml_autoregressor
from src.models.prophet_model import HAS_PROPHET, forecast_prophet
from src.models.random_forest import forecast_random_forest
from src.models.sarima_model import forecast_sarima
from src.models.sarimax_model import forecast_sarimax_model
from src.models.seasonal_naive import forecast_seasonal_naive

__all__ = [
    "forecast_seasonal_naive",
    "forecast_holt_winters",
    "forecast_ar",
    "forecast_ma",
    "forecast_arima",
    "forecast_sarima",
    "forecast_arimax",
    "forecast_sarimax_model",
    "forecast_prophet",
    "forecast_gradient_boosting",
    "forecast_random_forest",
    "forecast_ml_autoregressor",
    "HAS_PROPHET",
]
