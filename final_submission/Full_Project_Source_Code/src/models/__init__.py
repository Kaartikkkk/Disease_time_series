"""Agricultural Disease Forecasting — Modular Model Suite.

Organized into dedicated model families:
1. Classical Models: `src.models.classical` (AR, MA, ARIMA, SARIMA, ARIMAX, SARIMAX, Holt-Winters, Seasonal Naive)
2. Machine Learning: `src.models.machine_learning` (Random Forest, XGBoost, Global Models, Hurdle XGBoost, Gradient Boosting)
3. Deep Learning: `src.models.deep_learning` (PyTorch LSTM, GRU)
4. Probabilistic: `src.models.probabilistic` (Meta Prophet)
5. Common Infrastructure: `src.models.common` (ForecastContext, validation schemas, adapters)

Re-exports provided for backward-compatibility.
"""

from src.models.classical import (
    forecast_ar,
    forecast_arima,
    forecast_arimax,
    forecast_holt_winters,
    forecast_ma,
    forecast_sarima,
    forecast_sarimax_model,
    forecast_seasonal_naive,
)
from src.models.machine_learning import (
    HAS_XGBOOST,
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_gradient_boosting,
    forecast_hurdle_xgboost,
    forecast_ml_autoregressor,
    forecast_random_forest,
    forecast_xgboost,
)
from src.models.deep_learning import (
    HAS_TORCH,
    create_lstm_sequences,
    forecast_gru,
    forecast_lstm,
)
from src.models.probabilistic import (
    HAS_PROPHET,
    forecast_prophet,
    prepare_prophet_dataframe,
)
from src.models.common import (
    ForecastContext,
    ModelAdapterFn,
    build_failure_output,
    validate_forecast_output,
)

__all__ = [
    # Classical
    "forecast_seasonal_naive",
    "forecast_holt_winters",
    "forecast_ar",
    "forecast_ma",
    "forecast_arima",
    "forecast_sarima",
    "forecast_arimax",
    "forecast_sarimax_model",
    # Probabilistic
    "forecast_prophet",
    "prepare_prophet_dataframe",
    "HAS_PROPHET",
    # Machine Learning
    "forecast_gradient_boosting",
    "forecast_random_forest",
    "forecast_xgboost",
    "forecast_global_xgboost",
    "forecast_global_random_forest",
    "forecast_hurdle_xgboost",
    "forecast_ml_autoregressor",
    "HAS_XGBOOST",
    # Deep Learning
    "forecast_lstm",
    "forecast_gru",
    "create_lstm_sequences",
    "HAS_TORCH",
    # Common
    "ForecastContext",
    "ModelAdapterFn",
    "build_failure_output",
    "validate_forecast_output",
]
