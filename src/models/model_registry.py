"""Model registry and factory for time series forecasting models."""

from typing import Dict, Any, List, Optional
from src.models.baselines import (
    BaseForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    HistoricalSeasonalMeanForecaster,
)
from src.models.statistical_models import (
    ARIMAForecaster,
    SARIMAForecaster,
    ExponentialSmoothingForecaster,
)
from src.models.machine_learning_models import MLForecaster

from src.models.advanced_classical_models import (
    ARForecaster,
    MAForecaster,
    ARMAForecaster,
    ARIMAXForecaster,
    SARIMAXForecaster,
    VARForecaster,
    ProphetForecaster,
    AutoTSForecaster,
)

AVAILABLE_MODELS = {
    # Baselines
    "persistence": "Persistence (Last Observation Naïve)",
    "seasonal_naive": "Seasonal Naïve",
    "moving_average": "Moving Average",
    "historical_mean": "Historical Seasonal Mean",
    # Classical Statistical Models (Course Syllabus Models)
    "ar": "AR(p) - Autoregressive Model",
    "ma": "MA(q) - Moving Average Model",
    "arma": "ARMA(p,q) - Autoregressive Moving Average",
    "arima": "ARIMA(p,d,q) - Auto-Selected AIC",
    "sarima": "SARIMA(p,d,q)(P,D,Q)s - Seasonal ARIMA",
    "exponential_smoothing": "Holt-Winters - Exponential Smoothing",
    "arimax": "ARIMAX - ARIMA with Weather Regressors",
    "sarimax": "SARIMAX - Seasonal ARIMA with Weather",
    "var": "VAR - Vector Autoregression (Multivariate)",
    # Modern & Automated Frameworks
    "prophet": "Prophet - Additive Trend & Holiday Model (Meta)",
    "autots": "AutoTS - Automated Evolutionary Model Selection",
    # Machine Learning
    "random_forest": "Random Forest Regressor",
    "hist_gradient_boosting": "HistGradientBoosting Regressor",
    "xgboost": "XGBoost Regressor",
}


def get_model(
    model_name: str,
    seasonal_period: int = 7,
    frequency: str = "D",
    random_seed: int = 42,
    include_weather: bool = True,
) -> BaseForecaster:
    """Instantiate a forecaster by identifier with appropriate configuration."""
    m_name = model_name.lower().replace("-", "_")

    if m_name == "persistence":
        return PersistenceForecaster()
    elif m_name == "seasonal_naive":
        return SeasonalNaiveForecaster(seasonal_period=seasonal_period)
    elif m_name == "moving_average":
        return MovingAverageForecaster(window=seasonal_period or 7)
    elif m_name == "historical_mean":
        return HistoricalSeasonalMeanForecaster(frequency=frequency)
    elif m_name == "ar":
        return ARForecaster(p=min(3, seasonal_period))
    elif m_name == "ma":
        return MAForecaster(q=2)
    elif m_name == "arma":
        return ARMAForecaster(p=2, q=1)
    elif m_name == "arima":
        return ARIMAForecaster(auto_order=True)
    elif m_name == "sarima":
        s = seasonal_period if seasonal_period > 1 else 7
        return SARIMAForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 0, s))
    elif m_name in ["exponential_smoothing", "holt_winters"]:
        s = seasonal_period if seasonal_period > 1 else 7
        return ExponentialSmoothingForecaster(trend="add", seasonal="add", seasonal_periods=s)
    elif m_name == "arimax":
        return ARIMAXForecaster(order=(1, 1, 1))
    elif m_name == "sarimax":
        s = seasonal_period if seasonal_period > 1 else 7
        return SARIMAXForecaster(order=(1, 1, 1), seasonal_order=(1, 1, 0, s))
    elif m_name == "var":
        return VARForecaster(p=min(2, seasonal_period))
    elif m_name == "prophet":
        return ProphetForecaster()
    elif m_name == "autots":
        return AutoTSForecaster(max_generations=1, num_validations=1)
    elif m_name in ["random_forest", "hist_gradient_boosting", "xgboost"]:
        return MLForecaster(
            model_type=m_name,
            random_seed=random_seed,
            frequency=frequency,
            include_weather=include_weather,
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(AVAILABLE_MODELS.keys())}"
        )


def list_available_models() -> Dict[str, str]:
    """Return dictionary of available model codes and human-readable names."""
    return AVAILABLE_MODELS.copy()
