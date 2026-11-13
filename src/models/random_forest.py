"""Random Forest Autoregressive Model.

Algorithm:
    Bootstrap aggregated ensemble of 100 decision trees (RandomForestRegressor).
    Captures non-linear feature interactions between historical disease lags
    and meteorological thresholds (e.g. Temperature x Humidity interaction).

    Features:
    - Autoregressive target lags: y_{t-1}, y_{t-2}, y_{t-3}, y_{t-12}
    - Annual calendar harmonics: sin(2*pi*m/12), cos(2*pi*m/12)
    - 1-month-lagged NASA POWER satellite weather regressors
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.models.ml_autoregressor import forecast_ml_autoregressor


def forecast_random_forest(
    train_df: pd.DataFrame,
    horizon: int,
    lags: Optional[List[int]] = None,
    weather_lag_cols: Optional[List[str]] = None,
    future_weather_df: Optional[pd.DataFrame] = None,
    rf_params: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit Random Forest autoregressive model and generate recursive predictions.

    Args:
        train_df: Historical training dataframe.
        horizon: Multi-step forecast horizon in months.
        lags: Target lag periods (default: [1, 2, 3, 12]).
        weather_lag_cols: Exogenous lagged climate column names.
        future_weather_df: Projected climate regressors for forecast horizon.
        rf_params: Optional dictionary of RandomForestRegressor hyperparameters.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model).
    """
    return forecast_ml_autoregressor(
        train_df=train_df,
        horizon=horizon,
        model_type="random_forest",
        lags=lags,
        weather_lag_cols=weather_lag_cols,
        future_weather_df=future_weather_df,
        model_params=rf_params,
    )
