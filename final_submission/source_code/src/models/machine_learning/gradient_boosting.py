"""Histogram-Based Gradient Boosting Autoregressive Model.

Algorithm:
    LightGBM-inspired tree boosting (HistGradientBoostingRegressor).
    Optimizes least-squares loss using histogram-based binning of continuous
    autoregressive lags and lagged meteorological features.

    Features:
    - Autoregressive target lags: y_{t-1}, y_{t-2}, y_{t-3}, y_{t-12}
    - Annual calendar harmonics: sin(2*pi*m/12), cos(2*pi*m/12)
    - 1-month-lagged NASA POWER satellite weather regressors
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.models.machine_learning.ml_autoregressor import forecast_ml_autoregressor


def forecast_gradient_boosting(
    train_df: pd.DataFrame,
    horizon: int,
    lags: Optional[List[int]] = None,
    weather_lag_cols: Optional[List[str]] = None,
    future_weather_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit HistGradientBoosting autoregressive model and generate recursive predictions.

    Args:
        train_df: Historical training dataframe.
        horizon: Multi-step forecast horizon in months.
        lags: Target lag periods (default: [1, 2, 3, 12]).
        weather_lag_cols: Exogenous lagged climate column names.
        future_weather_df: Projected climate regressors for forecast horizon.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model).
    """
    return forecast_ml_autoregressor(
        train_df=train_df,
        horizon=horizon,
        model_type="gradient_boosting",
        lags=lags,
        weather_lag_cols=weather_lag_cols,
        future_weather_df=future_weather_df,
    )
