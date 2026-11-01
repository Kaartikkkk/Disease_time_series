"""Holt-Winters Triple Exponential Smoothing Model.

Mathematical Formulation (Additive Trend and Additive Seasonality):
    Level:    l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
    Trend:    b_t = beta  * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
    Seasonal: s_t = gamma * (y_t - l_t)     + (1 - gamma) * s_{t-m}
    Forecast: y_hat_{t+h} = l_t + h * b_t + s_{t+h-m}

where:
    alpha: Level smoothing coefficient [0, 1]
    beta:  Trend smoothing coefficient [0, 1]
    gamma: Seasonal smoothing coefficient [0, 1]
    m:     Seasonal period (12 for annual monthly cycles)
"""

from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing


def forecast_holt_winters(
    train_series: pd.Series,
    horizon: int,
    seasonal_periods: int = 12,
    trend: str = "add",
    seasonal: Optional[str] = "add",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit Holt-Winters Exponential Smoothing model and generate predictions.

    Args:
        train_series: Historical series of disease reports.
        horizon: Forecast horizon (number of months).
        seasonal_periods: Number of periods in a complete seasonal cycle (12 for monthly).
        trend: Type of trend component ("add" or None).
        seasonal: Type of seasonal component ("add" or None).

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_series.astype(float).values

    # If series length is less than 2 seasonal cycles, disable seasonal term
    if len(y) < 2 * seasonal_periods:
        seasonal = None

    model = ExponentialSmoothing(
        y,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods if seasonal else None,
        initialization_method="estimated",
    )
    res = model.fit()
    point_pred = np.maximum(0.0, res.forecast(horizon))

    # Empirical interval estimate from training residuals
    residuals = res.resid
    se = float(np.std(residuals)) if len(residuals) > 0 else 1.0
    lower_95 = np.maximum(0.0, point_pred - 1.96 * se)
    upper_95 = point_pred + 1.96 * se

    return point_pred, lower_95, upper_95, res
