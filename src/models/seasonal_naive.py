"""Seasonal Naive Forecasting Model.

Mathematical Formulation:
    y_hat_{T+h} = y_{T+h-s}
where:
    s = seasonal period (12 months for annual agricultural crop cycles)
    h = forecast horizon step

Serves as the empirical persistence benchmark baseline. Every advanced statistical
and machine learning model must beat this baseline on out-of-sample RMSE.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def forecast_seasonal_naive(
    train_series: pd.Series,
    horizon: int,
    season_length: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seasonal Naive baseline: repeats observation from s periods ago.

    Args:
        train_series: Historical observation counts.
        horizon: Number of future months to predict.
        season_length: Periodicity (default 12 for monthly seasonal cycle).

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci).
    """
    y = train_series.values
    preds = []
    for h in range(1, horizon + 1):
        idx = len(y) + h - 1 - season_length
        if 0 <= idx < len(y):
            preds.append(y[idx])
        else:
            preds.append(y[-1])  # Fallback to simple persistence if history < 12

    preds_arr = np.array(preds, dtype=float)

    # Estimate empirical standard error from seasonal differences
    residuals = y[season_length:] - y[:-season_length] if len(y) > season_length else y[1:] - y[:-1]
    se = float(np.std(residuals)) if len(residuals) > 0 else 1.0

    lower_95 = np.maximum(0.0, preds_arr - 1.96 * se)
    upper_95 = preds_arr + 1.96 * se

    return preds_arr, lower_95, upper_95
