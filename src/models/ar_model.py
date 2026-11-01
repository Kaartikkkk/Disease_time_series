"""Autoregressive AR(1) Model.

Mathematical Formulation:
    y_t = c + phi_1 * y_{t-1} + epsilon_t
where:
    c: Constant intercept
    phi_1: First-order autoregressive coefficient (|phi_1| < 1 for stationarity)
    epsilon_t: White noise shock ~ N(0, sigma^2)
"""

from typing import Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_ar(
    train_y: pd.Series,
    horizon: int,
    p: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit pure Autoregressive AR(p) model and generate point forecast with intervals.

    Args:
        train_y: Historical series.
        horizon: Forecast horizon (months).
        p: Autoregressive lag order (default 1).

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values
    model = SARIMAX(
        y,
        order=(p, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=100)
    forecast_res = res.get_forecast(steps=horizon)

    point_pred = np.maximum(0.0, forecast_res.predicted_mean)
    conf_int = forecast_res.conf_int(alpha=0.05)
    lower_95 = np.maximum(0.0, conf_int[:, 0])
    upper_95 = conf_int[:, 1]

    return point_pred, lower_95, upper_95, res
