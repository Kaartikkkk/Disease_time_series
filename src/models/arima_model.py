"""Autoregressive Integrated Moving Average ARIMA(p, d, q) Model.

Mathematical Formulation:
    (1 - sum_{i=1}^p phi_i * B^i) (1 - B)^d y_t = c + (1 + sum_{j=1}^q theta_j * B^j) epsilon_t

In the benchmark, the default configuration is ARIMA(1, 1, 1):
    (1 - phi_1 * B) (1 - B) y_t = c + (1 + theta_1 * B) epsilon_t
where:
    d = 1 handles non-stationarity via first-differencing (nabla y_t = y_t - y_{t-1})
    p = 1 captures short-term autoregressive autocorrelation
    q = 1 captures shock persistence
"""

from typing import Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_arima(
    train_y: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit ARIMA(p, d, q) model and generate point forecast with prediction intervals.

    Args:
        train_y: Historical training series.
        horizon: Forecast horizon (months).
        order: (p, d, q) tuple. Default is (1, 1, 1).

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values
    model = SARIMAX(
        y,
        order=order,
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
