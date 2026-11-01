"""Seasonal Autoregressive Integrated Moving Average SARIMA(p,d,q)(P,D,Q)[s] Model.

Mathematical Formulation:
    Phi_P(B^s) * phi_p(B) * (1 - B^s)^D * (1 - B)^d y_t = c + Theta_Q(B^s) * theta_q(B) * epsilon_t
where:
    s = 12 (annual agricultural cycle in months)
    (p, d, q) = Non-seasonal orders: (1, 0, 1)
    (P, D, Q) = Seasonal orders: (1, 0, 1) with period 12
"""

from typing import Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_sarima(
    train_y: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 12),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit SARIMA(p, d, q)(P, D, Q)[s] model and generate point forecast with intervals.

    Args:
        train_y: Historical series.
        horizon: Forecast horizon (months).
        order: Non-seasonal (p, d, q) orders.
        seasonal_order: Seasonal (P, D, Q, s) orders.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
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
