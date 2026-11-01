"""Seasonal ARIMAX with Exogenous Climate Regressors (SARIMAX).

Mathematical Formulation:
    Phi_P(B^s) * phi_p(B) * (1 - B^s)^D * (1 - B)^d y_t = c + sum_{k=1}^K beta_k * X_{k, t-1} + Theta_Q(B^s) * theta_q(B) * epsilon_t
where:
    s = 12 (annual agricultural cycle in months)
    (p, d, q) = (1, 0, 1)
    (P, D, Q) = (1, 0, 1)
    X_{k, t-1} = 1-month-lagged NASA POWER meteorological regressors:
      - 2m Air Temperature (deg C)
      - Total Precipitation (mm/month)
      - Relative Humidity (%)
"""

from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_sarimax_model(
    train_y: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 12),
    train_exog: Optional[pd.DataFrame] = None,
    future_exog: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit full SARIMAX model (Seasonal + Exogenous Climate) and generate forecast.

    Args:
        train_y: Historical disease series.
        horizon: Forecast horizon (months).
        order: Non-seasonal (p, d, q) orders.
        seasonal_order: Seasonal (P, D, Q, s) orders.
        train_exog: Historical lagged climate regressors.
        future_exog: Future horizon lagged climate regressors.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values
    exog_data = train_exog.values if train_exog is not None and not train_exog.empty else None
    fut_exog_data = future_exog.values if future_exog is not None and not future_exog.empty else None

    model = SARIMAX(
        y,
        exog=exog_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=100)
    forecast_res = res.get_forecast(steps=horizon, exog=fut_exog_data)

    point_pred = np.maximum(0.0, forecast_res.predicted_mean)
    conf_int = forecast_res.conf_int(alpha=0.05)
    lower_95 = np.maximum(0.0, conf_int[:, 0])
    upper_95 = conf_int[:, 1]

    return point_pred, lower_95, upper_95, res
