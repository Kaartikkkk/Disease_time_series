"""Moving Average MA(1) Model.

Mathematical Formulation:
    y_t = mu + epsilon_t + theta_1 * epsilon_{t-1}
where:
    mu: Mean of the series
    theta_1: First-order moving average parameter (|theta_1| < 1 for invertibility)
    epsilon_t: Unobserved white noise shock at time t
"""

from typing import Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_ma(
    train_y: pd.Series,
    horizon: int,
    q: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit pure Moving Average MA(q) model and generate point forecast with intervals.

    Args:
        train_y: Historical series.
        horizon: Forecast horizon (months).
        q: Moving average error order (default 1).

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values
    model = SARIMAX(
        y,
        order=(0, 0, q),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=100)
    forecast_res = res.get_forecast(steps=horizon)

    point_pred = np.maximum(0.0, forecast_res.predicted_mean)
    conf_int = forecast_res.conf_int(alpha=0.05)
    lower_95 = np.maximum(0.0, conf_int[:, 0])
    upper_95 = conf_int[:, 1]

    setattr(res, "param_summary", f"p=0, d=0, q={q} (mu={res.params[0]:.2f}, theta1={res.params[1] if len(res.params)>1 else 0.0:.2f})")

    return point_pred, lower_95, upper_95, res
