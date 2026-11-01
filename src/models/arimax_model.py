"""Autoregressive Integrated Moving Average with Exogenous Climate Regressors (ARIMAX).

Mathematical Formulation:
    phi(B) * (1 - B)^d y_t = c + sum_{k=1}^K beta_k * X_{k, t-1} + theta(B) * epsilon_t
where:
    X_{k, t-1} represents 1-month-lagged exogenous agro-climatic variables:
    - Lagged Mean Temperature (deg C)
    - Lagged Monthly Precipitation (mm)
    - Lagged Relative Humidity (%)
Lagging climate regressors by 1 month prevents future information leakage (lookahead bias).
"""

from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_arimax(
    train_y: pd.Series,
    horizon: int,
    train_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 0, 1),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit ARIMAX model with exogenous climate features and generate forecast.

    Args:
        train_y: Historical disease series.
        horizon: Forecast horizon (months).
        train_exog: Historical lagged weather regressors.
        future_exog: Projected or scenario lagged weather regressors for horizon.
        order: (p, d, q) orders. Default is (1, 0, 1).

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
