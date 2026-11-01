"""Seasonal Autoregressive Integrated Moving Average SARIMA(p,d,q)(P,D,Q)[s] Model.

Mathematical Formulation:
    Phi_P(B^s) * phi_p(B) * (1 - B^s)^D * (1 - B)^d y_t = c + Theta_Q(B^s) * theta_q(B) * epsilon_t
where:
    s = 12 (annual agricultural seasonal cycle in monthly observations)
    (p, d, q) = Non-seasonal orders capturing immediate monthly autocorrelation
    (P, D, Q) = Seasonal orders capturing annual 12-month cyclical memory

Optimization:
    Evaluates parsimonious seasonal candidate grids against Akaike Information Criterion (AIC).
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def optimize_sarima_order(
    y: np.ndarray,
    seasonal_period: int = 12,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Select the best (order, seasonal_order) minimizing AIC on training data."""
    candidates = [
        ((1, 0, 1), (1, 0, 1, seasonal_period)),
        ((1, 1, 1), (1, 0, 1, seasonal_period)),
        ((1, 0, 1), (0, 1, 1, seasonal_period)),
        ((0, 1, 1), (1, 0, 1, seasonal_period)),
        ((1, 1, 0), (1, 0, 1, seasonal_period)),
        ((1, 0, 0), (1, 0, 0, seasonal_period)),
        ((1, 0, 1), (1, 0, 0, seasonal_period)),
    ]

    best_aic = float("inf")
    best_order = (1, 0, 1)
    best_seasonal = (1, 0, 1, seasonal_period)

    for ord_cand, s_cand in candidates:
        try:
            model = SARIMAX(
                y,
                order=ord_cand,
                seasonal_order=s_cand,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=50)
            if res.aic is not None and not np.isnan(res.aic) and res.aic < best_aic:
                best_aic = res.aic
                best_order = ord_cand
                best_seasonal = s_cand
        except Exception:
            continue

    return best_order, best_seasonal


def forecast_sarima(
    train_y: pd.Series,
    horizon: int,
    order: Optional[Tuple[int, int, int]] = (1, 0, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = (1, 0, 1, 12),
    auto_optimize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit SARIMA(p, d, q)(P, D, Q)[s] model and generate point forecast with intervals.

    Args:
        train_y: Historical series.
        horizon: Forecast horizon (months).
        order: Non-seasonal (p, d, q) orders.
        seasonal_order: Seasonal (P, D, Q, s) orders.
        auto_optimize: Whether to search for optimal AIC seasonal order.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values

    if auto_optimize:
        sel_order, sel_seasonal = optimize_sarima_order(y, seasonal_period=12)
    else:
        sel_order = order or (1, 0, 1)
        sel_seasonal = seasonal_order or (1, 0, 1, 12)

    model = SARIMAX(
        y,
        order=sel_order,
        seasonal_order=sel_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=100)
    forecast_res = res.get_forecast(steps=horizon)

    point_pred = np.maximum(0.0, forecast_res.predicted_mean)
    conf_int = forecast_res.conf_int(alpha=0.05)
    lower_95 = np.maximum(0.0, conf_int[:, 0])
    upper_95 = conf_int[:, 1]

    setattr(res, "selected_order", sel_order)
    setattr(res, "selected_seasonal_order", sel_seasonal)
    setattr(res, "param_summary", f"Order={sel_order}, Seasonal={sel_seasonal} (AIC={res.aic:.1f})")

    return point_pred, lower_95, upper_95, res
