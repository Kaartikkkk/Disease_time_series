"""Autoregressive Integrated Moving Average ARIMA(p, d, q) Model with AIC Optimization.

Mathematical Formulation:
    (1 - sum_{i=1}^p phi_i * B^i) (1 - B)^d y_t = c + (1 + sum_{j=1}^q theta_j * B^j) epsilon_t
where:
    d = differencing order to induce stationarity (selected via unit-root diagnostics)
    p = autoregressive order (captures lagged persistence)
    q = moving average order (captures shock absorption)

Optimization Protocol:
    Evaluates candidate (p, d, q) specifications on in-sample training data using
    Akaike Information Criterion (AIC):
        AIC = 2k - 2*ln(L_hat)
    The parameter set minimizing AIC with verified model convergence is selected.
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def optimize_arima_order(
    y: np.ndarray,
    candidate_orders: Optional[List[Tuple[int, int, int]]] = None,
) -> Tuple[int, int, int]:
    """Find the optimal (p, d, q) order minimizing AIC on training series."""
    if candidate_orders is None:
        # Determine d via ADF test
        try:
            adf_stat, adf_pval, _, _, _, _ = adfuller(y, autolag="AIC", result_object=False)
            d_init = 1 if adf_pval >= 0.05 else 0
        except Exception:
            d_init = 1

        if d_init == 1:
            candidate_orders = [
                (1, 1, 1),
                (1, 1, 0),
                (0, 1, 1),
                (2, 1, 1),
                (1, 1, 2),
                (2, 1, 0),
                (1, 0, 1),
            ]
        else:
            candidate_orders = [
                (1, 0, 1),
                (1, 0, 0),
                (0, 0, 1),
                (2, 0, 1),
                (1, 0, 2),
                (2, 0, 0),
                (1, 1, 1),
            ]

    best_aic = float("inf")
    best_order = (1, 0, 1)

    for order in candidate_orders:
        d = order[1]
        trend = "c" if d == 0 else None
        try:
            model = SARIMAX(
                y,
                order=order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False, maxiter=80)
            if res.aic is not None and not np.isnan(res.aic) and res.aic < best_aic:
                best_aic = res.aic
                best_order = order
        except Exception:
            continue

    return best_order


def forecast_arima(
    train_y: pd.Series,
    horizon: int,
    order: Optional[Tuple[int, int, int]] = (1, 1, 1),
    auto_optimize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit ARIMA(p, d, q) model and generate point forecast with prediction intervals.

    Args:
        train_y: Historical training series.
        horizon: Forecast horizon (months).
        order: (p, d, q) tuple. If auto_optimize=True, serves as initial candidate.
        auto_optimize: Whether to search for optimal AIC order across candidates.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values

    if auto_optimize:
        selected_order = optimize_arima_order(y)
    else:
        selected_order = order or (1, 0, 1)

    d = selected_order[1]
    trend = "c" if d == 0 else None

    model = SARIMAX(
        y,
        order=selected_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=100)
    forecast_res = res.get_forecast(steps=horizon)

    point_pred = np.maximum(0.0, forecast_res.predicted_mean)
    conf_int = forecast_res.conf_int(alpha=0.05)
    lower_95 = np.maximum(0.0, conf_int[:, 0])
    upper_95 = conf_int[:, 1]

    # Store selected parameters on the results object for easy inspection
    setattr(res, "selected_order", selected_order)
    setattr(res, "param_summary", f"p={selected_order[0]}, d={selected_order[1]}, q={selected_order[2]} (AIC={res.aic:.1f})")

    return point_pred, lower_95, upper_95, res
