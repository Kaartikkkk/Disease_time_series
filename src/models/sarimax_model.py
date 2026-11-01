"""Seasonal ARIMAX with Exogenous Climate Regressors (SARIMAX).

Mathematical Formulation:
    Phi_P(B^s) * phi_p(B) * (1 - B^s)^D * (1 - B)^d y_t = c + sum_{k=1}^K beta_k * X_{k, t-1} + Theta_Q(B^s) * theta_q(B) * epsilon_t
where:
    s = 12 (annual agricultural seasonal cycle in months)
    (p, d, q) = Non-seasonal orders
    (P, D, Q) = Seasonal orders (s=12)
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
    auto_optimize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit full SARIMAX model (Seasonal + Exogenous Climate) and generate forecast.

    Args:
        train_y: Historical disease series.
        horizon: Forecast horizon (months).
        order: Non-seasonal (p, d, q) orders.
        seasonal_order: Seasonal (P, D, Q, s) orders.
        train_exog: Historical lagged climate regressors.
        future_exog: Future horizon lagged climate regressors.
        auto_optimize: Whether to test candidate seasonal orders against AIC.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values
    exog_data = train_exog.bfill().fillna(0).values if train_exog is not None and not train_exog.empty else None
    fut_exog_data = future_exog.bfill().fillna(0).values if future_exog is not None and not future_exog.empty else None

    selected_order = order
    selected_seasonal = seasonal_order
    best_res = None

    if auto_optimize and exog_data is not None:
        candidates = [
            ((1, 0, 1), (1, 0, 1, 12)),
            ((1, 1, 1), (1, 0, 1, 12)),
            ((1, 0, 1), (0, 1, 1, 12)),
            ((1, 0, 0), (1, 0, 0, 12)),
        ]
        best_aic = float("inf")
        for ord_c, seas_c in candidates:
            try:
                m = SARIMAX(
                    y,
                    exog=exog_data,
                    order=ord_c,
                    seasonal_order=seas_c,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_cand = m.fit(disp=False, maxiter=50)
                if fit_cand.aic is not None and not np.isnan(fit_cand.aic) and fit_cand.aic < best_aic:
                    best_aic = fit_cand.aic
                    selected_order = ord_c
                    selected_seasonal = seas_c
                    best_res = fit_cand
            except Exception:
                continue

    if best_res is None:
        model = SARIMAX(
            y,
            exog=exog_data,
            order=selected_order,
            seasonal_order=selected_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        best_res = model.fit(disp=False, maxiter=100)

    forecast_res = best_res.get_forecast(steps=horizon, exog=fut_exog_data)

    point_pred = np.maximum(0.0, forecast_res.predicted_mean)
    conf_int = forecast_res.conf_int(alpha=0.05)
    lower_95 = np.maximum(0.0, conf_int[:, 0])
    upper_95 = conf_int[:, 1]

    n_feats = train_exog.shape[1] if train_exog is not None else 0
    setattr(best_res, "param_summary", f"Order={selected_order}, Seasonal={selected_seasonal}, Exog={n_feats} lags (AIC={best_res.aic:.1f})")

    return point_pred, lower_95, upper_95, best_res
