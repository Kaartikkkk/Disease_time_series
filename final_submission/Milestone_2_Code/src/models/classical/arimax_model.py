"""Autoregressive Integrated Moving Average with Exogenous Climate Regressors (ARIMAX).

Mathematical Formulation:
    phi(B) * (1 - B)^d y_t = c + sum_{k=1}^K beta_k * X_{k, t-1} + theta(B) * epsilon_t
where:
    X_{k, t-1} represents 1-month-lagged exogenous agro-climatic variables:
    - Lagged Mean Temperature (deg C)
    - Lagged Monthly Precipitation (mm)
    - Lagged Relative Humidity (%)

Leakage Prevention:
    Exogenous regressors are strictly lagged by 1 month (t-1). At time t, only weather
    through month t-1 is utilized, completely eliminating lookahead bias.
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


from sklearn.preprocessing import StandardScaler


def forecast_arimax(
    train_y: pd.Series,
    horizon: int,
    train_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 0, 1),
    auto_optimize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit ARIMAX model with exogenous climate features and generate forecast.

    Args:
        train_y: Historical disease series.
        horizon: Forecast horizon (months).
        train_exog: Historical lagged weather regressors.
        future_exog: Projected or scenario lagged weather regressors for horizon.
        order: Non-seasonal (p, d, q) orders.
        auto_optimize: Whether to test candidate orders against AIC.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_y.astype(float).values

    exog_data = None
    fut_exog_data = None
    if train_exog is not None and not train_exog.empty:
        scaler = StandardScaler()
        clean_train_exog = train_exog.ffill().bfill().fillna(0)
        exog_data = scaler.fit_transform(clean_train_exog)

        if future_exog is not None and not future_exog.empty:
            train_medians = clean_train_exog.median()
            clean_fut_exog = future_exog.ffill().fillna(train_medians).fillna(0)
            fut_exog_data = scaler.transform(clean_fut_exog)

    selected_order = order
    best_res = None

    if auto_optimize and exog_data is not None:
        candidates = [(1, 0, 0), (1, 0, 1), (0, 0, 1), (1, 1, 1), (0, 1, 1)]
        best_aic = float("inf")
        for cand in candidates:
            trend = "c" if cand[1] == 0 else None
            try:
                m = SARIMAX(
                    y,
                    exog=exog_data,
                    order=cand,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_cand = m.fit(disp=False, maxiter=60)
                if fit_cand.aic is not None and not np.isnan(fit_cand.aic) and fit_cand.aic < best_aic:
                    best_aic = fit_cand.aic
                    selected_order = cand
                    best_res = fit_cand
            except Exception:
                continue

    if best_res is None:
        trend = "c" if selected_order[1] == 0 else None
        model = SARIMAX(
            y,
            exog=exog_data,
            order=selected_order,
            trend=trend,
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
    setattr(best_res, "param_summary", f"Order={selected_order}, ExogFeatures={n_feats} lags (AIC={best_res.aic:.1f})")

    return point_pred, lower_95, upper_95, best_res
