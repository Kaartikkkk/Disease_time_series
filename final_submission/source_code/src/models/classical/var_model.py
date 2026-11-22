"""Vector Autoregression (VAR) Model for Joint Pathogen-Weather Endogenous Systems.

Mathematical Formulation:
    Y_t = c + sum_{i=1}^p A_i Y_{t-i} + epsilon_t
where:
    Y_t = [y_{t, disease}, x_{t, temp}, x_{t, humidity}]^T is a K-dimensional endogenous vector
    c is a K x 1 intercept vector
    A_i are K x K coefficient matrices capturing dynamic cross-variable feedback
    epsilon_t ~ WN(0, Sigma) is white noise disturbance

Degrees-of-Freedom Management:
    A K-variable VAR(p) estimates K + p * K^2 parameters. With K=3 variables:
    - VAR(1): 3 + 1 * 9 = 12 parameters
    - VAR(2): 3 + 2 * 9 = 21 parameters
    - VAR(3): 3 + 3 * 9 = 30 parameters
    To prevent singular covariance matrices on N ~ 80-120 observations, lag selection
    is restricted to maxlag=2 or 3 using Akaike Information Criterion (AIC).
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def forecast_var(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    exog_cols: Optional[List[str]] = None,
    maxlags: int = 2,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Any, Dict[str, Any]]:
    """Fit a Vector Autoregression (VAR) model and generate out-of-sample multi-step forecasts.

    Args:
        train_df: Historical training DataFrame.
        horizon: Forecast horizon (months).
        target_col: Name of disease occurrence count column.
        exog_cols: Optional list of additional endogenous driver columns (e.g. temp, humidity).
        maxlags: Maximum lag order to consider for AIC order selection (default 2).

    Returns:
        Tuple of (point_predictions, lower_bound, upper_bound, fitted_model, diagnostics).
    """
    t_start = time.perf_counter()

    df_sorted = train_df.copy()
    if "date" in df_sorted.columns:
        df_sorted["date"] = pd.to_datetime(df_sorted["date"])
        df_sorted = df_sorted.sort_values("date").reset_index(drop=True)

    # Determine endogenous system variables
    candidate_drivers = ["temp_mean_c", "relative_humidity_pct", "precipitation_monthly_mm"]
    if exog_cols:
        active_drivers = [c for c in exog_cols if c in df_sorted.columns and c != target_col]
    else:
        active_drivers = [c for c in candidate_drivers if c in df_sorted.columns]

    # Limit to top 2 drivers so total variables K <= 3, preserving degrees of freedom
    active_drivers = active_drivers[:2]
    var_cols = [target_col] + active_drivers

    data_matrix = df_sorted[var_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    n_obs, k_vars = data_matrix.shape

    if n_obs < 15:
        raise ValueError(f"Insufficient observations ({n_obs}) to estimate VAR model.")

    # Safe maxlag bounding to ensure N > K * p + 5
    safe_maxlag = max(1, min(maxlags, (n_obs - 10) // (k_vars * k_vars + 1)))

    model = VAR(data_matrix.values)
    try:
        lag_order_results = model.select_order(maxlags=safe_maxlag)
        selected_lag = lag_order_results.aic
        if selected_lag is None or selected_lag < 1:
            selected_lag = 1
    except Exception:
        selected_lag = 1

    selected_lag = max(1, min(selected_lag, safe_maxlag))

    # Fit VAR model
    res = model.fit(maxlags=selected_lag, ic=None, trend="c")

    # Forecast horizon steps
    history_window = data_matrix.values[-selected_lag:]
    forecast_values = res.forecast(history_window, steps=horizon)

    # Target column is at index 0
    target_point_pred = forecast_values[:, 0]
    target_point_pred = np.maximum(0.0, target_point_pred)

    # Forecast intervals using analytical forecast error variance
    try:
        forecast_intervals = res.forecast_interval(history_window, steps=horizon, alpha=0.05)
        # forecast_intervals is (mid, lower, upper)
        lower_bound = np.maximum(0.0, forecast_intervals[1][:, 0])
        upper_bound = np.maximum(0.0, forecast_intervals[2][:, 0])
    except Exception:
        # Fallback to residual standard deviation
        res_std = float(np.std(res.resid[:, 0])) if len(res.resid) > 0 else 1.0
        lower_bound = np.maximum(0.0, target_point_pred - 1.96 * res_std)
        upper_bound = np.maximum(0.0, target_point_pred + 1.96 * res_std)

    fit_time = time.perf_counter() - t_start
    diagnostics: Dict[str, Any] = {
        "model_type": "VectorAutoregression",
        "variables": var_cols,
        "k_endogenous": k_vars,
        "lag_order": int(selected_lag),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "fpe": float(res.fpe),
        "fit_time_seconds": round(fit_time, 3),
        "parameters": f"VAR({selected_lag}) on {k_vars} vars ({', '.join(var_cols)})",
    }

    return target_point_pred, lower_bound, upper_bound, res, diagnostics
