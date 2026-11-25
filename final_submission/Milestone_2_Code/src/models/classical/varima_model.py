"""Vector Autoregressive Integrated Moving Average (VARIMA) Model.

Mathematical Formulation:
    Let Y_t = [y_{t, 1}, y_{t, 2}, ..., y_{t, K}]^T be a K-dimensional endogenous system
    comprising disease occurrences and climate drivers (e.g. temperature, relative humidity).

    For an integration order d >= 1:
        Delta^d Y_t = (1 - L)^d Y_t

    The differenced stationary vector series follows a Vector ARMA (VARMA(p, q)):
        Delta^d Y_t = c + sum_{i=1}^p A_i Delta^d Y_{t-i} + epsilon_t + sum_{j=1}^q M_j epsilon_{t-j}
    where:
        c is a K x 1 drift vector
        A_i are K x K autoregressive transition matrices
        M_j are K x K moving average parameter matrices capturing propagation of unobserved shocks
        epsilon_t ~ WN(0, Sigma) is multivariate white noise

Degrees of Freedom & Identifiability Management:
    VARMA estimation involves K^2 * (p + q) parameters plus covariance matrix terms.
    To ensure numerical stability on observational records (N ~ 50-130 months),
    orders default to p=1, d=1, q=1 with diagonal error covariance.
    If MLE optimization encounters numerical non-invertibility or ill-conditioned Hessians,
    the pipeline gracefully falls back to integrated VAR(p).
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX


def forecast_varima(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    exog_cols: Optional[List[str]] = None,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    trend: str = "c",
    maxiter: int = 100,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Any, Dict[str, Any]]:
    """Fit a Vector Autoregressive Integrated Moving Average (VARIMA) model and forecast.

    Args:
        train_df: Historical training DataFrame.
        horizon: Forecast horizon (months).
        target_col: Name of disease occurrence count column.
        exog_cols: Optional list of additional endogenous driver columns (e.g. temp, humidity).
        p: Autoregressive order (default 1).
        d: Integration/differencing order (default 1).
        q: Moving average order (default 1).
        trend: Deterministic trend component ('c' for constant, 'n' for none).
        maxiter: Maximum iterations for MLE optimization.

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

    # Limit to top 2 drivers so total endogenous variables K <= 3
    active_drivers = active_drivers[:2]
    var_cols = [target_col] + active_drivers

    data_matrix = df_sorted[var_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    n_obs, k_vars = data_matrix.shape

    if n_obs < 15:
        raise ValueError(f"Insufficient observations ({n_obs}) to estimate VARIMA model.")

    values = data_matrix.values.astype(float)
    fallback_used = False
    fitted_model = None
    aic_val = float("nan")
    bic_val = float("nan")

    # 1. Apply integration differencing if d > 0
    if d > 0:
        working_series = np.diff(values, n=d, axis=0)
    else:
        working_series = values.copy()

    # 2. Fit VARMAX(p, q) via State-Space MLE
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            varmax_mod = VARMAX(
                working_series,
                order=(p, q),
                trend=trend,
                error_cov_type="diagonal",
            )
            res = varmax_mod.fit(maxiter=maxiter, disp=False)
            fc = res.get_forecast(steps=horizon)
            diff_pred = np.asarray(fc.predicted_mean)
            diff_se = np.asarray(fc.se_mean)

            fitted_model = res
            aic_val = float(res.aic) if hasattr(res, "aic") else float("nan")
            bic_val = float(res.bic) if hasattr(res, "bic") else float("nan")
    except Exception as exc:
        # Fallback: estimate integrated VAR(p) on differenced data
        fallback_used = True
        try:
            var_mod = VAR(working_series)
            res_var = var_mod.fit(maxlags=max(1, p), ic=None, trend="c")
            hist_win = working_series[-max(1, p):]
            diff_pred = res_var.forecast(hist_win, steps=horizon)
            res_std = np.std(res_var.resid, axis=0) if len(res_var.resid) > 0 else np.ones(k_vars)
            diff_se = np.tile(res_std, (horizon, 1))
            fitted_model = res_var
            aic_val = float(res_var.aic)
            bic_val = float(res_var.bic)
        except Exception:
            # Secondary fallback: naive mean difference
            mean_diff = np.mean(working_series, axis=0)
            diff_pred = np.tile(mean_diff, (horizon, 1))
            diff_se = np.tile(np.std(working_series, axis=0), (horizon, 1))
            fitted_model = None

    # 3. Integrate back if d > 0
    if d > 0:
        pred_levels = np.zeros_like(diff_pred)
        lower_levels = np.zeros_like(diff_pred)
        upper_levels = np.zeros_like(diff_pred)

        last_level = values[-1].copy()
        cum_variance = np.zeros(k_vars)

        for h in range(horizon):
            last_level = last_level + diff_pred[h]
            # Uncertainty accumulates under integration: Var(Y_{T+h}) = sum_{k=1}^h Var(Delta Y_{T+k})
            se_h = diff_se[h] if len(diff_se.shape) > 1 else diff_se
            cum_variance += np.square(se_h)
            se_cumulative = np.sqrt(cum_variance)

            pred_levels[h] = last_level
            lower_levels[h] = last_level - 1.96 * se_cumulative
            upper_levels[h] = last_level + 1.96 * se_cumulative
    else:
        pred_levels = diff_pred
        lower_levels = diff_pred - 1.96 * diff_se
        upper_levels = diff_pred + 1.96 * diff_se

    # Target column is index 0
    target_preds = np.maximum(0.0, pred_levels[:, 0])
    target_lb = np.maximum(0.0, lower_levels[:, 0])
    target_ub = np.maximum(0.0, upper_levels[:, 0])

    fit_time = time.perf_counter() - t_start
    diagnostics: Dict[str, Any] = {
        "model_type": "VectorAutoregressiveIntegratedMovingAverage",
        "order": (p, d, q),
        "variables": var_cols,
        "k_endogenous": k_vars,
        "aic": aic_val,
        "bic": bic_val,
        "fallback_used": fallback_used,
        "fit_time_seconds": round(fit_time, 3),
        "parameters": f"VARIMA({p},{d},{q}) on {k_vars} vars ({', '.join(var_cols)})",
    }

    return target_preds, target_lb, target_ub, fitted_model, diagnostics
