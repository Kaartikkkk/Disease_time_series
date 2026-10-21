"""Forecast diagnostics: residual error analysis, autocorrelation, and horizon-wise performance."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from src.evaluation.metrics import calculate_metrics, MetricScores


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag: int = 10,
) -> Dict[str, Any]:
    """Compute comprehensive diagnostic statistics on forecast residuals (e = y - y_hat).

    Analyzes:
    - Mean error (forecast bias)
    - Mean absolute error & standard error
    - Skewness and kurtosis of residuals
    - Ljung-Box test for white noise / absence of residual autocorrelation
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    residuals = yt - yp
    n = len(residuals)

    if n < 4:
        return {"error": "Insufficient residual count."}

    mean_err = float(np.mean(residuals))
    std_err = float(np.std(residuals))
    skew = float(stats.skew(residuals))
    kurt = float(stats.kurtosis(residuals))

    # Ljung-Box test for residual autocorrelation
    actual_lags = min(max_lag, max(1, n // 4))
    try:
        lb_res = acorr_ljungbox(residuals, lags=[actual_lags], return_df=True)
        lb_stat = float(lb_res["lb_stat"].iloc[0])
        lb_pvalue = float(lb_res["lb_pvalue"].iloc[0])
        is_white_noise = lb_pvalue > 0.05
    except Exception:
        lb_stat, lb_pvalue, is_white_noise = np.nan, np.nan, True

    return {
        "count": n,
        "mean_error_bias": round(mean_err, 4),
        "std_error": round(std_err, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "ljung_box_stat": round(lb_stat, 4) if not np.isnan(lb_stat) else None,
        "ljung_box_pvalue": round(lb_pvalue, 4) if not np.isnan(lb_pvalue) else None,
        "residuals_are_white_noise": is_white_noise,
        "diagnostic_interpretation": (
            "Residuals resemble uncorrelated white noise (p > 0.05), indicating model captured systematic patterns."
            if is_white_noise
            else "Significant residual autocorrelation detected (p < 0.05); model leaves systematic signal uncaptured."
        ),
    }


def compute_horizon_wise_metrics(
    fold_results: List[Any],
    horizon: int,
) -> pd.DataFrame:
    """Compute step-by-step forecast degradation across horizons (h = 1, 2, ..., H).

    Reveals how error compounds as the forecast extends further into the future.
    """
    rows = []
    for h_step in range(1, horizon + 1):
        step_true = []
        step_pred = []
        for f in fold_results:
            if len(f.y_true) >= h_step and len(f.y_pred) >= h_step:
                step_true.append(f.y_true[h_step - 1])
                step_pred.append(f.y_pred[h_step - 1])

        if step_true:
            yt = np.array(step_true)
            yp = np.array(step_pred)
            mae = float(np.mean(np.abs(yt - yp)))
            rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            rows.append({
                "Step (Horizon)": h_step,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "Sample Count": len(yt),
            })

    return pd.DataFrame(rows)
