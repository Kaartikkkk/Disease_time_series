"""Forecasting Evaluation Metrics and Comparison Module.

Implements rigorous, mathematically verified metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE) — strictly masked/omitted when denominator is 0
- Symmetric Mean Absolute Percentage Error (sMAPE)
- Weighted Absolute Percentage Error (WAPE) — recommended for count/sparse time series
- Forecast Bias (Mean Error): positive indicates over-forecasting, negative indicates under-forecasting
- Prediction Interval Coverage Probability (PICP) for 80% and 95% intervals
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Calculate all standard and robust time series forecast accuracy metrics."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    # Valid mask for non-NaN pairs
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    n = len(y_true)
    if n == 0:
        return {
            "model_name": model_name,
            "n_observations": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "wape": None,
            "bias": None,
            "picp": None,
        }

    # 1. MAE
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))

    # 2. RMSE
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # 3. Forecast Bias (Mean Error)
    bias = float(np.mean(errors))

    # 4. MAPE (Only defined when actual != 0)
    non_zero_mask = (y_true != 0)
    if non_zero_mask.sum() > 0:
        mape_val = float(np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100)
    else:
        mape_val = None

    # 5. sMAPE (Symmetric MAPE bounded between 0% and 200%)
    denom_smape = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape_val = float(np.mean(2.0 * abs_errors / denom_smape) * 100)

    # 6. WAPE (Weighted Absolute Percentage Error = sum(|errors|) / sum(|actual|))
    sum_actual = float(np.sum(np.abs(y_true)))
    if sum_actual > 0:
        wape_val = float(np.sum(abs_errors) / sum_actual * 100)
    else:
        wape_val = None

    # 7. Prediction Interval Coverage Probability (PICP)
    picp_val = None
    if lower_bound is not None and upper_bound is not None:
        lb = np.asarray(lower_bound, dtype=float)[valid_mask]
        ub = np.asarray(upper_bound, dtype=float)[valid_mask]
        within_bounds = (y_true >= lb) & (y_true <= ub)
        picp_val = float(np.mean(within_bounds) * 100)

    return {
        "model_name": model_name,
        "n_observations": n,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape_val, 2) if mape_val is not None else None,
        "smape": round(smape_val, 2),
        "wape": round(wape_val, 2) if wape_val is not None else None,
        "bias": round(bias, 3),
        "picp": round(picp_val, 1) if picp_val is not None else None,
    }


def build_model_comparison_table(eval_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format multiple model evaluation dictionaries into a clean comparison DataFrame."""
    rows = []
    for res in eval_results:
        rows.append({
            "Model": res.get("model_name"),
            "RMSE": res.get("rmse"),
            "MAE": res.get("mae"),
            "sMAPE (%)": res.get("smape"),
            "WAPE (%)": res.get("wape") if res.get("wape") is not None else "N/A",
            "MAPE (%)": f"{res.get('mape')}%" if res.get("mape") is not None else "Undefined (Zeros)",
            "Bias (Mean Error)": res.get("bias"),
            "Interval Coverage (%)": f"{res.get('picp')}%" if res.get("picp") is not None else "N/A",
        })

    df = pd.DataFrame(rows)
    if not df.empty and "RMSE" in df.columns:
        # Sort by RMSE ascending (best model first)
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
    return df
