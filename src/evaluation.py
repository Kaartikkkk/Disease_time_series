"""Forecasting Evaluation Metrics and Comparison Module.

Implements rigorous, mathematically verified metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE) — strictly masked/omitted when denominator is 0
- Symmetric Mean Absolute Percentage Error (sMAPE)
- Weighted Absolute Percentage Error (WAPE) — recommended for count/sparse time series
- Mean Absolute Scaled Error (MASE) — scaled against in-sample seasonal naive benchmark
- Forecast Bias (Mean Error): positive indicates over-forecasting, negative indicates under-forecasting
- Prediction Interval Coverage Probability (PICP) for 80% and 95% intervals
- Volume-Bounded Forecast Accuracy (%) = max(0, 100 - WAPE)
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
    y_train: Optional[np.ndarray] = None,
    season_length: int = 12,
    parameters: str = "",
    model_family: str = "",
    model_category: str = "",
) -> Dict[str, Any]:
    """Calculate all standard and robust time series forecast accuracy metrics.

    Args:
        actual: Ground truth holdout observations.
        predicted: Model point forecasts.
        lower_bound: Lower prediction interval boundary.
        upper_bound: Upper prediction interval boundary.
        model_name: Display label for the forecasting method.
        y_train: Optional historical training observations (for MASE scaling).
        season_length: Seasonal lag period (default 12 for annual monthly cycles).
        parameters: Selected parameter specification string.
        model_family: Architectural family (e.g. Box-Jenkins, ML, Baseline).

    Returns:
        Dictionary of computed metrics with transparent, verified values.
    """
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
            "parameters": parameters,
            "model_family": model_family,
            "model_category": model_category,
            "n_observations": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "wape": None,
            "mase": None,
            "accuracy_pct": None,
            "bias": None,
            "picp": None,
        }

    # 1. Mean Absolute Error (MAE)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))

    # 2. Root Mean Squared Error (RMSE)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # 3. Forecast Bias (Mean Error)
    bias = float(np.mean(errors))

    # 4. MAPE (Only defined when actual != 0 to prevent division by zero)
    non_zero_mask = (y_true != 0)
    if non_zero_mask.sum() > 0:
        mape_val = float(np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100)
    else:
        mape_val = None

    # 5. sMAPE (Symmetric MAPE bounded between 0% and 200%)
    # Standard time-series convention: if both actual and predicted are zero, error is 0.0%
    both_zero_mask = (np.abs(y_true) < 1e-6) & (np.abs(y_pred) < 1e-6)
    denom_smape = np.abs(y_true) + np.abs(y_pred)
    denom_smape = np.where(both_zero_mask, 1.0, denom_smape)
    smape_terms = 2.0 * abs_errors / denom_smape
    smape_terms = np.where(both_zero_mask, 0.0, smape_terms)
    smape_val = float(np.mean(smape_terms) * 100)

    # 6. WAPE (Weighted Absolute Percentage Error = sum(|errors|) / sum(|actual|))
    sum_actual = float(np.sum(np.abs(y_true)))
    if sum_actual > 0:
        wape_val = float(np.sum(abs_errors) / sum_actual * 100)
    else:
        wape_val = None

    # 7. MASE (Mean Absolute Scaled Error, Hyndman & Koehler, 2006)
    # Scaled by in-sample seasonal naive mean absolute difference
    mase_val = None
    if y_train is not None and len(y_train) > season_length:
        train_arr = np.asarray(y_train, dtype=float)
        seasonal_diff = np.abs(train_arr[season_length:] - train_arr[:-season_length])
        scale = float(np.mean(seasonal_diff))
        if scale > 1e-6:
            mase_val = float(mae / scale)
    elif y_train is not None and len(y_train) > 1:
        train_arr = np.asarray(y_train, dtype=float)
        diff = np.abs(train_arr[1:] - train_arr[:-1])
        scale = float(np.mean(diff))
        if scale > 1e-6:
            mase_val = float(mae / scale)

    # 8. Prediction Interval Coverage Probability (PICP)
    picp_val = None
    if lower_bound is not None and upper_bound is not None:
        lb = np.asarray(lower_bound, dtype=float)[valid_mask]
        ub = np.asarray(upper_bound, dtype=float)[valid_mask]
        within_bounds = (y_true >= lb) & (y_true <= ub)
        picp_val = float(np.mean(within_bounds) * 100)

    # 9. Volume-Bounded Forecast Accuracy (%)
    # Rigorously defined as max(0.0, 100.0 - WAPE). Valid when sum_actual > 0.
    accuracy_pct = None
    if wape_val is not None:
        accuracy_pct = round(max(0.0, 100.0 - wape_val), 1)

    return {
        "model_name": model_name,
        "parameters": parameters,
        "model_family": model_family,
        "model_category": model_category,
        "n_observations": n,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape_val, 2) if mape_val is not None else None,
        "smape": round(smape_val, 2),
        "wape": round(wape_val, 2) if wape_val is not None else None,
        "mase": round(mase_val, 3) if mase_val is not None else None,
        "accuracy_pct": accuracy_pct,
        "bias": round(bias, 3),
        "picp": round(picp_val, 1) if picp_val is not None else None,
    }


def build_model_comparison_table(eval_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format multiple model evaluation dictionaries into a clean comparison DataFrame."""
    rows = []
    baseline_rmse = next((r.get("rmse") for r in eval_results if "Seasonal Naive" in r.get("model_name", "")), None)

    for res in eval_results:
        m_rmse = res.get("rmse")
        m_mase = res.get("mase")
        m_wape = res.get("wape")
        m_acc = res.get("accuracy_pct")

        # Performance badge and baseline improvement relative to Seasonal Naive
        if baseline_rmse and m_rmse is not None and baseline_rmse > 0:
            if "Seasonal Naive" in res.get("model_name", ""):
                perf_str = "Baseline Reference"
                imp_str = "0.0% (Ref)"
            else:
                imp = ((baseline_rmse - m_rmse) / baseline_rmse) * 100
                if imp > 0:
                    perf_str = f"Beats Baseline (+{imp:.1f}%)"
                    imp_str = f"+{imp:.1f}%"
                else:
                    perf_str = f"Below Baseline ({imp:.1f}%)"
                    imp_str = f"{imp:.1f}%"
        else:
            perf_str = "N/A"
            imp_str = "N/A"

        rows.append({
            "Model": res.get("model_name"),
            "Category": res.get("model_category") or res.get("model_family") or "Time Series",
            "Selected Parameters": res.get("parameters", "Auto-Optimized"),
            "RMSE": m_rmse,
            "MAE": res.get("mae"),
            "sMAPE (%)": f"{res.get('smape'):.2f}%" if res.get("smape") is not None else "N/A",
            "Improvement vs Baseline": imp_str,
            "MASE": f"{m_mase:.3f}" if m_mase is not None else "N/A",
            "Performance": perf_str,
            "Forecast Score (%)": f"{m_acc:.1f}%" if m_acc is not None else "N/A",
            "Interval Coverage (%)": f"{res.get('picp')}%" if res.get("picp") is not None else "N/A",
        })

    df = pd.DataFrame(rows)
    if not df.empty and "RMSE" in df.columns:
        # Sort by RMSE ascending (best model first)
        df = df.sort_values("RMSE", ascending=True).reset_index(drop=True)
        df.insert(0, "Rank", [f"#{i+1}" for i in range(len(df))])
    return df
