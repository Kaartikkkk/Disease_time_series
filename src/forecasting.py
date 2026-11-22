"""Comprehensive Time Series Forecasting Engine for Indian Plant Disease Surveillance.

Architecture:
- Individual forecasting models are modularized in `src/models/`:
    * `src/models/seasonal_naive.py` (Persistence baseline)
    * `src/models/holt_winters.py` (Triple exponential smoothing)
    * `src/models/ar_model.py` (AR(1) Autoregressive)
    * `src/models/ma_model.py` (MA(1) Moving average)
    * `src/models/arima_model.py` (ARIMA(1,1,1) Integrated Box-Jenkins)
    * `src/models/sarima_model.py` (SARIMA seasonal cycle)
    * `src/models/arimax_model.py` (ARIMAX with NASA weather lags)
    * `src/models/sarimax_model.py` (SARIMAX seasonal + NASA weather lags)
    * `src/models/gradient_boosting.py` (HistGradientBoosting autoregressor)
    * `src/models/random_forest.py` (RandomForest autoregressor)
- This module orchestrates:
    1. Chronological holdout splitting (70% train / 15% validation / 15% test)
    2. Comprehensive 10-model benchmarking with out-of-sample RMSE/MAE rankings
    3. Future scenario weather simulation (Monsoon, Drought, Climatology)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

from src.evaluation import compute_forecast_metrics
from src.model_registry import get_available_models, run_registered_model
from src.models import (
    forecast_ar,
    forecast_arima,
    forecast_arimax,
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_gradient_boosting,
    forecast_gru,
    forecast_holt_winters,
    forecast_hurdle_xgboost,
    forecast_lstm,
    forecast_ma,
    forecast_prophet,
    forecast_random_forest,
    forecast_sarima,
    forecast_sarimax_model,
    forecast_seasonal_naive,
    forecast_xgboost,
)
from src.models.base import ForecastContext, build_failure_output

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


def _load_forecasting_hyperparameters() -> Dict[str, Any]:
    """Load centralized forecasting hyperparameters from config/settings.yaml."""
    defaults = {
        "seasonal_period": 12,
        "autoregressive_lags": [1, 2, 3, 12],
        "weather_exog_cols": [
            "temp_mean_c_lag1",
            "precipitation_monthly_mm_lag1",
            "relative_humidity_pct_lag1",
        ],
    }
    if _SETTINGS_PATH.exists():
        try:
            cfg = yaml.safe_load(_SETTINGS_PATH.read_text(encoding="utf-8")) or {}
            hp = cfg.get("forecasting", {}).get("hyperparameters", {})
            return {**defaults, **hp}
        except Exception:
            pass
    return defaults


# ==============================================================================
# CHRONOLOGICAL SPLIT HELPER
# ==============================================================================

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically without shuffling to prevent leakage."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


# ==============================================================================
# GENERAL SARIMAX ADAPTER (Backwards Compatibility)
# ==============================================================================

def forecast_sarimax(
    train_y: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = (1, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    train_exog: Optional[pd.DataFrame] = None,
    future_exog: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """General SARIMAX dispatcher supporting flexible (p,d,q)(P,D,Q,s) orders and exogenous lags."""
    return forecast_sarimax_model(
        train_y=train_y,
        horizon=horizon,
        order=order,
        seasonal_order=seasonal_order,
        train_exog=train_exog,
        future_exog=future_exog,
    )


# ==============================================================================
# BENCHMARK AND MODEL COMPARISON RUNNER
# ==============================================================================

def run_model_comparison(
    df: pd.DataFrame,
    horizon: Optional[int] = None,
    target_col: str = "documented_occurrence_count",
) -> Dict[str, Any]:
    """Execute all eligible models on fixed chronological split and return comparative metrics."""
    # Split into train (70%), val (15%), test (15%)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15)

    # We evaluate models on the validation set
    y_train = train_df[target_col]

    # If a specific horizon is requested and within validation set length, evaluate on that horizon
    if horizon is not None and 0 < horizon <= len(val_df):
        eval_horizon = horizon
        y_val = val_df[target_col].values[:eval_horizon]
        val_dates = pd.to_datetime(val_df["date"]).iloc[:eval_horizon]
        eval_val_df = val_df.iloc[:eval_horizon].copy()
    else:
        eval_horizon = len(val_df)
        y_val = val_df[target_col].values
        val_dates = pd.to_datetime(val_df["date"])
        eval_val_df = val_df.copy()

    hp = _load_forecasting_hyperparameters()
    season_len = int(hp["seasonal_period"])
    lags = list(hp["autoregressive_lags"])
    weather_exog_cols = list(hp["weather_exog_cols"])
    has_weather = all(c in train_df.columns for c in weather_exog_cols)

    models_results: List[Dict[str, Any]] = []
    forecasts_dict: Dict[str, np.ndarray] = {}

    ctx = ForecastContext(
        train_df=train_df,
        horizon=eval_horizon,
        target_col=target_col,
        exog_cols=weather_exog_cols if has_weather else None,
        future_df=eval_val_df if has_weather else None,
        config={"season_length": season_len, "lags": lags, "auto_optimize": True},
    )

    for spec in get_available_models():
        if spec.requires_exog and not has_weather:
            continue
        try:
            out = run_registered_model(spec.model_id, ctx, clip_non_negative=True)
            meta = out["metadata"]
            metrics = compute_forecast_metrics(
                y_val,
                out["predictions"],
                out["lower_bound"],
                out["upper_bound"],
                model_name=out["model_name"],
                y_train=y_train.values,
                season_length=season_len,
                parameters=meta.get("parameters", ""),
                model_family=meta.get("model_family", spec.model_family),
                model_category=spec.category,
            )
            models_results.append(metrics)
            forecasts_dict[out["model_name"]] = out["predictions"]
        except Exception as e:
            logger.warning("%s failed in fixed holdout: %s", spec.display_name, e)

    # Rank models by RMSE
    models_results = sorted(models_results, key=lambda x: x.get("rmse") if x.get("rmse") is not None else float("inf"))
    best_model_name = models_results[0]["model_name"] if models_results else "None"

    return {
        "evaluation_metrics": models_results,
        "best_model": best_model_name,
        "validation_actual": y_val,
        "validation_dates": val_dates,
        "forecasts": forecasts_dict,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


# ==============================================================================
# ROLLING WALK-FORWARD VALIDATION RUNNER
# ==============================================================================

def run_rolling_walk_forward_evaluation(
    df: pd.DataFrame,
    horizon: Optional[int] = None,
    step_ahead: int = 1,
    window_type: str = "expanding",
    fixed_window_size: int = 92,
    target_col: str = "documented_occurrence_count",
    apply_threshold: bool = False,
    tau: float = 0.5,
    use_log1p: bool = False,
    eval_start_idx: Optional[int] = None,
    filter_model_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute rolling-origin walk-forward evaluation across all models without temporal lookahead.

    In each step k:
        1. Historical slice: [0 : curr_train_end] (expanding) or [curr_train_start : curr_train_end] (rolling).
        2. Fits each model strictly on the historical slice.
        3. Forecasts step_ahead months ahead.
        4. Out-of-sample prediction is compared with ground truth at target_idx.
        5. Origin rolls forward by 1 month, incorporating the newly available actual observation.

    Args:
        df: Pathogen time series DataFrame.
        horizon: Optional number of walk-forward steps (defaults to validation window length).
        step_ahead: Forecasting horizon (1 for next month).
        window_type: "expanding" (accumulates history) or "rolling" (fixed history length).
        fixed_window_size: Window size when window_type == "rolling" (default 92).
        target_col: Name of disease occurrence count column.
        apply_threshold: If True, small non-negative predictions below tau are set to 0.0.
        tau: Rounding/dormancy threshold (default 0.5 reports).
        use_log1p: If True, transforms target z = log(1+y) before fitting and back-transforms.
        eval_start_idx: Starting index in df for evaluation (defaults to start of validation set = 92).
        filter_model_ids: Optional subset of model_ids to evaluate.

    Returns:
        Dictionary containing comparative walk-forward metrics, actual values, and step forecasts.
    """
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    n_train = len(train_df)
    total_val_len = len(val_df)
    max_total_len = n_train + total_val_len

    start_idx = eval_start_idx if eval_start_idx is not None else n_train
    max_eval_steps = max(1, max_total_len - start_idx - step_ahead + 1)
    if horizon is not None and 0 < horizon <= max_eval_steps:
        eval_steps = horizon
    else:
        eval_steps = max_eval_steps

    hp = _load_forecasting_hyperparameters()
    season_len = int(hp["seasonal_period"])
    lags = list(hp["autoregressive_lags"])
    weather_exog_cols = list(hp["weather_exog_cols"])
    has_weather = all(c in df.columns for c in weather_exog_cols)

    # Initial order optimization on baseline training window strictly prior to evaluation origin
    initial_y_train = df.iloc[:start_idx][target_col].copy()
    if use_log1p:
        initial_y_fit = np.log1p(np.maximum(0.0, initial_y_train.values))
    else:
        initial_y_fit = initial_y_train.values

    try:
        _, _, _, r_arima = forecast_arima(pd.Series(initial_y_fit), horizon=step_ahead, auto_optimize=True)
        cached_arima_order = getattr(r_arima, "best_order", (1, 0, 1))
    except Exception:
        cached_arima_order = (1, 0, 1)

    try:
        _, _, _, r_sarima = forecast_sarima(pd.Series(initial_y_fit), horizon=step_ahead, auto_optimize=True)
        cached_sarima_order = getattr(r_sarima, "best_order", ((1, 0, 1), (1, 0, 1, season_len)))
    except Exception:
        cached_sarima_order = ((1, 0, 1), (1, 0, 1, season_len))

    active_specs = [
        spec for spec in get_available_models()
        if not (spec.requires_exog and not has_weather)
        and (filter_model_ids is None or spec.model_id in filter_model_ids)
    ]

    step_predictions: Dict[str, List[float]] = {spec.display_name: [] for spec in active_specs}
    model_failures: Dict[str, List[str]] = {}
    actuals: List[float] = []
    eval_dates: List[pd.Timestamp] = []

    def _record_failure(model_label: str, step_idx: int, exc: Exception) -> None:
        logger.warning("Walk-forward step %d failed for %s: %s", step_idx, model_label, exc)
        fail_out = build_failure_output(model_label, step_ahead, exc)
        model_failures.setdefault(model_label, []).append(f"step {step_idx}: {exc}")
        step_predictions[model_label].append(float(fail_out["predictions"][-1]))

    def _invert_and_clean(pred_array: np.ndarray) -> float:
        val = float(pred_array[-1])
        if use_log1p:
            val = float(np.expm1(val))
        val = max(0.0, val)
        if apply_threshold and val < tau:
            val = 0.0
        return val

    for step in range(eval_steps):
        target_idx = start_idx + step + step_ahead - 1
        curr_train_end = start_idx + step

        if window_type == "rolling":
            curr_train_start = max(0, curr_train_end - fixed_window_size)
            sub_df = df.iloc[curr_train_start:curr_train_end].copy().reset_index(drop=True)
        else:
            sub_df = df.iloc[:curr_train_end].copy().reset_index(drop=True)

        future_sub_df = df.iloc[curr_train_end:target_idx + 1].copy().reset_index(drop=True)

        y_true = float(df[target_col].iloc[target_idx])
        dt_val = pd.to_datetime(df["date"].iloc[target_idx])
        actuals.append(y_true)
        eval_dates.append(dt_val)

        sub_df_fit = sub_df.copy()
        if use_log1p:
            sub_df_fit[target_col] = np.log1p(np.maximum(0.0, sub_df_fit[target_col].values))

        for spec in active_specs:
            step_cfg: Dict[str, Any] = {
                "season_length": season_len,
                "lags": lags,
                "auto_optimize": False,
            }
            if spec.model_id in ("arima", "arimax"):
                step_cfg["order"] = cached_arima_order
            elif spec.model_id in ("sarima", "sarimax"):
                step_cfg["order"] = cached_sarima_order[0]
                step_cfg["seasonal_order"] = cached_sarima_order[1]

            step_ctx = ForecastContext(
                train_df=sub_df_fit,
                horizon=step_ahead,
                target_col=target_col,
                exog_cols=weather_exog_cols if has_weather else None,
                future_df=future_sub_df if has_weather else None,
                config=step_cfg,
            )
            try:
                out = run_registered_model(spec.model_id, step_ctx, clip_non_negative=False)
                step_predictions[spec.display_name].append(_invert_and_clean(out["predictions"]))
            except Exception as e:
                _record_failure(spec.display_name, step, e)

    y_actual_arr = np.array(actuals, dtype=float)
    y_train_init = df.iloc[:start_idx][target_col].values

    eval_results: List[Dict[str, Any]] = []
    forecasts_dict: Dict[str, np.ndarray] = {}

    for spec in active_specs:
        name = spec.display_name
        preds = np.array(step_predictions[name], dtype=float)
        if len(preds) == len(y_actual_arr):
            m = compute_forecast_metrics(
                actual=y_actual_arr,
                predicted=preds,
                model_name=name,
                y_train=y_train_init,
                season_length=season_len,
                parameters=f"Walk-Forward {step_ahead}-Step Ahead ({window_type.capitalize()} Window)",
                model_family="Rolling Origin Benchmark",
                model_category=spec.category,
            )
            eval_results.append(m)
            forecasts_dict[name] = preds

    # Sort models by RMSE ascending
    eval_results = sorted(eval_results, key=lambda x: x.get("rmse") if x.get("rmse") is not None else float("inf"))
    best_model_name = eval_results[0]["model_name"] if eval_results else "None"

    return {
        "evaluation_metrics": eval_results,
        "best_model": best_model_name,
        "validation_actual": y_actual_arr,
        "validation_dates": pd.Series(eval_dates),
        "forecasts": forecasts_dict,
        "model_failures": model_failures,
        "step_ahead": step_ahead,
        "window_type": window_type,
        "n_eval_steps": eval_steps,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


# ==============================================================================
# FUTURE SCENARIO GENERATOR
# ==============================================================================

def generate_future_weather_scenario(
    historical_df: pd.DataFrame,
    horizon_months: int = 12,
    scenario_type: str = "baseline_climatology",
    temp_shift_c: float = 0.0,
    precip_factor: float = 1.0,
) -> pd.DataFrame:
    """Generate future weather regressors based on transparent, documented climate scenarios."""
    last_date = pd.to_datetime(historical_df["date"].max())
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

    # Group historical data by calendar month
    hist = historical_df.copy()
    hist["cal_month"] = pd.to_datetime(hist["date"]).dt.month
    monthly_clim = hist.groupby("cal_month").agg({
        "temp_mean_c": "mean",
        "temp_max_c": "mean",
        "temp_min_c": "mean",
        "precipitation_monthly_mm": "mean",
        "relative_humidity_pct": "mean",
        "dew_point_c": "mean",
    }).reset_index()

    future_rows = []
    for dt in future_dates:
        m = dt.month
        clim_row = monthly_clim[monthly_clim["cal_month"] == m].iloc[0]

        t_mean = clim_row["temp_mean_c"] + temp_shift_c
        t_max = clim_row["temp_max_c"] + temp_shift_c
        t_min = clim_row["temp_min_c"] + temp_shift_c
        precip = max(0.0, clim_row["precipitation_monthly_mm"] * precip_factor)
        rh = min(100.0, max(15.0, clim_row["relative_humidity_pct"] + (precip_factor - 1.0) * 10.0))
        dew = clim_row["dew_point_c"] + temp_shift_c * 0.5

        future_rows.append({
            "date": dt,
            "year": dt.year,
            "month": dt.month,
            "temp_mean_c": round(t_mean, 2),
            "temp_max_c": round(t_max, 2),
            "temp_min_c": round(t_min, 2),
            "precipitation_monthly_mm": round(precip, 2),
            "relative_humidity_pct": round(rh, 1),
            "dew_point_c": round(dew, 2),
        })

    df_fut = pd.DataFrame(future_rows)
    # Construct lagged regressors from end of historical plus scenario
    combined_for_lags = pd.concat([historical_df[["date", "temp_mean_c", "precipitation_monthly_mm", "relative_humidity_pct"]].tail(3), df_fut], ignore_index=True)
    combined_for_lags["temp_mean_c_lag1"] = combined_for_lags["temp_mean_c"].shift(1)
    combined_for_lags["precipitation_monthly_mm_lag1"] = combined_for_lags["precipitation_monthly_mm"].shift(1)
    combined_for_lags["relative_humidity_pct_lag1"] = combined_for_lags["relative_humidity_pct"].shift(1)

    df_res = combined_for_lags.iloc[3:].copy().reset_index(drop=True)
    df_res["scenario_label"] = scenario_type
    return df_res
