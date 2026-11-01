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
    * `src/models/prophet_model.py` (Meta Prophet piecewise trend + Fourier)
    * `src/models/gradient_boosting.py` (HistGradientBoosting autoregressor)
    * `src/models/random_forest.py` (RandomForest autoregressor)
- This module orchestrates:
    1. Chronological holdout splitting (70% train / 15% validation / 15% test)
    2. Comprehensive 11-model benchmarking with out-of-sample RMSE/MAE rankings
    3. Future scenario weather simulation (Monsoon, Drought, Climatology)
"""

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluation import compute_forecast_metrics
from src.models import (
    HAS_PROPHET,
    forecast_ar,
    forecast_arima,
    forecast_arimax,
    forecast_gradient_boosting,
    forecast_holt_winters,
    forecast_ma,
    forecast_ml_autoregressor,
    forecast_prophet,
    forecast_random_forest,
    forecast_sarima,
    forecast_sarimax_model,
    forecast_seasonal_naive,
)

logger = logging.getLogger(__name__)


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
    horizon: int = 6,
    target_col: str = "documented_occurrence_count",
) -> Dict[str, Any]:
    """Execute all eligible models on fixed chronological split and return comparative metrics."""
    # Split into train (70%), val (15%), test (15%)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15)

    # We evaluate models on the validation set
    y_train = train_df[target_col]
    y_val = val_df[target_col].values
    val_dates = pd.to_datetime(val_df["date"])
    eval_horizon = len(y_val)

    # Weather regressors
    weather_exog_cols = ["temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"]
    has_weather = all(c in train_df.columns for c in weather_exog_cols)

    train_exog = train_df[weather_exog_cols].bfill().fillna(0) if has_weather else None
    val_exog = val_df[weather_exog_cols].bfill().fillna(0) if has_weather else None

    models_results: List[Dict[str, Any]] = []
    forecasts_dict: Dict[str, np.ndarray] = {}

    # 1. Seasonal Naive Baseline (src/models/seasonal_naive.py)
    try:
        p_snaive, lb_snaive, ub_snaive = forecast_seasonal_naive(y_train, horizon=eval_horizon, season_length=12)
        m_snaive = compute_forecast_metrics(
            y_val, p_snaive, lb_snaive, ub_snaive,
            model_name="Seasonal Naive (Baseline)",
            y_train=y_train.values,
            season_length=12,
            parameters="Same-Month Prior Year (s=12)",
            model_family="Persistence Baseline",
        )
        models_results.append(m_snaive)
        forecasts_dict["Seasonal Naive (Baseline)"] = p_snaive
    except Exception as e:
        logger.warning("Seasonal Naive failed: %s", e)

    # 2. AR(1) / ARIMA(1,0,0) (src/models/ar_model.py)
    try:
        p_ar, lb_ar, ub_ar, res_ar = forecast_ar(y_train, horizon=eval_horizon, p=1)
        m_ar = compute_forecast_metrics(
            y_val, p_ar, lb_ar, ub_ar,
            model_name="AR(1)",
            y_train=y_train.values,
            season_length=12,
            parameters="p=1, d=0, q=0",
            model_family="Box-Jenkins Univariate",
        )
        models_results.append(m_ar)
        forecasts_dict["AR(1)"] = p_ar
    except Exception as e:
        logger.warning("AR(1) failed: %s", e)

    # 3. MA(1) / ARIMA(0,0,1) (src/models/ma_model.py)
    try:
        p_ma, lb_ma, ub_ma, res_ma = forecast_ma(y_train, horizon=eval_horizon, q=1)
        m_ma = compute_forecast_metrics(
            y_val, p_ma, lb_ma, ub_ma,
            model_name="MA(1)",
            y_train=y_train.values,
            season_length=12,
            parameters="p=0, d=0, q=1",
            model_family="Box-Jenkins Univariate",
        )
        models_results.append(m_ma)
        forecasts_dict["MA(1)"] = p_ma
    except Exception as e:
        logger.warning("MA(1) failed: %s", e)

    # 4. ARIMA(1, 1, 1) (src/models/arima_model.py)
    try:
        p_arima, lb_arima, ub_arima, res_arima = forecast_arima(y_train, horizon=eval_horizon, order=(1, 1, 1), auto_optimize=True)
        param_str = getattr(res_arima, "param_summary", "p=1, d=1, q=1")
        m_arima = compute_forecast_metrics(
            y_val, p_arima, lb_arima, ub_arima,
            model_name="ARIMA(1,1,1)",
            y_train=y_train.values,
            season_length=12,
            parameters=param_str,
            model_family="Box-Jenkins Integrated",
        )
        models_results.append(m_arima)
        forecasts_dict["ARIMA(1,1,1)"] = p_arima
    except Exception as e:
        logger.warning("ARIMA(1,1,1) failed: %s", e)

    # 5. SARIMA(1, 0, 1)(1, 0, 1, 12) (src/models/sarima_model.py)
    try:
        p_sarima, lb_sarima, ub_sarima, res_sarima = forecast_sarima(
            y_train, horizon=eval_horizon, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), auto_optimize=True
        )
        param_str = getattr(res_sarima, "param_summary", "(1,0,1)(1,0,1)[12]")
        m_sarima = compute_forecast_metrics(
            y_val, p_sarima, lb_sarima, ub_sarima,
            model_name="SARIMA(1,0,1)(1,0,1)[12]",
            y_train=y_train.values,
            season_length=12,
            parameters=param_str,
            model_family="Seasonal Box-Jenkins",
        )
        models_results.append(m_sarima)
        forecasts_dict["SARIMA(1,0,1)(1,0,1)[12]"] = p_sarima
    except Exception as e:
        logger.warning("SARIMA failed: %s", e)

    # 6. Holt-Winters Exponential Smoothing (src/models/holt_winters.py)
    try:
        p_hw, lb_hw, ub_hw, res_hw = forecast_holt_winters(y_train, horizon=eval_horizon, seasonal_periods=12, auto_optimize=True)
        param_str = getattr(res_hw, "param_summary", "Additive Level + Trend + Season (m=12)")
        m_hw = compute_forecast_metrics(
            y_val, p_hw, lb_hw, ub_hw,
            model_name="Holt-Winters Exponential Smoothing",
            y_train=y_train.values,
            season_length=12,
            parameters=param_str,
            model_family="Exponential Smoothing",
        )
        models_results.append(m_hw)
        forecasts_dict["Holt-Winters Exponential Smoothing"] = p_hw
    except Exception as e:
        logger.warning("Holt-Winters failed: %s", e)

    # 7. ARIMAX with Exogenous Weather Lags (src/models/arimax_model.py)
    if has_weather and train_exog is not None and val_exog is not None:
        try:
            p_arimax, lb_arimax, ub_arimax, res_arimax = forecast_arimax(
                y_train, horizon=eval_horizon, train_exog=train_exog, future_exog=val_exog, order=(1, 0, 1), auto_optimize=True
            )
            param_str = getattr(res_arimax, "param_summary", "ARIMA + 3 NASA Weather Lags (t-1)")
            m_arimax = compute_forecast_metrics(
                y_val, p_arimax, lb_arimax, ub_arimax,
                model_name="ARIMAX (Weather Regressors)",
                y_train=y_train.values,
                season_length=12,
                parameters=param_str,
                model_family="Exogenous Econometric",
            )
            models_results.append(m_arimax)
            forecasts_dict["ARIMAX (Weather Regressors)"] = p_arimax
        except Exception as e:
            logger.warning("ARIMAX failed: %s", e)

    # 8. SARIMAX Seasonal + Exogenous Weather Lags (src/models/sarimax_model.py)
    if has_weather and train_exog is not None and val_exog is not None:
        try:
            p_sarimax, lb_sarimax, ub_sarimax, res_sarimax = forecast_sarimax_model(
                y_train,
                horizon=eval_horizon,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12),
                train_exog=train_exog,
                future_exog=val_exog,
                auto_optimize=True,
            )
            param_str = getattr(res_sarimax, "param_summary", "SARIMA(1,0,1)(1,0,1)[12] + NASA Weather")
            m_sarimax = compute_forecast_metrics(
                y_val, p_sarimax, lb_sarimax, ub_sarimax,
                model_name="SARIMAX (Seasonal + Weather)",
                y_train=y_train.values,
                season_length=12,
                parameters=param_str,
                model_family="Exogenous Econometric",
            )
            models_results.append(m_sarimax)
            forecasts_dict["SARIMAX (Seasonal + Weather)"] = p_sarimax
        except Exception as e:
            logger.warning("SARIMAX failed: %s", e)

    # 9. Meta Prophet (src/models/prophet_model.py)
    if HAS_PROPHET:
        try:
            p_prophet, lb_prophet, ub_prophet, _ = forecast_prophet(
                train_df, horizon=eval_horizon, future_dates=val_dates,
                weather_cols=weather_exog_cols if has_weather else None,
                future_weather_df=val_df if has_weather else None,
            )
            m_prophet = compute_forecast_metrics(
                y_val, p_prophet, lb_prophet, ub_prophet,
                model_name="Prophet (Trend + Seasonality)",
                y_train=y_train.values,
                season_length=12,
                parameters="Piecewise Trend + Yearly Fourier (m=12) + Weather",
                model_family="Generalized Additive",
            )
            models_results.append(m_prophet)
            forecasts_dict["Prophet (Trend + Seasonality)"] = p_prophet
        except Exception as e:
            logger.warning("Prophet failed: %s", e)

    # 10. Gradient Boosting Regressor (src/models/gradient_boosting.py)
    try:
        p_gb, lb_gb, ub_gb, _ = forecast_gradient_boosting(
            train_df, horizon=eval_horizon,
            lags=[1, 2, 3, 12],
            weather_lag_cols=weather_exog_cols if has_weather else None,
            future_weather_df=val_df if has_weather else None,
        )
        m_gb = compute_forecast_metrics(
            y_val, p_gb, lb_gb, ub_gb,
            model_name="Hist Gradient Boosting (Autoregressive)",
            y_train=y_train.values,
            season_length=12,
            parameters="Lags [1, 2, 3, 12] + Cyclical Harmonics + Weather",
            model_family="Tree Ensemble ML",
        )
        models_results.append(m_gb)
        forecasts_dict["Hist Gradient Boosting (Autoregressive)"] = p_gb
    except Exception as e:
        logger.warning("HistGradientBoosting autoregressor failed: %s", e)

    # 11. Random Forest Regressor (src/models/random_forest.py)
    try:
        p_rf, lb_rf, ub_rf, _ = forecast_random_forest(
            train_df, horizon=eval_horizon,
            lags=[1, 2, 3, 12],
            weather_lag_cols=weather_exog_cols if has_weather else None,
            future_weather_df=val_df if has_weather else None,
        )
        m_rf = compute_forecast_metrics(
            y_val, p_rf, lb_rf, ub_rf,
            model_name="Random Forest (Autoregressive)",
            y_train=y_train.values,
            season_length=12,
            parameters="100 Trees + Lags [1, 2, 3, 12] + Weather",
            model_family="Tree Ensemble ML",
        )
        models_results.append(m_rf)
        forecasts_dict["Random Forest (Autoregressive)"] = p_rf
    except Exception as e:
        logger.warning("Random Forest autoregressor failed: %s", e)

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
