"""Meta Prophet Generalized Additive Forecasting Model.

Mathematical Formulation:
    y(t) = g(t) + s(t) + epsilon_t
where:
    g(t): Piecewise linear growth trend (conservative changepoint_prior_scale)
    s(t): Yearly Fourier periodic seasonality (additive mode, weekly/daily disabled)
    epsilon_t: Residual noise

Configured conservatively for sparse monthly agricultural disease counts (132 months,
71.8% dataset-wide zeros). Primary mode is strictly univariate to avoid dependence on
unrealized future weather regressors.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Silence verbose cmdstanpy/prophet stdout logs during walk-forward fitting
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    Prophet = None  # type: ignore[assignment]


DEFAULT_PROPHET_PARAMS: Dict[str, Any] = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "seasonality_mode": "additive",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 0.1,
    "interval_width": 0.95,
}


def prepare_prophet_dataframe(
    train_df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
) -> pd.DataFrame:
    """Convert historical monthly series into chronologically sorted Prophet (ds, y) format."""
    if "date" not in train_df.columns:
        raise ValueError("train_df must contain a 'date' column for Prophet.")
    if target_col not in train_df.columns:
        raise ValueError(f"train_df must contain target column '{target_col}'.")

    df_p = pd.DataFrame({
        "ds": pd.to_datetime(train_df["date"], errors="coerce"),
        "y": pd.to_numeric(train_df[target_col], errors="coerce"),
    })
    df_p = df_p.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    if len(df_p) < 12:
        raise ValueError(f"Insufficient valid monthly observations ({len(df_p)}) for Prophet.")
    return df_p


def forecast_prophet(
    train_df: pd.DataFrame,
    horizon: int,
    future_dates: Optional[pd.DatetimeIndex] = None,
    weather_cols: Optional[List[str]] = None,
    future_weather_df: Optional[pd.DataFrame] = None,
    target_col: str = "documented_occurrence_count",
    prophet_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Forecast using Meta Prophet with conservative monthly settings.

    Args:
        train_df: Historical dataframe with 'date' and target_col.
        horizon: Forecast horizon (months).
        future_dates: Optional DatetimeIndex for future prediction dates (MS frequency).
        weather_cols: Optional exogenous regressor columns (omitted in primary univariate mode).
        future_weather_df: Optional future exogenous values matching future_dates.
        target_col: Target column name.
        prophet_params: Optional parameter overrides for Prophet.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_prophet_model).
    """
    if not HAS_PROPHET or Prophet is None:
        raise ImportError("Prophet is not installed or unavailable in this environment.")

    params = {**DEFAULT_PROPHET_PARAMS, **(prophet_params or {})}
    df_p = prepare_prophet_dataframe(train_df, target_col=target_col)

    model = Prophet(
        yearly_seasonality=params["yearly_seasonality"],
        weekly_seasonality=params["weekly_seasonality"],
        daily_seasonality=params["daily_seasonality"],
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        interval_width=params["interval_width"],
    )

    active_exog: List[str] = []
    if weather_cols:
        df_sorted = train_df.copy()
        df_sorted["date"] = pd.to_datetime(df_sorted["date"])
        df_sorted = df_sorted.sort_values("date").reset_index(drop=True)
        for col in weather_cols:
            if col in df_sorted.columns:
                df_p[col] = pd.to_numeric(df_sorted[col], errors="coerce").bfill().ffill().fillna(0.0).values[:len(df_p)]
                model.add_regressor(col)
                active_exog.append(col)

    model.fit(df_p)

    if future_dates is None:
        last_ds = df_p["ds"].max()
        future_dates = pd.date_range(start=last_ds + pd.DateOffset(months=1), periods=horizon, freq="MS")
    else:
        future_dates = pd.DatetimeIndex(pd.to_datetime(future_dates))[:horizon]

    future_p = pd.DataFrame({"ds": future_dates})
    for col in active_exog:
        if future_weather_df is not None and col in future_weather_df.columns and len(future_weather_df) >= len(future_p):
            future_p[col] = pd.to_numeric(future_weather_df[col].iloc[:len(future_p)], errors="coerce").fillna(df_p[col].mean()).values
        else:
            future_p[col] = float(df_p[col].mean())

    forecast = model.predict(future_p)

    raw_yhat = forecast["yhat"].to_numpy(dtype=float)
    raw_lb = forecast["yhat_lower"].to_numpy(dtype=float)
    raw_ub = forecast["yhat_upper"].to_numpy(dtype=float)

    negative_raw_count = int(np.sum(raw_yhat < 0.0))
    point_pred = np.maximum(0.0, raw_yhat)
    lower_95 = np.maximum(0.0, raw_lb)
    upper_95 = np.maximum(point_pred, raw_ub)
    lower_95 = np.minimum(lower_95, upper_95)

    diagnostics: Dict[str, Any] = {
        "pred_std": round(float(np.std(point_pred)), 4),
        "near_zero_pct": round(float(np.mean(point_pred < 0.05) * 100.0), 2),
        "pred_mean": round(float(np.mean(point_pred)), 4),
        "pred_min": round(float(np.min(point_pred)), 4),
        "pred_max": round(float(np.max(point_pred)), 4),
        "negative_raw_count": negative_raw_count,
        "train_samples": int(len(df_p)),
        "forecast_horizon": int(horizon),
        "selected_params": {
            "yearly_seasonality": params["yearly_seasonality"],
            "weekly_seasonality": params["weekly_seasonality"],
            "daily_seasonality": params["daily_seasonality"],
            "seasonality_mode": params["seasonality_mode"],
            "changepoint_prior_scale": params["changepoint_prior_scale"],
            "seasonality_prior_scale": params["seasonality_prior_scale"],
            "univariate": len(active_exog) == 0,
        },
        "status": "ok",
    }

    setattr(model, "diagnostics", diagnostics)
    setattr(
        model,
        "param_summary",
        f"Prophet ({params['seasonality_mode']}, yearly=True, cp={params['changepoint_prior_scale']}, sp={params['seasonality_prior_scale']})",
    )

    return point_pred, lower_95, upper_95, model
