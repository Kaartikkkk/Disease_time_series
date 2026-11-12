"""XGBoost Autoregressive Forecasting Model with Lag, Rolling, Calendar, and Weather Features.

Designed for sparse, zero-inflated monthly plant disease surveillance counts:
- Strictly chronological lag features: [1, 2, 3, 6, 12]
- Strictly shifted rolling statistics (3m and 6m mean/std using y_{t-1}..y_{t-w})
- Calendar harmonics & time index: month, quarter, sin/cos(2*pi*m/12), time_idx
- Lagged NASA POWER meteorological regressors (when available)
- Conservative tree regularization to prevent overfitting on ~80-120 monthly samples
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

HAS_XGBOOST = False
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    XGBRegressor = None  # type: ignore[assignment]


DEFAULT_XGB_LAGS = [1, 2, 3, 6, 12]
DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 120,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": 1,
}


def build_xgboost_features(
    df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
    lags: Optional[List[int]] = None,
    exog_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build leakage-free autoregressive, shifted rolling, calendar, and weather features.

    All lag and rolling features at row t use strictly observations <= t-1.
    """
    if lags is None:
        lags = list(DEFAULT_XGB_LAGS)

    df_feat = df.copy()
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    series = df_feat[target_col].astype(float)
    feature_cols: List[str] = []

    # 1. Target lag features
    for lag in lags:
        col = f"lag_{lag}"
        df_feat[col] = series.shift(lag)
        feature_cols.append(col)

    # 2. Strictly shifted rolling statistics (never includes current target y_t)
    shifted = series.shift(1)
    for w in (3, 6):
        mean_col = f"roll_mean_{w}"
        std_col = f"roll_std_{w}"
        df_feat[mean_col] = shifted.rolling(window=w, min_periods=1).mean()
        df_feat[std_col] = shifted.rolling(window=w, min_periods=1).std().fillna(0.0)
        feature_cols.extend([mean_col, std_col])

    # 3. Calendar & trend index features
    months = df_feat["date"].dt.month
    df_feat["month"] = months.astype(float)
    df_feat["quarter"] = df_feat["date"].dt.quarter.astype(float)
    df_feat["month_sin"] = np.sin(2.0 * np.pi * months / 12.0)
    df_feat["month_cos"] = np.cos(2.0 * np.pi * months / 12.0)
    df_feat["time_idx"] = np.arange(len(df_feat), dtype=float)
    feature_cols.extend(["month", "quarter", "month_sin", "month_cos", "time_idx"])

    # 4. Existing lagged weather features
    if exog_cols:
        for w_col in exog_cols:
            if w_col in df_feat.columns:
                df_feat[w_col] = pd.to_numeric(df_feat[w_col], errors="coerce").bfill().ffill().fillna(0.0)
                feature_cols.append(w_col)

    return df_feat, feature_cols


def forecast_xgboost(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    lags: Optional[List[int]] = None,
    weather_lag_cols: Optional[List[str]] = None,
    future_weather_df: Optional[pd.DataFrame] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """Fit XGBRegressor on strictly historical features and run recursive forecasting.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model, diagnostics).
    """
    if not HAS_XGBOOST or XGBRegressor is None:
        raise ImportError("xgboost is not installed or unavailable in this environment.")

    if lags is None:
        lags = list(DEFAULT_XGB_LAGS)

    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    df_sorted = train_df.copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values("date").reset_index(drop=True)

    active_exog = [c for c in (weather_lag_cols or []) if c in df_sorted.columns]
    df_feat, feature_cols = build_xgboost_features(
        df_sorted,
        target_col=target_col,
        lags=lags,
        exog_cols=active_exog,
    )

    df_train_clean = df_feat.dropna(subset=feature_cols + [target_col]).copy()
    if len(df_train_clean) < 12:
        raise ValueError(
            f"Insufficient training rows ({len(df_train_clean)}) after lag construction for XGBoost."
        )

    X_train = df_train_clean[feature_cols].values
    y_train = df_train_clean[target_col].astype(float).values

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Recursive multi-step out-of-sample forecast
    history = df_sorted[target_col].astype(float).tolist()
    last_date = df_sorted["date"].iloc[-1]
    base_time_idx = len(df_sorted)
    exog_means = {c: float(df_train_clean[c].mean()) for c in active_exog}

    preds: List[float] = []
    for step in range(1, horizon + 1):
        step_date = (
            pd.to_datetime(future_weather_df["date"].iloc[step - 1])
            if (future_weather_df is not None and "date" in future_weather_df.columns and len(future_weather_df) >= step)
            else last_date + pd.DateOffset(months=step)
        )
        row_vals: List[float] = []

        # Lags
        for lag in lags:
            row_vals.append(float(history[-lag]) if len(history) >= lag else float(history[-1]))

        # Shifted rolling mean/std over 3 and 6 months from history
        for w in (3, 6):
            window_slice = np.asarray(history[-w:], dtype=float)
            row_vals.append(float(np.mean(window_slice)))
            row_vals.append(float(np.std(window_slice, ddof=1)) if len(window_slice) > 1 else 0.0)

        # Calendar & time index
        m = float(step_date.month)
        q = float((step_date.month - 1) // 3 + 1)
        row_vals.extend([
            m,
            q,
            float(np.sin(2.0 * np.pi * m / 12.0)),
            float(np.cos(2.0 * np.pi * m / 12.0)),
            float(base_time_idx + step - 1),
        ])

        # Exogenous lagged weather
        for w_col in active_exog:
            if future_weather_df is not None and w_col in future_weather_df.columns and len(future_weather_df) >= step:
                val = pd.to_numeric(future_weather_df[w_col].iloc[step - 1], errors="coerce")
                row_vals.append(float(val) if np.isfinite(val) else exog_means[w_col])
            else:
                row_vals.append(exog_means[w_col])

        step_pred = float(model.predict(np.asarray([row_vals], dtype=float))[0])
        step_pred = max(0.0, step_pred)
        preds.append(step_pred)
        history.append(step_pred)

    preds_arr = np.asarray(preds, dtype=float)

    # Residual-based 95% confidence interval
    train_preds = np.maximum(0.0, model.predict(X_train))
    se = float(np.std(y_train - train_preds)) if len(y_train) > 1 else 1.0
    lower_95 = np.maximum(0.0, preds_arr - 1.96 * se)
    upper_95 = preds_arr + 1.96 * se

    diagnostics: Dict[str, Any] = {
        "pred_std": round(float(np.std(preds_arr)), 4),
        "near_zero_pct": round(float(np.mean(preds_arr < 0.05) * 100.0), 2),
        "pred_mean": round(float(np.mean(preds_arr)), 4),
        "pred_min": round(float(np.min(preds_arr)), 4),
        "pred_max": round(float(np.max(preds_arr)), 4),
        "train_samples": int(len(X_train)),
        "feature_count": int(len(feature_cols)),
        "feature_names": feature_cols,
        "hyperparameters": {k: v for k, v in params.items() if k != "n_jobs"},
        "objective": str(params.get("objective", "reg:squarederror")),
        "status": "ok",
    }

    return preds_arr, lower_95, upper_95, model, diagnostics

