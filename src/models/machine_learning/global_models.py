"""Global Pooled Machine Learning & Two-Stage Hurdle Forecasting Models.

Provides multi-pathogen pooled panel modeling and zero-inflated hurdle architectures:
1. Global XGBoost: Trained on pooled panel data across all 10 crop pathogens.
2. Global Random Forest: Regularized ensemble trained on pooled panel data.
3. Hurdle XGBoost: Two-stage architecture decomposing disease incidence into:
   - Stage 1: XGBClassifier predicting outbreak occurrence probability P(y > 0)
   - Stage 2: XGBRegressor modeling positive outbreak magnitude E[y | y > 0]
   - Expected forecast: y_hat = P(y > 0) * max(0, E[y | y > 0])

Strict Temporal Leakage Protection:
- All target lags (1, 2, 3, 6, 12) and shifted rolling statistics (3, 6 months)
  are computed strictly within each individual pathogen's chronological timeline.
- Never computes cross-pathogen lags or rolling metrics.
- Slices pooled historical data strictly up to max(train_df['date']), guaranteeing
  zero future evaluation target leakage.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

HAS_XGBOOST = False
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None  # type: ignore[assignment]
    XGBRegressor = None  # type: ignore[assignment]


# Tuned via expanding-window cross-validation on internal training data (date <= 2022-08-01)
DEFAULT_GLOBAL_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 130,
    "max_depth": 4,
    "learning_rate": 0.025,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.4,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_jobs": 1,
}

DEFAULT_GLOBAL_RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 120,
    "max_depth": 4,
    "min_samples_leaf": 4,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": 1,
}

DEFAULT_HURDLE_CLF_PARAMS: Dict[str, Any] = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1,
}

DEFAULT_HURDLE_REG_PARAMS: Dict[str, Any] = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.5,
    "reg_lambda": 2.5,
    "random_state": 42,
    "n_jobs": 1,
}

MASTER_PANEL_PATH = Path("data/processed/india_plant_diseases_monthly.csv")


def build_global_panel_features(
    panel_df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
    lags: Tuple[int, ...] = (1, 2, 3, 6, 12),
    rolling_windows: Tuple[int, ...] = (3, 6),
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct chronological lag, shifted rolling, calendar, and weather features per pathogen.

    Strict Guarantee: All rolling and lag features at row t are calculated strictly
    from that specific pathogen's history at <= t-1.
    """
    df_work = panel_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["date"]):
        df_work["date"] = pd.to_datetime(df_work["date"])

    if "pathogen_slug" not in df_work.columns:
        df_work["pathogen_slug"] = "default_pathogen"

    all_slugs = sorted(df_work["pathogen_slug"].unique())
    grouped_dfs: List[pd.DataFrame] = []

    for slug, grp in df_work.groupby("pathogen_slug"):
        g = grp.sort_values("date").copy().reset_index(drop=True)
        y = g[target_col].astype(float)

        # 1. Target lags
        for l in lags:
            g[f"lag_{l}"] = y.shift(l)

        # 2. Shifted rolling statistics (never contains current y_t)
        sy = y.shift(1)
        for w in rolling_windows:
            g[f"roll_mean_{w}"] = sy.rolling(window=w, min_periods=1).mean()
            g[f"roll_std_{w}"] = sy.rolling(window=w, min_periods=1).std().fillna(0.0)
            g[f"roll_freq_{w}"] = (sy > 0).rolling(window=w, min_periods=1).mean().fillna(0.0)

        # 3. Calendar & cyclical
        m = g["date"].dt.month
        g["month"] = m.astype(float)
        g["quarter"] = g["date"].dt.quarter.astype(float)
        g["month_sin"] = np.sin(2.0 * np.pi * m / 12.0)
        g["month_cos"] = np.cos(2.0 * np.pi * m / 12.0)

        # 4. Lagged weather variables
        for w_col in ("temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"):
            if w_col in g.columns:
                g[w_col] = pd.to_numeric(g[w_col], errors="coerce").bfill().ffill().fillna(0.0)

        # 5. One-hot pathogen indicators
        for other_slug in all_slugs:
            g[f"is_{other_slug}"] = 1.0 if slug == other_slug else 0.0

        grouped_dfs.append(g)

    feat_df = pd.concat(grouped_dfs, axis=0).sort_values(["date", "pathogen_slug"]).reset_index(drop=True)

    base_feature_cols = [
        f"lag_{l}" for l in lags
    ] + [
        f"roll_mean_{w}" for w in rolling_windows
    ] + [
        f"roll_std_{w}" for w in rolling_windows
    ] + [
        f"roll_freq_{w}" for w in rolling_windows
    ] + [
        "month", "quarter", "month_sin", "month_cos",
        "temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1",
    ]
    pathogen_indicators = [f"is_{s}" for s in all_slugs]
    active_feature_cols = [c for c in base_feature_cols if c in feat_df.columns] + pathogen_indicators

    return feat_df, active_feature_cols


def _load_historical_panel(train_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, str]:
    """Retrieve historical panel up to max(train_df['date']) without lookahead."""
    df_sorted = train_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted["date"]):
        df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values("date").reset_index(drop=True)

    max_train_date = df_sorted["date"].max()
    target_slug = (
        str(df_sorted["pathogen_slug"].dropna().iloc[0])
        if "pathogen_slug" in df_sorted.columns and not df_sorted["pathogen_slug"].dropna().empty
        else "target_series"
    )

    panel_candidate = (
        Path("data/processed/improved_india_plant_diseases_monthly.csv")
        if Path("data/processed/improved_india_plant_diseases_monthly.csv").exists()
        else MASTER_PANEL_PATH
    )
    if panel_candidate.exists():
        try:
            mdf = pd.read_csv(panel_candidate)
            mdf["date"] = pd.to_datetime(mdf["date"])
            # Strict slice: observations dated <= max_train_date only
            panel_slice = mdf[mdf["date"] <= max_train_date].copy()
            if target_slug in panel_slice["pathogen_slug"].values:
                return panel_slice, target_slug
        except Exception:
            pass

    # Fallback to single series if master panel is unreadable or during isolated testing
    fallback = df_sorted.copy()
    if "pathogen_slug" not in fallback.columns:
        fallback["pathogen_slug"] = target_slug
    return fallback, target_slug


def forecast_global_xgboost(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    future_weather_df: Optional[pd.DataFrame] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """Fit Global XGBoost across pooled pathogen histories and forecast target series."""
    if not HAS_XGBOOST or XGBRegressor is None:
        raise ImportError("xgboost is not installed or unavailable in this environment.")

    t_start = time.perf_counter()
    panel_slice, target_slug = _load_historical_panel(train_df, target_col)
    feat_panel, feature_cols = build_global_panel_features(panel_slice, target_col=target_col)

    params = {**DEFAULT_GLOBAL_XGB_PARAMS, **(xgb_params or {})}
    exclude_weather = bool(params.pop("exclude_weather", False))
    exclude_rolling = bool(params.pop("exclude_rolling", False))
    exclude_lag12 = bool(params.pop("exclude_lag12", False))
    exclude_pathogen_id = bool(params.pop("exclude_pathogen_id", False))

    if exclude_weather:
        weather_cols = {"temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"}
        feature_cols = [c for c in feature_cols if c not in weather_cols]
    if exclude_rolling:
        feature_cols = [c for c in feature_cols if not c.startswith("roll_")]
    if exclude_lag12:
        feature_cols = [c for c in feature_cols if c != "lag_12"]
    if exclude_pathogen_id:
        feature_cols = [c for c in feature_cols if not c.startswith("is_")]

    clean_train = feat_panel.dropna(subset=feature_cols + [target_col]).copy()
    if len(clean_train) < 12:
        raise ValueError(f"Insufficient training observations ({len(clean_train)}) for Global XGBoost.")

    X_train = clean_train[feature_cols].values
    y_train = clean_train[target_col].astype(float).values

    model = XGBRegressor(**params)
    sample_weight = (
        np.where(clean_train["is_missing"] == 1, 0.2, 1.0)
        if "is_missing" in clean_train.columns
        else None
    )
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train)

    # Recursive multi-step forecast for the target pathogen
    target_history_df = panel_slice[panel_slice["pathogen_slug"] == target_slug].sort_values("date").reset_index(drop=True)
    history = target_history_df[target_col].astype(float).tolist()
    last_date = target_history_df["date"].iloc[-1]
    all_slugs = [(c[3:] if c.startswith("is_") else c) for c in feature_cols if c.startswith("is_")]

    preds: List[float] = []
    for step in range(1, horizon + 1):
        step_date = (
            pd.to_datetime(future_weather_df["date"].iloc[step - 1])
            if (future_weather_df is not None and "date" in future_weather_df.columns and len(future_weather_df) >= step)
            else last_date + pd.DateOffset(months=step)
        )
        row_dict: Dict[str, float] = {}
        for l in (1, 2, 3, 6, 12):
            row_dict[f"lag_{l}"] = float(history[-l]) if len(history) >= l else float(history[-1])
        for w in (3, 6):
            w_slice = np.asarray(history[-w:], dtype=float)
            row_dict[f"roll_mean_{w}"] = float(np.mean(w_slice))
            row_dict[f"roll_std_{w}"] = float(np.std(w_slice, ddof=1)) if len(w_slice) > 1 else 0.0
            row_dict[f"roll_freq_{w}"] = float(np.mean(w_slice > 0))

        m = float(step_date.month)
        row_dict["month"] = m
        row_dict["quarter"] = float(step_date.quarter)
        row_dict["month_sin"] = float(np.sin(2.0 * np.pi * m / 12.0))
        row_dict["month_cos"] = float(np.cos(2.0 * np.pi * m / 12.0))

        # Weather features
        for w_col in ("temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"):
            if future_weather_df is not None and w_col in future_weather_df.columns and len(future_weather_df) >= step:
                row_dict[w_col] = float(future_weather_df[w_col].iloc[step - 1])
            else:
                row_dict[w_col] = float(clean_train[w_col].mean()) if w_col in clean_train.columns else 0.0

        for s in all_slugs:
            row_dict[f"is_{s}"] = 1.0 if s == target_slug else 0.0

        row_vals = [row_dict[c] for c in feature_cols]
        pred_step = max(0.0, float(model.predict(np.asarray([row_vals], dtype=float))[0]))
        preds.append(pred_step)
        history.append(pred_step)

    preds_arr = np.asarray(preds, dtype=float)
    train_preds = np.maximum(0.0, model.predict(X_train))
    resids = y_train - train_preds
    se = float(np.std(resids)) if len(resids) > 0 else 1.0

    lower_95 = np.maximum(0.0, preds_arr - 1.96 * se)
    upper_95 = preds_arr + 1.96 * se
    runtime_sec = round(time.perf_counter() - t_start, 4)

    diagnostics: Dict[str, Any] = {
        "model_type": "Global XGBoost (Multi-Pathogen)",
        "target_pathogen": target_slug,
        "pooled_train_samples": int(len(clean_train)),
        "n_features": len(feature_cols),
        "hyperparameters": params,
        "training_runtime_sec": runtime_sec,
        "pred_mean": round(float(np.mean(preds_arr)), 4),
        "pred_std": round(float(np.std(preds_arr)), 4),
        "near_zero_pct": round(float(np.mean(preds_arr < 0.05) * 100.0), 2),
        "status": "ok",
    }
    return preds_arr, lower_95, upper_95, model, diagnostics


def forecast_global_random_forest(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    future_weather_df: Optional[pd.DataFrame] = None,
    rf_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """Fit Global Random Forest across pooled pathogen histories and forecast target series."""
    t_start = time.perf_counter()
    panel_slice, target_slug = _load_historical_panel(train_df, target_col)
    feat_panel, feature_cols = build_global_panel_features(panel_slice, target_col=target_col)

    params = {**DEFAULT_GLOBAL_RF_PARAMS, **(rf_params or {})}
    exclude_weather = bool(params.pop("exclude_weather", False))
    exclude_rolling = bool(params.pop("exclude_rolling", False))
    exclude_lag12 = bool(params.pop("exclude_lag12", False))
    exclude_pathogen_id = bool(params.pop("exclude_pathogen_id", False))

    if exclude_weather:
        weather_cols = {"temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"}
        feature_cols = [c for c in feature_cols if c not in weather_cols]
    if exclude_rolling:
        feature_cols = [c for c in feature_cols if not c.startswith("roll_")]
    if exclude_lag12:
        feature_cols = [c for c in feature_cols if c != "lag_12"]
    if exclude_pathogen_id:
        feature_cols = [c for c in feature_cols if not c.startswith("is_")]

    clean_train = feat_panel.dropna(subset=feature_cols + [target_col]).copy()
    if len(clean_train) < 12:
        raise ValueError(f"Insufficient training observations ({len(clean_train)}) for Global Random Forest.")

    X_train = clean_train[feature_cols].values
    y_train = clean_train[target_col].astype(float).values

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    target_history_df = panel_slice[panel_slice["pathogen_slug"] == target_slug].sort_values("date").reset_index(drop=True)
    history = target_history_df[target_col].astype(float).tolist()
    last_date = target_history_df["date"].iloc[-1]
    all_slugs = [(c[3:] if c.startswith("is_") else c) for c in feature_cols if c.startswith("is_")]

    preds: List[float] = []
    for step in range(1, horizon + 1):
        step_date = (
            pd.to_datetime(future_weather_df["date"].iloc[step - 1])
            if (future_weather_df is not None and "date" in future_weather_df.columns and len(future_weather_df) >= step)
            else last_date + pd.DateOffset(months=step)
        )
        row_dict: Dict[str, float] = {}
        for l in (1, 2, 3, 6, 12):
            row_dict[f"lag_{l}"] = float(history[-l]) if len(history) >= l else float(history[-1])
        for w in (3, 6):
            w_slice = np.asarray(history[-w:], dtype=float)
            row_dict[f"roll_mean_{w}"] = float(np.mean(w_slice))
            row_dict[f"roll_std_{w}"] = float(np.std(w_slice, ddof=1)) if len(w_slice) > 1 else 0.0
            row_dict[f"roll_freq_{w}"] = float(np.mean(w_slice > 0))

        m = float(step_date.month)
        row_dict["month"] = m
        row_dict["quarter"] = float(step_date.quarter)
        row_dict["month_sin"] = float(np.sin(2.0 * np.pi * m / 12.0))
        row_dict["month_cos"] = float(np.cos(2.0 * np.pi * m / 12.0))

        for w_col in ("temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"):
            if future_weather_df is not None and w_col in future_weather_df.columns and len(future_weather_df) >= step:
                row_dict[w_col] = float(future_weather_df[w_col].iloc[step - 1])
            else:
                row_dict[w_col] = float(clean_train[w_col].mean()) if w_col in clean_train.columns else 0.0

        for s in all_slugs:
            row_dict[f"is_{s}"] = 1.0 if s == target_slug else 0.0

        row_vals = [row_dict[c] for c in feature_cols]
        pred_step = max(0.0, float(model.predict(np.asarray([row_vals], dtype=float))[0]))
        preds.append(pred_step)
        history.append(pred_step)

    preds_arr = np.asarray(preds, dtype=float)
    train_preds = np.maximum(0.0, model.predict(X_train))
    resids = y_train - train_preds
    se = float(np.std(resids)) if len(resids) > 0 else 1.0

    lower_95 = np.maximum(0.0, preds_arr - 1.96 * se)
    upper_95 = preds_arr + 1.96 * se
    runtime_sec = round(time.perf_counter() - t_start, 4)

    diagnostics: Dict[str, Any] = {
        "model_type": "Global Random Forest (Multi-Pathogen)",
        "target_pathogen": target_slug,
        "pooled_train_samples": int(len(clean_train)),
        "n_features": len(feature_cols),
        "hyperparameters": params,
        "training_runtime_sec": runtime_sec,
        "pred_mean": round(float(np.mean(preds_arr)), 4),
        "pred_std": round(float(np.std(preds_arr)), 4),
        "near_zero_pct": round(float(np.mean(preds_arr < 0.05) * 100.0), 2),
        "status": "ok",
    }
    return preds_arr, lower_95, upper_95, model, diagnostics


def forecast_hurdle_xgboost(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    future_weather_df: Optional[pd.DataFrame] = None,
    clf_params: Optional[Dict[str, Any]] = None,
    reg_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Dict[str, Any]]:
    """Fit Two-Stage Hurdle XGBoost (P(y>0) * E[y|y>0]) and forecast target series."""
    if not HAS_XGBOOST or XGBClassifier is None or XGBRegressor is None:
        raise ImportError("xgboost is not installed or unavailable in this environment.")

    t_start = time.perf_counter()
    panel_slice, target_slug = _load_historical_panel(train_df, target_col)
    feat_panel, feature_cols = build_global_panel_features(panel_slice, target_col=target_col)

    clean_train = feat_panel.dropna(subset=feature_cols + [target_col]).copy()
    if len(clean_train) < 12:
        raise ValueError(f"Insufficient training observations ({len(clean_train)}) for Hurdle XGBoost.")

    X_train = clean_train[feature_cols].values
    y_train = clean_train[target_col].astype(float).values

    # Stage 1: Binary Occurrence Classifier
    c_params = {**DEFAULT_HURDLE_CLF_PARAMS, **(clf_params or {})}
    y_train_binary = (y_train > 0.0).astype(int)
    unique_classes = np.unique(y_train_binary)
    if len(unique_classes) < 2:
        clf = None
        constant_prob = float(unique_classes[0])
    else:
        clf = XGBClassifier(**c_params)
        sample_weight = (
            np.where(clean_train["is_missing"] == 1, 0.2, 1.0)
            if "is_missing" in clean_train.columns
            else None
        )
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        try:
            clf.fit(X_train, y_train_binary, **fit_kwargs)
        except TypeError:
            clf.fit(X_train, y_train_binary)
        constant_prob = None

    # Stage 2: Positive Magnitude Regressor
    r_params = {**DEFAULT_HURDLE_REG_PARAMS, **(reg_params or {})}
    pos_mask = y_train > 0.0
    if pos_mask.sum() >= 5:
        reg = XGBRegressor(**r_params)
        reg.fit(X_train[pos_mask], y_train[pos_mask])
    elif pos_mask.sum() > 0:
        reg = None
        default_mag = float(np.mean(y_train[pos_mask]))
    else:
        reg = None
        default_mag = 0.0

    target_history_df = panel_slice[panel_slice["pathogen_slug"] == target_slug].sort_values("date").reset_index(drop=True)
    history = target_history_df[target_col].astype(float).tolist()
    last_date = target_history_df["date"].iloc[-1]
    all_slugs = [(c[3:] if c.startswith("is_") else c) for c in feature_cols if c.startswith("is_")]

    preds: List[float] = []
    probs: List[float] = []
    mags: List[float] = []

    for step in range(1, horizon + 1):
        step_date = (
            pd.to_datetime(future_weather_df["date"].iloc[step - 1])
            if (future_weather_df is not None and "date" in future_weather_df.columns and len(future_weather_df) >= step)
            else last_date + pd.DateOffset(months=step)
        )
        row_vals: List[float] = []
        for l in (1, 2, 3, 6, 12):
            row_vals.append(float(history[-l]) if len(history) >= l else float(history[-1]))
        for w in (3, 6):
            w_slice = np.asarray(history[-w:], dtype=float)
            row_vals.append(float(np.mean(w_slice)))
            row_vals.append(float(np.std(w_slice, ddof=1)) if len(w_slice) > 1 else 0.0)
            row_vals.append(float(np.mean(w_slice > 0)))

        m = float(step_date.month)
        row_vals.extend([
            m,
            float(step_date.quarter),
            float(np.sin(2.0 * np.pi * m / 12.0)),
            float(np.cos(2.0 * np.pi * m / 12.0)),
        ])

        for w_col in ("temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"):
            if w_col in feature_cols:
                if future_weather_df is not None and w_col in future_weather_df.columns and len(future_weather_df) >= step:
                    row_vals.append(float(future_weather_df[w_col].iloc[step - 1]))
                else:
                    row_vals.append(float(clean_train[w_col].mean()) if w_col in clean_train.columns else 0.0)

        for s in all_slugs:
            row_vals.append(1.0 if s == target_slug else 0.0)

        x_step = np.asarray([row_vals], dtype=float)
        prob_occ = constant_prob if constant_prob is not None else float(clf.predict_proba(x_step)[0, 1])
        if reg is not None:
            pred_mag = max(0.0, float(reg.predict(x_step)[0]))
        else:
            pred_mag = default_mag

        expected_count = max(0.0, prob_occ * pred_mag)
        preds.append(expected_count)
        probs.append(round(prob_occ, 4))
        mags.append(round(pred_mag, 4))
        history.append(expected_count)

    preds_arr = np.asarray(preds, dtype=float)

    # In-sample residuals for prediction interval estimation
    if clf is not None:
        train_probs = clf.predict_proba(X_train)[:, 1]
    else:
        train_probs = np.full(len(X_train), constant_prob)
    if reg is not None:
        train_mags = np.maximum(0.0, reg.predict(X_train))
    else:
        train_mags = np.full(len(X_train), default_mag)
    train_preds = train_probs * train_mags
    resids = y_train - train_preds
    se = float(np.std(resids)) if len(resids) > 0 else 1.0

    lower_95 = np.maximum(0.0, preds_arr - 1.96 * se)
    upper_95 = preds_arr + 1.96 * se
    runtime_sec = round(time.perf_counter() - t_start, 4)

    diagnostics: Dict[str, Any] = {
        "model_type": "Two-Stage Hurdle XGBoost",
        "target_pathogen": target_slug,
        "pooled_train_samples": int(len(clean_train)),
        "positive_train_samples": int(pos_mask.sum()),
        "stage1_probabilities": probs,
        "stage2_magnitudes": mags,
        "training_runtime_sec": runtime_sec,
        "pred_mean": round(float(np.mean(preds_arr)), 4),
        "pred_std": round(float(np.std(preds_arr)), 4),
        "near_zero_pct": round(float(np.mean(preds_arr < 0.05) * 100.0), 2),
        "status": "ok",
    }
    return preds_arr, lower_95, upper_95, (clf, reg), diagnostics
