"""Leakage-free lag, rolling window, and environmental feature extraction."""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from src.features.calendar_features import add_calendar_features
from src.utils.logging_config import logger


def generate_lag_features(
    df: pd.DataFrame,
    target_col: str = "incidence_rate",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    weather_lags: Optional[List[int]] = None,
    include_weather: bool = True,
    frequency: str = "D",
) -> pd.DataFrame:
    """Generate strictly backward-looking lag and rolling features.

    LEAKAGE PREVENTION RULES:
    1. Lag k always satisfies k >= 1 (shift(k)).
    2. Rolling means and rolling standard deviations are computed strictly on
       the shifted series: series.shift(1).rolling(window).mean(), ensuring
       observation at time t is never used to predict itself.
    3. Weather variables are lagged by >= 1 step unless explicitly running a
       scenario-based future weather forecast mode.
    """
    if df.empty or target_col not in df.columns:
        return df

    if lags is None:
        if frequency in ["D", "DAILY"]:
            lags = [1, 2, 3, 7, 14, 21, 28]
        elif frequency in ["W", "WEEKLY", "W-SUN"]:
            lags = [1, 2, 4, 8, 12]
        else:
            lags = [1, 2, 3, 6, 12]

    if rolling_windows is None:
        if frequency in ["D", "DAILY"]:
            rolling_windows = [3, 7, 14, 28]
        elif frequency in ["W", "WEEKLY", "W-SUN"]:
            rolling_windows = [2, 4, 8]
        else:
            rolling_windows = [3, 6]

    if weather_lags is None:
        weather_lags = [1, 3, 7]

    df_out = df.sort_values("date").copy()

    # 1. Calendar & Cyclical features
    df_out = add_calendar_features(df_out, date_col="date", frequency=frequency)

    # 2. Target lag features: shift(k)
    for lag in lags:
        df_out[f"target_lag_{lag}"] = df_out[target_col].shift(lag)

    # 3. Rolling statistics on strictly lagged series (shift(1))
    shifted_target = df_out[target_col].shift(1)
    for win in rolling_windows:
        df_out[f"target_roll_mean_{win}"] = shifted_target.rolling(window=win, min_periods=max(2, win // 2)).mean()
        df_out[f"target_roll_std_{win}"] = shifted_target.rolling(window=win, min_periods=max(2, win // 2)).std()

    # 4. Lagged environmental covariates
    weather_cols = ["temperature", "humidity", "rainfall"]
    if include_weather:
        for w_col in weather_cols:
            if w_col in df_out.columns:
                shifted_w = df_out[w_col].shift(1)
                for w_lag in weather_lags:
                    df_out[f"{w_col}_lag_{w_lag}"] = df_out[w_col].shift(w_lag)
                df_out[f"{w_col}_roll_mean_7"] = shifted_w.rolling(window=7, min_periods=2).mean()

    return df_out


def create_supervised_matrix(
    df: pd.DataFrame,
    target_col: str = "incidence_rate",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    include_weather: bool = True,
    frequency: str = "D",
    drop_na_rows: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Compile feature matrix X and target vector y for supervised ML models.

    Returns:
    - X: feature DataFrame with clean column names and metadata excluded
    - y: continuous target Series
    - feature_names: list of predictor names
    """
    df_features = generate_lag_features(
        df,
        target_col=target_col,
        lags=lags,
        rolling_windows=rolling_windows,
        include_weather=include_weather,
        frequency=frequency,
    )

    # Exclude non-feature metadata columns from X
    metadata_cols = [
        "date",
        "crop",
        "disease",
        "location_id",
        "source",
        "is_synthetic",
        "cases_or_affected",
        "inspected_or_total",
        target_col,  # y is separated
    ]

    feature_cols = [c for c in df_features.columns if c not in metadata_cols]

    if drop_na_rows:
        valid_mask = df_features[feature_cols + [target_col]].notna().all(axis=1)
        df_clean = df_features[valid_mask].copy()
    else:
        df_clean = df_features.copy()

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    logger.info(
        f"Generated supervised matrix: X shape {X.shape}, y shape {y.shape} ({len(feature_cols)} features)."
    )
    return X, y, feature_cols
