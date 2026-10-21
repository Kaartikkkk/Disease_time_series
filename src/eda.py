"""Comprehensive Exploratory Data Analysis (EDA) Module for Plant Disease Forecasting.

Computes:
- Dataset shape, schema, summary statistics, and missing value profiles
- Temporal coverage, observation density, and multi-year gaps
- Rolling central tendencies (rolling mean, rolling std, rolling volatility)
- Outlier detection via robust IQR fences and Z-score thresholds
- Autocorrelation (ACF) and Partial Autocorrelation (PACF) with 95% Bartlett confidence limits
- Seasonal decomposition (additive and multiplicative components)
- Weather-disease cross-correlations across distributed lags (0 to 3 months)
- Formal assessment of whether a time series is suitable for forecasting
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

logger = logging.getLogger(__name__)


def compute_series_summary(df: pd.DataFrame, target_col: str = "documented_occurrence_count") -> Dict[str, Any]:
    """Compute exhaustive descriptive summary statistics for a time-series dataframe."""
    series = df[target_col].dropna()
    dates = pd.to_datetime(df["date"])

    total_months = len(df)
    missing_months = int(df["is_missing"].sum()) if "is_missing" in df.columns else 0
    non_zero_months = int((series > 0).sum())
    zero_months = int((series == 0).sum())

    mean_val = float(series.mean()) if not series.empty else 0.0
    std_val = float(series.std()) if len(series) > 1 else 0.0
    median_val = float(series.median()) if not series.empty else 0.0
    iqr_val = float(series.quantile(0.75) - series.quantile(0.25)) if not series.empty else 0.0
    skew_val = float(stats.skew(series)) if len(series) > 2 and std_val > 0 else 0.0
    kurt_val = float(stats.kurtosis(series)) if len(series) > 3 and std_val > 0 else 0.0

    return {
        "total_months": total_months,
        "missing_months": missing_months,
        "active_months": total_months - missing_months,
        "non_zero_months": non_zero_months,
        "zero_months": zero_months,
        "pct_non_zero": round(non_zero_months / max(1, total_months) * 100, 2),
        "total_occurrences": int(series.sum()),
        "mean_monthly_occurrences": round(mean_val, 2),
        "std_monthly_occurrences": round(std_val, 2),
        "median": round(median_val, 2),
        "iqr": round(iqr_val, 2),
        "min": float(series.min()) if not series.empty else 0.0,
        "max": float(series.max()) if not series.empty else 0.0,
        "skewness": round(skew_val, 3),
        "kurtosis": round(kurt_val, 3),
        "start_date": dates.min().strftime("%Y-%m-%d") if not dates.empty else "N/A",
        "end_date": dates.max().strftime("%Y-%m-%d") if not dates.empty else "N/A",
    }


def compute_rolling_statistics(
    df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
    window: int = 6,
) -> pd.DataFrame:
    """Calculate rolling statistics (mean, std, min, max) for trend and volatility tracking."""
    df_out = df[["date", target_col]].copy()
    df_out[f"rolling_mean_{window}m"] = df_out[target_col].rolling(window=window, min_periods=1).mean()
    df_out[f"rolling_std_{window}m"] = df_out[target_col].rolling(window=window, min_periods=1).std().fillna(0)
    df_out[f"rolling_min_{window}m"] = df_out[target_col].rolling(window=window, min_periods=1).min()
    df_out[f"rolling_max_{window}m"] = df_out[target_col].rolling(window=window, min_periods=1).max()
    return df_out


def detect_outliers(
    df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
    iqr_multiplier: float = 1.5,
) -> Dict[str, Any]:
    """Identify outliers using IQR upper fence and Z-score criteria."""
    s = df[target_col]
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)
    iqr = q75 - q25
    upper_fence = q75 + (iqr_multiplier * iqr)
    lower_fence = max(0.0, q25 - (iqr_multiplier * iqr))

    # Standardized Z-Score
    mean = s.mean()
    std = s.std() if s.std() > 0 else 1.0
    z_scores = (s - mean) / std

    iqr_outliers = df[s > upper_fence][["date", target_col]].copy()
    z_outliers = df[z_scores > 3.0][["date", target_col]].copy()

    return {
        "iqr_upper_fence": round(upper_fence, 2),
        "iqr_outliers_count": len(iqr_outliers),
        "iqr_outliers": iqr_outliers.to_dict(orient="records"),
        "z_outliers_count": len(z_outliers),
        "z_outliers": z_outliers.to_dict(orient="records"),
    }


def compute_acf_pacf_values(
    df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
    nlags: int = 24,
) -> Dict[str, Any]:
    """Calculate Autocorrelation and Partial Autocorrelation coefficients with 95% CI."""
    series = df[target_col].values
    nlags = min(nlags, len(series) // 2 - 1)
    if nlags < 2:
        return {"lags": [], "acf": [], "pacf": [], "confidence_interval": 0.0}

    acf_vals = acf(series, nlags=nlags, fft=True)
    try:
        pacf_vals = pacf(series, nlags=nlags, method="yule_walker")
    except Exception:
        pacf_vals = acf_vals  # Fallback if matrix singular

    # 95% Bartlett confidence band is approx 1.96 / sqrt(N)
    ci = 1.96 / np.sqrt(len(series))

    return {
        "lags": list(range(nlags + 1)),
        "acf": [round(float(v), 4) for v in acf_vals],
        "pacf": [round(float(v), 4) for v in pacf_vals],
        "confidence_interval": round(float(ci), 4),
    }


def perform_seasonal_decomposition(
    df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
    period: int = 12,
    model: str = "additive",
) -> Optional[pd.DataFrame]:
    """Decompose series into Trend, Seasonal, and Residual components."""
    if len(df) < period * 2:
        logger.warning("Series length (%d) is insufficient for seasonal decomposition (requires >= %d).", len(df), period * 2)
        return None

    # Series must be indexed by continuous DatetimeIndex with frequency
    ts = df.set_index("date")[target_col].asfreq("MS")
    # If zeros exist and model is multiplicative, fallback to additive
    if model == "multiplicative" and (ts <= 0).any():
        model = "additive"

    try:
        res = seasonal_decompose(ts, model=model, period=period)
        decomp_df = pd.DataFrame({
            "date": ts.index,
            "observed": res.observed.values,
            "trend": res.trend.values,
            "seasonal": res.seasonal.values,
            "residual": res.resid.values,
        })
        return decomp_df
    except Exception as exc:
        logger.error("Seasonal decomposition failed: %s", exc)
        return None


def compute_weather_disease_correlations(
    df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
    max_lag: int = 3,
) -> Dict[str, Any]:
    """Compute Pearson & Spearman correlation between target and climate factors across lags."""
    weather_vars = [
        ("temp_mean_c", "Mean Temperature (°C)"),
        ("temp_max_c", "Max Temperature (°C)"),
        ("temp_min_c", "Min Temperature (°C)"),
        ("precipitation_monthly_mm", "Precipitation (mm)"),
        ("relative_humidity_pct", "Relative Humidity (%)"),
        ("dew_point_c", "Dew Point (°C)"),
    ]

    results = []
    target_series = df[target_col]

    for col, label in weather_vars:
        if col not in df.columns:
            continue

        for lag in range(max_lag + 1):
            w_series = df[col].shift(lag)
            valid_mask = target_series.notna() & w_series.notna()

            if valid_mask.sum() > 5:
                p_corr, p_val = stats.pearsonr(w_series[valid_mask], target_series[valid_mask])
                s_corr, s_val = stats.spearmanr(w_series[valid_mask], target_series[valid_mask])
                results.append({
                    "weather_variable": col,
                    "label": label,
                    "lag_months": lag,
                    "pearson_r": round(float(p_corr), 3),
                    "pearson_p_value": round(float(p_val), 4),
                    "spearman_rho": round(float(s_corr), 3),
                    "spearman_p_value": round(float(s_val), 4),
                    "is_statistically_significant": bool(p_val < 0.05),
                })

    return {
        "correlations": results,
        "scientific_disclaimer": (
            "Empirical correlation indicates statistical co-occurrence and does NOT prove causation. "
            "Biophysical pathogen proliferation requires specific host susceptibility, spore viability, "
            "and incubation windows in conjunction with macro-climatic permissive thresholds."
        ),
    }


def assess_forecasting_suitability(
    df: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
    pathogen_name: str = "",
) -> Dict[str, Any]:
    """Assess whether a time-series series meets statistical criteria for forecasting."""
    series = df[target_col].dropna()
    total_months = len(series)
    non_zeros = int((series > 0).sum())
    variance = float(series.var())

    criteria = {
        "sufficient_length": bool(total_months >= 24),
        "non_trivial_variance": bool(variance > 0.01),
        "nonzero_observation_density": bool(non_zeros >= 12),
    }

    is_suitable = all(criteria.values())

    if is_suitable:
        recommendation = (
            f"Series for '{pathogen_name}' is statistically viable for time-series forecasting. "
            f"Observed {non_zeros} active months out of {total_months}."
        )
    else:
        recommendation = (
            f"Series for '{pathogen_name}' exhibits severe sparsity ({non_zeros} non-zero months). "
            f"Classical seasonal models (e.g. SARIMA) may struggle with over-parameterization; "
            f"intermittent demand models or baseline persistence are recommended."
        )

    return {
        "pathogen_name": pathogen_name,
        "is_suitable": is_suitable,
        "criteria": criteria,
        "recommendation": recommendation,
    }
