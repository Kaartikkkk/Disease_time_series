"""Time Series Exploratory Data Analysis, Autocorrelation, Seasonality, and Environmental Correlation."""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from src.utils.logging_config import logger


def compute_eda_summary(
    df: pd.DataFrame,
    target_col: str = "incidence_rate",
) -> Dict[str, Any]:
    """Compute comprehensive statistical moments and distributions for the target variable."""
    if df.empty or target_col not in df.columns:
        return {}

    s = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if s.empty:
        return {}

    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "p25": float(np.percentile(s, 25)),
        "p75": float(np.percentile(s, 75)),
        "iqr": float(np.percentile(s, 75) - np.percentile(s, 25)),
        "skewness": float(stats.skew(s)) if len(s) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(s)) if len(s) > 3 else 0.0,
        "zero_count": int((s == 0.0).sum()),
        "zero_percentage": round(float((s == 0.0).sum() / len(s)) * 100, 2),
    }


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 28,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute ACF and PACF values with (1-alpha) confidence bounds.

    Returns dictionary with lags, acf_values, pacf_values, and confidence bounds.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna().values
    n = len(clean_s)

    if n < 8:
        return {
            "lags": [],
            "acf": [],
            "pacf": [],
            "conf_int": 0.0,
            "error": f"Series has only {n} observations (minimum 8 required).",
        }

    actual_lags = min(nlags, n // 3, 40)
    if actual_lags < 2:
        actual_lags = 2

    try:
        acf_vals, acf_conf = acf(clean_s, nlags=actual_lags, alpha=alpha)
        pacf_vals, pacf_conf = pacf(clean_s, nlags=actual_lags, alpha=alpha, method="yule_walker")

        # Bartlett's approx standard error for white noise bounds: +- 1.96 / sqrt(N)
        ci_bound = 1.96 / np.sqrt(n)

        return {
            "lags": list(range(len(acf_vals))),
            "acf": acf_vals.tolist(),
            "pacf": pacf_vals.tolist(),
            "conf_bound": float(ci_bound),
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Error computing ACF/PACF: {e}")
        return {
            "lags": [],
            "acf": [],
            "pacf": [],
            "conf_bound": 0.0,
            "error": str(e),
        }


def compute_monthly_seasonality_matrix(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "incidence_rate",
) -> pd.DataFrame:
    """Generate Year-by-Month matrix of mean disease incidence for heatmaps."""
    if df.empty or date_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    df_m = df.copy()
    df_m[date_col] = pd.to_datetime(df_m[date_col])
    df_m["year"] = df_m[date_col].dt.year
    df_m["month"] = df_m[date_col].dt.month

    pivot = df_m.pivot_table(index="year", columns="month", values=target_col, aggfunc="mean")
    # Rename months to 3-letter codes
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    pivot = pivot.rename(columns=month_names)
    return pivot


def compute_seasonal_subseries(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "incidence_rate",
) -> pd.DataFrame:
    """Compute monthly seasonal subseries: mean, standard deviation, and years recorded."""
    if df.empty or date_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame()

    df_s = df.copy()
    df_s[date_col] = pd.to_datetime(df_s[date_col])
    df_s["month_num"] = df_s[date_col].dt.month
    df_s["month"] = df_s[date_col].dt.strftime("%b")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    subseries = df_s.groupby(["month_num", "month"], as_index=False).agg(
        mean=(target_col, "mean"),
        std=(target_col, "std"),
        median=(target_col, "median"),
        count=(target_col, "count"),
    ).sort_values("month_num")

    return subseries


def compute_weather_disease_cross_correlation(
    df: pd.DataFrame,
    target_col: str = "incidence_rate",
    weather_cols: Optional[List[str]] = None,
    max_lag: int = 14,
) -> Dict[str, Dict[int, float]]:
    """Compute cross-correlation function (CCF) between weather variables and disease incidence at lags.

    Lag k means weather at time (t - k) predicting disease at time t.
    """
    if weather_cols is None:
        weather_cols = ["temperature", "humidity", "rainfall"]

    results = {}
    if df.empty or target_col not in df.columns:
        return results

    clean_df = df.sort_values("date").dropna(subset=[target_col]).copy()
    y = clean_df[target_col].values

    for w_col in weather_cols:
        if w_col not in clean_df.columns:
            continue
        w = pd.to_numeric(clean_df[w_col], errors="coerce").fillna(method="ffill").fillna(0.0).values
        if len(w) != len(y) or len(y) < 10:
            continue

        lag_corrs = {}
        for lag in range(0, max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(w, y)[0, 1]
            else:
                w_lagged = w[:-lag]
                y_target = y[lag:]
                if len(w_lagged) > 5 and np.std(w_lagged) > 1e-6 and np.std(y_target) > 1e-6:
                    corr = np.corrcoef(w_lagged, y_target)[0, 1]
                else:
                    corr = 0.0
            lag_corrs[lag] = float(np.nan_to_num(corr))
        results[w_col] = lag_corrs

    return results
