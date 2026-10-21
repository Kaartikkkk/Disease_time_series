"""Stationarity testing, rolling diagnostics, differencing, and invertible transformations."""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from src.utils.logging_config import logger


@dataclass
class StationarityTestResult:
    is_stationary: bool
    verdict: str
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: Dict[str, float]
    adf_used_lag: int
    kpss_statistic: float
    kpss_pvalue: float
    kpss_critical_values: Dict[str, float]
    summary_explanation: str


def compute_rolling_statistics(
    series: pd.Series,
    window: int = 7,
) -> pd.DataFrame:
    """Compute rolling mean, rolling standard deviation, and rolling min/max."""
    clean_s = series.dropna()
    rolling_obj = clean_s.rolling(window=window, min_periods=max(2, window // 2))
    df_roll = pd.DataFrame({
        "original": clean_s,
        "rolling_mean": rolling_obj.mean(),
        "rolling_std": rolling_obj.std(),
        "rolling_min": rolling_obj.min(),
        "rolling_max": rolling_obj.max(),
    })
    return df_roll


def test_stationarity(
    series: pd.Series,
    significance_level: float = 0.05,
) -> StationarityTestResult:
    """Perform rigorous joint ADF and KPSS tests for stationarity.

    Methodological framework:
    - ADF Null Hypothesis (H0): The series has a unit root (NON-STATIONARY).
      Rejecting H0 (p < 0.05) suggests stationarity.
    - KPSS Null Hypothesis (H0): The series is level or trend STATIONARY.
      Rejecting H0 (p < 0.05) suggests non-stationarity.

    Joint outcomes:
    1. ADF rejects H0 (p < 0.05) and KPSS fails to reject H0 (p >= 0.05) -> Strictly Stationary.
    2. ADF fails to reject (p >= 0.05) and KPSS rejects (p < 0.05) -> Non-Stationary (unit root).
    3. Both reject H0 -> Difference stationary with structural breaks or heteroscedasticity.
    4. Neither rejects H0 -> Indeterminate (insufficient statistical power).
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna().values
    n = len(clean_s)

    if n < 10:
        return StationarityTestResult(
            is_stationary=False,
            verdict="Insufficient Data",
            adf_statistic=np.nan,
            adf_pvalue=np.nan,
            adf_critical_values={},
            adf_used_lag=0,
            kpss_statistic=np.nan,
            kpss_pvalue=np.nan,
            kpss_critical_values={},
            summary_explanation=f"Series has only {n} valid points (minimum 10 required for asymptotic tests).",
        )

    # 1. ADF Test
    try:
        max_lag = min(int(12 * (n / 100) ** 0.25), n // 3)
        adf_res = adfuller(clean_s, maxlag=max_lag, autolag="AIC")
        adf_stat = float(adf_res[0])
        adf_pval = float(adf_res[1])
        adf_lag = int(adf_res[2])
        adf_crit = {k: float(v) for k, v in adf_res[4].items()}
    except Exception as e:
        logger.warning(f"ADF test failed: {e}")
        adf_stat, adf_pval, adf_lag, adf_crit = np.nan, 1.0, 0, {}

    # 2. KPSS Test
    try:
        # nlags='auto' uses Hobijn et al. bandwidth selection
        kpss_res = kpss(clean_s, regression="c", nlags="auto")
        kpss_stat = float(kpss_res[0])
        kpss_pval = float(kpss_res[1])
        kpss_crit = {k: float(v) for k, v in kpss_res[3].items()}
    except Exception as e:
        logger.warning(f"KPSS test failed: {e}")
        kpss_stat, kpss_pval, kpss_crit = np.nan, 0.0, {}

    adf_rejects = adf_pval < significance_level
    kpss_rejects = kpss_pval < significance_level

    if adf_rejects and not kpss_rejects:
        verdict = "Strictly Stationary"
        is_stat = True
        explanation = (
            f"Both tests concur: ADF rejects unit root (p={adf_pval:.4f}) and "
            f"KPSS confirms stationarity (p={kpss_pval:.4f})."
        )
    elif not adf_rejects and kpss_rejects:
        verdict = "Non-Stationary (Unit Root Present)"
        is_stat = False
        explanation = (
            f"Both tests confirm non-stationarity: ADF fails to reject unit root (p={adf_pval:.4f}) "
            f"and KPSS rejects stationarity (p={kpss_pval:.4f}). Differencing is recommended."
        )
    elif adf_rejects and kpss_rejects:
        verdict = "Trend Stationary / Structural Break"
        is_stat = False
        explanation = (
            f"Conflicting signals: ADF indicates mean-reversion (p={adf_pval:.4f}) but "
            f"KPSS rejects stationarity (p={kpss_pval:.4f}), indicating possible deterministic trend or structural shift."
        )
    else:
        verdict = "Indeterminate"
        is_stat = False
        explanation = (
            f"Neither test reaches {significance_level:.0%} significance (ADF p={adf_pval:.4f}, KPSS p={kpss_pval:.4f}). "
            "Data series exhibits high variance or insufficient observation length."
        )

    return StationarityTestResult(
        is_stationary=is_stat,
        verdict=verdict,
        adf_statistic=adf_stat,
        adf_pvalue=adf_pval,
        adf_critical_values=adf_crit,
        adf_used_lag=adf_lag,
        kpss_statistic=kpss_stat,
        kpss_pvalue=kpss_pval,
        kpss_critical_values=kpss_crit,
        summary_explanation=explanation,
    )


def apply_difference(
    series: pd.Series,
    order: int = 1,
    seasonal_period: Optional[int] = None,
) -> pd.Series:
    """Apply first-order differencing and optional seasonal differencing."""
    diff_s = series.diff(periods=order)
    if seasonal_period and seasonal_period > 1:
        diff_s = diff_s.diff(periods=seasonal_period)
    return diff_s.dropna()


def log_transform_safe(series: pd.Series, offset: float = 1.0) -> Tuple[pd.Series, float]:
    """Apply variance-stabilizing natural log transformation log(y + offset) safely handling zeros."""
    min_val = series.min()
    safe_offset = offset
    if min_val < 0:
        safe_offset = abs(min_val) + offset
    transformed = np.log(series + safe_offset)
    return transformed, safe_offset


def inverse_log_transform(series_or_array: Any, offset: float = 1.0) -> np.ndarray:
    """Invert safe log transformation: exp(z) - offset."""
    arr = np.asarray(series_or_array)
    return np.maximum(0.0, np.exp(arr) - offset)
