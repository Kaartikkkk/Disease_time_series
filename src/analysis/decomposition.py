"""Time series decomposition into Trend, Seasonal, and Residual components."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from src.utils.logging_config import logger


@dataclass
class DecompositionResult:
    is_valid: bool
    method: str
    period: int
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series
    observed: pd.Series
    seasonal_strength: float  # Var(seasonal) / Var(observed) proxy
    residual_variance: float
    error_message: Optional[str] = None


def decompose_series(
    series: pd.Series,
    period: int = 7,
    method: str = "stl",  # 'stl' or 'classical'
    model_type: str = "additive",
) -> DecompositionResult:
    """Decompose time series into Trend, Seasonal, and Residual components.

    Handles minimum period requirements and missing values gracefully.
    Requires at least 2 full seasonal periods (2 * period) observations.
    """
    clean_s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean_s)

    if n < max(14, 2 * period):
        return DecompositionResult(
            is_valid=False,
            method=method,
            period=period,
            trend=pd.Series(dtype=float),
            seasonal=pd.Series(dtype=float),
            resid=pd.Series(dtype=float),
            observed=clean_s,
            seasonal_strength=0.0,
            residual_variance=0.0,
            error_message=f"Series has {n} observations; at least {2 * period} required for period={period}.",
        )

    try:
        if method.lower() == "stl":
            # STL requires odd seasonal window
            s_window = 7 if period % 2 == 1 else 9
            stl = STL(clean_s, period=period, seasonal=s_window, robust=True)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid
        else:
            res = seasonal_decompose(clean_s, period=period, model=model_type, extrapolate_trend="freq")
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid

        # Compute seasonal strength metric: max(0, 1 - Var(resid) / Var(trend + resid))
        detrended = clean_s - trend
        var_detrended = np.var(detrended)
        var_resid = np.var(resid)
        seasonal_strength = float(max(0.0, 1.0 - (var_resid / max(1e-8, var_detrended))))
        resid_var = float(np.var(resid))

        return DecompositionResult(
            is_valid=True,
            method=method,
            period=period,
            trend=trend,
            seasonal=seasonal,
            resid=resid,
            observed=clean_s,
            seasonal_strength=round(seasonal_strength, 4),
            residual_variance=round(resid_var, 4),
            error_message=None,
        )
    except Exception as e:
        logger.warning(f"Decomposition failed: {e}")
        return DecompositionResult(
            is_valid=False,
            method=method,
            period=period,
            trend=pd.Series(dtype=float),
            seasonal=pd.Series(dtype=float),
            resid=pd.Series(dtype=float),
            observed=clean_s,
            seasonal_strength=0.0,
            residual_variance=0.0,
            error_message=str(e),
        )
