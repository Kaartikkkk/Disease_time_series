"""Stationarity, Transformation, and Diagnostic Testing Suite for Time Series.

Implements rigorous econometrics tests:
1. Augmented Dickey-Fuller (ADF) Test
   - H0: Series possesses a unit root (Non-stationary)
   - H1: Series is stationary
2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
   - H0: Series is level/trend stationary
   - H1: Series possesses a unit root (Non-stationary)
3. Joint Decision Matrix (combining ADF and KPSS outcomes)
4. First-order Differencing (Δy_t) and Seasonal Differencing (Δ_s y_t)
5. Variance Stabilizing Transformations (log1p, sqrt, Box-Cox)
6. Ljung-Box Portmanteau Test for Residual Autocorrelation
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


def perform_adf_test(
    series: pd.Series,
    regression: str = "c",
    maxlag: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute Augmented Dickey-Fuller unit-root test with hypothesis explanations."""
    clean_s = pd.Series(series).dropna()
    if len(clean_s) < 10:
        return {"error": "Insufficient observations for ADF test (minimum 10 required)."}

    try:
        stat, pval, usedlag, nobs, crit_vals, icbest = adfuller(
            clean_s, regression=regression, maxlag=maxlag, autolag="AIC"
        )
        is_stationary = bool(pval < 0.05)

        return {
            "test_name": "Augmented Dickey-Fuller (ADF)",
            "null_hypothesis": "H0: The series possesses a unit root (Non-Stationary).",
            "alternative_hypothesis": "H1: The series has no unit root (Stationary).",
            "test_statistic": round(float(stat), 4),
            "p_value": round(float(pval), 5),
            "is_stationary": is_stationary,
            "used_lag": int(usedlag),
            "num_observations": int(nobs),
            "critical_values": {k: round(float(v), 4) for k, v in crit_vals.items()},
            "interpretation": (
                f"Reject H0 at 5% significance level (p={pval:.4f} < 0.05). "
                "Evidence favors stationarity."
                if is_stationary
                else f"Fail to reject H0 at 5% level (p={pval:.4f} >= 0.05). "
                "Series exhibits non-stationary unit root behavior."
            ),
        }
    except Exception as exc:
        logger.error("ADF test failed: %s", exc)
        return {"error": str(exc)}


def perform_kpss_test(
    series: pd.Series,
    regression: str = "c",
) -> Dict[str, Any]:
    """Execute KPSS stationarity test with hypothesis explanations."""
    clean_s = pd.Series(series).dropna()
    if len(clean_s) < 10:
        return {"error": "Insufficient observations for KPSS test (minimum 10 required)."}

    try:
        # Suppress benign statsmodels InterpolationWarning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pval, lags, crit_vals = kpss(clean_s, regression=regression, nlags="auto")

        is_stationary = bool(pval >= 0.05)

        return {
            "test_name": "Kwiatkowski-Phillips-Schmidt-Shin (KPSS)",
            "null_hypothesis": "H0: The series is stationary around a deterministic constant/trend.",
            "alternative_hypothesis": "H1: The series possesses a unit root (Non-Stationary).",
            "test_statistic": round(float(stat), 4),
            "p_value": round(float(pval), 5),
            "is_stationary": is_stationary,
            "lags_used": int(lags),
            "critical_values": {k: round(float(v), 4) for k, v in crit_vals.items()},
            "interpretation": (
                f"Fail to reject H0 at 5% significance level (p={pval:.4f} >= 0.05). "
                "Evidence is consistent with stationarity."
                if is_stationary
                else f"Reject H0 at 5% level (p={pval:.4f} < 0.05). "
                "Evidence indicates non-stationarity."
            ),
        }
    except Exception as exc:
        logger.error("KPSS test failed: %s", exc)
        return {"error": str(exc)}


def synthesize_stationarity_decision(
    adf_result: Dict[str, Any],
    kpss_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine ADF and KPSS outcomes into formal econometric classification."""
    if "error" in adf_result or "error" in kpss_result:
        return {
            "classification": "Indeterminate",
            "action": "Ensure sufficient continuous observations are available.",
            "explanation": "One or both statistical tests encountered numerical failure.",
        }

    adf_rejects = adf_result.get("is_stationary", False)  # p < 0.05 rejects unit root
    kpss_rejects = not kpss_result.get("is_stationary", True)  # p < 0.05 rejects stationarity

    if adf_rejects and not kpss_rejects:
        # Case 1: Both tests indicate stationarity
        classification = "Strictly Stationary"
        action = "Fit ARMA or SARMA models without differencing (d=0, D=0)."
        explanation = (
            "Both ADF and KPSS tests concur: ADF rejects unit root, while KPSS fails "
            "to reject stationarity. The series can be modeled in original levels."
        )
    elif not adf_rejects and kpss_rejects:
        # Case 2: Both tests indicate non-stationarity
        classification = "Difference Non-Stationary (Unit Root)"
        action = "Apply first-order differencing (d=1) or seasonal differencing (D=1)."
        explanation = (
            "Both tests confirm non-stationarity: ADF fails to reject unit root, and KPSS "
            "rejects stationarity. Differencing is statistically necessary."
        )
    elif not adf_rejects and not kpss_rejects:
        # Case 3: Inconclusive
        classification = "Inconclusive / Weak Evidence"
        action = "Inspect ACF/PACF decay; check for low sample size or near-unit root."
        explanation = (
            "ADF finds insufficient power to reject unit root, but KPSS also cannot "
            "reject stationarity. Common in shorter or sparse count series."
        )
    else:
        # Case 4: Trend stationary or structural break
        classification = "Trend Stationary / Structural Break"
        action = "Include deterministic linear trend or test for regime shifts."
        explanation = (
            "ADF rejects unit root, but KPSS also rejects level stationarity. "
            "Suggests stationarity around a deterministic trend or regime shifts."
        )

    return {
        "classification": classification,
        "action": action,
        "explanation": explanation,
        "adf_p_value": adf_result.get("p_value"),
        "kpss_p_value": kpss_result.get("p_value"),
    }


def analyze_differencing(series: pd.Series, seasonal_period: int = 12) -> Dict[str, Any]:
    """Evaluate variance and stationarity under level, first difference, and seasonal difference."""
    s = pd.Series(series).dropna()
    s_diff1 = s.diff().dropna()
    s_diff_seasonal = s.diff(seasonal_period).dropna()

    results = {
        "original": {
            "variance": round(float(s.var()), 4),
            "adf": perform_adf_test(s),
        },
        "first_difference": {
            "variance": round(float(s_diff1.var()), 4) if len(s_diff1) > 5 else None,
            "adf": perform_adf_test(s_diff1) if len(s_diff1) > 10 else None,
        },
        "seasonal_difference": {
            "variance": round(float(s_diff_seasonal.var()), 4) if len(s_diff_seasonal) > 5 else None,
            "adf": perform_adf_test(s_diff_seasonal) if len(s_diff_seasonal) > 10 else None,
        },
    }

    # Variance comparison recommendation:
    # Over-differencing increases variance compared to preceding differencing step
    orig_var = results["original"]["variance"]
    diff1_var = results["first_difference"]["variance"] or orig_var
    diff_s_var = results["seasonal_difference"]["variance"] or orig_var

    if diff1_var < orig_var:
        rec_diff = "First-order differencing (d=1) successfully reduces variance."
    else:
        rec_diff = "First-order differencing increases variance; caution against over-differencing."

    results["recommendation"] = rec_diff
    return results


def apply_transformations(series: pd.Series) -> Dict[str, pd.Series]:
    """Calculate standard variance-stabilizing transformations."""
    s = pd.Series(series)
    return {
        "original": s,
        "log1p": np.log1p(np.maximum(0, s)),
        "sqrt": np.sqrt(np.maximum(0, s)),
    }


def perform_ljung_box_test(residuals: pd.Series, lags: List[int] = [6, 12]) -> Dict[str, Any]:
    """Execute Ljung-Box test for residual autocorrelation."""
    clean_res = pd.Series(residuals).dropna()
    if len(clean_res) < max(lags) + 2:
        return {"error": "Residual series is too short for specified Ljung-Box lags."}

    try:
        lb_df = acorr_ljungbox(clean_res, lags=lags, return_df=True)
        results = []
        for lag in lags:
            if lag in lb_df.index:
                stat = float(lb_df.loc[lag, "lb_stat"])
                pval = float(lb_df.loc[lag, "lb_pvalue"])
                results.append({
                    "lag": lag,
                    "statistic": round(stat, 4),
                    "p_value": round(pval, 5),
                    "is_white_noise": bool(pval >= 0.05),
                })

        all_white_noise = all(r["is_white_noise"] for r in results)
        interpretation = (
            "Residuals are uncorrelated (white noise, p >= 0.05 across all checked lags). "
            "Model has successfully captured dynamic autocorrelation."
            if all_white_noise
            else "Residuals exhibit significant remaining autocorrelation (p < 0.05). "
            "Consider adding AR/MA terms or seasonal lags."
        )

        return {
            "test_name": "Ljung-Box Test for Autocorrelation",
            "null_hypothesis": "H0: Residuals are independently distributed (no serial correlation).",
            "alternative_hypothesis": "H1: Residuals exhibit serial autocorrelation.",
            "results_by_lag": results,
            "interpretation": interpretation,
        }
    except Exception as exc:
        logger.error("Ljung-Box test failed: %s", exc)
        return {"error": str(exc)}
