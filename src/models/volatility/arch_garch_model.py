"""ARCH and GARCH Volatility Models for Disease Time Series Forecasting.

Implements univariate autoregressive conditional heteroskedasticity (ARCH)
and generalized autoregressive conditional heteroskedasticity (GARCH) models
to quantify time-varying uncertainty, outbreak volatility clustering, and
heteroskedastic variance dynamics in monthly plant disease time series.

Supported Model Specifications:
    - ARCH(1), ARCH(2), ARCH(5)
    - GARCH(1,1), GARCH(1,2), GARCH(2,1), GARCH(2,2)
Supported Error Distributions:
    - Normal (Gaussian)
    - Student's t (Heavy-tailed)

Statistically Appropriate Methodologies:
    - Approach A: Residual Volatility Modelling
      Fits an ARIMA/SARIMA mean model to occurrence counts, extracts residuals,
      tests for ARCH effects (Engle's ARCH-LM test), fits ARCH/GARCH to residuals,
      and constructs dynamic prediction intervals: y_hat +/- z * sigma_hat.
    - Approach B: Changes in Disease Occurrence (Differencing)
      Models changes Delta y_t = y_t - y_{t-1} directly with constant/AR mean
      and ARCH/GARCH conditional variance.

Note:
    ARCH and GARCH estimate how disease-series uncertainty changes over time.
    They do not directly predict disease occurrence counts.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from arch import arch_model
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)

# Valid ARCH/GARCH specification tuples: (p, q) where q=0 for ARCH
VOLATILITY_SPECS = {
    "arch_1": (1, 0),
    "arch_2": (2, 0),
    "arch_5": (5, 0),
    "garch_1_1": (1, 1),
    "garch_1_2": (1, 2),
    "garch_2_1": (2, 1),
    "garch_2_2": (2, 2),
}

DistributionType = Literal["Normal", "StudentsT", "normal", "t"]
ApproachType = Literal["approach_a", "approach_b", "residuals", "diff"]


@dataclass
class VolatilityModelResult:
    """Standardized output container for fitted ARCH/GARCH models."""

    spec_name: str
    p: int
    q: int
    dist: str
    approach: str
    converged: bool
    convergence_flag: int
    log_likelihood: float
    aic: float
    bic: float
    params: Dict[str, float]
    pvalues: Dict[str, float]
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    arch_lm_stat: float
    arch_lm_pvalue: float
    arch_lm_f_stat: float
    arch_lm_f_pvalue: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    ljung_box_sq_stat: float
    ljung_box_sq_pvalue: float
    fitted_arch_result: Any
    mean_model_info: Dict[str, Any] = field(default_factory=dict)
    summary_text: str = ""


@dataclass
class VolatilityForecastResult:
    """Standardized container for ARCH/GARCH out-of-sample forecasts."""

    spec_name: str
    horizon: int
    variance_forecast: np.ndarray
    volatility_forecast: np.ndarray
    mean_forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)


def perform_arch_lm_test(
    series: Union[np.ndarray, pd.Series],
    lags: int = 5,
) -> Dict[str, float]:
    """Perform Engle's Lagrange Multiplier (LM) test for autoregressive conditional heteroskedasticity.

    Null Hypothesis (H0): No ARCH effects present (homoskedastic variance).
    Alternative (H1): ARCH effects present (conditional heteroskedasticity / volatility clustering).

    Returns:
        Dict containing lm_stat, lm_pvalue, f_stat, f_pvalue, and has_arch_effects boolean.
    """
    arr = np.asarray(series, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < lags * 2 + 2:
        lags = max(1, n // 5)

    if n < 5 or np.all(arr == arr[0]):
        return {
            "lm_stat": 0.0,
            "lm_pvalue": 1.0,
            "f_stat": 0.0,
            "f_pvalue": 1.0,
            "has_arch_effects": False,
        }

    try:
        lm_stat, lm_pval, f_stat, f_pval = het_arch(arr, nlags=lags)
        return {
            "lm_stat": float(lm_stat),
            "lm_pvalue": float(lm_pval),
            "f_stat": float(f_stat),
            "f_pvalue": float(f_pval),
            "has_arch_effects": bool(lm_pval < 0.05),
        }
    except Exception as exc:
        logger.warning(f"Engle ARCH-LM test encountered exception: {exc}")
        return {
            "lm_stat": np.nan,
            "lm_pvalue": np.nan,
            "f_stat": np.nan,
            "f_pvalue": np.nan,
            "has_arch_effects": False,
        }


def compute_volatility_metrics(
    actuals: np.ndarray,
    predicted_means: np.ndarray,
    predicted_volatilities: np.ndarray,
) -> Dict[str, float]:
    """Compute realized volatility proxy and evaluate volatility forecasting performance.

    Realized Volatility Proxy:
        Realized Absolute Error: |e_t| = |y_t - y_hat_t|
        Realized Variance Proxy: e_t^2 = (y_t - y_hat_t)^2

    Metrics:
        - RMSVE: Root Mean Squared Volatility Error vs |e_t|
        - MAVE: Mean Absolute Volatility Error vs |e_t|
        - MSE_var: Mean Squared Error on variance vs e_t^2
        - QLIKE: Quasi-Likelihood loss (strictly consistent for variance estimation)
    """
    actuals = np.asarray(actuals, dtype=float).reshape(-1)
    means = np.asarray(predicted_means, dtype=float).reshape(-1)
    vols = np.asarray(predicted_volatilities, dtype=float).reshape(-1)

    errors = actuals - means
    realized_vol = np.abs(errors)
    realized_var = errors ** 2
    predicted_var = np.maximum(1e-6, vols ** 2)

    # RMSVE
    rmsve = float(np.sqrt(np.mean((vols - realized_vol) ** 2)))
    # MAVE
    mave = float(np.mean(np.abs(vols - realized_vol)))
    # MSE on variance
    mse_var = float(np.mean((predicted_var - realized_var) ** 2))

    # QLIKE loss: L(var, h) = h/var - log(h/var) - 1
    # Avoid log(0) with small epsilon
    ratio = np.maximum(1e-8, realized_var) / predicted_var
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))

    return {
        "rmsve": rmsve,
        "mave": mave,
        "mse_var": mse_var,
        "qlike": qlike,
    }


class ARCHGARCHModel:
    """Unified ARCH and GARCH Estimator and Uncertainty Forecaster."""

    def __init__(
        self,
        p: int = 1,
        q: int = 0,
        dist: str = "StudentsT",
        approach: str = "approach_a",
        mean: str = "Zero",
        arima_order: Tuple[int, int, int] = (1, 0, 1),
    ):
        """Initialize ARCH/GARCH model.

        Args:
            p: Order of ARCH lag terms (autoregressive squared innovations).
            q: Order of GARCH lag terms (conditional variance lags). q=0 yields ARCH(p).
            dist: 'Normal' or 'StudentsT'.
            approach: 'approach_a' (residual volatility on ARIMA) or
                      'approach_b' (differenced series Delta y_t).
            mean: 'Zero' or 'Constant' (when approach_a, defaults to 'Zero' for residuals).
            arima_order: (p, d, q) for the mean model in Approach A.
        """
        if p < 1:
            raise ValueError(f"ARCH order p must be >= 1, got {p}.")
        if q < 0:
            raise ValueError(f"GARCH order q must be >= 0, got {q}.")

        norm_dist = "StudentsT" if dist.lower() in ("student", "studentst", "t") else "Normal"
        self.p = p
        self.q = q
        self.dist = norm_dist
        self.approach = approach.lower()
        self.mean = mean
        self.arima_order = arima_order

        # Internal state
        self.fitted_arch_res: Optional[Any] = None
        self.fitted_mean_res: Optional[Any] = None
        self.model_result: Optional[VolatilityModelResult] = None
        self.last_train_series: Optional[np.ndarray] = None
        self.last_train_target: Optional[np.ndarray] = None

    def fit(self, y: Union[np.ndarray, pd.Series]) -> VolatilityModelResult:
        """Fit ARCH/GARCH model according to the chosen approach.

        Args:
            y: 1D array-like of historical occurrence counts (documented_occurrence_count).

        Returns:
            VolatilityModelResult containing parameters, diagnostics, and fitted volatility.
        """
        arr = np.asarray(y, dtype=float).reshape(-1)
        if len(arr) < max(self.p + self.q + 4, 15):
            raise ValueError(
                f"Time series too short for ARCH/GARCH estimation (length={len(arr)}, needed >={max(self.p + self.q + 4, 15)})."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input series contains NaN or infinite values.")

        self.last_train_target = arr.copy()
        vol_type = "GARCH" if self.q > 0 else "ARCH"
        mean_info: Dict[str, Any] = {}

        if self.approach in ("approach_a", "residuals"):
            # Approach A: Residual Volatility Modelling
            # 1. Fit ARIMA mean model
            try:
                mean_model = ARIMA(arr, order=self.arima_order)
                self.fitted_mean_res = mean_model.fit()
                residuals = self.fitted_mean_res.resid
                mean_info = {
                    "type": "ARIMA",
                    "order": self.arima_order,
                    "aic": float(self.fitted_mean_res.aic),
                    "bic": float(self.fitted_mean_res.bic),
                }
            except Exception as exc:
                logger.warning(f"ARIMA mean model fitting failed ({exc}), falling back to demeaned series.")
                mean_val = float(np.mean(arr))
                residuals = arr - mean_val
                self.fitted_mean_res = None
                mean_info = {"type": "Constant", "mean": mean_val}

            modeling_series = residuals
            arch_mean = "Zero"
        else:
            # Approach B: Changes in disease occurrence (Differencing)
            diff_y = np.diff(arr)
            modeling_series = diff_y
            arch_mean = "Constant"
            mean_info = {
                "type": "FirstDifference",
                "original_len": len(arr),
                "diff_len": len(diff_y),
            }

        self.last_train_series = modeling_series

        # 2. Fit ARCH/GARCH with arch library
        am = arch_model(
            modeling_series,
            mean=arch_mean,
            vol=vol_type,
            p=self.p,
            q=self.q if self.q > 0 else 0,
            dist=self.dist,
            rescale=False,
        )

        try:
            res = am.fit(disp="off", show_warning=False)
            self.fitted_arch_res = res
        except Exception as exc:
            # If estimation fails without rescaling, retry with rescale=True or default
            logger.warning(f"ARCH/GARCH fit without rescaling failed: {exc}. Retrying with rescale=True.")
            am_rescaled = arch_model(
                modeling_series,
                mean=arch_mean,
                vol=vol_type,
                p=self.p,
                q=self.q if self.q > 0 else 0,
                dist=self.dist,
                rescale=True,
            )
            res = am_rescaled.fit(disp="off", show_warning=False)
            self.fitted_arch_res = res

        # Extract parameters and statistical diagnostics
        converged = bool(res.convergence_flag == 0)
        params = {str(k): float(v) for k, v in res.params.items()}
        pvalues = {str(k): float(v) for k, v in res.pvalues.items()}
        cond_vol = np.asarray(res.conditional_volatility, dtype=float)
        std_resid = np.asarray(res.std_resid, dtype=float)
        std_resid_clean = std_resid[np.isfinite(std_resid)]

        # Diagnostic checks on standardized residuals
        arch_lm_test = perform_arch_lm_test(std_resid_clean, lags=min(5, max(1, len(std_resid_clean) // 5)))

        # Ljung-Box test on std_resid (mean autocorrelation)
        try:
            lb_res = acorr_ljungbox(std_resid_clean, lags=[min(5, len(std_resid_clean) // 5)], return_df=True)
            lb_stat = float(lb_res["lb_stat"].iloc[0])
            lb_pval = float(lb_res["lb_pvalue"].iloc[0])
        except Exception:
            lb_stat, lb_pval = np.nan, np.nan

        # Ljung-Box test on std_resid^2 (residual variance autocorrelation)
        try:
            lb_sq_res = acorr_ljungbox(std_resid_clean ** 2, lags=[min(5, len(std_resid_clean) // 5)], return_df=True)
            lb_sq_stat = float(lb_sq_res["lb_stat"].iloc[0])
            lb_sq_pval = float(lb_sq_res["lb_pvalue"].iloc[0])
        except Exception:
            lb_sq_stat, lb_sq_pval = np.nan, np.nan

        spec_name = f"ARCH({self.p})" if self.q == 0 else f"GARCH({self.p},{self.q})"
        self.model_result = VolatilityModelResult(
            spec_name=spec_name,
            p=self.p,
            q=self.q,
            dist=self.dist,
            approach=self.approach,
            converged=converged,
            convergence_flag=int(res.convergence_flag),
            log_likelihood=float(res.loglikelihood),
            aic=float(res.aic),
            bic=float(res.bic),
            params=params,
            pvalues=pvalues,
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            arch_lm_stat=arch_lm_test["lm_stat"],
            arch_lm_pvalue=arch_lm_test["lm_pvalue"],
            arch_lm_f_stat=arch_lm_test["f_stat"],
            arch_lm_f_pvalue=arch_lm_test["f_pvalue"],
            ljung_box_stat=lb_stat,
            ljung_box_pvalue=lb_pval,
            ljung_box_sq_stat=lb_sq_stat,
            ljung_box_sq_pvalue=lb_sq_pval,
            fitted_arch_result=res,
            mean_model_info=mean_info,
            summary_text=str(res.summary()),
        )
        return self.model_result

    def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95,
    ) -> VolatilityForecastResult:
        """Generate out-of-sample conditional variance, volatility, and uncertainty bands.

        Args:
            horizon: Number of forward periods to forecast (h >= 1).
            confidence_level: Interval coverage probability (default: 0.95).

        Returns:
            VolatilityForecastResult containing variance, volatility, mean forecasts,
            and confidence intervals.
        """
        if self.fitted_arch_res is None or self.model_result is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if horizon <= 0:
            raise ValueError(f"Forecast horizon must be positive, got {horizon}.")

        # 1. Forecast conditional variance with arch
        fcast = self.fitted_arch_res.forecast(horizon=horizon, reindex=False)
        var_fcast = fcast.variance.iloc[-1].values.astype(float)
        # Ensure finite, positive variance
        var_fcast = np.maximum(1e-6, np.nan_to_num(var_fcast, nan=1.0, posinf=100.0, neginf=1e-6))
        vol_fcast = np.sqrt(var_fcast)

        # 2. Compute Mean Forecast
        if self.approach in ("approach_a", "residuals"):
            if self.fitted_mean_res is not None:
                try:
                    mean_fcast = np.asarray(self.fitted_mean_res.forecast(steps=horizon), dtype=float)
                except Exception:
                    mean_fcast = np.full(horizon, np.mean(self.last_train_target))
            else:
                mean_fcast = np.full(horizon, np.mean(self.last_train_target))
        else:
            # Approach B: Differenced series
            # Drift / constant forecast for Delta y_t
            mu = self.model_result.params.get("mu", 0.0)
            last_val = self.last_train_target[-1] if self.last_train_target is not None else 0.0
            # Cumulative sums of changes
            mean_fcast = np.maximum(0.0, last_val + np.cumsum(np.full(horizon, mu)))

        # 3. Dynamic Prediction Intervals based on conditional volatility and distribution
        alpha = 1.0 - confidence_level
        if self.dist == "StudentsT" and "nu" in self.model_result.params:
            nu = max(2.1, self.model_result.params["nu"])
            # Student-t critical value adjusted for variance scaling
            crit_val = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu) * np.sqrt((nu - 2.0) / nu))
        else:
            crit_val = float(stats.norm.ppf(1.0 - alpha / 2.0))

        margin = crit_val * vol_fcast
        lower_bound = np.maximum(0.0, mean_fcast - margin)
        upper_bound = np.maximum(0.0, mean_fcast + margin)

        spec_name = f"ARCH({self.p})" if self.q == 0 else f"GARCH({self.p},{self.q})"
        return VolatilityForecastResult(
            spec_name=spec_name,
            horizon=horizon,
            variance_forecast=var_fcast,
            volatility_forecast=vol_fcast,
            mean_forecast=mean_fcast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            metadata={
                "p": self.p,
                "q": self.q,
                "dist": self.dist,
                "approach": self.approach,
                "crit_val": crit_val,
                "converged": self.model_result.converged,
                "aic": self.model_result.aic,
                "bic": self.model_result.bic,
            },
        )


def fit_and_compare_all_specifications(
    y: Union[np.ndarray, pd.Series],
    dist: str = "StudentsT",
    approach: str = "approach_a",
) -> Dict[str, VolatilityModelResult]:
    """Fit all 7 required ARCH/GARCH specifications and return their diagnostic results.

    Specifications:
        ARCH(1), ARCH(2), ARCH(5)
        GARCH(1,1), GARCH(1,2), GARCH(2,1), GARCH(2,2)
    """
    results: Dict[str, VolatilityModelResult] = {}
    for spec_name, (p, q) in VOLATILITY_SPECS.items():
        try:
            model = ARCHGARCHModel(p=p, q=q, dist=dist, approach=approach)
            res = model.fit(y)
            results[spec_name] = res
        except Exception as exc:
            logger.warning(f"Specification {spec_name} failed: {exc}")
    return results


def run_volatility_walk_forward_evaluation(
    series: Union[np.ndarray, pd.Series],
    spec_name: str = "garch_1_1",
    p: int = 1,
    q: int = 1,
    dist: str = "StudentsT",
    approach: str = "approach_a",
    horizon: int = 1,
    min_train_len: int = 60,
    step: int = 1,
) -> Dict[str, Any]:
    """Execute rigorous chronological walk-forward out-of-sample evaluation.

    Guarantees:
        - Expanding chronological window.
        - Zero future-data leakage (fit on past only).
        - Independent out-of-sample volatility and point evaluation.
    """
    arr = np.asarray(series, dtype=float).reshape(-1)
    n = len(arr)
    if n <= min_train_len + horizon:
        raise ValueError(f"Series length {n} insufficient for walk-forward validation with min_train_len={min_train_len}.")

    pred_means: List[float] = []
    pred_vols: List[float] = []
    actual_vals: List[float] = []
    eval_dates_idx: List[int] = []

    for t in range(min_train_len, n - horizon + 1, step):
        train_y = arr[:t]
        test_y = arr[t : t + horizon]

        try:
            model = ARCHGARCHModel(p=p, q=q, dist=dist, approach=approach)
            model.fit(train_y)
            fcast = model.forecast(horizon=horizon)
            pred_means.extend(fcast.mean_forecast.tolist())
            pred_vols.extend(fcast.volatility_forecast.tolist())
            actual_vals.extend(test_y.tolist())
            eval_dates_idx.extend(list(range(t, t + horizon)))
        except Exception as exc:
            logger.warning(f"Walk-forward origin t={t} failed: {exc}")
            continue

    actuals = np.array(actual_vals)
    means = np.array(pred_means)
    vols = np.array(pred_vols)

    if len(actuals) == 0:
        raise RuntimeError("Walk-forward evaluation produced 0 valid forecasts.")

    # 1. Volatility evaluation metrics
    vol_metrics = compute_volatility_metrics(actuals, means, vols)

    # 2. Point forecast metrics
    errors = actuals - means
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    sum_actual = float(np.sum(np.abs(actuals)))
    wape = float(np.sum(np.abs(errors)) / sum_actual) if sum_actual > 0 else np.nan
    denom = np.abs(actuals) + np.abs(means)
    smape = float(np.mean(2.0 * np.abs(errors) / np.maximum(1e-6, denom)) * 100.0)

    # Naive benchmark for MASE
    naive_err = np.abs(np.diff(arr[:min_train_len]))
    mean_naive = float(np.mean(naive_err)) if len(naive_err) > 0 and np.mean(naive_err) > 0 else 1.0
    mase = float(mae / mean_naive)

    return {
        "spec_name": spec_name,
        "n_predictions": len(actuals),
        "point_metrics": {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "mase": mase,
            "wape": wape,
            "smape": smape,
        },
        "volatility_metrics": vol_metrics,
        "actuals": actuals,
        "predictions": means,
        "volatilities": vols,
    }
