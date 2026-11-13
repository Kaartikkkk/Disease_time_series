"""Holt-Winters Triple Exponential Smoothing Model with Zero-Safe Optimization.

Mathematical Formulation (Additive Trend and Additive Seasonality):
    Level:    l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
    Trend:    b_t = beta  * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
    Seasonal: s_t = gamma * (y_t - l_t)     + (1 - gamma) * s_{t-m}
    Forecast: y_hat_{t+h} = l_t + h * b_t + s_{t+h-m}

Mathematical Constraint for Agricultural Zero Counts:
    When zero-count months are present (common in off-season crop cycles), multiplicative
    trend or seasonality is mathematically undefined (division by zero / non-positive base).
    The algorithm safely selects additive formulation and tests damped vs undamped trend.
"""

from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing


def forecast_holt_winters(
    train_series: pd.Series,
    horizon: int,
    seasonal_periods: int = 12,
    trend: Optional[str] = "add",
    seasonal: Optional[str] = "add",
    auto_optimize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit Holt-Winters Exponential Smoothing model and generate predictions.

    Args:
        train_series: Historical series of disease reports.
        horizon: Forecast horizon (number of months).
        seasonal_periods: Periods in complete seasonal cycle (12 for monthly data).
        trend: Trend type ("add", None).
        seasonal: Seasonal type ("add", None).
        auto_optimize: Automatically select between damped/undamped and additive configurations.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model_result).
    """
    y = train_series.astype(float).values
    n = len(y)

    # Disable seasonality if series is shorter than 2 complete seasonal cycles
    use_seasonal = seasonal if n >= 2 * seasonal_periods else None
    has_zeros = bool(np.any(y <= 0))

    best_res = None
    best_aic = float("inf")
    best_config = {"trend": "add", "damped": False, "seasonal": use_seasonal}

    if auto_optimize:
        # Candidate configurations:
        # 1. Additive trend + Additive season (undamped)
        # 2. Additive trend + Additive season (damped)
        # 3. No trend + Additive season
        # 4. Holt's Linear (trend="add", seasonal=None, damped) - protects sparse series from seasonal overfitting
        # 5. Simple Exponential Smoothing (trend=None, seasonal=None)
        configs = [
            {"trend": "add", "damped": False, "seasonal": use_seasonal},
            {"trend": "add", "damped": True, "seasonal": use_seasonal},
            {"trend": None, "damped": False, "seasonal": use_seasonal},
            {"trend": "add", "damped": True, "seasonal": None},
            {"trend": "add", "damped": False, "seasonal": None},
            {"trend": None, "damped": False, "seasonal": None},
        ]

        # If data is strictly positive, also test multiplicative seasonal
        if not has_zeros and use_seasonal is not None:
            configs.append({"trend": "add", "damped": False, "seasonal": "mul"})

        for cfg in configs:
            try:
                model = ExponentialSmoothing(
                    y,
                    trend=cfg["trend"],
                    damped_trend=cfg["damped"],
                    seasonal=cfg["seasonal"],
                    seasonal_periods=seasonal_periods if cfg["seasonal"] else None,
                    initialization_method="estimated",
                )
                res = model.fit(optimized=True)
                aic_val = float(res.aic) if hasattr(res, "aic") and res.aic is not None and not np.isnan(res.aic) else float(np.sum(res.resid ** 2))
                if aic_val < best_aic:
                    best_aic = aic_val
                    best_res = res
                    best_config = cfg
            except Exception:
                continue

    # Fallback to standard additive if optimization failed or not requested
    if best_res is None:
        model = ExponentialSmoothing(
            y,
            trend=trend,
            seasonal=use_seasonal,
            seasonal_periods=seasonal_periods if use_seasonal else None,
            initialization_method="estimated",
        )
        best_res = model.fit(optimized=True)

    point_pred = np.maximum(0.0, best_res.forecast(horizon))

    # Empirical interval estimate from training residuals
    residuals = best_res.resid
    se = float(np.std(residuals)) if len(residuals) > 0 else 1.0
    lower_95 = np.maximum(0.0, point_pred - 1.96 * se)
    upper_95 = point_pred + 1.96 * se

    # Extract smoothing params
    alpha = getattr(best_res.params, "get", lambda k, d: d)("smoothing_level", 0.0)
    beta = getattr(best_res.params, "get", lambda k, d: d)("smoothing_trend", 0.0)
    gamma = getattr(best_res.params, "get", lambda k, d: d)("smoothing_seasonal", 0.0)

    setattr(best_res, "param_summary", (
        f"Trend={best_config['trend']}{'(damped)' if best_config.get('damped') else ''}, "
        f"Seasonal={best_config['seasonal']} [m={seasonal_periods}], "
        f"alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}"
    ))

    return point_pred, lower_95, upper_95, best_res
