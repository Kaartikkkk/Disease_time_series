"""AutoTS (Automated Time Series) Forecasting Model.

Mathematical Formulation & Architecture:
    AutoTS executes an automated tournament across multiple time series forecasting
    algorithms (GLM, Exponential Smoothing, ARIMA, Theta, Seasonal Naive, Ensembles).
    It utilizes cross-validation loss search to optimize:
        1. Preprocessing transformations (detrending, differencing, box-cox, scaling)
        2. Model hyperparameter configuration
        3. Model ensembling (simple average, weighted, or distance-based)
        4. Non-negativity constraints (no_negatives=True for agricultural counts)
"""

import io
import logging
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HAS_AUTOTS = False
try:
    from autots import AutoTS
    HAS_AUTOTS = True
except ImportError:
    pass


class AutoTSForecastResult:
    """Wrapper encapsulating AutoTS fitted results and prediction intervals."""

    def __init__(
        self,
        model: Any,
        point_forecast: np.ndarray,
        lower_95: np.ndarray,
        upper_95: np.ndarray,
        lower_80: np.ndarray,
        upper_80: np.ndarray,
    ):
        self.model = model
        self.point_forecast = point_forecast
        self.lower_95 = lower_95
        self.upper_95 = upper_95
        self.lower_80 = lower_80
        self.upper_80 = upper_80
        self.best_model_name = getattr(model, "best_model_name", "AutoTS") if model else "Seasonal Naive"
        self.best_model_params = getattr(model, "best_model_params", {}) if model else {}

        # Concise human-readable parameter string for tables and cards
        if isinstance(self.best_model_params, dict) and self.best_model_params:
            params_str = ", ".join(f"{k}={v}" for k, v in list(self.best_model_params.items())[:3])
            self.param_summary = f"{self.best_model_name} ({params_str})"
        else:
            self.param_summary = f"{self.best_model_name} (Auto-Selected)"


def forecast_autots(
    train_df: pd.DataFrame,
    horizon: int,
    future_dates: Optional[pd.DatetimeIndex] = None,
    max_generations: int = 0,
    model_list: str = "superfast",
    target_col: str = "documented_occurrence_count",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, AutoTSForecastResult]:
    """Forecast using AutoTS Automated Machine Learning engine.

    Args:
        train_df: Dataframe with 'date' and target occurrences.
        horizon: Forecast horizon (months ahead).
        future_dates: DatetimeIndex for future prediction timestamps (optional).
        max_generations: Search generation iterations (default: 0 for instant evaluation of candidate templates).
        model_list: Model catalog ('superfast', 'fast', or curated list).
        target_col: Column name containing the occurrence counts.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, AutoTSForecastResult).
    """
    if not HAS_AUTOTS:
        raise ImportError("AutoTS is not installed or unavailable in this environment.")

    # Guard against insufficient history for AutoML
    y_series = train_df[target_col] if target_col in train_df.columns else train_df.iloc[:, -1]
    if len(train_df) < 12:
        from src.models.seasonal_naive import forecast_seasonal_naive
        logger.warning("History too short for AutoTS (<12 months); falling back to seasonal naive.")
        p, lb, ub = forecast_seasonal_naive(y_series, horizon=horizon)
        res = AutoTSForecastResult(None, p, lb, ub, lb, ub)
        res.best_model_name = "SeasonalNaive (Insufficient History)"
        res.param_summary = "SeasonalNaive Baseline"
        return p, lb, ub, res

    df_in = pd.DataFrame({
        "date": pd.to_datetime(train_df["date"]),
        target_col: y_series.values,
    })
    df_wide = df_in.set_index("date")[[target_col]]

    # Suppress verbose trial search output from stdout/stderr during tournament
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        model = AutoTS(
            forecast_length=horizon,
            frequency="MS",
            prediction_interval=0.95,
            max_generations=max_generations,
            num_validations=0,
            model_list=model_list,
            no_negatives=True,
            verbose=-1,
            random_seed=42,
        )
        model.fit(df_wide)
        prediction = model.predict()

        point_pred = np.maximum(0.0, prediction.forecast.values.flatten()[:horizon])
        lower_95 = np.maximum(0.0, prediction.lower_forecast.values.flatten()[:horizon])
        upper_95 = np.maximum(point_pred, prediction.upper_forecast.values.flatten()[:horizon])

        # Approximate 80% prediction intervals using 95% dispersion (z_80 / z_95 = 1.28 / 1.96)
        sigma_est = (upper_95 - lower_95) / (2.0 * 1.95996)
        lower_80 = np.maximum(0.0, point_pred - 1.28155 * sigma_est)
        upper_80 = np.maximum(point_pred, point_pred + 1.28155 * sigma_est)

        res_obj = AutoTSForecastResult(
            model=model,
            point_forecast=point_pred,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_80=lower_80,
            upper_80=upper_80,
        )

        return point_pred, lower_95, upper_95, res_obj

    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.warning("AutoTS execution note (%s); using seasonal naive fallback.", e)
        from src.models.seasonal_naive import forecast_seasonal_naive
        p, lb, ub = forecast_seasonal_naive(y_series, horizon=horizon)
        res_obj = AutoTSForecastResult(None, p, lb, ub, lb, ub)
        res_obj.best_model_name = "SeasonalNaive (AutoTS Fallback)"
        res_obj.param_summary = "Seasonal Naive (AutoTS Fallback)"
        return p, lb, ub, res_obj

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
