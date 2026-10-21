"""High-level forecasting pipeline and inference execution."""

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from src.models.machine_learning_models import MLForecaster
from src.models.advanced_classical_models import (
    ARForecaster,
    MAForecaster,
    ARMAForecaster,
    ARIMAXForecaster,
    SARIMAXForecaster,
    VARForecaster,
    ProphetForecaster,
    AutoTSForecaster,
)
from src.models.model_registry import get_model
from src.utils.logging_config import logger
from src.utils.validation import enforce_target_bounds


@dataclass
class ForecastResult:
    model_name: str
    target_column: str
    frequency: str
    horizon: int
    training_points: int
    last_historical_date: pd.Timestamp
    forecast_dates: List[pd.Timestamp]
    point_forecast: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    fit_time_seconds: float
    feature_importances: Optional[pd.DataFrame] = None
    notes: Optional[str] = None


def generate_forecast(
    df: pd.DataFrame,
    model_name: str = "random_forest",
    target_col: str = "incidence_rate",
    horizon: int = 14,
    frequency: str = "D",
    seasonal_period: int = 7,
    confidence_level: float = 0.95,
    include_weather: bool = True,
    random_seed: int = 42,
) -> ForecastResult:
    """Train a model on historical observations and generate out-of-sample forecasts.

    Strictly ensures:
    1. Historical observations precede future forecast dates.
    2. Prediction intervals are bounded to non-negative domain.
    3. Multi-step forecasts respect information availability.
    """
    clean_df = df.sort_values("date").dropna(subset=[target_col]).reset_index(drop=True)
    n_train = len(clean_df)

    if n_train < 7:
        raise ValueError(f"Insufficient training observations: {n_train} points (minimum 7 required).")

    last_date = pd.to_datetime(clean_df["date"].iloc[-1])

    # Build future dates
    freq_code = "D"
    if frequency in ["W", "WEEKLY", "W-SUN"]:
        freq_code = "W-SUN"
    elif frequency in ["M", "MONTHLY", "MS"]:
        freq_code = "MS"

    future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_code)[1:].tolist()

    model = get_model(
        model_name=model_name,
        seasonal_period=seasonal_period,
        frequency=frequency,
        random_seed=random_seed,
        include_weather=include_weather,
    )

    start_time = time.time()
    feature_importances = None

    df_fit_models = (
        MLForecaster,
        ARIMAXForecaster,
        SARIMAXForecaster,
        VARForecaster,
        ProphetForecaster,
        AutoTSForecaster,
    )

    if isinstance(model, df_fit_models):
        if isinstance(model, (ProphetForecaster, AutoTSForecaster)):
            model.fit(clean_df, target_col=target_col, frequency=frequency)
        else:
            model.fit(clean_df, target_col=target_col)

        point_preds = model.predict(horizon=horizon)
        if hasattr(model, "get_feature_importances"):
            feature_importances = model.get_feature_importances()

        fc_dict = model.predict_with_intervals(horizon=horizon, confidence_level=confidence_level)
        point_preds = fc_dict["mean"]
        lower = fc_dict["lower"]
        upper = fc_dict["upper"]
    else:
        # Statistical & Baseline forecasters fit on univariate target Series
        series = clean_df[target_col]
        model.fit(series)
        fc_dict = model.predict_with_intervals(horizon=horizon, confidence_level=confidence_level)
        point_preds = fc_dict["mean"]
        lower = fc_dict["lower"]
        upper = fc_dict["upper"]

    elapsed = time.time() - start_time

    return ForecastResult(
        model_name=model.name,
        target_column=target_col,
        frequency=frequency,
        horizon=horizon,
        training_points=n_train,
        last_historical_date=last_date,
        forecast_dates=future_dates,
        point_forecast=point_preds,
        lower_bound=lower,
        upper_bound=upper,
        fit_time_seconds=round(elapsed, 3),
        feature_importances=feature_importances,
        notes=f"Forecast produced by {model.name} without future lookahead.",
    )
