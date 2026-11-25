"""AutoTS Automated Time Series Forecasting Adapter for Crop Disease Intelligence.

Leverages AutoTS genetic model search across lightweight baselines (Theta, ETS, GLM, Naive)
configured with non-negative constraints and execution safeguards.
"""

import contextlib
import io
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from autots import AutoTS
    HAS_AUTOTS = True
except ImportError:
    AutoTS = None
    HAS_AUTOTS = False


def forecast_autots(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    date_col: str = "date",
    model_list: str = "superfast",
    max_generations: int = 1,
    num_validations: int = 1,
    ensemble: str = "simple",
    prediction_interval: float = 0.9,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit AutoTS on series and return (predictions, lower_bound, upper_bound, metadata)."""
    if not HAS_AUTOTS:
        raise ImportError("AutoTS is not installed. Install it via `pip install autots`.")

    df_clean = train_df[[date_col, target_col]].copy()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce").fillna(0.0)

    model = AutoTS(
        forecast_length=int(horizon),
        frequency="infer",
        prediction_interval=prediction_interval,
        ensemble=ensemble,
        model_list=model_list,
        max_generations=max_generations,
        num_validations=num_validations,
        no_negatives=True,
        constraint=None,
        random_seed=random_seed,
        verbose=-1,
    )

    # Suppress console output during generation search
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = model.fit(df_clean, date_col=date_col, value_col=target_col)
        prediction = model.predict()

    forecast_series = prediction.forecast[target_col].values
    preds = np.clip(np.nan_to_num(forecast_series, nan=0.0), 0.0, None)

    try:
        lower_series = prediction.lower_forecast[target_col].values
        lb = np.clip(np.nan_to_num(lower_series, nan=0.0), 0.0, None)
    except Exception:
        lb = np.maximum(0.0, preds * 0.8)

    try:
        upper_series = prediction.upper_forecast[target_col].values
        ub = np.clip(np.nan_to_num(upper_series, nan=0.0), 0.0, None)
    except Exception:
        ub = np.maximum(lb, preds * 1.2)

    best_name = str(getattr(model, "best_model_name", "AutoTS Ensemble"))
    metadata = {
        "best_model_name": best_name,
        "model_list": model_list,
        "generations": max_generations,
        "fitted_model": model,
    }

    return preds, lb, ub, metadata
