"""Evaluation metrics for time series forecasting without data leakage."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class MetricScores:
    mae: float
    rmse: float
    mse: float
    r2: float
    smape: float
    mape: float
    mase: Optional[float] = None


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-4) -> float:
    """Compute standard Mean Absolute Percentage Error (MAPE).

    MAPE = 100% * (1 / N) * sum(|y_true - y_pred| / |y_true|)
    Filters near-zero values to avoid division by zero.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.abs(yt) > epsilon
    if not np.any(mask):
        # Fall back to zero error if all ground truth is zero and predictions match
        return 0.0 if np.all(np.abs(yp) <= epsilon) else 100.0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (sMAPE) handling zeros safely.

    sMAPE = 100% * (2 / N) * sum(|y_true - y_pred| / (|y_true| + |y_pred| + eps))
    Bounded between 0% and 200%.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.abs(yt) + np.abs(yp) + epsilon
    diff = np.abs(yp - yt)
    return float(np.mean(2.0 * diff / denom) * 100.0)


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None,
    seasonal_period: int = 1,
) -> Optional[float]:
    """Compute Mean Absolute Scaled Error (MASE).

    Scales MAE by the in-sample mean absolute error of a naive baseline (or seasonal naive).
    MASE < 1 indicates the model outperforms the naive baseline.
    """
    if y_train_history is None or len(y_train_history) <= seasonal_period:
        return None

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mae_model = np.mean(np.abs(yt - yp))

    # In-sample naive benchmark error
    train_diffs = np.abs(y_train_history[seasonal_period:] - y_train_history[:-seasonal_period])
    scale = np.mean(train_diffs)

    if scale < 1e-8:
        return None
    return float(mae_model / scale)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None,
    seasonal_period: int = 1,
) -> MetricScores:
    """Calculate all standard time series evaluation metrics.

    Metrics:
    - MAE: Mean Absolute Error (lower is better, same units as target)
    - RMSE: Root Mean Squared Error (lower is better, penalizes large deviations)
    - MSE: Mean Squared Error
    - R²: Coefficient of determination (higher is better, 1.0 = perfect, can be < 0 for poor out-of-sample fit)
    - sMAPE: Symmetric Mean Absolute Percentage Error (0% to 200%, lower is better)
    - MASE: Mean Absolute Scaled Error (< 1.0 beats persistence baseline)
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if len(yt) == 0 or len(yp) == 0:
        return MetricScores(mae=np.nan, rmse=np.nan, mse=np.nan, r2=np.nan, smape=np.nan, mape=np.nan, mase=None)

    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))

    # Out-of-sample R² can be negative if model performs worse than the historical mean
    try:
        r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 1e-8 else 0.0
    except Exception:
        r2 = 0.0

    smape = compute_smape(yt, yp)
    mape = compute_mape(yt, yp)
    mase = compute_mase(yt, yp, y_train_history, seasonal_period=seasonal_period)

    return MetricScores(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        mse=round(mse, 4),
        r2=round(r2, 4),
        smape=round(smape, 2),
        mape=round(mape, 2),
        mase=round(mase, 4) if mase is not None else None,
    )


METRIC_DESCRIPTIONS = {
    "mae": "Mean Absolute Error: Average absolute forecast error in natural target units (lower is better).",
    "rmse": "Root Mean Squared Error: Penalizes larger outbreak prediction errors quadratically (lower is better).",
    "r2": "Coefficient of Determination: Variance explained by forecast out-of-sample (higher is better, <= 1.0).",
    "smape": "Symmetric Mean Absolute Percentage Error: Scale-independent percentage error bounded 0-200% (lower is better).",
    "mase": "Mean Absolute Scaled Error: Scaled relative to persistence baseline. MASE < 1.0 demonstrates superiority over naïve reference.",
}
