"""Baseline benchmark forecasting models: Persistence, Seasonal Naive, Moving Average, Historical Mean."""

from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from src.utils.validation import enforce_target_bounds


class BaseForecaster:
    """Base class for all univariate time series forecasters."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.last_date: Optional[pd.Timestamp] = None
        self.train_history: Optional[np.ndarray] = None

    def fit(self, train_series: pd.Series) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def predict_with_intervals(
        self,
        horizon: int,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """Generate point forecast with empirical residual uncertainty intervals."""
        preds = self.predict(horizon)
        # Baseline heuristic uncertainty based on training standard deviation
        std_err = float(np.std(self.train_history)) if self.train_history is not None and len(self.train_history) > 1 else 1.0
        z = 1.96 if confidence_level >= 0.95 else 1.645
        # Increasing uncertainty with horizon
        horizon_factors = np.sqrt(np.arange(1, horizon + 1))
        margin = z * (std_err * 0.25) * horizon_factors

        lower = enforce_target_bounds(preds - margin)
        upper = preds + margin
        return {
            "mean": preds,
            "lower": lower,
            "upper": upper,
        }


class PersistenceForecaster(BaseForecaster):
    """Last-observation persistence forecast (Naïve 1): y_{t+h} = y_t."""

    def __init__(self):
        super().__init__(name="Persistence (Naïve)")
        self.last_value: float = 0.0

    def fit(self, train_series: pd.Series) -> "PersistenceForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna()
        if clean_s.empty:
            raise ValueError("Cannot fit PersistenceForecaster on empty series.")
        self.train_history = clean_s.values
        self.last_value = float(clean_s.iloc[-1])
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        preds = np.full(shape=(horizon,), fill_value=self.last_value, dtype=float)
        return enforce_target_bounds(preds)


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naïve forecast: y_{t+h} = y_{t+h - m} where m is the seasonal period."""

    def __init__(self, seasonal_period: int = 7):
        super().__init__(name=f"Seasonal Naïve (m={seasonal_period})")
        self.seasonal_period = max(1, seasonal_period)
        self.seasonal_values: Optional[np.ndarray] = None

    def fit(self, train_series: pd.Series) -> "SeasonalNaiveForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna()
        if len(clean_s) < self.seasonal_period:
            # Fall back to persistence if fewer observations than one season
            self.seasonal_values = np.array([clean_s.iloc[-1]])
        else:
            self.seasonal_values = clean_s.iloc[-self.seasonal_period :].values
        self.train_history = clean_s.values
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.seasonal_values is None:
            raise RuntimeError("Model must be fitted before predict.")
        reps = int(np.ceil(horizon / len(self.seasonal_values)))
        tiled = np.tile(self.seasonal_values, reps)[:horizon]
        return enforce_target_bounds(tiled)


class MovingAverageForecaster(BaseForecaster):
    """Moving Average forecast: y_{t+h} = mean of the last k historical observations."""

    def __init__(self, window: int = 7):
        super().__init__(name=f"Moving Average (k={window})")
        self.window = max(1, window)
        self.mean_value: float = 0.0

    def fit(self, train_series: pd.Series) -> "MovingAverageForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna()
        if clean_s.empty:
            raise ValueError("Cannot fit on empty series.")
        actual_k = min(self.window, len(clean_s))
        self.train_history = clean_s.values
        self.mean_value = float(clean_s.iloc[-actual_k:].mean())
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        preds = np.full(shape=(horizon,), fill_value=self.mean_value, dtype=float)
        return enforce_target_bounds(preds)


class HistoricalSeasonalMeanForecaster(BaseForecaster):
    """Historical Seasonal Mean forecast: computes mean per calendar month or day of week."""

    def __init__(self, frequency: str = "D"):
        super().__init__(name="Historical Seasonal Mean")
        self.frequency = frequency
        self.seasonal_means: Dict[int, float] = {}
        self.global_mean: float = 0.0

    def fit(self, train_series: pd.Series, dates: Optional[pd.Series] = None) -> "HistoricalSeasonalMeanForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna()
        if clean_s.empty:
            raise ValueError("Cannot fit on empty series.")
        self.train_history = clean_s.values
        self.global_mean = float(clean_s.mean())

        if dates is not None and len(dates) == len(train_series):
            d = pd.to_datetime(dates.loc[clean_s.index])
            group_keys = d.dt.dayofweek if self.frequency in ["D", "DAILY"] else d.dt.month
            grouped = clean_s.groupby(group_keys).mean()
            self.seasonal_means = grouped.to_dict()
        else:
            self.seasonal_means = {}

        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        # If dates were not provided, use global historical mean
        preds = np.full(shape=(horizon,), fill_value=self.global_mean, dtype=float)
        return enforce_target_bounds(preds)
