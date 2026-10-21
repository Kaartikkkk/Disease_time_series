"""Statistical time series forecasting models: ARIMA, SARIMA, and Exponential Smoothing."""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.models.baselines import BaseForecaster
from src.utils.logging_config import logger
from src.utils.validation import enforce_target_bounds


class ARIMAForecaster(BaseForecaster):
    """Autoregressive Integrated Moving Average (ARIMA) Forecaster.

    Orders are selected strictly using training-data AIC to prevent test leakage.
    """

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        auto_order: bool = True,
    ):
        super().__init__(name="ARIMA")
        self.order = order or (1, 1, 1)
        self.auto_order = auto_order
        self.fitted_model_ = None

    def fit(self, train_series: pd.Series) -> "ARIMAForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna().values
        n = len(clean_s)
        if n < 10:
            raise ValueError(f"Insufficient observations for ARIMA ({n} provided, min 10 required).")

        self.train_history = clean_s

        best_aic = float("inf")
        best_order = self.order
        best_res = None

        if self.auto_order:
            candidate_orders = [
                (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1),
                (2, 1, 1), (1, 1, 2), (0, 1, 2), (2, 1, 2),
            ]
            for o in candidate_orders:
                try:
                    mod = ARIMA(clean_s, order=o)
                    res = mod.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = o
                        best_res = res
                except Exception:
                    continue

        if best_res is None:
            # Fallback to default specified order
            mod = ARIMA(clean_s, order=self.order)
            best_res = mod.fit()
            best_order = self.order

        self.order = best_order
        self.name = f"ARIMA{self.order}"
        self.fitted_model_ = best_res
        self.is_fitted = True
        logger.info(f"Fitted ARIMA with order {self.order} (AIC: {self.fitted_model_.aic:.1f})")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        forecast_res = self.fitted_model_.forecast(steps=horizon)
        preds = np.asarray(forecast_res)
        return enforce_target_bounds(preds)

    def predict_with_intervals(
        self,
        horizon: int,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        alpha = 1.0 - confidence_level
        fc_obj = self.fitted_model_.get_forecast(steps=horizon)
        preds = np.asarray(fc_obj.predicted_mean)
        conf_int = fc_obj.conf_int(alpha=alpha)
        lower = enforce_target_bounds(np.asarray(conf_int[:, 0]))
        upper = np.asarray(conf_int[:, 1])
        return {
            "mean": enforce_target_bounds(preds),
            "lower": lower,
            "upper": upper,
        }


class SARIMAForecaster(BaseForecaster):
    """Seasonal Autoregressive Integrated Moving Average (SARIMA) Forecaster.

    Fits SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,s)).
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 7),
    ):
        super().__init__(name=f"SARIMA{order}x{seasonal_order}")
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model_ = None

    def fit(self, train_series: pd.Series) -> "SARIMAForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna().values
        n = len(clean_s)
        s = self.seasonal_order[3]
        if n < 2 * s + 5:
            # Fall back to standard ARIMA if insufficient seasons
            logger.warning(
                f"Series length {n} insufficient for SARIMA with period {s}. Using ARIMA(1,1,1)."
            )
            self.order = (1, 1, 1)
            self.seasonal_order = (0, 0, 0, 0)

        self.train_history = clean_s
        try:
            mod = SARIMAX(
                clean_s,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.fitted_model_ = mod.fit(disp=False)
            self.is_fitted = True
            logger.info(f"Fitted SARIMA (AIC: {self.fitted_model_.aic:.1f})")
            return self
        except Exception as e:
            logger.warning(f"SARIMA fit failed: {e}. Falling back to ARIMA(1,1,1).")
            fallback = ARIMA(clean_s, order=(1, 1, 1)).fit()
            self.fitted_model_ = fallback
            self.name = "ARIMA(1,1,1) [SARIMA fallback]"
            self.is_fitted = True
            return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        preds = np.asarray(self.fitted_model_.forecast(steps=horizon))
        return enforce_target_bounds(preds)

    def predict_with_intervals(
        self,
        horizon: int,
        confidence_level: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        alpha = 1.0 - confidence_level
        fc_obj = self.fitted_model_.get_forecast(steps=horizon)
        preds = np.asarray(fc_obj.predicted_mean)
        conf_int = fc_obj.conf_int(alpha=alpha)
        lower = enforce_target_bounds(np.asarray(conf_int[:, 0]))
        upper = np.asarray(conf_int[:, 1])
        return {
            "mean": enforce_target_bounds(preds),
            "lower": lower,
            "upper": upper,
        }


class ExponentialSmoothingForecaster(BaseForecaster):
    """Holt-Winters Exponential Smoothing Forecaster."""

    def __init__(
        self,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
    ):
        super().__init__(name="Exponential Smoothing")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model_ = None

    def fit(self, train_series: pd.Series) -> "ExponentialSmoothingForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna().values
        n = len(clean_s)
        self.train_history = clean_s

        s = self.seasonal_periods or 7
        actual_seasonal = self.seasonal
        if actual_seasonal is not None and n < 2 * s + 2:
            actual_seasonal = None

        try:
            mod = ExponentialSmoothing(
                clean_s,
                trend=self.trend,
                seasonal=actual_seasonal,
                seasonal_periods=s if actual_seasonal else None,
                initialization_method="estimated",
            )
            self.fitted_model_ = mod.fit()
            self.is_fitted = True
            self.name = f"Holt-Winters (trend={self.trend}, seasonal={actual_seasonal})"
            return self
        except Exception as e:
            logger.warning(f"ExponentialSmoothing with trend failed: {e}. Fitting simple level.")
            mod = ExponentialSmoothing(clean_s, trend=None, seasonal=None)
            self.fitted_model_ = mod.fit()
            self.is_fitted = True
            self.name = "Simple Exponential Smoothing"
            return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        preds = np.asarray(self.fitted_model_.forecast(horizon))
        return enforce_target_bounds(preds)
