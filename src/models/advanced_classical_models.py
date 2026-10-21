"""Advanced Classical & Modern Forecasters: AR(p), MA(q), ARMA(p,q), ARIMAX, SARIMAX, VAR, Prophet, AutoTS."""

import warnings
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from src.models.baselines import BaseForecaster
from src.utils.logging_config import logger
from src.utils.validation import enforce_target_bounds


class ARForecaster(BaseForecaster):
    """Pure Autoregressive model AR(p): y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} + e_t."""

    def __init__(self, p: int = 3):
        super().__init__(name=f"AR({p})")
        self.p = max(1, p)
        self.fitted_model_ = None

    def fit(self, train_series: pd.Series) -> "ARForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna().values
        self.train_history = clean_s
        n = len(clean_s)
        actual_p = min(self.p, n // 3)
        if actual_p < 1:
            actual_p = 1
        self.p = actual_p
        self.name = f"AR({self.p})"

        try:
            mod = AutoReg(clean_s, lags=self.p, old_names=False)
            self.fitted_model_ = mod.fit()
        except Exception:
            mod = ARIMA(clean_s, order=(self.p, 0, 0))
            self.fitted_model_ = mod.fit()

        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        n = len(self.train_history)
        preds = self.fitted_model_.predict(start=n, end=n + horizon - 1)
        return enforce_target_bounds(np.asarray(preds))


class MAForecaster(BaseForecaster):
    """Pure Moving Average model MA(q): y_t = mu + e_t + theta_1*e_{t-1} + ... + theta_q*e_{t-q}."""

    def __init__(self, q: int = 2):
        super().__init__(name=f"MA({q})")
        self.q = max(1, q)
        self.fitted_model_ = None

    def fit(self, train_series: pd.Series) -> "MAForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna().values
        self.train_history = clean_s
        n = len(clean_s)
        actual_q = min(self.q, n // 3)
        self.q = max(1, actual_q)
        self.name = f"MA({self.q})"

        mod = ARIMA(clean_s, order=(0, 0, self.q))
        self.fitted_model_ = mod.fit()
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        preds = self.fitted_model_.forecast(steps=horizon)
        return enforce_target_bounds(np.asarray(preds))


class ARMAForecaster(BaseForecaster):
    """Autoregressive Moving Average model ARMA(p, q)."""

    def __init__(self, p: int = 2, q: int = 1):
        super().__init__(name=f"ARMA({p},{q})")
        self.p = max(1, p)
        self.q = max(1, q)
        self.fitted_model_ = None

    def fit(self, train_series: pd.Series) -> "ARMAForecaster":
        clean_s = pd.to_numeric(train_series, errors="coerce").dropna().values
        self.train_history = clean_s
        mod = ARIMA(clean_s, order=(self.p, 0, self.q))
        self.fitted_model_ = mod.fit()
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        preds = self.fitted_model_.forecast(steps=horizon)
        return enforce_target_bounds(np.asarray(preds))


class ARIMAXForecaster(BaseForecaster):
    """ARIMA with Exogenous Covariates (ARIMAX).

    Integrates environmental variables (temperature, relative humidity, rainfall) as external regressors.
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__(name="ARIMAX (Weather Regressors)")
        self.order = order
        self.fitted_model_ = None
        self.exog_cols_: List[str] = []
        self.last_exog_vals_: Optional[np.ndarray] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "incidence_rate",
        exog_cols: Optional[List[str]] = None,
    ) -> "ARIMAXForecaster":
        if exog_cols is None:
            exog_cols = [c for c in ["temperature", "humidity", "rainfall"] if c in train_df.columns]

        clean_df = train_df.dropna(subset=[target_col] + exog_cols).copy()
        y = clean_df[target_col].values
        self.train_history = y
        self.exog_cols_ = exog_cols

        if not exog_cols:
            # Fall back to univariate ARIMA if no exogenous columns
            mod = ARIMA(y, order=self.order)
            self.fitted_model_ = mod.fit()
            self.name = f"ARIMA{self.order} (No Exog)"
        else:
            X = clean_df[exog_cols].values
            self.last_exog_vals_ = X[-1, :]
            mod = ARIMA(y, exog=X, order=self.order)
            self.fitted_model_ = mod.fit()
            self.name = f"ARIMAX{self.order} ({', '.join(exog_cols)})"

        self.is_fitted = True
        return self

    def predict(self, horizon: int, future_exog: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")

        if self.exog_cols_:
            if future_exog is None:
                # Use persistence of last observed weather conditions
                future_exog = np.tile(self.last_exog_vals_, (horizon, 1))
            preds = self.fitted_model_.forecast(steps=horizon, exog=future_exog)
        else:
            preds = self.fitted_model_.forecast(steps=horizon)

        return enforce_target_bounds(np.asarray(preds))


class SARIMAXForecaster(BaseForecaster):
    """Seasonal ARIMA with Exogenous Weather Regressors (SARIMAX)."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 7),
    ):
        super().__init__(name=f"SARIMAX{order}x{seasonal_order}")
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model_ = None
        self.exog_cols_: List[str] = []
        self.last_exog_vals_: Optional[np.ndarray] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "incidence_rate",
        exog_cols: Optional[List[str]] = None,
    ) -> "SARIMAXForecaster":
        if exog_cols is None:
            exog_cols = [c for c in ["temperature", "humidity", "rainfall"] if c in train_df.columns]

        clean_df = train_df.dropna(subset=[target_col] + exog_cols).copy()
        y = clean_df[target_col].values
        self.train_history = y
        self.exog_cols_ = exog_cols

        X = clean_df[exog_cols].values if exog_cols else None
        if X is not None:
            self.last_exog_vals_ = X[-1, :]

        s = self.seasonal_order[3]
        s_order = self.seasonal_order if len(y) >= 2 * s + 5 else (0, 0, 0, 0)

        mod = SARIMAX(
            y,
            exog=X,
            order=self.order,
            seasonal_order=s_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted_model_ = mod.fit(disp=False)
        self.is_fitted = True
        return self

    def predict(self, horizon: int, future_exog: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")

        if self.exog_cols_:
            if future_exog is None:
                future_exog = np.tile(self.last_exog_vals_, (horizon, 1))
            preds = self.fitted_model_.forecast(steps=horizon, exog=future_exog)
        else:
            preds = self.fitted_model_.forecast(steps=horizon)

        return enforce_target_bounds(np.asarray(preds))


class VARForecaster(BaseForecaster):
    """Vector Autoregression (VAR): Multidimensional linear model between disease and microclimate.

    Simultaneously models:
    Y_t = [incidence_rate_t, temperature_t, humidity_t] = C + A_1*Y_{t-1} + ... + A_p*Y_{t-p} + E_t
    """

    def __init__(self, p: int = 2):
        super().__init__(name=f"VAR({p}) Multivariate")
        self.p = p
        self.fitted_model_ = None
        self.var_cols_: List[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "incidence_rate",
        additional_cols: Optional[List[str]] = None,
    ) -> "VARForecaster":
        if additional_cols is None:
            additional_cols = [c for c in ["temperature", "humidity"] if c in train_df.columns]

        cols = [target_col] + additional_cols
        clean_df = train_df[cols].dropna().copy()

        if len(clean_df) < 15 or len(cols) < 2:
            raise ValueError(f"Insufficient multivariate data for VAR: {len(clean_df)} rows, {len(cols)} columns.")

        self.var_cols_ = cols
        self.train_history = clean_df[target_col].values

        mod = VAR(clean_df.values)
        self.fitted_model_ = mod.fit(maxlags=min(self.p, len(clean_df) // 5))
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        k_ar = self.fitted_model_.k_ar
        history_vals = self.fitted_model_.endog[-k_ar:]
        forecast_matrix = self.fitted_model_.forecast(y=history_vals, steps=horizon)
        # Column 0 is target (incidence_rate)
        preds = forecast_matrix[:, 0]
        return enforce_target_bounds(preds)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet Bayesian structural time series model."""

    def __init__(self):
        super().__init__(name="Prophet (Meta)")
        self.fitted_model_ = None
        self.frequency_ = "D"

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "incidence_rate",
        frequency: str = "D",
    ) -> "ProphetForecaster":
        from prophet import Prophet

        self.frequency_ = frequency
        clean_df = train_df[["date", target_col]].dropna().copy()
        clean_df = clean_df.rename(columns={"date": "ds", target_col: "y"})
        self.train_history = clean_df["y"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=(frequency in ["D", "DAILY"]),
                yearly_seasonality=(len(clean_df) > 180),
            )
            m.fit(clean_df)

        self.fitted_model_ = m
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")

        freq_code = "D" if self.frequency_ in ["D", "DAILY"] else ("W-SUN" if self.frequency_ in ["W", "WEEKLY"] else "MS")
        future = self.fitted_model_.make_future_dataframe(periods=horizon, freq=freq_code, include_history=False)
        forecast = self.fitted_model_.predict(future)
        preds = forecast["yhat"].values
        return enforce_target_bounds(np.asarray(preds))


class AutoTSForecaster(BaseForecaster):
    """AutoTS Automated Machine Learning & Time Series Algorithm Selection."""

    def __init__(self, max_generations: int = 1, num_validations: int = 1):
        super().__init__(name="AutoTS (AutoML Ensemble)")
        self.max_generations = max_generations
        self.num_validations = num_validations
        self.fitted_model_ = None
        self.best_model_name_: str = ""

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "incidence_rate",
        frequency: str = "D",
    ) -> "AutoTSForecaster":
        from autots import AutoTS

        clean_df = train_df[["date", target_col]].dropna().copy()
        self.train_history = clean_df[target_col].values

        freq_str = "D" if frequency in ["D", "DAILY"] else ("W" if frequency in ["W", "WEEKLY"] else "M")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoTS(
                forecast_length=14,
                frequency=freq_str,
                prediction_interval=0.95,
                ensemble="simple",
                model_list="fast",
                max_generations=self.max_generations,
                num_validations=self.num_validations,
                models_to_validate=0.15,
                verbose=0,
            )
            model = model.fit(clean_df, date_col="date", value_col=target_col)

        self.fitted_model_ = model
        self.best_model_name_ = getattr(model, "best_model_name", "AutoTS Best")
        self.name = f"AutoTS [{self.best_model_name_}]"
        self.is_fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise RuntimeError("Model is not fitted.")
        pred_obj = self.fitted_model_.predict()
        forecast_df = pred_obj.forecast
        preds = forecast_df.iloc[:horizon].values.flatten()
        return enforce_target_bounds(np.asarray(preds))
