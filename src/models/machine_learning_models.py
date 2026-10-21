"""Supervised machine learning forecasters (Random Forest, HistGradientBoosting, XGBoost)."""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb
from src.features.lag_features import generate_lag_features, create_supervised_matrix
from src.models.baselines import BaseForecaster
from src.utils.logging_config import logger
from src.utils.validation import enforce_target_bounds


class MLForecaster(BaseForecaster):
    """Recursive autoregressive machine learning time series forecaster.

    Supported model types:
    - 'random_forest': RandomForestRegressor
    - 'hist_gradient_boosting': HistGradientBoostingRegressor
    - 'xgboost': XGBRegressor

    MULTI-STEP FORECASTING STRATEGY:
    Recursive multi-step autoregressive iteration. At each forward step h:
    1. The feature vector is assembled using historical observations and previously
       predicted steps (y_{T+1}, ..., y_{T+h-1}).
    2. Calendar features are computed for the future timestamp.
    3. The model predicts y_{T+h}.
    4. y_{T+h} is appended to the rolling history buffer for subsequent step predictions.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: Optional[int] = 6,
        random_seed: int = 42,
        frequency: str = "D",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        include_weather: bool = True,
    ):
        type_names = {
            "random_forest": "Random Forest Regressor",
            "hist_gradient_boosting": "HistGradientBoosting Regressor",
            "xgboost": "XGBoost Regressor",
        }
        name = type_names.get(model_type.lower(), f"ML_{model_type}")
        super().__init__(name=name)

        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.frequency = frequency
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.include_weather = include_weather

        self.model_ = self._init_underlying_model()
        self.feature_names_: List[str] = []
        self.last_history_df_: Optional[pd.DataFrame] = None
        self.target_col_: str = "incidence_rate"

    def _init_underlying_model(self):
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_seed,
                n_jobs=-1,
            )
        elif self.model_type == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_seed,
            )
        elif self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth or 5,
                learning_rate=0.08,
                random_state=self.random_seed,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported ML model type: {self.model_type}")

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "incidence_rate",
    ) -> "MLForecaster":
        """Fit the supervised model strictly on training data."""
        self.target_col_ = target_col
        self.last_history_df_ = train_df.sort_values("date").copy()

        X, y, feature_cols = create_supervised_matrix(
            train_df,
            target_col=target_col,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            include_weather=self.include_weather,
            frequency=self.frequency,
            drop_na_rows=True,
        )

        if len(X) < 10:
            raise ValueError(f"Insufficient training rows after lag feature creation ({len(X)} rows).")

        self.feature_names_ = feature_cols
        self.train_history = y.values

        self.model_.fit(X, y)
        self.is_fitted = True
        logger.info(f"Fitted {self.name} on {len(X)} training samples with {len(feature_cols)} features.")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Recursive multi-step out-of-sample forecasting."""
        if not self.is_fitted or self.last_history_df_ is None:
            raise RuntimeError("Model must be fitted before predict.")

        # Determine date frequency
        freq_str = "D"
        if self.frequency in ["W", "WEEKLY", "W-SUN"]:
            freq_str = "W-SUN"
        elif self.frequency in ["M", "MONTHLY", "MS"]:
            freq_str = "MS"

        last_date = pd.to_datetime(self.last_history_df_["date"].iloc[-1])
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_str)[1:]

        # Maintain a rolling simulation dataframe
        sim_df = self.last_history_df_.copy()
        predictions = []

        # Default weather persistence for future steps
        last_temp = float(sim_df["temperature"].iloc[-1]) if "temperature" in sim_df.columns and pd.notna(sim_df["temperature"].iloc[-1]) else 20.0
        last_hum = float(sim_df["humidity"].iloc[-1]) if "humidity" in sim_df.columns and pd.notna(sim_df["humidity"].iloc[-1]) else 65.0
        last_rain = 0.0

        for step, step_date in enumerate(future_dates):
            # 1. Append placeholder row for step_date
            new_row = {
                "date": step_date,
                self.target_col_: np.nan,  # unobserved
                "temperature": last_temp,
                "humidity": last_hum,
                "rainfall": last_rain,
            }
            # Copy categorical attributes if present
            for meta_col in ["crop", "disease", "location_id", "source", "is_synthetic"]:
                if meta_col in sim_df.columns:
                    new_row[meta_col] = sim_df[meta_col].iloc[-1]

            sim_df = pd.concat([sim_df, pd.DataFrame([new_row])], ignore_index=True)

            # 2. Generate lag features
            feat_df = generate_lag_features(
                sim_df,
                target_col=self.target_col_,
                lags=self.lags,
                rolling_windows=self.rolling_windows,
                include_weather=self.include_weather,
                frequency=self.frequency,
            )

            # 3. Extract feature vector for the current step (the last row)
            current_x = feat_df.iloc[[-1]][self.feature_names_].fillna(0.0)

            # 4. Predict target
            pred_val = float(self.model_.predict(current_x)[0])
            pred_bounded = float(enforce_target_bounds(np.array([pred_val]))[0])
            predictions.append(pred_bounded)

            # 5. Update the placeholder row in sim_df with the predicted value for subsequent steps!
            sim_df.loc[sim_df.index[-1], self.target_col_] = pred_bounded

        return np.array(predictions)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importance ranking for tree-based models."""
        if not self.is_fitted:
            return pd.DataFrame()

        importances = None
        if hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
        elif hasattr(self.model_, "coef_"):
            importances = np.abs(self.model_.coef_)

        if importances is not None and len(importances) == len(self.feature_names_):
            df_imp = pd.DataFrame({
                "feature": self.feature_names_,
                "importance": importances,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            return df_imp
        return pd.DataFrame()
