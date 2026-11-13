"""Machine Learning Autoregressive Framework.

Transforms univariate time series and lagged exogenous weather regressors into
supervised learning feature matrices [X_t, y_t] with recursive multi-step forward rollouts:
    Features:
    - Autoregressive Lags: [y_{t-1}, y_{t-2}, y_{t-3}, y_{t-12}]
    - Cyclical Month Encodings: sin(2*pi*m/12), cos(2*pi*m/12)
    - Lagged Climate: 1-month-lagged Temperature, Humidity, Precipitation
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


def forecast_ml_autoregressor(
    train_df: pd.DataFrame,
    horizon: int,
    model_type: str = "gradient_boosting",
    lags: Optional[List[int]] = None,
    weather_lag_cols: Optional[List[str]] = None,
    future_weather_df: Optional[pd.DataFrame] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Autoregressive ML model (Gradient Boosting / Random Forest) with strictly lagged regressors.

    Args:
        train_df: Historical training data with 'date' and 'documented_occurrence_count'.
        horizon: Multi-step forecast horizon in months.
        model_type: "gradient_boosting" or "random_forest".
        lags: Autoregressive target lag periods.
        weather_lag_cols: Lagged weather feature names.
        future_weather_df: Future weather values across the horizon.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_model).
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12]
    target_col = "documented_occurrence_count"

    # Feature construction on training data
    df_feat = train_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_feat["date"]):
        df_feat["date"] = pd.to_datetime(df_feat["date"])
    feature_cols = []

    for l in lags:
        col_name = f"lag_{l}"
        df_feat[col_name] = df_feat[target_col].shift(l)
        feature_cols.append(col_name)

    # Cyclical month features
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["date"].dt.month / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["date"].dt.month / 12.0)
    feature_cols.extend(["month_sin", "month_cos"])

    # Lagged weather features (strictly lagged to prevent leakage)
    if weather_lag_cols:
        for w_col in weather_lag_cols:
            if w_col in df_feat.columns:
                feature_cols.append(w_col)

    # Drop initial NaN rows created by lags
    df_train_clean = df_feat.dropna(subset=feature_cols).copy()
    X_train = df_train_clean[feature_cols].values
    y_train = df_train_clean[target_col].values

    if model_type == "random_forest":
        # Regularized tree depth and leaf size to prevent memorizing sparse counts
        rf_kwargs: Dict[str, Any] = {
            "n_estimators": 180,
            "max_depth": 5,
            "min_samples_leaf": 4,
            "max_features": 0.70,
            "random_state": 42,
        }
        if model_params:
            rf_kwargs.update(model_params)
        model = RandomForestRegressor(**rf_kwargs)
    else:
        # Regularized gradient boosting with depth limits and L2 penalty
        hgb_kwargs: Dict[str, Any] = {
            "max_depth": 3,
            "min_samples_leaf": 5,
            "l2_regularization": 2.0,
            "random_state": 42,
        }
        if model_params:
            hgb_kwargs.update(model_params)
        model = HistGradientBoostingRegressor(**hgb_kwargs)

    model.fit(X_train, y_train)

    # Recursive multi-step out-of-sample forecasting
    current_history = list(train_df[target_col].values)
    preds = []

    last_date = pd.to_datetime(train_df["date"].iloc[-1])
    for step in range(1, horizon + 1):
        step_date = last_date + pd.DateOffset(months=step)
        row_feat = []
        for l in lags:
            val = current_history[-l] if len(current_history) >= l else current_history[-1]
            row_feat.append(val)

        row_feat.append(np.sin(2 * np.pi * step_date.month / 12.0))
        row_feat.append(np.cos(2 * np.pi * step_date.month / 12.0))

        if weather_lag_cols:
            for w_col in weather_lag_cols:
                if future_weather_df is not None and w_col in future_weather_df.columns and len(future_weather_df) >= step:
                    row_feat.append(future_weather_df[w_col].iloc[step - 1])
                else:
                    # Fallback to mean historical lagged value
                    row_feat.append(df_train_clean[w_col].mean() if w_col in df_train_clean else 0.0)

        step_pred = float(model.predict([row_feat])[0])
        step_pred = max(0.0, step_pred)
        preds.append(step_pred)
        current_history.append(step_pred)

    preds_arr = np.array(preds)

    # Estimate prediction intervals from training residuals
    train_preds = model.predict(X_train)
    resids = y_train - train_preds
    se = float(np.std(resids)) if len(resids) > 0 else 1.0

    lower_95 = np.maximum(0.0, preds_arr - 1.96 * se)
    upper_95 = preds_arr + 1.96 * se

    return preds_arr, lower_95, upper_95, model
