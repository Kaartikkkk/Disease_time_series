"""Meta Prophet Generalized Additive Forecasting Model.

Mathematical Formulation:
    y(t) = g(t) + s(t) + h(t) + sum_{k=1}^K beta_k * x_k(t) + epsilon_t
where:
    g(t): Piecewise linear growth trend
    s(t): Yearly Fourier periodic changes (Fourier order = 10 by default)
    h(t): Holiday / intervention effects
    beta_k: Regression coefficients for exogenous weather variables
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd

HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    pass


def forecast_prophet(
    train_df: pd.DataFrame,
    horizon: int,
    future_dates: pd.DatetimeIndex,
    weather_cols: Optional[List[str]] = None,
    future_weather_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Forecast using Meta Prophet with yearly seasonality and climate regressors.

    Args:
        train_df: Historical dataframe with 'date' and 'documented_occurrence_count'.
        horizon: Forecast horizon (months).
        future_dates: DatetimeIndex for the future prediction intervals.
        weather_cols: List of climate columns to add as exogenous regressors.
        future_weather_df: Projected future climate regressors.

    Returns:
        Tuple of (point_predictions, lower_95_ci, upper_95_ci, fitted_prophet_model).
    """
    if not HAS_PROPHET:
        raise ImportError("Prophet is not installed or unavailable in this environment.")

    df_p = pd.DataFrame({
        "ds": pd.to_datetime(train_df["date"]),
        "y": train_df["documented_occurrence_count"].values,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )

    if weather_cols:
        for col in weather_cols:
            if col in train_df.columns:
                df_p[col] = train_df[col].fillna(train_df[col].mean()).values
                model.add_regressor(col)

    model.fit(df_p)

    future_p = pd.DataFrame({"ds": future_dates})
    if weather_cols and future_weather_df is not None:
        for col in weather_cols:
            if col in future_weather_df.columns:
                future_p[col] = future_weather_df[col].values
            else:
                future_p[col] = df_p[col].mean()

    forecast = model.predict(future_p)
    point_pred = np.maximum(0.0, forecast["yhat"].values)
    lower_95 = np.maximum(0.0, forecast["yhat_lower"].values)
    upper_95 = forecast["yhat_upper"].values

    return point_pred, lower_95, upper_95, model
