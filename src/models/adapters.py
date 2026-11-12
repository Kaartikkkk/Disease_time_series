"""Lightweight Adapters Connecting Forecasting Implementations to the Common Contract.

Each adapter wraps an existing model function in `src/models/` without altering
its underlying mathematical algorithm, ensuring Milestone 2 behavior is preserved.
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.models.ar_model import forecast_ar
from src.models.arima_model import forecast_arima
from src.models.arimax_model import forecast_arimax
from src.models.base import ForecastContext
from src.models.global_models import (
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_hurdle_xgboost,
)
from src.models.gradient_boosting import forecast_gradient_boosting
from src.models.gru_model import forecast_gru
from src.models.holt_winters import forecast_holt_winters
from src.models.lstm_model import HAS_TORCH, forecast_lstm
from src.models.ma_model import forecast_ma
from src.models.prophet_model import HAS_PROPHET, forecast_prophet
from src.models.random_forest import forecast_random_forest
from src.models.sarima_model import forecast_sarima
from src.models.sarimax_model import forecast_sarimax_model
from src.models.seasonal_naive import forecast_seasonal_naive
from src.models.xgboost_model import HAS_XGBOOST, forecast_xgboost


def _extract_exog(ctx: ForecastContext) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Extract aligned training and future exogenous DataFrames if available."""
    if not ctx.exog_cols:
        return None, None
    if not all(c in ctx.train_df.columns for c in ctx.exog_cols):
        return None, None
    train_exog = ctx.train_df[ctx.exog_cols].bfill().fillna(0)
    future_exog = None
    if ctx.future_df is not None and all(c in ctx.future_df.columns for c in ctx.exog_cols):
        future_exog = ctx.future_df[ctx.exog_cols].bfill().fillna(0)
    return train_exog, future_exog


def _build_output(
    preds: Any,
    lb: Any,
    ub: Any,
    model_name: str,
    parameters: str,
    model_family: str,
    fitted_model: Any = None,
) -> Dict[str, Any]:
    return {
        "predictions": preds,
        "lower_bound": lb,
        "upper_bound": ub,
        "model_name": model_name,
        "metadata": {
            "parameters": parameters,
            "model_family": model_family,
            "fitted_model": fitted_model,
            "status": "ok",
        },
    }


def adapt_seasonal_naive(ctx: ForecastContext) -> Dict[str, Any]:
    season_len = int(ctx.config.get("season_length", 12))
    preds, lb, ub = forecast_seasonal_naive(
        ctx.train_df[ctx.target_col], horizon=ctx.horizon, season_length=season_len
    )
    return _build_output(
        preds, lb, ub,
        model_name="Seasonal Naive (Baseline)",
        parameters=f"Same-Month Prior Year (s={season_len})",
        model_family="Persistence Baseline",
    )


def adapt_ar(ctx: ForecastContext) -> Dict[str, Any]:
    p = int(ctx.config.get("p", 1))
    preds, lb, ub, res = forecast_ar(ctx.train_df[ctx.target_col], horizon=ctx.horizon, p=p)
    return _build_output(
        preds, lb, ub,
        model_name="AR(1)",
        parameters=getattr(res, "param_summary", f"p={p}, d=0, q=0"),
        model_family="Box-Jenkins Univariate",
        fitted_model=res,
    )


def adapt_ma(ctx: ForecastContext) -> Dict[str, Any]:
    q = int(ctx.config.get("q", 1))
    preds, lb, ub, res = forecast_ma(ctx.train_df[ctx.target_col], horizon=ctx.horizon, q=q)
    return _build_output(
        preds, lb, ub,
        model_name="MA(1)",
        parameters=getattr(res, "param_summary", f"p=0, d=0, q={q}"),
        model_family="Box-Jenkins Univariate",
        fitted_model=res,
    )


def adapt_arima(ctx: ForecastContext) -> Dict[str, Any]:
    order = tuple(ctx.config.get("order", (1, 1, 1)))
    auto_opt = bool(ctx.config.get("auto_optimize", True))
    preds, lb, ub, res = forecast_arima(
        ctx.train_df[ctx.target_col], horizon=ctx.horizon, order=order, auto_optimize=auto_opt
    )
    return _build_output(
        preds, lb, ub,
        model_name="ARIMA(1,1,1)",
        parameters=getattr(res, "param_summary", f"p={order[0]}, d={order[1]}, q={order[2]}"),
        model_family="Box-Jenkins Integrated",
        fitted_model=res,
    )


def adapt_sarima(ctx: ForecastContext) -> Dict[str, Any]:
    order = tuple(ctx.config.get("order", (1, 0, 1)))
    seasonal_order = tuple(ctx.config.get("seasonal_order", (1, 0, 1, 12)))
    auto_opt = bool(ctx.config.get("auto_optimize", True))
    preds, lb, ub, res = forecast_sarima(
        ctx.train_df[ctx.target_col],
        horizon=ctx.horizon,
        order=order,
        seasonal_order=seasonal_order,
        auto_optimize=auto_opt,
    )
    return _build_output(
        preds, lb, ub,
        model_name="SARIMA(1,0,1)(1,0,1)[12]",
        parameters=getattr(res, "param_summary", f"Order={order}, Seasonal={seasonal_order}"),
        model_family="Seasonal Box-Jenkins",
        fitted_model=res,
    )


def adapt_holt_winters(ctx: ForecastContext) -> Dict[str, Any]:
    season_len = int(ctx.config.get("season_length", 12))
    auto_opt = bool(ctx.config.get("auto_optimize", True))
    preds, lb, ub, res = forecast_holt_winters(
        ctx.train_df[ctx.target_col],
        horizon=ctx.horizon,
        seasonal_periods=season_len,
        auto_optimize=auto_opt,
    )
    return _build_output(
        preds, lb, ub,
        model_name="Holt-Winters Exponential Smoothing",
        parameters=getattr(res, "param_summary", f"Additive Level + Trend + Season (m={season_len})"),
        model_family="Exponential Smoothing",
        fitted_model=res,
    )


def adapt_arimax(ctx: ForecastContext) -> Dict[str, Any]:
    train_exog, future_exog = _extract_exog(ctx)
    if train_exog is None or future_exog is None:
        raise ValueError("ARIMAX requires aligned exogenous training and future regressors.")
    order = tuple(ctx.config.get("order", (1, 0, 1)))
    auto_opt = bool(ctx.config.get("auto_optimize", True))
    preds, lb, ub, res = forecast_arimax(
        ctx.train_df[ctx.target_col],
        horizon=ctx.horizon,
        train_exog=train_exog,
        future_exog=future_exog,
        order=order,
        auto_optimize=auto_opt,
    )
    return _build_output(
        preds, lb, ub,
        model_name="ARIMAX (Weather Regressors)",
        parameters=getattr(res, "param_summary", "ARIMA + 3 NASA Weather Lags (t-1)"),
        model_family="Exogenous Econometric",
        fitted_model=res,
    )


def adapt_sarimax(ctx: ForecastContext) -> Dict[str, Any]:
    train_exog, future_exog = _extract_exog(ctx)
    if train_exog is None or future_exog is None:
        raise ValueError("SARIMAX requires aligned exogenous training and future regressors.")
    order = tuple(ctx.config.get("order", (1, 0, 1)))
    seasonal_order = tuple(ctx.config.get("seasonal_order", (1, 0, 1, 12)))
    auto_opt = bool(ctx.config.get("auto_optimize", True))
    preds, lb, ub, res = forecast_sarimax_model(
        ctx.train_df[ctx.target_col],
        horizon=ctx.horizon,
        order=order,
        seasonal_order=seasonal_order,
        train_exog=train_exog,
        future_exog=future_exog,
        auto_optimize=auto_opt,
    )
    return _build_output(
        preds, lb, ub,
        model_name="SARIMAX (Seasonal + Weather)",
        parameters=getattr(res, "param_summary", "SARIMA(1,0,1)(1,0,1)[12] + NASA Weather"),
        model_family="Exogenous Econometric",
        fitted_model=res,
    )


def adapt_random_forest(ctx: ForecastContext) -> Dict[str, Any]:
    lags = list(ctx.config.get("lags", [1, 2, 3, 12]))
    exog_cols = ctx.exog_cols if ctx.exog_cols and all(c in ctx.train_df.columns for c in ctx.exog_cols) else None
    preds, lb, ub, model = forecast_random_forest(
        ctx.train_df,
        horizon=ctx.horizon,
        lags=lags,
        weather_lag_cols=exog_cols,
        future_weather_df=ctx.future_df if exog_cols else None,
    )
    return _build_output(
        preds, lb, ub,
        model_name="Random Forest (Autoregressive)",
        parameters=f"Regularized Trees (Depth 3) + Lags {lags} + Weather",
        model_family="Tree Ensemble ML",
        fitted_model=model,
    )


def adapt_gradient_boosting(ctx: ForecastContext) -> Dict[str, Any]:
    lags = list(ctx.config.get("lags", [1, 2, 3, 12]))
    exog_cols = ctx.exog_cols if ctx.exog_cols and all(c in ctx.train_df.columns for c in ctx.exog_cols) else None
    preds, lb, ub, model = forecast_gradient_boosting(
        ctx.train_df,
        horizon=ctx.horizon,
        lags=lags,
        weather_lag_cols=exog_cols,
        future_weather_df=ctx.future_df if exog_cols else None,
    )
    return _build_output(
        preds, lb, ub,
        model_name="Hist Gradient Boosting (Autoregressive)",
        parameters=f"Lags {lags} + Cyclical Harmonics + Weather",
        model_family="Tree Ensemble ML",
        fitted_model=model,
    )


def adapt_xgboost(ctx: ForecastContext) -> Dict[str, Any]:
    """Adapter for XGBoost autoregressive forecaster with lag, rolling, and weather features."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed or unavailable in this environment.")

    lags = list(ctx.config.get("xgb_lags", [1, 2, 3, 6, 12]))
    xgb_params = ctx.config.get("xgb_params")
    exog_cols = ctx.exog_cols if ctx.exog_cols and all(c in ctx.train_df.columns for c in ctx.exog_cols) else None

    preds, lb, ub, model, diagnostics = forecast_xgboost(
        ctx.train_df,
        horizon=ctx.horizon,
        target_col=ctx.target_col,
        lags=lags,
        weather_lag_cols=exog_cols,
        future_weather_df=ctx.future_df if exog_cols else None,
        xgb_params=xgb_params,
    )
    hp = diagnostics.get("hyperparameters", {})
    param_str = (
        f"XGBoost ({diagnostics.get('objective', 'reg:squarederror')}, "
        f"n={hp.get('n_estimators', 120)}, depth={hp.get('max_depth', 3)}, "
        f"lr={hp.get('learning_rate', 0.05)})"
    )
    out = _build_output(
        preds, lb, ub,
        model_name="XGBoost (Autoregressive)",
        parameters=param_str,
        model_family="Gradient Boosted Trees (XGBoost)",
        fitted_model=model,
    )
    out["metadata"]["diagnostics"] = diagnostics
    return out


def adapt_global_xgboost(ctx: ForecastContext) -> Dict[str, Any]:
    """Adapter for pooled multi-pathogen Global XGBoost."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed or unavailable in this environment.")

    xgb_params = ctx.config.get("xgb_params")
    preds, lb, ub, model, diagnostics = forecast_global_xgboost(
        ctx.train_df,
        horizon=ctx.horizon,
        target_col=ctx.target_col,
        future_weather_df=ctx.future_df,
        xgb_params=xgb_params,
    )
    hp = diagnostics.get("hyperparameters", {})
    param_str = (
        f"Global XGBoost (n={hp.get('n_estimators', 110)}, depth={hp.get('max_depth', 3)}, "
        f"lr={hp.get('learning_rate', 0.03)}, pooled_N={diagnostics.get('pooled_train_samples', 0)})"
    )
    out = _build_output(
        preds, lb, ub,
        model_name="Global XGBoost (Multi-Pathogen)",
        parameters=param_str,
        model_family="Pooled Gradient Boosted Trees",
        fitted_model=model,
    )
    out["metadata"]["diagnostics"] = diagnostics
    return out


def adapt_global_random_forest(ctx: ForecastContext) -> Dict[str, Any]:
    """Adapter for pooled multi-pathogen Global Random Forest."""
    rf_params = ctx.config.get("rf_params")
    preds, lb, ub, model, diagnostics = forecast_global_random_forest(
        ctx.train_df,
        horizon=ctx.horizon,
        target_col=ctx.target_col,
        future_weather_df=ctx.future_df,
        rf_params=rf_params,
    )
    hp = diagnostics.get("hyperparameters", {})
    param_str = (
        f"Global Random Forest (n={hp.get('n_estimators', 120)}, depth={hp.get('max_depth', 4)}, "
        f"leaf={hp.get('min_samples_leaf', 4)}, pooled_N={diagnostics.get('pooled_train_samples', 0)})"
    )
    out = _build_output(
        preds, lb, ub,
        model_name="Global Random Forest (Multi-Pathogen)",
        parameters=param_str,
        model_family="Pooled Tree Ensemble",
        fitted_model=model,
    )
    out["metadata"]["diagnostics"] = diagnostics
    return out


def adapt_hurdle_xgboost(ctx: ForecastContext) -> Dict[str, Any]:
    """Adapter for Two-Stage Hurdle XGBoost (Occurrence Classifier + Positive Regressor)."""
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed or unavailable in this environment.")

    clf_params = ctx.config.get("clf_params")
    reg_params = ctx.config.get("reg_params")
    preds, lb, ub, model, diagnostics = forecast_hurdle_xgboost(
        ctx.train_df,
        horizon=ctx.horizon,
        target_col=ctx.target_col,
        future_weather_df=ctx.future_df,
        clf_params=clf_params,
        reg_params=reg_params,
    )
    param_str = (
        f"Hurdle XGBoost (Classifier + Regressor, pos_N={diagnostics.get('positive_train_samples', 0)})"
    )
    out = _build_output(
        preds, lb, ub,
        model_name="Hurdle XGBoost (Classifier + Regressor)",
        parameters=param_str,
        model_family="Two-Stage Zero-Inflated ML",
        fitted_model=model,
    )
    out["metadata"]["diagnostics"] = diagnostics
    return out



def adapt_prophet(ctx: ForecastContext) -> Dict[str, Any]:
    """Adapter for univariate Meta Prophet with conservative monthly seasonality."""
    if not HAS_PROPHET:
        raise ImportError("Prophet is not installed or unavailable in this environment.")

    if ctx.future_df is not None and "date" in ctx.future_df.columns:
        future_dates = pd.DatetimeIndex(pd.to_datetime(ctx.future_df["date"]))
    else:
        last_date = pd.to_datetime(ctx.train_df["date"].max())
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=ctx.horizon, freq="MS")

    # Primary Prophet implementation is strictly univariate (no unrealized future weather required)
    use_exog = bool(ctx.config.get("use_exog", False))
    exog_cols = (
        ctx.exog_cols
        if use_exog and ctx.exog_cols and all(c in ctx.train_df.columns for c in ctx.exog_cols)
        else None
    )
    prophet_params = ctx.config.get("prophet_params")

    preds, lb, ub, model = forecast_prophet(
        ctx.train_df,
        horizon=ctx.horizon,
        future_dates=future_dates,
        weather_cols=exog_cols,
        future_weather_df=ctx.future_df if exog_cols else None,
        target_col=ctx.target_col,
        prophet_params=prophet_params,
    )
    out = _build_output(
        preds, lb, ub,
        model_name="Prophet (Trend + Seasonality)",
        parameters=getattr(model, "param_summary", "Prophet (additive, yearly=True, cp=0.05, sp=0.1)"),
        model_family="Generalized Additive (Prophet)",
        fitted_model=model,
    )
    out["metadata"]["diagnostics"] = getattr(model, "diagnostics", {})
    return out


def adapt_lstm(ctx: ForecastContext) -> Dict[str, Any]:
    """Adapter for compact univariate PyTorch LSTM."""
    if not HAS_TORCH:
        raise ImportError("PyTorch (torch) is not installed or unavailable in this environment.")

    lstm_config = ctx.config.get("lstm_config")
    preds, lb, ub, model, diagnostics = forecast_lstm(
        ctx.train_df,
        horizon=ctx.horizon,
        target_col=ctx.target_col,
        lstm_config=lstm_config,
    )
    param_str = (
        f"PyTorch LSTM (lookback={diagnostics['lookback']}, units={diagnostics['units']}, "
        f"dropout={diagnostics['dropout']}, best_ep={diagnostics['best_epoch']}/{diagnostics['epochs_trained']})"
    )
    out = _build_output(
        preds, lb, ub,
        model_name="LSTM (Sequence Neural Net)",
        parameters=param_str,
        model_family="Recurrent Neural Network (PyTorch)",
        fitted_model=model,
    )
    out["metadata"]["diagnostics"] = diagnostics
    return out


def adapt_gru(ctx: ForecastContext) -> Dict[str, Any]:
    """Adapter for compact univariate PyTorch GRU."""
    if not HAS_TORCH:
        raise ImportError("PyTorch (torch) is not installed or unavailable in this environment.")

    gru_config = ctx.config.get("gru_config")
    preds, lb, ub, model, diagnostics = forecast_gru(
        ctx.train_df,
        horizon=ctx.horizon,
        target_col=ctx.target_col,
        gru_config=gru_config,
    )
    param_str = (
        f"PyTorch GRU (lookback={diagnostics['lookback']}, units={diagnostics['units']}, "
        f"dropout={diagnostics['dropout']}, best_ep={diagnostics['best_epoch']}/{diagnostics['epochs_trained']})"
    )
    out = _build_output(
        preds, lb, ub,
        model_name="GRU (Gated Recurrent Unit)",
        parameters=param_str,
        model_family="Recurrent Neural Network (PyTorch)",
        fitted_model=model,
    )
    out["metadata"]["diagnostics"] = diagnostics
    return out




