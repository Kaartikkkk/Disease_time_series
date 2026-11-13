"""Dedicated unit, leakage-prevention, and integration tests for XGBoost."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import run_model_comparison, run_rolling_walk_forward_evaluation
from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.xgboost_model import HAS_XGBOOST, build_xgboost_features, forecast_xgboost


def _sample_xgb_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n)
    counts = np.maximum(0, np.round(4.0 + 3.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.5, n))).astype(int)
    return pd.DataFrame({
        "date": dates,
        "documented_occurrence_count": counts,
        "temp_mean_c_lag1": 25.0 + 4.0 * np.sin(2 * np.pi * t / 12.0),
        "precipitation_monthly_mm_lag1": np.maximum(0.0, 110.0 + 90.0 * np.sin(2 * np.pi * t / 12.0)),
        "relative_humidity_pct_lag1": 70.0 + 12.0 * np.sin(2 * np.pi * t / 12.0),
    })


def test_xgboost_registration_and_dependency_flag():
    spec = get_model_spec("xgboost")
    assert spec.display_name == "XGBoost (Autoregressive)"
    assert spec.category == "Machine Learning"
    assert spec.available is HAS_XGBOOST
    assert spec.supports_exog is True
    assert spec.has_ci is True


def test_xgboost_adapter_contract_and_diagnostics():
    df = _sample_xgb_df(60)
    train_df = df.iloc[:54].copy()
    future_df = df.iloc[54:].copy()
    exog = ["temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"]

    ctx = ForecastContext(train_df=train_df, horizon=6, exog_cols=exog, future_df=future_df)
    out = run_registered_model("xgboost", ctx)
    validated = validate_forecast_output(out, expected_horizon=6, clip_non_negative=True)

    assert validated["predictions"].shape == (6,)
    assert np.all(np.isfinite(validated["predictions"]))
    assert np.all(validated["predictions"] >= 0.0)
    assert validated["lower_bound"] is not None and validated["upper_bound"] is not None
    assert np.all(validated["lower_bound"] <= validated["upper_bound"])

    diag = validated["metadata"]["diagnostics"]
    for key in ("pred_std", "near_zero_pct", "pred_mean", "pred_min", "pred_max",
                "train_samples", "feature_count", "hyperparameters", "objective", "status"):
        assert key in diag, f"Missing diagnostic key: {key}"
    assert diag["status"] == "ok"
    assert diag["objective"] == "reg:squarederror"
    assert diag["train_samples"] > 0
    assert diag["feature_count"] >= 14  # 5 lags + 4 rolling + 5 calendar + weather


def test_lag_and_rolling_features_no_temporal_leakage():
    """Modifying target at row t or later must NOT alter lag/rolling features at row <= t."""
    df_orig = _sample_xgb_df(36)
    df_perturbed = df_orig.copy()
    cutoff = 25
    df_perturbed.loc[cutoff:, "documented_occurrence_count"] = 9999

    f_orig, cols = build_xgboost_features(df_orig)
    f_pert, _ = build_xgboost_features(df_perturbed)

    # At row == cutoff, ONLY current target y[cutoff] changed; all lag_* and roll_* at row cutoff must be identical!
    lag_roll_cols = [c for c in cols if c.startswith("lag_") or c.startswith("roll_")]
    pd.testing.assert_frame_equal(
        f_orig.loc[:cutoff, lag_roll_cols],
        f_pert.loc[:cutoff, lag_roll_cols],
    )


def test_xgboost_walk_forward_no_leakage_and_fixed_holdout(monkeypatch):
    df = _sample_xgb_df(60)
    seen_max_dates = []
    orig_fn = forecast_xgboost

    def spy_xgb(train_df, horizon, **kwargs):
        seen_max_dates.append(pd.to_datetime(train_df["date"].max()))
        return orig_fn(train_df, horizon, **kwargs)

    monkeypatch.setattr("src.models.adapters.forecast_xgboost", spy_xgb)

    res_wf = run_rolling_walk_forward_evaluation(df, horizon=6, step_ahead=1, window_type="expanding")
    assert "XGBoost (Autoregressive)" in res_wf["forecasts"]
    val_dates = pd.to_datetime(res_wf["validation_dates"]).tolist()
    assert len(seen_max_dates) == 6
    for tr_max, ev_dt in zip(seen_max_dates, val_dates):
        assert tr_max < ev_dt, f"Walk-forward leakage: {tr_max} >= {ev_dt}"

    res_hold = run_model_comparison(df, horizon=6)
    assert "XGBoost (Autoregressive)" in res_hold["forecasts"]
    assert len(res_hold["forecasts"]["XGBoost (Autoregressive)"]) == 6


def test_xgboost_failure_not_converted_to_zero(monkeypatch):
    df = _sample_xgb_df(60)

    def broken_xgb(*args, **kwargs):
        raise RuntimeError("simulated xgboost crash")

    monkeypatch.setattr("src.models.adapters.forecast_xgboost", broken_xgb)
    res_wf = run_rolling_walk_forward_evaluation(df, horizon=3, step_ahead=1)
    assert "XGBoost (Autoregressive)" in res_wf["model_failures"]
    preds = res_wf["forecasts"]["XGBoost (Autoregressive)"]
    assert np.all(np.isnan(preds))

