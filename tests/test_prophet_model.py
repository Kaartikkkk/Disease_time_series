"""Unit, leakage-prevention, and integration tests for Meta Prophet."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import run_model_comparison, run_rolling_walk_forward_evaluation
from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.prophet_model import HAS_PROPHET, forecast_prophet, prepare_prophet_dataframe


def _sample_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n)
    counts = np.maximum(0, np.round(3.5 + 2.5 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.5, n))).astype(int)
    return pd.DataFrame({"date": dates, "documented_occurrence_count": counts})


class _DummyProphet:
    """Fast deterministic mock Prophet for unit tests (supports negative raw output testing)."""

    seen_ds_max = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def add_regressor(self, col):
        pass

    def fit(self, df_p):
        _DummyProphet.seen_ds_max.append(pd.to_datetime(df_p["ds"]).max())
        return self

    def predict(self, future_p):
        n = len(future_p)
        # Deliberately include negative yhat to test clipping & interval consistency
        yhat = np.array([-2.0 if i % 2 == 0 else 3.5 for i in range(n)], dtype=float)
        return pd.DataFrame({
            "ds": future_p["ds"],
            "yhat": yhat,
            "yhat_lower": yhat - 1.5,
            "yhat_upper": yhat + 1.5,
        })


def test_prophet_registration_and_availability():
    spec = get_model_spec("prophet")
    assert spec.display_name == "Prophet (Trend + Seasonality)"
    assert spec.category == "Statistical"
    assert spec.available is HAS_PROPHET
    assert spec.supports_exog is False
    assert spec.has_ci is True


def test_prepare_prophet_dataframe_chronological_ms():
    df = _sample_df(36).sample(frac=1.0, random_state=1)  # shuffled input
    df_p = prepare_prophet_dataframe(df)
    assert list(df_p.columns) == ["ds", "y"]
    assert df_p["ds"].is_monotonic_increasing
    assert len(df_p) == 36


def test_prophet_negative_clipping_intervals_and_diagnostics(monkeypatch):
    _DummyProphet.seen_ds_max = []
    monkeypatch.setattr("src.models.prophet_model.Prophet", _DummyProphet)
    monkeypatch.setattr("src.models.prophet_model.HAS_PROPHET", True)
    monkeypatch.setattr("src.models.adapters.HAS_PROPHET", True)

    df = _sample_df(48)
    ctx = ForecastContext(train_df=df.iloc[:42], horizon=6, future_df=df.iloc[42:])
    out = run_registered_model("prophet", ctx)
    validated = validate_forecast_output(out, expected_horizon=6, clip_non_negative=True)

    assert validated["predictions"].shape == (6,)
    assert np.all(np.isfinite(validated["predictions"]))
    assert np.all(validated["predictions"] >= 0.0)
    assert np.all(validated["lower_bound"] <= validated["predictions"])
    assert np.all(validated["predictions"] <= validated["upper_bound"])

    diag = validated["metadata"]["diagnostics"]
    for key in ("pred_std", "near_zero_pct", "pred_mean", "pred_min", "pred_max",
                "negative_raw_count", "train_samples", "forecast_horizon", "selected_params", "status"):
        assert key in diag
    assert diag["negative_raw_count"] == 3
    assert diag["status"] == "ok"


def test_prophet_walk_forward_isolation_and_fixed_holdout(monkeypatch):
    _DummyProphet.seen_ds_max = []
    monkeypatch.setattr("src.models.prophet_model.Prophet", _DummyProphet)
    monkeypatch.setattr("src.models.prophet_model.HAS_PROPHET", True)
    monkeypatch.setattr("src.models.adapters.HAS_PROPHET", True)

    df = _sample_df(60)
    res_wf = run_rolling_walk_forward_evaluation(df, horizon=6, step_ahead=1, window_type="expanding")
    assert "Prophet (Trend + Seasonality)" in res_wf["forecasts"]
    val_dates = pd.to_datetime(res_wf["validation_dates"]).tolist()
    assert len(_DummyProphet.seen_ds_max) == 6
    for tr_max, ev_dt in zip(_DummyProphet.seen_ds_max, val_dates):
        assert tr_max < ev_dt, f"Prophet walk-forward leakage: {tr_max} >= {ev_dt}"

    res_hold = run_model_comparison(df, horizon=6)
    assert "Prophet (Trend + Seasonality)" in res_hold["forecasts"]
    assert len(res_hold["forecasts"]["Prophet (Trend + Seasonality)"]) == 6


def test_prophet_failure_not_converted_to_zero(monkeypatch):
    def broken_prophet(*args, **kwargs):
        raise RuntimeError("simulated prophet crash")

    monkeypatch.setattr("src.models.adapters.forecast_prophet", broken_prophet)
    df = _sample_df(60)
    res_wf = run_rolling_walk_forward_evaluation(df, horizon=3, step_ahead=1)
    assert "Prophet (Trend + Seasonality)" in res_wf["model_failures"]
    assert np.all(np.isnan(res_wf["forecasts"]["Prophet (Trend + Seasonality)"]))


@pytest.mark.skipif(not HAS_PROPHET, reason="Prophet not installed")
def test_prophet_real_execution_contract():
    df = _sample_df(48)
    ctx = ForecastContext(train_df=df.iloc[:42], horizon=6, future_df=df.iloc[42:])
    out = run_registered_model("prophet", ctx)
    assert out["predictions"].shape == (6,)
    assert np.all(np.isfinite(out["predictions"]))
    assert np.all(out["predictions"] >= 0.0)
    assert out["metadata"]["diagnostics"]["status"] == "ok"

