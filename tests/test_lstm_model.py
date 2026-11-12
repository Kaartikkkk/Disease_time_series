"""Unit, scaling-leakage, and integration tests for PyTorch LSTM."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import run_model_comparison, run_rolling_walk_forward_evaluation
from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.lstm_model import HAS_TORCH, create_lstm_sequences, forecast_lstm


def _sample_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(77)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n)
    counts = np.maximum(0, np.round(4.0 + 3.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.5, n))).astype(int)
    return pd.DataFrame({"date": dates, "documented_occurrence_count": counts})


def test_lstm_registration_and_availability():
    spec = get_model_spec("lstm")
    assert spec.display_name == "LSTM (Sequence Neural Net)"
    assert spec.category == "Deep Learning"
    assert spec.available is HAS_TORCH
    assert spec.supports_exog is False
    assert spec.has_ci is False


def test_create_lstm_sequences_chronological():
    vals = np.arange(15, dtype=float)
    X, y = create_lstm_sequences(vals, lookback=12)
    assert X.shape == (3, 12, 1)
    assert y.shape == (3, 1)
    assert np.allclose(X[0, :, 0], np.arange(0, 12))
    assert y[0, 0] == 12.0
    assert np.allclose(X[2, :, 0], np.arange(2, 14))
    assert y[2, 0] == 14.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_lstm_adapter_contract_and_diagnostics():
    df = _sample_df(60)
    ctx = ForecastContext(train_df=df.iloc[:54], horizon=6, future_df=df.iloc[54:])
    out = run_registered_model("lstm", ctx)
    validated = validate_forecast_output(out, expected_horizon=6, clip_non_negative=True)

    assert validated["predictions"].shape == (6,)
    assert np.all(np.isfinite(validated["predictions"]))
    assert np.all(validated["predictions"] >= 0.0)
    assert validated["lower_bound"] is None
    assert validated["upper_bound"] is None

    diag = validated["metadata"]["diagnostics"]
    for key in (
        "framework", "epochs_trained", "best_epoch", "train_samples", "val_samples",
        "lookback", "units", "dropout", "learning_rate", "batch_size",
        "scaling_method", "training_runtime_sec", "pred_std", "near_zero_pct", "status",
    ):
        assert key in diag, f"Missing LSTM diagnostic key: {key}"
    assert diag["status"] == "ok"
    assert diag["lookback"] == 12
    assert diag["units"] == 16
    assert diag["scaling_method"] == "MinMaxScaler(0, 1)"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_lstm_walk_forward_no_leakage_and_fixed_holdout(monkeypatch):
    df = _sample_df(60)
    seen_max_dates = []
    orig_fn = forecast_lstm

    def spy_lstm(train_df, horizon, **kwargs):
        seen_max_dates.append(pd.to_datetime(train_df["date"].max()))
        return orig_fn(train_df, horizon, **kwargs)

    monkeypatch.setattr("src.models.adapters.forecast_lstm", spy_lstm)

    res_wf = run_rolling_walk_forward_evaluation(df, horizon=6, step_ahead=1, window_type="expanding")
    assert "LSTM (Sequence Neural Net)" in res_wf["forecasts"]
    val_dates = pd.to_datetime(res_wf["validation_dates"]).tolist()
    assert len(seen_max_dates) == 6
    for tr_max, ev_dt in zip(seen_max_dates, val_dates):
        assert tr_max < ev_dt, f"LSTM walk-forward leakage: {tr_max} >= {ev_dt}"

    res_hold = run_model_comparison(df, horizon=6)
    assert "LSTM (Sequence Neural Net)" in res_hold["forecasts"]
    assert len(res_hold["forecasts"]["LSTM (Sequence Neural Net)"]) == 6


def test_lstm_failure_not_converted_to_zero(monkeypatch):
    def broken_lstm(*args, **kwargs):
        raise RuntimeError("simulated lstm crash")

    monkeypatch.setattr("src.models.adapters.forecast_lstm", broken_lstm)
    df = _sample_df(60)
    res_wf = run_rolling_walk_forward_evaluation(df, horizon=3, step_ahead=1)
    assert "LSTM (Sequence Neural Net)" in res_wf["model_failures"]
    assert np.all(np.isnan(res_wf["forecasts"]["LSTM (Sequence Neural Net)"]))

