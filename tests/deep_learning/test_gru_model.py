"""Unit, scaling-leakage, and integration tests for PyTorch GRU."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.forecasting import run_model_comparison, run_rolling_walk_forward_evaluation
from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.gru_model import HAS_TORCH, forecast_gru
from src.models.lstm_model import create_lstm_sequences


def _sample_df(n: int = 60, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n)
    counts = np.maximum(0, np.round(4.0 + 3.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.5, n))).astype(int)
    return pd.DataFrame({"date": dates, "documented_occurrence_count": counts})


def test_gru_registration_and_availability():
    spec = get_model_spec("gru")
    assert spec.display_name == "GRU (Gated Recurrent Unit)"
    assert spec.category == "Deep Learning"
    assert spec.available is HAS_TORCH
    assert spec.supports_exog is False
    assert spec.has_ci is False
    assert spec.default_config.get("lookback") == 12


def test_gru_sequence_shape_and_chronological_lookback():
    vals = np.arange(15, dtype=float)
    X, y = create_lstm_sequences(vals, lookback=12)
    assert X.shape == (3, 12, 1)
    assert y.shape == (3, 1)
    assert np.allclose(X[0, :, 0], np.arange(0, 12))
    assert y[0, 0] == 12.0
    assert np.allclose(X[2, :, 0], np.arange(2, 14))
    assert y[2, 0] == 14.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_gru_training_only_scaler_and_insufficient_history(monkeypatch):
    df = _sample_df(40)
    # Place extreme outlier strictly in the internal validation tail (last 6 rows)
    df.loc[df.index[-6:], "documented_occurrence_count"] = 9999

    fitted_max_values = []
    from sklearn.preprocessing import MinMaxScaler
    orig_fit = MinMaxScaler.fit

    def spy_fit(self, X, y=None):
        fitted_max_values.append(float(np.max(X)))
        return orig_fit(self, X, y=y)

    monkeypatch.setattr(MinMaxScaler, "fit", spy_fit)
    preds, lb, ub, _, diag = forecast_gru(
        df, horizon=3, gru_config={"max_epochs": 5, "patience": 2}
    )
    assert len(preds) == 3
    assert lb is None and ub is None
    assert len(fitted_max_values) == 1
    assert fitted_max_values[0] < 100.0, "Scaler leaked internal validation tail values!"
    assert diag["scaling_method"] == "MinMaxScaler(0, 1)"

    short_df = _sample_df(16)
    with pytest.raises(ValueError, match="Insufficient training observations"):
        forecast_gru(short_df, horizon=3)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_gru_adapter_contract_clipping_and_diagnostics(monkeypatch):
    df = _sample_df(60)
    ctx = ForecastContext(
        train_df=df.iloc[:54],
        horizon=6,
        future_df=df.iloc[54:],
        config={"gru_config": {"max_epochs": 10, "patience": 3}},
    )
    out = run_registered_model("gru", ctx)
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
        "scaling_method", "training_runtime_sec", "training_runtime",
        "pred_std", "near_zero_pct", "pred_mean", "pred_min", "pred_max", "status",
    ):
        assert key in diag, f"Missing GRU diagnostic key: {key}"
    assert diag["status"] == "ok"
    assert diag["lookback"] == 12
    assert diag["units"] in (8, 16)
    assert diag["scaling_method"] == "MinMaxScaler(0, 1)"

    # Verify negative raw output gets clipped to 0.0
    def neg_gru(*args, **kwargs):
        return np.array([-2.0, -0.5, 1.5]), None, None, None, {
            "lookback": 12, "units": 16, "dropout": 0.1,
            "best_epoch": 1, "epochs_trained": 1, "status": "ok",
        }

    monkeypatch.setattr("src.models.adapters.forecast_gru", neg_gru)
    clipped_out = run_registered_model("gru", ForecastContext(train_df=df.iloc[:54], horizon=3))
    assert np.allclose(clipped_out["predictions"], [0.0, 0.0, 1.5])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_gru_walk_forward_no_leakage_and_fixed_holdout(monkeypatch):
    df = _sample_df(60)
    seen_max_dates = []
    orig_fn = forecast_gru

    def spy_gru(train_df, horizon, **kwargs):
        seen_max_dates.append(pd.to_datetime(train_df["date"].max()))
        kwargs.setdefault("gru_config", {"max_epochs": 6, "patience": 2})
        return orig_fn(train_df, horizon, **kwargs)

    monkeypatch.setattr("src.models.adapters.forecast_gru", spy_gru)

    res_wf = run_rolling_walk_forward_evaluation(df, horizon=6, step_ahead=1, window_type="expanding")
    assert "GRU (Gated Recurrent Unit)" in res_wf["forecasts"]
    val_dates = pd.to_datetime(res_wf["validation_dates"]).tolist()
    assert len(seen_max_dates) == 6
    for tr_max, ev_dt in zip(seen_max_dates, val_dates):
        assert tr_max < ev_dt, f"GRU walk-forward leakage: {tr_max} >= {ev_dt}"

    res_hold = run_model_comparison(df, horizon=6)
    assert "GRU (Gated Recurrent Unit)" in res_hold["forecasts"]
    assert len(res_hold["forecasts"]["GRU (Gated Recurrent Unit)"]) == 6


def test_gru_failure_not_converted_to_zero_and_dashboard_wired(monkeypatch):
    def broken_gru(*args, **kwargs):
        raise RuntimeError("simulated gru crash")

    monkeypatch.setattr("src.models.adapters.forecast_gru", broken_gru)
    df = _sample_df(60)
    res_wf = run_rolling_walk_forward_evaluation(df, horizon=3, step_ahead=1)
    assert "GRU (Gated Recurrent Unit)" in res_wf["model_failures"]
    assert np.all(np.isnan(res_wf["forecasts"]["GRU (Gated Recurrent Unit)"]))

    page4_src = Path("pages/4_Future_Forecasts.py").read_text(encoding="utf-8")
    assert "GRU (Gated Recurrent Unit)" in page4_src
    assert "forecast_gru" in page4_src

