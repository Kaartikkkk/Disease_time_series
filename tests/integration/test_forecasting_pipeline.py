"""Regression tests for the forecasting orchestrator (src/forecasting.py)."""

import numpy as np
import pandas as pd

from src.forecasting import (
    chronological_split,
    run_model_comparison,
    run_rolling_walk_forward_evaluation,
)


def _make_monthly_df(n_months: int = 60) -> pd.DataFrame:
    """Create deterministic seasonal monthly DataFrame matching schema."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    counts = np.maximum(0, np.round(4.0 + 3.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.6, n_months))).astype(int)
    return pd.DataFrame({
        "date": dates,
        "documented_occurrence_count": counts,
        "temp_mean_c_lag1": 25.0 + 5.0 * np.sin(2 * np.pi * t / 12.0),
        "precipitation_monthly_mm_lag1": np.maximum(0.0, 120.0 + 100.0 * np.sin(2 * np.pi * (t - 5) / 12.0)),
        "relative_humidity_pct_lag1": 70.0 + 15.0 * np.sin(2 * np.pi * (t - 5) / 12.0),
    })


def test_chronological_split_no_leakage():
    df = _make_monthly_df(100)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()


def test_model_comparison_runs_all_models():
    df = _make_monthly_df(60)
    res = run_model_comparison(df, horizon=6)
    names = {m["model_name"] for m in res["evaluation_metrics"]}
    assert len(names) == 17
    assert "Holt-Winters Exponential Smoothing" in names
    assert "SARIMAX (Seasonal + Weather)" in names
    assert "XGBoost (Autoregressive)" in names
    assert "Prophet (Trend + Seasonality)" in names
    assert "LSTM (Sequence Neural Net)" in names
    assert "GRU (Gated Recurrent Unit)" in names
    assert "Global XGBoost (Multi-Pathogen)" in names
    assert "Global Random Forest (Multi-Pathogen)" in names
    assert "Hurdle XGBoost (Classifier + Regressor)" in names


def test_walk_forward_consistent_naming_and_log1p_no_inflation():
    df = _make_monthly_df(60)
    res_raw = run_rolling_walk_forward_evaluation(df, horizon=3, step_ahead=1, use_log1p=False)
    res_log = run_rolling_walk_forward_evaluation(df, horizon=3, step_ahead=1, use_log1p=True)

    names_raw = {m["model_name"] for m in res_raw["evaluation_metrics"]}
    names_log = {m["model_name"] for m in res_log["evaluation_metrics"]}
    assert "Holt-Winters Exponential Smoothing" in names_raw
    assert len(names_raw) == 17
    assert len(names_log) == 17

    # Verify ML models are NOT exponentially inflated when use_log1p=True
    max_target = float(df["documented_occurrence_count"].max())
    for ml_name in ("Random Forest (Autoregressive)", "Hist Gradient Boosting (Autoregressive)"):
        preds = res_log["forecasts"][ml_name]
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0.0)
        assert float(np.max(preds)) < max_target * 3.0, f"{ml_name} inflated under log1p: {preds}"

