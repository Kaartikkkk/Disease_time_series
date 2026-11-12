"""Unit, isolation, and integration tests for Global ML and Two-Stage Hurdle models."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import run_model_comparison, run_rolling_walk_forward_evaluation
from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.global_models import (
    HAS_XGBOOST,
    build_global_panel_features,
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_hurdle_xgboost,
)


def _make_two_pathogen_panel(n: int = 50) -> pd.DataFrame:
    """Create a controlled 2-pathogen panel to test cross-pathogen isolation."""
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    # Pathogen A: constant 10.0
    df_a = pd.DataFrame({
        "date": dates,
        "pathogen_slug": "pathogen_a",
        "documented_occurrence_count": 10.0,
        "temp_mean_c_lag1": 25.0,
        "precipitation_monthly_mm_lag1": 100.0,
        "relative_humidity_pct_lag1": 70.0,
    })
    # Pathogen B: constant 100.0
    df_b = pd.DataFrame({
        "date": dates,
        "pathogen_slug": "pathogen_b",
        "documented_occurrence_count": 100.0,
        "temp_mean_c_lag1": 20.0,
        "precipitation_monthly_mm_lag1": 50.0,
        "relative_humidity_pct_lag1": 60.0,
    })
    return pd.concat([df_a, df_b], axis=0).sort_values(["date", "pathogen_slug"]).reset_index(drop=True)


def test_global_features_strictly_isolated_by_pathogen():
    """Verify that lag and rolling statistics NEVER mix data between pathogens."""
    panel = _make_two_pathogen_panel(40)
    feat_df, cols = build_global_panel_features(panel, lags=(1, 2), rolling_windows=(3,))

    a_rows = feat_df[feat_df["pathogen_slug"] == "pathogen_a"].dropna(subset=cols)
    b_rows = feat_df[feat_df["pathogen_slug"] == "pathogen_b"].dropna(subset=cols)

    # Pathogen A lags and rolling means must be strictly 10.0
    assert np.allclose(a_rows["lag_1"], 10.0)
    assert np.allclose(a_rows["lag_2"], 10.0)
    assert np.allclose(a_rows["roll_mean_3"], 10.0)

    # Pathogen B lags and rolling means must be strictly 100.0
    assert np.allclose(b_rows["lag_1"], 100.0)
    assert np.allclose(b_rows["lag_2"], 100.0)
    assert np.allclose(b_rows["roll_mean_3"], 100.0)


@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
def test_global_models_adapter_contracts_and_non_negative():
    panel = _make_two_pathogen_panel(40)
    sub_a = panel[panel["pathogen_slug"] == "pathogen_a"].iloc[:30].copy().reset_index(drop=True)
    # Ensure both zero and positive occurrence classes exist
    sub_a["documented_occurrence_count"] = [0.0 if i % 2 == 0 else 10.0 for i in range(len(sub_a))]

    ctx = ForecastContext(train_df=sub_a, horizon=6)

    for m_id in ("global_xgboost", "global_random_forest", "hurdle_xgboost"):
        out = run_registered_model(m_id, ctx)
        validated = validate_forecast_output(out, expected_horizon=6, clip_non_negative=True)
        assert validated["predictions"].shape == (6,)
        assert np.all(np.isfinite(validated["predictions"]))
        assert np.all(validated["predictions"] >= 0.0)
        assert validated["lower_bound"] is not None
        assert validated["upper_bound"] is not None
        assert "diagnostics" in validated["metadata"]


@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
def test_hurdle_probability_and_magnitude_multiplication(monkeypatch):
    """Verify Stage 1 P(y>0) and Stage 2 positive magnitude multiplication."""
    panel = _make_two_pathogen_panel(40)
    sub_a = panel[panel["pathogen_slug"] == "pathogen_a"].iloc[:30].copy().reset_index(drop=True)
    # Ensure both zero and positive occurrence classes exist
    sub_a["documented_occurrence_count"] = [0.0 if i % 2 == 0 else 10.0 for i in range(len(sub_a))]

    # Mock classifier to output probability 0.50 and regressor to output magnitude 8.0
    class MockClf:
        def fit(self, X, y, *args, **kwargs):
            return self

        def predict_proba(self, X):
            return np.array([[0.50, 0.50]] * len(X))

    class MockReg:
        def fit(self, X, y, *args, **kwargs):
            return self

        def predict(self, X):
            return np.array([8.0] * len(X))

    monkeypatch.setattr("src.models.global_models.XGBClassifier", lambda **kwargs: MockClf())
    monkeypatch.setattr("src.models.global_models.XGBRegressor", lambda **kwargs: MockReg())

    # We test forecast_hurdle_xgboost directly with mock
    from src.models.global_models import _load_historical_panel, build_global_panel_features
    # Run hurdle model
    preds, lb, ub, _, diag = forecast_hurdle_xgboost(sub_a, horizon=3)
    # Expected: 0.50 * 8.0 = 4.0
    assert np.allclose(preds, 4.0)
    assert np.all(preds >= 0.0)
    assert len(diag["stage1_probabilities"]) == 3
    assert len(diag["stage2_magnitudes"]) == 3


@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not installed")
def test_global_models_walk_forward_no_leakage(monkeypatch):
    """Verify walk-forward evaluation incorporates historical observations strictly prior to eval date."""
    df = _make_two_pathogen_panel(40)
    sub_a = df[df["pathogen_slug"] == "pathogen_a"].copy()

    seen_max_dates = []
    orig_fn = forecast_global_xgboost

    def spy_g_xgb(train_df, horizon, **kwargs):
        seen_max_dates.append(pd.to_datetime(train_df["date"].max()))
        return orig_fn(train_df, horizon, **kwargs)

    monkeypatch.setattr("src.models.adapters.forecast_global_xgboost", spy_g_xgb)

    res_wf = run_rolling_walk_forward_evaluation(sub_a, horizon=3, step_ahead=1, window_type="expanding")
    assert "Global XGBoost (Multi-Pathogen)" in res_wf["forecasts"]
    val_dates = pd.to_datetime(res_wf["validation_dates"]).tolist()
    assert len(seen_max_dates) == 3
    for tr_max, ev_dt in zip(seen_max_dates, val_dates):
        assert tr_max < ev_dt, f"Global model leakage: {tr_max} >= {ev_dt}"


def test_global_models_failure_handling(monkeypatch):
    """Verify failing global model produces NaNs and registers in model_failures."""
    def broken_fn(*args, **kwargs):
        raise RuntimeError("simulated global model failure")

    monkeypatch.setattr("src.models.adapters.forecast_global_xgboost", broken_fn)
    df = _make_two_pathogen_panel(30)
    sub_a = df[df["pathogen_slug"] == "pathogen_a"].copy()

    res_wf = run_rolling_walk_forward_evaluation(sub_a, horizon=3, step_ahead=1)
    assert "Global XGBoost (Multi-Pathogen)" in res_wf["model_failures"]
    assert np.all(np.isnan(res_wf["forecasts"]["Global XGBoost (Multi-Pathogen)"]))
