"""Tests for the unified forecasting architecture: contract, registry, validation, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import build_model_comparison_table
from src.forecasting import run_model_comparison, run_rolling_walk_forward_evaluation
from src.model_registry import (
    MODEL_SPECS,
    get_available_models,
    get_model_spec,
    list_registered_models,
    run_registered_model,
)
from src.models.base import ForecastContext, build_failure_output, validate_forecast_output


def _sample_df(n: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=n, freq="MS")
    t = np.arange(n)
    y = np.maximum(0, np.round(3.0 + 2.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.5, n))).astype(int)
    return pd.DataFrame({
        "date": dates,
        "documented_occurrence_count": y,
        "temp_mean_c_lag1": 24.0 + 4.0 * np.sin(2 * np.pi * t / 12.0),
        "precipitation_monthly_mm_lag1": np.maximum(0.0, 100.0 + 80.0 * np.sin(2 * np.pi * t / 12.0)),
        "relative_humidity_pct_lag1": 72.0 + 10.0 * np.sin(2 * np.pi * t / 12.0),
    })


def test_model_registration_counts_and_categories():
    all_specs = list_registered_models(include_unavailable=True)
    avail_specs = get_available_models()
    assert len(all_specs) == 17
    assert len(avail_specs) == 17
    assert {s.model_id for s in all_specs if not s.available} == set()
    assert get_model_spec("xgboost").available is True
    assert get_model_spec("prophet").available is True
    assert get_model_spec("lstm").available is True
    assert get_model_spec("lstm").category == "Deep Learning"
    assert get_model_spec("gru").available is True
    assert get_model_spec("gru").category == "Deep Learning"
    assert get_model_spec("global_xgboost").available is True
    assert get_model_spec("global_xgboost").category == "Machine Learning"
    assert get_model_spec("global_random_forest").available is True
    assert get_model_spec("hurdle_xgboost").available is True
    assert get_model_spec("seasonal_naive").category == "Classical"
    assert get_model_spec("arima").category == "Statistical"
    assert get_model_spec("random_forest").category == "Machine Learning"


def test_unavailable_model_raises_runtime_error(monkeypatch):
    from src.model_registry import MODEL_SPECS, ModelSpec

    dummy_spec = ModelSpec(
        model_id="_dummy_unavailable",
        display_name="Dummy Unavailable",
        category="Deep Learning",
        model_family="Test",
        available=False,
        supports_exog=False,
        has_ci=False,
        notes="test unavailable",
    )
    monkeypatch.setitem(MODEL_SPECS, "_dummy_unavailable", dummy_spec)
    df = _sample_df(36)
    ctx = ForecastContext(train_df=df, horizon=3)
    with pytest.raises(RuntimeError, match="unavailable"):
        run_registered_model("_dummy_unavailable", ctx)


def test_standardized_output_and_optional_ci():
    raw = {
        "predictions": [1.0, -0.5, 3.2],
        "lower_bound": None,
        "upper_bound": None,
        "model_name": "TestModel",
        "metadata": {"k": 1},
    }
    validated = validate_forecast_output(raw, expected_horizon=3, clip_non_negative=True)
    assert validated["predictions"].shape == (3,)
    assert np.allclose(validated["predictions"], [1.0, 0.0, 3.2])
    assert validated["lower_bound"] is None
    assert validated["upper_bound"] is None
    assert validated["model_name"] == "TestModel"


def test_incorrect_horizon_rejected():
    raw = {"predictions": [1.0, 2.0], "lower_bound": None, "upper_bound": None, "model_name": "X"}
    with pytest.raises(ValueError, match="length"):
        validate_forecast_output(raw, expected_horizon=3)


def test_invalid_numeric_predictions_rejected():
    for bad in ([1.0, np.nan, 2.0], [1.0, np.inf, 2.0], ["a", "b", "c"]):
        raw = {"predictions": bad, "lower_bound": None, "upper_bound": None, "model_name": "X"}
        with pytest.raises(ValueError):
            validate_forecast_output(raw, expected_horizon=3)


def test_confidence_interval_validation():
    # lower > upper rejected
    bad_order = {
        "predictions": [2.0, 2.0],
        "lower_bound": [3.0, 1.0],
        "upper_bound": [2.5, 3.0],
        "model_name": "X",
    }
    with pytest.raises(ValueError, match="lower_bound exceeds upper_bound"):
        validate_forecast_output(bad_order, expected_horizon=2)

    # Half-specified CI rejected
    half_ci = {
        "predictions": [2.0, 2.0],
        "lower_bound": [1.0, 1.0],
        "upper_bound": None,
        "model_name": "X",
    }
    with pytest.raises(ValueError, match="provided together"):
        validate_forecast_output(half_ci, expected_horizon=2)


def test_failed_model_handling_uses_nan_not_zero():
    fail = build_failure_output("BrokenModel", expected_horizon=4, error=RuntimeError("boom"), model_family="Test")
    assert len(fail["predictions"]) == 4
    assert np.all(np.isnan(fail["predictions"]))
    assert fail["lower_bound"] is None
    assert fail["metadata"]["status"] == "failed"
    assert "boom" in fail["metadata"]["error"]


def test_all_10_classical_models_compatible_via_registry():
    df = _sample_df(48)
    train_df = df.iloc[:42].copy()
    future_df = df.iloc[42:].copy()
    exog_cols = ["temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"]
    ctx = ForecastContext(
        train_df=train_df,
        horizon=6,
        exog_cols=exog_cols,
        future_df=future_df,
    )
    for spec in get_available_models():
        out = run_registered_model(spec.model_id, ctx)
        assert out["predictions"].shape == (6,)
        assert np.all(np.isfinite(out["predictions"]))
        assert np.all(out["predictions"] >= 0.0)
        assert out["model_name"] == spec.display_name
        assert out["metadata"]["status"] == "ok"


def test_walk_forward_no_future_leakage_and_category_table(monkeypatch):
    df = _sample_df(60)
    seen_max_dates = []
    orig_runner = run_registered_model

    def spy_runner(model_id, ctx, clip_non_negative=True):
        if model_id == "seasonal_naive":
            seen_max_dates.append(pd.to_datetime(ctx.train_df["date"].max()))
        return orig_runner(model_id, ctx, clip_non_negative=clip_non_negative)

    monkeypatch.setattr("src.forecasting.run_registered_model", spy_runner)

    res = run_rolling_walk_forward_evaluation(df, horizon=3, step_ahead=1)
    val_dates = pd.to_datetime(res["validation_dates"]).tolist()
    assert len(seen_max_dates) == len(val_dates) == 3
    for train_max, eval_dt in zip(seen_max_dates, val_dates):
        assert train_max < eval_dt, f"Leakage detected: {train_max} >= {eval_dt}"

    table = build_model_comparison_table(res["evaluation_metrics"])
    assert "Category" in table.columns
    assert set(table["Category"]).issubset({"Classical", "Statistical", "Machine Learning", "Deep Learning"})

