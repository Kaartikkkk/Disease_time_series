"""Dedicated unit and integration tests for AutoTS AutoML Model Adapter."""

import numpy as np
import pandas as pd
import pytest

from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.machine_learning.autots_model import HAS_AUTOTS, forecast_autots


def _sample_series_df(n: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n)
    counts = np.maximum(0, np.round(3.0 + 2.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.4, n))).astype(int)
    return pd.DataFrame({
        "date": dates,
        "documented_occurrence_count": counts,
    })


def test_autots_registration():
    """Verify that AutoTS is properly registered in MODEL_SPECS."""
    spec = get_model_spec("autots")
    assert spec is not None
    assert spec.display_name == "AutoTS (AutoML Benchmark)"
    assert spec.category == "Machine Learning"
    assert spec.model_family == "Automated ML (AutoTS)"
    assert spec.available is HAS_AUTOTS
    assert spec.has_ci is True


@pytest.mark.skipif(not HAS_AUTOTS, reason="AutoTS not installed")
def test_autots_forecast_output_structure():
    """Test standalone forecast_autots output shapes, bounds, and non-negativity."""
    df = _sample_series_df(36)
    horizon = 3
    preds, lb, ub, meta = forecast_autots(
        df,
        horizon=horizon,
        model_list="superfast",
        max_generations=1,
    )
    assert len(preds) == horizon
    assert len(lb) == horizon
    assert len(ub) == horizon
    assert np.all(preds >= 0.0)
    assert np.all(lb >= 0.0)
    assert np.all(ub >= lb)
    assert "best_model_name" in meta


@pytest.mark.skipif(not HAS_AUTOTS, reason="AutoTS not installed")
def test_autots_via_registry():
    """Verify run_registered_model contract validation for autots."""
    df = _sample_series_df(36)
    ctx = ForecastContext(train_df=df, horizon=3, target_col="documented_occurrence_count")
    out = run_registered_model("autots", ctx)
    validated = validate_forecast_output(out, expected_horizon=3, clip_non_negative=True)
    assert len(validated["predictions"]) == 3
    assert validated["model_name"] == "AutoTS (AutoML Benchmark)"
