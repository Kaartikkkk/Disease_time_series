"""Unit and integration tests for Vector Autoregression (VAR)."""

import numpy as np
import pandas as pd
import pytest

from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.classical.var_model import forecast_var


def _sample_multivariate_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n)
    temp = 25.0 + 5.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.5, n)
    humidity = 70.0 + 15.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 1.0, n)
    counts = np.maximum(0, np.round(2.0 + 0.1 * humidity + rng.normal(0, 0.5, n))).astype(int)
    return pd.DataFrame({
        "date": dates,
        "documented_occurrence_count": counts,
        "temp_mean_c": temp,
        "relative_humidity_pct": humidity,
    })


def test_var_registration_and_availability():
    spec = get_model_spec("var")
    assert spec.display_name == "VAR (Vector Autoregression)"
    assert spec.category == "Statistical"
    assert spec.available is True
    assert spec.supports_exog is True
    assert spec.has_ci is True


def test_var_forecast_output_contract():
    df = _sample_multivariate_df(45)
    ctx = ForecastContext(
        train_df=df.iloc[:39],
        horizon=6,
        exog_cols=["temp_mean_c", "relative_humidity_pct"],
        future_df=df.iloc[39:],
    )
    out = run_registered_model("var", ctx)
    validated = validate_forecast_output(out, expected_horizon=6, clip_non_negative=True)

    assert validated["predictions"].shape == (6,)
    assert np.all(np.isfinite(validated["predictions"]))
    assert np.all(validated["predictions"] >= 0.0)
    assert "diagnostics" in validated["metadata"]
