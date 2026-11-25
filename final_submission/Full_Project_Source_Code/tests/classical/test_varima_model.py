"""Unit and integration tests for Vector Autoregressive Integrated Moving Average (VARIMA)."""

import numpy as np
import pandas as pd
import pytest

from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.classical.varima_model import forecast_varima


def _sample_multivariate_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(101)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n)
    temp = 24.0 + 6.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.4, n)
    humidity = 68.0 + 16.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.8, n)
    counts = np.maximum(0, np.round(1.5 + 0.12 * humidity + rng.normal(0, 0.6, n))).astype(int)
    return pd.DataFrame({
        "date": dates,
        "documented_occurrence_count": counts,
        "temp_mean_c": temp,
        "relative_humidity_pct": humidity,
    })


def test_varima_registration_and_availability():
    spec = get_model_spec("varima")
    assert spec.display_name == "VARIMA(1,1,1) (Multivariate System)"
    assert spec.category == "Statistical"
    assert spec.available is True
    assert spec.supports_exog is True
    assert spec.has_ci is True


def test_varima_forecast_output_contract():
    df = _sample_multivariate_df(45)
    ctx = ForecastContext(
        train_df=df.iloc[:39],
        horizon=6,
        exog_cols=["temp_mean_c", "relative_humidity_pct"],
        future_df=df.iloc[39:],
    )
    out = run_registered_model("varima", ctx)
    validated = validate_forecast_output(out, expected_horizon=6, clip_non_negative=True)

    assert validated["predictions"].shape == (6,)
    assert np.all(np.isfinite(validated["predictions"]))
    assert np.all(validated["predictions"] >= 0.0)
    assert "diagnostics" in validated["metadata"]
    diag = validated["metadata"]["diagnostics"]
    assert diag["order"] == (1, 1, 1)
    assert diag["k_endogenous"] >= 2


def test_varima_direct_function():
    df = _sample_multivariate_df(40)
    preds, lb, ub, model, diag = forecast_varima(
        train_df=df,
        horizon=5,
        target_col="documented_occurrence_count",
        exog_cols=["temp_mean_c", "relative_humidity_pct"],
        p=1,
        d=1,
        q=1,
    )
    assert len(preds) == 5
    assert lb is not None and ub is not None
    assert len(lb) == 5 and len(ub) == 5
    assert np.all(preds >= 0.0)
    assert np.all(lb <= ub + 1e-6)
