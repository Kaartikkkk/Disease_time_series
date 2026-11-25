"""Unit and integration tests for PyTorch Time Series Transformer (PatchTST / Attention)."""

import numpy as np
import pandas as pd
import pytest

from src.model_registry import get_model_spec, run_registered_model
from src.models.base import ForecastContext, validate_forecast_output
from src.models.deep_learning.transformer_model import (
    HAS_TORCH,
    create_transformer_sequences,
    forecast_transformer,
)


def _sample_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    t = np.arange(n)
    counts = np.maximum(0, np.round(5.0 + 4.0 * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.5, n))).astype(int)
    return pd.DataFrame({"date": dates, "documented_occurrence_count": counts})


def test_transformer_registration_and_availability():
    spec = get_model_spec("transformer")
    assert spec.display_name == "Temporal Transformer (PatchTST / Attention)"
    assert spec.category == "Deep Learning"
    assert spec.available is HAS_TORCH
    assert spec.supports_exog is False
    assert spec.has_ci is True


def test_create_transformer_sequences_chronological():
    vals = np.arange(16, dtype=float)
    X, y = create_transformer_sequences(vals, lookback=12)
    assert X.shape == (4, 12, 1)
    assert y.shape == (4, 1)
    assert np.allclose(X[0, :, 0], np.arange(0, 12))
    assert y[0, 0] == 12.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_transformer_forecast_output_contract():
    df = _sample_df(50)
    ctx = ForecastContext(train_df=df.iloc[:44], horizon=6, future_df=df.iloc[44:])
    out = run_registered_model("transformer", ctx)
    validated = validate_forecast_output(out, expected_horizon=6, clip_non_negative=True)

    assert validated["predictions"].shape == (6,)
    assert np.all(np.isfinite(validated["predictions"]))
    assert np.all(validated["predictions"] >= 0.0)
    assert "diagnostics" in validated["metadata"]
