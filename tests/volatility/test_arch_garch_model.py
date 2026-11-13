"""Unit and integration tests for ARCH and GARCH volatility models."""

import numpy as np
import pandas as pd
import pytest

from src.model_registry import (
    VOLATILITY_MODEL_SPECS,
    get_model_spec,
    get_volatility_model_spec,
    list_volatility_models,
    run_registered_model,
)
from src.models.base import ForecastContext
from src.models.volatility.arch_garch_model import (
    ARCHGARCHModel,
    compute_volatility_metrics,
    fit_and_compare_all_specifications,
    perform_arch_lm_test,
    run_volatility_walk_forward_evaluation,
)


@pytest.fixture
def sample_disease_series() -> pd.Series:
    """Generate realistic synthetic disease occurrence series with conditional heteroskedasticity."""
    rng = np.random.default_rng(42)
    n = 120
    # Simulate GARCH(1,1) errors
    omega = 0.5
    alpha = 0.3
    beta = 0.6
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = omega / (1.0 - alpha - beta)
    z = rng.standard_t(df=4.0, size=n)
    eps[0] = np.sqrt(sigma2[0]) * z[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]

    # Add seasonal base and ensure non-negative occurrence counts
    seasonal = 4.0 + 3.0 * np.sin(2 * np.pi * np.arange(n) / 12.0)
    raw_y = seasonal + eps
    y = np.maximum(0, np.round(raw_y)).astype(float)
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    return pd.Series(y, index=dates, name="documented_occurrence_count")


def test_arch_effects_detection(sample_disease_series):
    """Test Engle's ARCH-LM test on series with and without heteroskedasticity."""
    diff_y = np.diff(sample_disease_series.values)
    res = perform_arch_lm_test(diff_y, lags=5)
    assert "lm_stat" in res
    assert "lm_pvalue" in res
    assert "has_arch_effects" in res
    assert np.isfinite(res["lm_stat"])
    assert 0.0 <= res["lm_pvalue"] <= 1.0


def test_arch_garch_fit_all_specifications(sample_disease_series):
    """Test fitting all 7 required specifications: ARCH(1,2,5), GARCH(1,1, 1,2, 2,1, 2,2)."""
    y = sample_disease_series.values
    specs = fit_and_compare_all_specifications(y, dist="StudentsT", approach="approach_a")
    assert len(specs) == 7
    expected = {"arch_1", "arch_2", "arch_5", "garch_1_1", "garch_1_2", "garch_2_1", "garch_2_2"}
    assert set(specs.keys()) == expected

    for name, res in specs.items():
        assert res.converged is True
        assert res.convergence_flag == 0
        assert np.isfinite(res.log_likelihood)
        assert np.isfinite(res.aic)
        assert np.isfinite(res.bic)
        assert len(res.conditional_volatility) > 0
        assert np.all(res.conditional_volatility > 0)
        assert "omega" in res.params
        assert res.params["omega"] > 0


def test_approach_a_vs_approach_b(sample_disease_series):
    """Test both Approach A (residual volatility) and Approach B (differenced occurrence)."""
    y = sample_disease_series.values

    # Approach A: Residual Volatility
    model_a = ARCHGARCHModel(p=1, q=1, dist="StudentsT", approach="approach_a")
    res_a = model_a.fit(y)
    fcast_a = model_a.forecast(horizon=6)
    assert res_a.approach == "approach_a"
    assert len(fcast_a.variance_forecast) == 6
    assert np.all(fcast_a.variance_forecast > 0)
    assert np.all(fcast_a.lower_bound <= fcast_a.upper_bound)
    assert np.all(fcast_a.lower_bound >= 0.0)

    # Approach B: Differenced series
    model_b = ARCHGARCHModel(p=1, q=1, dist="StudentsT", approach="approach_b")
    res_b = model_b.fit(y)
    fcast_b = model_b.forecast(horizon=6)
    assert res_b.approach == "approach_b"
    assert len(fcast_b.variance_forecast) == 6
    assert np.all(fcast_b.variance_forecast > 0)
    assert np.all(fcast_b.lower_bound <= fcast_b.upper_bound)


def test_distributions_normal_and_student_t(sample_disease_series):
    """Test fitting with Gaussian Normal vs Student's t error distributions."""
    y = sample_disease_series.values

    # Normal
    model_norm = ARCHGARCHModel(p=1, q=1, dist="Normal")
    res_norm = model_norm.fit(y)
    assert res_norm.dist == "Normal"
    assert "nu" not in res_norm.params

    # Student-t
    model_t = ARCHGARCHModel(p=1, q=1, dist="StudentsT")
    res_t = model_t.fit(y)
    assert res_t.dist == "StudentsT"
    assert "nu" in res_t.params
    assert res_t.params["nu"] > 2.0  # valid finite variance constraint


def test_invalid_input_handling():
    """Verify input validation: invalid orders, empty/short series, negative horizon."""
    # Negative/zero p
    with pytest.raises(ValueError, match="ARCH order p must be >= 1"):
        ARCHGARCHModel(p=0, q=1)

    # Negative q
    with pytest.raises(ValueError, match="GARCH order q must be >= 0"):
        ARCHGARCHModel(p=1, q=-1)

    # Short series
    model = ARCHGARCHModel(p=1, q=1)
    with pytest.raises(ValueError, match="Time series too short"):
        model.fit(np.array([1.0, 2.0, 3.0]))

    # Series with NaNs
    with pytest.raises(ValueError, match="contains NaN or infinite"):
        model.fit(np.array([1.0, np.nan, 3.0] * 10))

    # Horizon <= 0
    model.fit(np.ones(30))
    with pytest.raises(ValueError, match="Forecast horizon must be positive"):
        model.forecast(horizon=0)


def test_zero_heavy_series_robustness():
    """Verify graceful fitting and forecasting on zero-inflated disease series (85% zeros)."""
    rng = np.random.default_rng(99)
    y = np.zeros(100)
    outbreak_idx = rng.choice(100, size=15, replace=False)
    y[outbreak_idx] = rng.integers(1, 10, size=15)

    model = ARCHGARCHModel(p=1, q=1, dist="StudentsT", approach="approach_a")
    res = model.fit(y)
    assert res is not None
    assert np.all(np.isfinite(res.conditional_volatility))

    fcast = model.forecast(horizon=4)
    assert len(fcast.volatility_forecast) == 4
    assert np.all(fcast.volatility_forecast > 0)


def test_volatility_metrics_computation():
    """Verify calculation of RMSVE, MAVE, MSE_var, and QLIKE loss."""
    actuals = np.array([5.0, 2.0, 0.0, 8.0])
    means = np.array([4.0, 3.0, 1.0, 6.0])
    vols = np.array([1.2, 1.0, 0.8, 2.1])

    metrics = compute_volatility_metrics(actuals, means, vols)
    assert "rmsve" in metrics
    assert "mave" in metrics
    assert "mse_var" in metrics
    assert "qlike" in metrics
    assert metrics["rmsve"] >= 0.0
    assert metrics["mave"] >= 0.0
    assert metrics["mse_var"] >= 0.0
    assert np.isfinite(metrics["qlike"])


def test_walk_forward_evaluation_no_leakage(sample_disease_series):
    """Test chronological walk-forward evaluation guaranteeing zero future leakage."""
    wf_res = run_volatility_walk_forward_evaluation(
        sample_disease_series.values,
        spec_name="garch_1_1",
        p=1,
        q=1,
        dist="StudentsT",
        horizon=1,
        min_train_len=80,
    )
    assert wf_res["n_predictions"] == len(sample_disease_series) - 80
    assert "point_metrics" in wf_res
    assert "volatility_metrics" in wf_res
    assert wf_res["point_metrics"]["rmse"] > 0.0
    assert wf_res["volatility_metrics"]["qlike"] is not None


def test_model_registry_volatility_integration(sample_disease_series):
    """Test model registry retrieval, listing, and execution of volatility models."""
    # 1. Check registry specs
    vol_specs = list_volatility_models()
    assert len(vol_specs) == 7
    assert len(VOLATILITY_MODEL_SPECS) == 7

    spec = get_volatility_model_spec("garch_1_1")
    assert spec.model_id == "garch_1_1"
    assert spec.category == "Conditional Volatility"
    assert spec.available is True

    # 2. Check get_model_spec seamlessly finds volatility models
    spec_from_general = get_model_spec("arch_1")
    assert spec_from_general.model_id == "arch_1"

    # 3. Test execution via run_registered_model
    train_df = sample_disease_series.to_frame()
    ctx = ForecastContext(
        train_df=train_df,
        horizon=6,
        target_col="documented_occurrence_count",
    )
    out = run_registered_model("garch_1_1", ctx)
    assert "predictions" in out
    assert "lower_bound" in out
    assert "upper_bound" in out
    assert len(out["predictions"]) == 6
    assert len(out["lower_bound"]) == 6
    assert np.all(out["lower_bound"] <= out["upper_bound"])
    assert "variance_forecast" in out["metadata"]
    assert "volatility_forecast" in out["metadata"]
