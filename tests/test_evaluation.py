"""Unit Tests for Forecast Evaluation Metrics and Interval Coverage."""

import numpy as np
import pytest

from src.evaluation import build_model_comparison_table, compute_forecast_metrics


def test_basic_error_metrics():
    """Verify MAE, RMSE, and bias on deterministic inputs."""
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    # Predicted is systematically 2 units higher
    predicted = np.array([12.0, 22.0, 32.0, 42.0])

    res = compute_forecast_metrics(actual, predicted, model_name="TestModel")
    assert res["mae"] == 2.0
    assert res["rmse"] == 2.0
    assert res["bias"] == 2.0  # Over-prediction by +2


def test_zero_denominator_safe_mape():
    """Verify that MAPE handles actual zero values safely without crashing with ZeroDivisionError."""
    actual_with_zeros = np.array([0.0, 0.0, 10.0, 20.0])
    predicted = np.array([1.0, 2.0, 12.0, 18.0])

    res = compute_forecast_metrics(actual_with_zeros, predicted, model_name="ZeroTest")
    # For actual != 0 (10 and 20):
    # |12-10|/10 = 2/10 = 0.20
    # |18-20|/20 = 2/20 = 0.10
    # Mean = 0.15 * 100 = 15.0%
    assert res["mape"] == 15.0

    # If all actuals are zero:
    all_zeros = np.array([0.0, 0.0, 0.0])
    res_zeros = compute_forecast_metrics(all_zeros, np.array([1.0, 2.0, 3.0]))
    assert res_zeros["mape"] is None


def test_smape_bounded_range():
    """Verify that sMAPE remains strictly within [0, 200%]."""
    actual = np.array([0.0, 10.0, 50.0])
    predicted = np.array([5.0, 0.0, 50.0])

    res = compute_forecast_metrics(actual, predicted)
    assert 0.0 <= res["smape"] <= 200.0


def test_wape_calculation():
    """Verify Weighted Absolute Percentage Error (WAPE)."""
    actual = np.array([10.0, 20.0, 30.0])  # sum = 60
    predicted = np.array([12.0, 18.0, 36.0])  # errors = 2, 2, 6; sum = 10
    # WAPE = 10 / 60 * 100 = 16.67%
    res = compute_forecast_metrics(actual, predicted)
    assert pytest.approx(res["wape"], 0.01) == 16.67


def test_prediction_interval_coverage():
    """Verify PICP coverage percentage calculation."""
    actual = np.array([10.0, 15.0, 20.0, 25.0])
    lower = np.array([8.0, 12.0, 18.0, 28.0])  # 25 is outside (28 > 25)
    upper = np.array([12.0, 18.0, 22.0, 35.0])

    # Values within: index 0 (10 in [8,12]), index 1 (15 in [12,18]), index 2 (20 in [18,22])
    # index 3 is not within [28, 35]
    # Coverage = 3 / 4 * 100 = 75.0%
    res = compute_forecast_metrics(actual, actual, lower_bound=lower, upper_bound=upper)
    assert res["picp"] == 75.0


def test_model_comparison_table_formatting():
    """Test generating sorted comparison DataFrame."""
    m1 = compute_forecast_metrics(np.array([10, 20]), np.array([15, 25]), model_name="Model_A")  # RMSE = 5
    m2 = compute_forecast_metrics(np.array([10, 20]), np.array([11, 21]), model_name="Model_B")  # RMSE = 1
    table = build_model_comparison_table([m1, m2])
    assert len(table) == 2
    # Best model (Model_B with RMSE 1) should be at top
    assert table.iloc[0]["Model"] == "Model_B"
    assert table.iloc[1]["Model"] == "Model_A"
