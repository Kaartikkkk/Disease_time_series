"""Unit tests for evaluation metrics (MAE, RMSE, sMAPE, R2, MASE) on known vectors."""

import numpy as np
import pytest
from src.evaluation.metrics import calculate_metrics, compute_smape, compute_mase


def test_metrics_exact_known_values():
    """Verify metrics return exact mathematical values on known test vectors."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])
    # Errors: [2, -2, 3, -4]
    # Absolute errors: [2, 2, 3, 4] -> Mean = 11 / 4 = 2.75
    # Squared errors: [4, 4, 9, 16] -> Mean = 33 / 4 = 8.25 -> RMSE = sqrt(8.25) ~ 2.8723

    history = np.array([5.0, 10.0, 15.0, 20.0])  # diffs are all 5.0 -> scale = 5.0
    scores = calculate_metrics(y_true, y_pred, y_train_history=history, seasonal_period=1)

    assert np.isclose(scores.mae, 2.75)
    assert np.isclose(scores.mse, 8.25)
    assert np.isclose(scores.rmse, np.sqrt(8.25), atol=1e-4)
    # MAPE = (2/10 + 2/20 + 3/30 + 4/40) / 4 * 100 = (0.2 + 0.1 + 0.1 + 0.1)/4 * 100 = 0.5/4 * 100 = 12.5%
    assert np.isclose(scores.mape, 12.5)
    # MASE = 2.75 / 5.0 = 0.55
    assert np.isclose(scores.mase, 0.55)


def test_smape_zero_handling():
    """Verify sMAPE handles zero and near-zero values safely without throwing ZeroDivisionError."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    val = compute_smape(y_true, y_pred)
    assert np.isclose(val, 0.0)

    # Extreme discrepancy
    y_t2 = np.array([0.0, 10.0])
    y_p2 = np.array([10.0, 0.0])
    val2 = compute_smape(y_t2, y_p2)
    assert 0.0 <= val2 <= 200.0


def test_mape_calculation():
    """Verify MAPE calculates accurate percentage error and handles zeros gracefully."""
    from src.evaluation.metrics import compute_mape
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 180.0, 330.0])
    # Relative errors: 10/100=0.1, 20/200=0.1, 30/300=0.1 -> Mean = 0.1 * 100 = 10.0%
    val = compute_mape(y_true, y_pred)
    assert np.isclose(val, 10.0)

    # With zero present in y_true, zero is masked
    y_t2 = np.array([0.0, 100.0])
    y_p2 = np.array([0.0, 110.0])
    val2 = compute_mape(y_t2, y_p2)
    assert np.isclose(val2, 10.0)
