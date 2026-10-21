"""Unit tests for outbreak risk categorization, thresholds, and probability calibration."""

import numpy as np
import pandas as pd
import pytest
from src.risk.thresholds import compute_empirical_outbreak_thresholds
from src.risk.risk_assessment import evaluate_outbreak_risk


def test_empirical_thresholds_strictly_ordered():
    """Verify data-driven thresholds satisfy baseline <= moderate < high < very_high."""
    history = pd.Series([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0, 80.0])
    thresh = compute_empirical_outbreak_thresholds(history)

    assert thresh.baseline_level <= thresh.moderate_threshold
    assert thresh.moderate_threshold < thresh.high_alert_threshold
    assert thresh.high_alert_threshold < thresh.very_high_threshold


def test_outbreak_risk_probabilities_bounded():
    """Verify outbreak probability is strictly bounded between 0.0% and 100.0%."""
    history = pd.Series(np.random.normal(20, 5, 100).clip(0, 100))
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    point_forecast = np.array([5.0, 15.0, 25.0, 40.0, 60.0] + [20.0] * 9)
    lower_bound = np.maximum(0.0, point_forecast - 5.0)
    upper_bound = point_forecast + 5.0

    report = evaluate_outbreak_risk(
        forecast_dates=dates,
        point_forecast=point_forecast,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        train_history=history,
    )

    assert report.peak_risk_level in ["Low", "Moderate", "High", "Very High"]
    for assessment in report.daily_assessments:
        assert 0.0 <= assessment.outbreak_probability_pct <= 100.0
        assert assessment.risk_level in ["Low", "Moderate", "High", "Very High"]
        assert len(assessment.recommended_action) > 5
