"""Unit tests for programmatic forecast insights, peak month detection, and activity levels."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.forecast_insights import (
    classify_predicted_activity,
    compute_calendar_month_baselines,
    detect_forecast_trend,
    generate_monthly_disease_outlook,
    identify_peak_forecast_month,
)


@pytest.fixture
def sample_historical_df():
    """Create 3 years of synthetic monthly data with distinct October seasonality."""
    dates = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
    records = []
    for d in dates:
        # October (month 10) has high values, others low
        val = 15.0 if d.month == 10 else 2.0
        records.append({
            "date": d,
            "documented_occurrence_count": val,
            "is_missing": False,
        })
    return pd.DataFrame(records)


@pytest.fixture
def sample_forecast_df():
    """Create 6-month forward forecast dataframe."""
    dates = pd.date_range("2023-01-01", periods=6, freq="MS")
    # Jan, Feb, Mar, Apr, May, Jun
    # Values: 1.0, 3.0, 5.0, 4.8, 2.0, 1.0
    forecasts = [1.0, 3.0, 5.0, 4.8, 2.0, 1.0]
    return pd.DataFrame({
        "date": dates,
        "forecast": forecasts,
        "lower_80": [max(0.0, f - 1.0) for f in forecasts],
        "upper_80": [f + 1.5 for f in forecasts],
        "lower_95": [max(0.0, f - 2.0) for f in forecasts],
        "upper_95": [f + 2.5 for f in forecasts],
    })


def test_compute_calendar_month_baselines(sample_historical_df):
    baselines = compute_calendar_month_baselines(sample_historical_df)
    assert len(baselines) == 12
    assert baselines[10]["mean"] == 15.0
    assert baselines[1]["mean"] == 2.0
    assert baselines[10]["count"] == 3


def test_classify_predicted_activity_insufficient_evidence():
    # Empty baselines
    cat, col, note = classify_predicted_activity(10.0, 5, {})
    assert cat == "Insufficient evidence"
    assert col == "#6B7280"


def test_classify_predicted_activity_levels(sample_historical_df):
    baselines = compute_calendar_month_baselines(sample_historical_df)
    # Month 10: mean=15, std=0
    # A forecast of 20 should be above typical
    cat_high, _, _ = classify_predicted_activity(20.0, 10, baselines)
    assert "above typical" in cat_high.lower()

    # A forecast of 15 should be within typical
    cat_mid, _, _ = classify_predicted_activity(15.0, 10, baselines)
    assert cat_mid == "Within typical activity"

    # A forecast of 5 should be below typical
    cat_low, _, _ = classify_predicted_activity(5.0, 10, baselines)
    assert cat_low == "Below typical activity"


def test_detect_forecast_trend():
    label_inc, arrow_inc, abs_c, pct_c = detect_forecast_trend(5.0, 2.0)
    assert label_inc == "Increasing"
    assert arrow_inc == "↗"
    assert abs_c == 3.0
    assert pct_c == 150.0

    label_dec, arrow_dec, _, _ = detect_forecast_trend(2.0, 5.0)
    assert label_dec == "Decreasing"
    assert arrow_dec == "↘"

    label_stab, arrow_stab, _, _ = detect_forecast_trend(2.1, 2.0)
    assert label_stab == "Broadly stable"
    assert arrow_stab == "→"


def test_identify_peak_forecast_month(sample_forecast_df):
    peak_info = identify_peak_forecast_month(sample_forecast_df, "Rice Blast")
    assert peak_info["peak_value"] == 5.0
    assert "March 2023" in peak_info["peak_month"]
    # Check that plateau is identified since April (4.8) is within 10% of 5.0
    assert peak_info["is_plateau"] is True
    assert "narrative" in peak_info


def test_generate_monthly_disease_outlook(sample_forecast_df, sample_historical_df):
    outlooks = generate_monthly_disease_outlook(
        sample_forecast_df,
        sample_historical_df,
        pathogen_name="Rice Blast (Pyricularia oryzae)",
    )
    assert len(outlooks) == 6
    assert outlooks[0]["month_label"] == "January 2023"
    assert "narrative" in outlooks[0]
    assert outlooks[0]["predicted_count"] == 1.0
    # Check trend detection on second month (from 1.0 to 3.0 is increasing)
    assert outlooks[1]["trend_label"] == "Increasing"
