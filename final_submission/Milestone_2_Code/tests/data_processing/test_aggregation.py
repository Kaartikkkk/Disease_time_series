"""Unit Tests for Monthly Time Series Aggregation and Weather Feature Alignment."""

import numpy as np
import pandas as pd
import pytest

from src.monthly_aggregation import build_continuous_monthly_series


def test_continuous_monthly_timeline():
    """Verify that aggregation produces a continuous Month-Start (MS) timeline with no missing dates."""
    clean_records = pd.DataFrame({
        "event_date_parsed": pd.to_datetime(["2020-03-15", "2020-03-20", "2020-07-10", "2021-01-05"]),
    })
    pathogen_info = {"scientific_name": "Pyricularia oryzae", "common_name": "Rice Blast", "host_crop": "Rice"}

    ts_df = build_continuous_monthly_series(
        df_clean=clean_records,
        pathogen_slug="pyricularia_oryzae",
        pathogen_info=pathogen_info,
        start_year=2020,
        end_year=2021,
    )

    # 2 years = 24 continuous monthly stamps
    assert len(ts_df) == 24
    expected_dates = pd.date_range("2020-01-01", "2021-12-01", freq="MS")
    pd.testing.assert_index_equal(pd.DatetimeIndex(ts_df["date"]), expected_dates, check_names=False)


def test_zero_versus_missing_observations():
    """Test that active surveillance months with 0 events have count=0, and unobserved periods are flagged."""
    clean_records = pd.DataFrame({
        "event_date_parsed": pd.to_datetime(["2020-05-15", "2020-08-20"]),
    })
    pathogen_info = {"scientific_name": "Bipolaris oryzae", "common_name": "Brown Spot", "host_crop": "Rice"}

    ts_df = build_continuous_monthly_series(
        df_clean=clean_records,
        pathogen_slug="bipolaris_oryzae",
        pathogen_info=pathogen_info,
        start_year=2020,
        end_year=2020,
    )

    # May 2020 count = 1
    may_row = ts_df[ts_df["date"] == "2020-05-01"].iloc[0]
    assert may_row["documented_occurrence_count"] == 1
    assert not may_row["is_missing"]

    # August 2020 count = 1
    aug_row = ts_df[ts_df["date"] == "2020-08-01"].iloc[0]
    assert aug_row["documented_occurrence_count"] == 1
    assert not aug_row["is_missing"]

    # June and July are within active span (May to Aug), so count = 0
    jun_row = ts_df[ts_df["date"] == "2020-06-01"].iloc[0]
    assert jun_row["documented_occurrence_count"] == 0
    assert not jun_row["is_missing"]

    # January 2020 is outside the active window (before May)
    jan_row = ts_df[ts_df["date"] == "2020-01-01"].iloc[0]
    assert jan_row["is_missing"] == True


def test_weather_join_and_lag_leakage_prevention():
    """Verify that lagged weather features are strictly shifted and contain no future leakage."""
    clean_records = pd.DataFrame({
        "event_date_parsed": pd.to_datetime(["2020-02-10", "2020-04-10"]),
    })
    pathogen_info = {"scientific_name": "Ralstonia solanacearum", "common_name": "Bacterial Wilt", "host_crop": "Potato"}

    weather_data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", "2020-12-01", freq="MS"),
        "temp_mean_c": [15.0, 18.0, 22.0, 28.0, 32.0, 34.0, 31.0, 29.0, 28.0, 25.0, 20.0, 16.0],
        "precipitation_monthly_mm": [10, 15, 20, 30, 50, 120, 250, 220, 180, 60, 20, 5],
        "relative_humidity_pct": [60, 55, 50, 45, 50, 70, 85, 88, 80, 70, 65, 62],
    })

    ts_df = build_continuous_monthly_series(
        df_clean=clean_records,
        pathogen_slug="ralstonia_solanacearum",
        pathogen_info=pathogen_info,
        weather_df=weather_data,
        start_year=2020,
        end_year=2020,
    )

    # First month (Jan) lag1 must be NaN (no past data available at t=0)
    assert np.isnan(ts_df.loc[0, "temp_mean_c_lag1"])

    # In February (index 1), temp_mean_c_lag1 must equal January's temp (15.0)
    assert ts_df.loc[1, "temp_mean_c_lag1"] == 15.0
    # In March (index 2), temp_mean_c_lag1 must equal February's temp (18.0)
    assert ts_df.loc[2, "temp_mean_c_lag1"] == 18.0
    # In March (index 2), temp_mean_c_lag2 must equal January's temp (15.0)
    assert ts_df.loc[2, "temp_mean_c_lag2"] == 15.0
