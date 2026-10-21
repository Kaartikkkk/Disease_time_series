"""Unit tests for data parsing, deduplication, missingness, aggregation, and synthetic labeling."""

import numpy as np
import pandas as pd
import pytest
from src.data.clean_data import clean_plant_disease_data
from src.data.aggregate_series import aggregate_time_series
from src.data.validate_schema import validate_canonical_schema, ensure_canonical_dataframe
from src.data.fetch_data import generate_synthetic_benchmark


def test_date_parsing_and_sorting():
    """Verify dates are parsed into datetime and chronologically sorted."""
    raw_data = pd.DataFrame({
        "date": ["2023-05-10", "2023-05-01", "2023-05-05"],
        "crop": ["Grapevine", "Grapevine", "Grapevine"],
        "disease": ["Downy Mildew", "Downy Mildew", "Downy Mildew"],
        "incidence_rate": [12.0, 5.0, 8.5],
        "source": ["test", "test", "test"],
    })
    cleaned, report = clean_plant_disease_data(raw_data)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert cleaned["date"].is_monotonic_increasing
    assert str(cleaned["date"].iloc[0].date()) == "2023-05-01"
    assert str(cleaned["date"].iloc[-1].date()) == "2023-05-10"


def test_duplicate_observations_handling():
    """Verify duplicate observations on the same date/crop/location are aggregated."""
    raw_data = pd.DataFrame({
        "date": ["2023-06-01", "2023-06-01", "2023-06-02"],
        "crop": ["Wheat", "Wheat", "Wheat"],
        "disease": ["Blast", "Blast", "Blast"],
        "location_id": ["SiteA", "SiteA", "SiteA"],
        "incidence_rate": [10.0, 20.0, 15.0],
        "source": ["test", "test", "test"],
    })
    cleaned, report = clean_plant_disease_data(raw_data, aggregate_duplicates=True)
    assert len(cleaned) == 2
    assert report.duplicate_rows_detected == 1
    # Average of 10.0 and 20.0 should be 15.0
    first_row_val = cleaned[cleaned["date"] == "2023-06-01"]["incidence_rate"].iloc[0]
    assert np.isclose(first_row_val, 15.0)


def test_missing_disease_reports_not_converted_to_zero():
    """Verify unobserved target reports are dropped or preserved as NaN, NEVER naively zero-filled."""
    raw_data = pd.DataFrame({
        "date": ["2023-07-01", "2023-07-02", "2023-07-03"],
        "crop": ["Potato", "Potato", "Potato"],
        "disease": ["Late Blight", "Late Blight", "Late Blight"],
        "incidence_rate": [10.0, np.nan, 25.0],
        "source": ["test", "test", "test"],
    })
    # When dropping unobserved targets, row 2 is dropped rather than converted to 0.0
    cleaned_drop, _ = clean_plant_disease_data(raw_data, drop_unobserved_targets=True)
    assert len(cleaned_drop) == 2
    assert (cleaned_drop["incidence_rate"] == 0.0).sum() == 0

    # When not dropping unobserved targets, NaN is strictly preserved
    cleaned_keep, _ = clean_plant_disease_data(raw_data, drop_unobserved_targets=False)
    assert len(cleaned_keep) == 3
    assert np.isnan(cleaned_keep.loc[1, "incidence_rate"])


def test_aggregation_daily_weekly_monthly():
    """Verify aggregation produces expected intervals without silent zero-fill."""
    # Starting on Monday (2023-01-02) gives exactly two 7-day Sunday-ended weeks
    dates = pd.date_range("2023-01-02", periods=14, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "crop": "Potato",
        "disease": "Late Blight",
        "incidence_rate": np.arange(14, dtype=float),
        "rainfall": np.ones(14, dtype=float) * 5.0,
        "source": "test",
    })

    # Weekly Sunday aggregation
    df_weekly = aggregate_time_series(df, target_freq="W-SUN")
    assert len(df_weekly) == 2  # exactly 2 weeks
    assert "incidence_rate" in df_weekly.columns
    # Rainfall is cumulative (sum)
    assert df_weekly["rainfall"].iloc[0] == 35.0  # 7 * 5.0

    # Monthly aggregation
    dates_multi_month = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    df_long = pd.DataFrame({
        "date": dates_multi_month,
        "crop": "Potato",
        "disease": "Late Blight",
        "incidence_rate": np.ones(len(dates_multi_month)),
        "source": "test",
    })
    df_monthly = aggregate_time_series(df_long, target_freq="MS")
    assert len(df_monthly) == 3


def test_synthetic_demonstration_data_is_labeled():
    """Verify synthetic demonstration benchmark has is_synthetic=True and explicit metadata."""
    df_synth = generate_synthetic_benchmark(start_date="2023-01-01", end_date="2023-03-31")
    assert "is_synthetic" in df_synth.columns
    assert df_synth["is_synthetic"].all() == True
    assert "SYNTHETIC" in df_synth["source"].iloc[0]
