"""Unit Tests for Date Parsing, Deduplication, and Spatial Preprocessing."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    assign_agro_climatic_region,
    deduplicate_records,
    parse_and_validate_dates,
)


def test_parse_valid_iso_dates():
    """Test parsing standard ISO 8601 date strings."""
    df = pd.DataFrame({
        "eventDate": ["2023-08-15T10:30:00", "2022-11-05", "2021-04-20/2021-04-25"],
    })
    res = parse_and_validate_dates(df)
    assert res["is_date_valid"].all()
    assert res["event_date_parsed"].iloc[0] == pd.Timestamp("2023-08-15")
    assert res["event_date_parsed"].iloc[1] == pd.Timestamp("2022-11-05")
    assert res["event_date_parsed"].iloc[2] == pd.Timestamp("2021-04-20")


def test_parse_year_month_fallback():
    """Test fallback date construction from separate year and month columns."""
    df = pd.DataFrame({
        "eventDate": [None, np.nan, "invalid-date-string"],
        "year": [2023, 2024, 2025],
        "month": [5, 10, 12],
    })
    res = parse_and_validate_dates(df)
    assert res["is_date_valid"].all()
    assert res["event_date_parsed"].iloc[0] == pd.Timestamp("2023-05-01")
    assert res["event_date_parsed"].iloc[1] == pd.Timestamp("2024-10-01")
    assert res["event_date_parsed"].iloc[2] == pd.Timestamp("2025-12-01")


def test_reject_future_or_impossible_dates():
    """Test that future dates and impossible calendar dates are flagged invalid."""
    df = pd.DataFrame({
        "eventDate": ["2099-01-01", "2023-02-31", "not-a-date"],
        "year": [2099, None, None],
        "month": [1, None, None],
    })
    res = parse_and_validate_dates(df)
    # 2099 is in the distant future -> rejected
    # 2023-02-31 is invalid -> rejected
    # not-a-date -> rejected
    assert not res["is_date_valid"].any()
    assert res["event_date_parsed"].isna().all()


def test_deduplicate_occurrences_by_gbif_id():
    """Test deduplication using unique GBIF occurrence IDs."""
    df = pd.DataFrame({
        "gbifID": [1001, 1002, 1001, 1003, 1002],
        "event_date_parsed": pd.to_datetime(["2023-01-01"] * 5),
    })
    clean_df, dropped = deduplicate_records(df)
    assert len(clean_df) == 3
    assert dropped == 2
    assert set(clean_df["gbifID"]) == {1001, 1002, 1003}


def test_deduplicate_by_spatiotemporal_hash():
    """Test deduplication when gbifID is absent but coords and dates match."""
    df = pd.DataFrame({
        "scientificName": ["Pyricularia oryzae", "Pyricularia oryzae", "Pyricularia oryzae"],
        "event_date_parsed": pd.to_datetime(["2023-09-01", "2023-09-01", "2023-10-01"]),
        "decimalLatitude": [20.5, 20.5, 20.5],
        "decimalLongitude": [85.5, 85.5, 85.5],
    })
    clean_df, dropped = deduplicate_records(df)
    assert len(clean_df) == 2
    assert dropped == 1


def test_assign_agro_climatic_regions():
    """Test regional zone assignment for Indian coordinates."""
    # Punjab / Indo-Gangetic Plains (lat=30.7, lon=76.8)
    assert assign_agro_climatic_region(30.7, 76.8, None) == "punjab_haryana"
    # Eastern Rice Belt (Cuttack, lat=20.5, lon=85.9)
    assert assign_agro_climatic_region(20.5, 85.9, None) == "odisha_bengal"
    # Southern Zone (lat=11.1, lon=78.7)
    assert assign_agro_climatic_region(11.1, 78.7, None) == "tamil_nadu"
    # Western/Central Semi-Arid (lat=19.8, lon=75.7)
    assert assign_agro_climatic_region(19.8, 75.7, None) == "maharashtra"
    # Missing coords but state string
    assert assign_agro_climatic_region(None, None, "Punjab, India") == "punjab_haryana"
    assert assign_agro_climatic_region(None, None, "West Bengal") == "odisha_bengal"
    # Out of bounds fallback
    assert assign_agro_climatic_region(50.0, 10.0, None) == "national_composite"
