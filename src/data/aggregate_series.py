"""Temporal aggregation pipeline for Daily, Weekly, and Monthly forecasting modes."""

from typing import Optional
import numpy as np
import pandas as pd
from src.utils.logging_config import logger


def aggregate_time_series(
    df: pd.DataFrame,
    target_freq: str = "W-SUN",  # 'D' (Daily), 'W-SUN' (Weekly), 'MS' (Monthly Start)
    target_col: str = "incidence_rate",
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    location_id: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate irregular or daily observations into regular temporal intervals.

    Documented Aggregation Strategy:
    - Incidence rates, severities, temperature, humidity: arithmetic mean (or denominator-weighted if available).
    - Counts (cases, rainfall): cumulative sum.
    - Metadata (crop, disease, location, source, is_synthetic): first observation.
    - Strict week boundary: Sunday week end ('W-SUN').
    - Strict month boundary: 1st of month ('MS').
    """
    df_filtered = df.copy()

    if crop and "crop" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["crop"] == crop]
    if disease and "disease" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["disease"] == disease]
    if location_id and "location_id" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["location_id"] == location_id]

    if df_filtered.empty:
        logger.warning("Filtered DataFrame for aggregation is empty.")
        return pd.DataFrame()

    df_filtered["date"] = pd.to_datetime(df_filtered["date"])
    df_filtered = df_filtered.sort_values("date")

    # If daily target frequency and series is already daily/irregular observations,
    # group by exact calendar day
    if target_freq.upper() in ["D", "DAILY"]:
        grouper = pd.Grouper(key="date", freq="D")
    elif target_freq.upper() in ["W", "WEEKLY", "W-SUN"]:
        grouper = pd.Grouper(key="date", freq="W-SUN")
    elif target_freq.upper() in ["M", "MONTHLY", "MS"]:
        grouper = pd.Grouper(key="date", freq="MS")
    else:
        grouper = pd.Grouper(key="date", freq=target_freq)

    # Build aggregation dictionary based on available columns
    agg_dict = {}

    # Target & rates -> mean
    for col in [target_col, "incidence_rate", "severity", "temperature", "humidity"]:
        if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col]):
            agg_dict[col] = "mean"

    # Cumulative quantities -> sum
    for col in ["cases_or_affected", "inspected_or_total", "rainfall"]:
        if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col]):
            agg_dict[col] = "sum"

    # Metadata columns -> first
    for col in ["crop", "disease", "location_id", "source", "is_synthetic"]:
        if col in df_filtered.columns:
            agg_dict[col] = "first"

    df_agg = df_filtered.groupby(grouper).agg(agg_dict).reset_index()

    # If numerator and denominator exist, recompute weighted incidence
    if "cases_or_affected" in df_agg.columns and "inspected_or_total" in df_agg.columns:
        valid_denom = df_agg["inspected_or_total"] > 0
        df_agg.loc[valid_denom, "incidence_rate"] = (
            df_agg.loc[valid_denom, "cases_or_affected"] / df_agg.loc[valid_denom, "inspected_or_total"]
        ) * 100.0

    # Drop unobserved aggregated intervals where target is NaN (no reports occurred)
    # Do NOT fill with zero automatically!
    if target_col in df_agg.columns:
        df_agg = df_agg.dropna(subset=[target_col]).reset_index(drop=True)

    logger.info(
        f"Aggregated {len(df_filtered)} records to {len(df_agg)} intervals at frequency '{target_freq}'."
    )
    return df_agg
