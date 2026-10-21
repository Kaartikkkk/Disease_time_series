"""Data cleaning, normalization, duplicate resolution, and data quality profiling."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.data.validate_schema import ensure_canonical_dataframe
from src.utils.logging_config import logger


@dataclass
class DataQualityReport:
    total_records: int
    date_start: str
    date_end: str
    unique_crops: int
    unique_diseases: int
    duplicate_rows_detected: int
    missing_target_count: int
    missing_target_pct: float
    zero_target_count: int
    negative_values_detected: int
    inferred_frequency: str
    irregular_intervals: bool
    is_synthetic: bool
    summary_notes: str


def clean_plant_disease_data(
    df: pd.DataFrame,
    target_col: str = "incidence_rate",
    drop_unobserved_targets: bool = True,
    aggregate_duplicates: bool = True,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Clean and standardize a raw plant disease dataset into an analysis-ready canonical DataFrame.

    Rules strictly followed:
    1. Chronological sorting and date validation.
    2. Explicit distinction between missing reports (NaN) and confirmed zeros (0.0). Missing observations are NOT naively zero-filled.
    3. Detection and resolution of duplicates (averaging multiple measurements on same day/crop/disease/location).
    4. Negative values bounded to 0.0.
    5. Unit standardization (proportions 0-1 converted or labeled as percentage).
    6. Quality profiling report returned.
    """
    if df.empty:
        raise ValueError("Cannot clean an empty DataFrame.")

    df_cleaned = df.copy()

    # 1. Parse date and sort
    df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], errors="coerce")
    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 2. Duplicate detection
    group_keys = ["date", "crop", "disease", "location_id"]
    available_keys = [k for k in group_keys if k in df_cleaned.columns]
    duplicate_count = df_cleaned.duplicated(subset=available_keys).sum()

    if duplicate_count > 0 and aggregate_duplicates:
        logger.info(f"Resolving {duplicate_count} duplicate timestamp records via mean aggregation.")
        # Group numeric features by mean, keep first metadata
        num_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        non_num_cols = [c for c in df_cleaned.columns if c not in num_cols and c not in available_keys]

        agg_dict = {col: "mean" for col in num_cols}
        for c in non_num_cols:
            agg_dict[c] = "first"

        df_cleaned = df_cleaned.groupby(available_keys, as_index=False).agg(agg_dict)

    # 3. Target sanitization
    if target_col in df_cleaned.columns:
        df_cleaned[target_col] = pd.to_numeric(df_cleaned[target_col], errors="coerce")
        missing_target = df_cleaned[target_col].isna().sum()
        zero_target = (df_cleaned[target_col] == 0.0).sum()
        negative_count = (df_cleaned[target_col] < 0.0).sum()

        # Enforce non-negative natural boundary for biological disease quantities
        if negative_count > 0:
            logger.warning(f"Found {negative_count} negative target values; bounding to 0.0.")
            df_cleaned[target_col] = df_cleaned[target_col].clip(lower=0.0)

        # Drop unobserved rows if requested, but DO NOT fill missing with zero
        if drop_unobserved_targets and missing_target > 0:
            logger.info(f"Dropping {missing_target} records with unobserved target values.")
            df_cleaned = df_cleaned.dropna(subset=[target_col]).reset_index(drop=True)
    else:
        missing_target = 0
        zero_target = 0
        negative_count = 0

    # 4. Standardize text normalization
    for col in ["crop", "disease", "location_id"]:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()

    # 5. Detect observation frequency & regularity
    inferred_freq = "Irregular"
    irregular = True
    if len(df_cleaned) >= 3 and "date" in df_cleaned.columns:
        unique_dates = df_cleaned["date"].drop_duplicates().sort_values()
        date_diffs = unique_dates.diff().dt.days.dropna()
        if not date_diffs.empty:
            median_diff = date_diffs.median()
            std_diff = date_diffs.std()
            if std_diff < 0.1:
                irregular = False
                if median_diff == 1:
                    inferred_freq = "Daily (Regular)"
                elif median_diff == 7:
                    inferred_freq = "Weekly (Regular)"
                elif 28 <= median_diff <= 31:
                    inferred_freq = "Monthly (Regular)"
                elif 360 <= median_diff <= 366:
                    inferred_freq = "Annual (Regular)"
                else:
                    inferred_freq = f"Periodic ({int(median_diff)} days)"
            else:
                irregular = True
                inferred_freq = f"Irregular (median step: {int(median_diff)} days, std: {std_diff:.1f})"

    # 6. Quality report
    start_str = str(df_cleaned["date"].min().date()) if not df_cleaned.empty else "N/A"
    end_str = str(df_cleaned["date"].max().date()) if not df_cleaned.empty else "N/A"
    is_synth = bool(df_cleaned["is_synthetic"].any()) if "is_synthetic" in df_cleaned.columns else False

    report = DataQualityReport(
        total_records=len(df_cleaned),
        date_start=start_str,
        date_end=end_str,
        unique_crops=df_cleaned["crop"].nunique() if "crop" in df_cleaned.columns else 0,
        unique_diseases=df_cleaned["disease"].nunique() if "disease" in df_cleaned.columns else 0,
        duplicate_rows_detected=int(duplicate_count),
        missing_target_count=int(missing_target),
        missing_target_pct=round(float(missing_target / max(1, initial_count)) * 100, 2),
        zero_target_count=int(zero_target),
        negative_values_detected=int(negative_count),
        inferred_frequency=inferred_freq,
        irregular_intervals=irregular,
        is_synthetic=is_synth,
        summary_notes="Cleaned and validated against canonical schema without synthetic zero-inflation.",
    )

    canonical_df = ensure_canonical_dataframe(df_cleaned)
    return canonical_df, report
