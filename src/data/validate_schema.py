"""Canonical Data Model specification and schema validation."""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

CANONICAL_COLUMNS = [
    "date",
    "crop",
    "disease",
    "location_id",
    "latitude",
    "longitude",
    "cases_or_affected",
    "inspected_or_total",
    "incidence_rate",
    "severity",
    "temperature",
    "humidity",
    "rainfall",
    "source",
    "is_synthetic",
]

CORE_REQUIRED_COLUMNS = ["date", "crop", "disease", "incidence_rate", "source"]


def validate_canonical_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that a DataFrame conforms to the canonical plant disease time series schema."""
    errors = []
    if df.empty:
        return False, ["DataFrame is completely empty."]

    # Check required core columns
    missing_cols = [c for c in CORE_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required canonical columns: {missing_cols}")

    if "date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            try:
                pd.to_datetime(df["date"])
            except Exception as e:
                errors.append(f"Column 'date' cannot be converted to datetime: {e}")

    if "incidence_rate" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["incidence_rate"]):
            errors.append("Column 'incidence_rate' must be numeric.")
        else:
            # Check for finite values among non-nulls
            valid_vals = df["incidence_rate"].dropna()
            if not np.isfinite(valid_vals).all():
                errors.append("Column 'incidence_rate' contains infinite values.")

    # Check that is_synthetic is explicitly boolean or integer flag
    if "is_synthetic" in df.columns:
        if not pd.api.types.is_bool_dtype(df["is_synthetic"]) and not pd.api.types.is_numeric_dtype(df["is_synthetic"]):
            errors.append("Column 'is_synthetic' must be boolean or integer flag (0 or 1).")

    return len(errors) == 0, errors


def ensure_canonical_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Format and reorder available columns according to the canonical schema.

    Missing optional columns are filled with NaN or appropriate defaults.
    """
    df_clean = df.copy()

    # Ensure date is datetime
    if "date" in df_clean.columns and not pd.api.types.is_datetime64_any_dtype(df_clean["date"]):
        df_clean["date"] = pd.to_datetime(df_clean["date"])

    # Ensure chronological order
    if "date" in df_clean.columns:
        df_clean = df_clean.sort_values(by="date").reset_index(drop=True)

    # Ensure is_synthetic flag exists
    if "is_synthetic" not in df_clean.columns:
        df_clean["is_synthetic"] = False

    # Standardize column presence
    for col in CANONICAL_COLUMNS:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    # Retain canonical column ordering followed by any dataset-specific extras
    extra_cols = [c for c in df_clean.columns if c not in CANONICAL_COLUMNS]
    ordered_cols = CANONICAL_COLUMNS + extra_cols
    return df_clean[ordered_cols]
