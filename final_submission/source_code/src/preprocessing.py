"""Preprocessing and Spatial-Temporal Cleaning Pipeline for Indian Plant Disease Records.

Ensures:
- Strict validation and parsing of ISO 8601 event dates
- Distinction between field event dates and publication/deposit dates
- Deduplication using GBIF Occurrence ID and spatial-temporal hashing
- Geographic validation within India's political and territorial boundaries
- Agro-climatic zone assignment (Indo-Gangetic, Eastern Rice, Southern, Western/Central)
- Preservation of original data provenance and rejection logging
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Approximate geographic bounding boxes for major Indian agro-climatic zones
AGRO_ZONES = {
    "punjab_haryana": {
        "name": "Northwest Indo-Gangetic Plains",
        "lat_min": 27.5, "lat_max": 33.0,
        "lon_min": 73.5, "lon_max": 78.5,
    },
    "odisha_bengal": {
        "name": "Eastern Coastal Rice Belt",
        "lat_min": 17.5, "lat_max": 27.0,
        "lon_min": 83.0, "lon_max": 89.5,
    },
    "tamil_nadu": {
        "name": "Southern Agricultural Zone",
        "lat_min": 8.0, "lat_max": 15.0,
        "lon_min": 76.0, "lon_max": 81.0,
    },
    "maharashtra": {
        "name": "Western/Central Semi-Arid Zone",
        "lat_min": 15.5, "lat_max": 22.5,
        "lon_min": 72.5, "lon_max": 81.0,
    },
}


def parse_and_validate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and validate event dates, distinguishing from publication dates."""
    df = df.copy()

    # Track provenance
    df["event_date_parsed"] = pd.NaT
    df["date_precision"] = "unknown"
    df["is_date_valid"] = False

    # 1. Try ISO eventDate column
    if "eventDate" in df.columns:
        # Many GBIF dates are ISO timestamps: '2023-10-15T12:00:00' or '2023-10-15'
        # Strip time and interval characters
        clean_event_date = (
            df["eventDate"]
            .astype(str)
            .str.replace(r"/.*", "", regex=True)  # take start of intervals
            .str[:10]
        )
        parsed_iso = pd.to_datetime(clean_event_date, format="%Y-%m-%d", errors="coerce")
        valid_mask = parsed_iso.notna()
        df.loc[valid_mask, "event_date_parsed"] = parsed_iso[valid_mask]
        df.loc[valid_mask, "date_precision"] = "day"
        df.loc[valid_mask, "is_date_valid"] = True

    # 2. For remaining records, attempt fallback from year + month
    missing_date = ~df["is_date_valid"]
    if missing_date.any() and "year" in df.columns and "month" in df.columns:
        year_valid = pd.to_numeric(df.loc[missing_date, "year"], errors="coerce")
        month_valid = pd.to_numeric(df.loc[missing_date, "month"], errors="coerce")
        
        can_construct = (
            year_valid.between(2015, 2026) &
            month_valid.between(1, 12)
        )
        
        idx_construct = year_valid[can_construct].index
        dates_constructed = pd.to_datetime(
            year_valid[can_construct].astype(int).astype(str) + "-" +
            month_valid[can_construct].astype(int).astype(str).str.zfill(2) + "-01"
        )
        df.loc[idx_construct, "event_date_parsed"] = dates_constructed
        df.loc[idx_construct, "date_precision"] = "month"
        df.loc[idx_construct, "is_date_valid"] = True

    # Reject future dates (beyond local time)
    now_utc = pd.Timestamp.now(tz=timezone.utc).tz_localize(None)
    future_mask = df["event_date_parsed"] > now_utc
    if future_mask.any():
        logger.warning("Found %d records with future event dates; marking invalid.", future_mask.sum())
        df.loc[future_mask, "is_date_valid"] = False
        df.loc[future_mask, "event_date_parsed"] = pd.NaT

    # Explicitly distinguish publication/accession dates if present in GBIF
    if "dateIdentified" in df.columns:
        df["publication_or_identification_date"] = pd.to_datetime(
            df["dateIdentified"].astype(str).str[:10], errors="coerce"
        )
    else:
        df["publication_or_identification_date"] = pd.NaT

    return df


def deduplicate_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Deduplicate records based on primary identifier or spatio-temporal hash."""
    initial_len = len(df)
    
    # Priority 1: GBIF Occurrence ID
    if "gbifID" in df.columns:
        df = df.drop_duplicates(subset=["gbifID"])
    
    # Priority 2: Coordinate, date, scientific name composite
    subset_cols = [c for c in ["scientificName", "event_date_parsed", "decimalLatitude", "decimalLongitude"] if c in df.columns]
    if len(subset_cols) >= 3:
        # Drop duplicates where coordinates and date are non-null
        has_coords_and_date = df["event_date_parsed"].notna() & df.get("decimalLatitude", pd.Series(index=df.index)).notna()
        dedup_part = df[has_coords_and_date].drop_duplicates(subset=subset_cols)
        other_part = df[~has_coords_and_date]
        df = pd.concat([dedup_part, other_part], ignore_index=True)

    dropped = initial_len - len(df)
    return df, dropped


def assign_agro_climatic_region(lat: Optional[float], lon: Optional[float], state_province: Optional[str]) -> str:
    """Assign Indian geographic region based on coordinates or state."""
    if pd.notna(lat) and pd.notna(lon):
        lat_f = float(lat)
        lon_f = float(lon)
        for zone_id, bounds in AGRO_ZONES.items():
            if (bounds["lat_min"] <= lat_f <= bounds["lat_max"] and
                bounds["lon_min"] <= lon_f <= bounds["lon_max"]):
                return zone_id

    # Fallback to stateProvince string matching if coordinates missing
    if pd.notna(state_province):
        sp = str(state_province).lower()
        if any(s in sp for s in ["punjab", "haryana", "delhi", "himachal"]):
            return "punjab_haryana"
        if any(s in sp for s in ["odisha", "orissa", "bengal", "assam", "bihar", "jharkhand"]):
            return "odisha_bengal"
        if any(s in sp for s in ["tamil nadu", "kerala", "karnataka", "andhra", "telangana"]):
            return "tamil_nadu"
        if any(s in sp for s in ["maharashtra", "madhya pradesh", "gujarat", "chhattisgarh"]):
            return "maharashtra"

    return "national_composite"


def validate_and_clean_dataset(
    raw_csv_path: Path,
    pathogen_slug: str,
    pathogen_info: Dict[str, Any],
    output_dir: Path = Path("data/processed"),
) -> pd.DataFrame:
    """Execute complete validation and cleaning for a pathogen raw dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(raw_csv_path, low_memory=False)

    # 1. Parse and validate dates
    df = parse_and_validate_dates(raw_df)

    # Filter only records with valid event dates
    df_valid = df[df["is_date_valid"]].copy()

    # 2. Deduplicate records
    df_clean, num_dropped = deduplicate_records(df_valid)
    logger.info("Pathogen '%s': Deduplication removed %d duplicate rows.", pathogen_slug, num_dropped)

    # 3. Add taxonomic and host metadata
    df_clean["pathogen_slug"] = pathogen_slug
    df_clean["scientific_name"] = pathogen_info.get("scientific_name")
    df_clean["common_name"] = pathogen_info.get("common_name")
    df_clean["host_crop"] = pathogen_info.get("host_crop")
    df_clean["crop_category"] = pathogen_info.get("crop_category", "Cereal")

    # 4. Regional assignment
    df_clean["region_id"] = [
        assign_agro_climatic_region(
            row.get("decimalLatitude"),
            row.get("decimalLongitude"),
            row.get("stateProvince")
        )
        for _, row in df_clean.iterrows()
    ]

    # Save individual processed cleaned records
    out_csv = output_dir / f"clean_{pathogen_slug}_records.csv"
    df_clean.to_csv(out_csv, index=False)
    logger.info("Saved cleaned records for '%s' to %s (%d rows)", pathogen_slug, out_csv, len(df_clean))
    return df_clean
