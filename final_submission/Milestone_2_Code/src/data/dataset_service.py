"""Authoritative Centralized Dataset Service.

Single source of truth for loading, validating, and managing the authoritative
improved plant disease time-series dataset collection for Indian agriculture.

Active Dataset Collection:
- 10 pathogen-specific continuous monthly files: `data/processed/improved_monthly_{slug}.csv`
- 1 cross-pathogen unified monthly master panel: `data/processed/improved_india_plant_diseases_monthly.csv`
- Granular event records for UI exploration: `data/processed/clean_{slug}_records.csv`
- Permanent untouched historical backup: `data/processed/original_backup/`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
BACKUP_DIR = PROCESSED_DIR / "original_backup"
REPORTS_DIR = REPO_ROOT / "data" / "reports"

# Canonical 10 target pathogens
CANONICAL_PATHOGENS: List[str] = [
    "alternaria_solani",
    "bipolaris_oryzae",
    "fusarium_oxysporum",
    "macrophomina_phaseolina",
    "phytophthora_infestans",
    "puccinia_striiformis",
    "pyricularia_oryzae",
    "ralstonia_solanacearum",
    "rhizoctonia_solani",
    "xanthomonas_oryzae",
]


@dataclass(frozen=True)
class DatasetSpecification:
    """Immutable specification defining authoritative master dataset structure and invariants."""

    version: str = "3.0.0-master"
    target_column: str = "documented_occurrence_count"
    date_column: str = "date"
    pathogen_column: str = "pathogen_slug"
    frequency: str = "MS"
    expected_months: int = 132
    start_date: str = "2015-01-01"
    end_date: str = "2025-12-01"
    expected_pathogens_count: int = 10
    total_master_records: int = 1320
    expected_features_count: int = 43

    master_file: Path = REPO_ROOT / "data" / "disease_timeseries_master.csv"
    clean_records_pattern: str = "clean_{slug}_records.csv"


SPECIFICATION = DatasetSpecification()


def get_authoritative_master_path() -> Path:
    """Return the absolute path to the authoritative master dataset."""
    return SPECIFICATION.master_file


def get_pathogen_dataset_path(slug: str) -> Path:
    """Return master dataset path as single source of truth for all pathogen series."""
    return SPECIFICATION.master_file


def get_clean_records_path(slug: str) -> Path:
    """Return the path to cleaned granular occurrence event records for geospatial mapping."""
    clean_slug = slug.strip().lower()
    return PROCESSED_DIR / SPECIFICATION.clean_records_pattern.format(slug=clean_slug)


def list_authoritative_pathogen_files() -> Dict[str, Path]:
    """Return a mapping of canonical pathogen slugs to the master dataset."""
    master = get_authoritative_master_path()
    if master.exists():
        return {slug: master for slug in CANONICAL_PATHOGENS}
    return {}


def load_authoritative_series(slug: str) -> Optional[pd.DataFrame]:
    """Load, parse dates, and return the continuous monthly series for a single pathogen.

    Extracts directly from the single master dataset (data/disease_timeseries_master.csv).

    Args:
        slug: Pathogen identifier slug (e.g. 'bipolaris_oryzae').

    Returns:
        DataFrame sorted by date with 43 features and zero missing values, or None if not found.
    """
    clean_slug = slug.strip().lower()
    df_master = load_authoritative_master_dataset()
    sub_df = df_master[df_master[SPECIFICATION.pathogen_column] == clean_slug].copy()
    if sub_df.empty:
        return None
    if SPECIFICATION.date_column in sub_df.columns:
        sub_df[SPECIFICATION.date_column] = pd.to_datetime(sub_df[SPECIFICATION.date_column])
    return sub_df.sort_values(SPECIFICATION.date_column).reset_index(drop=True)


def load_authoritative_master_dataset() -> pd.DataFrame:
    """Load the full authoritative master panel across all 10 pathogens.

    Single Source of Truth: `data/disease_timeseries_master.csv`.

    Returns:
        Unified DataFrame containing 1,320 rows and 43 features.
    """
    master_path = get_authoritative_master_path()
    if not master_path.exists():
        raise FileNotFoundError(
            f"Authoritative master dataset file missing at: {master_path}!"
        )

    df = pd.read_csv(master_path)
    if SPECIFICATION.date_column in df.columns:
        df[SPECIFICATION.date_column] = pd.to_datetime(df[SPECIFICATION.date_column])
    return df.sort_values([SPECIFICATION.pathogen_column, SPECIFICATION.date_column]).reset_index(drop=True)


def load_clean_event_records(slug: str) -> Optional[pd.DataFrame]:
    """Load occurrence records for a pathogen from the master dataset."""
    return load_authoritative_series(slug)


def verify_dataset_integrity() -> Dict[str, Any]:
    """Execute complete validation across the master dataset and all 10 pathogen series."""
    results: Dict[str, Any] = {
        "status": "PASS",
        "pathogens_checked": len(CANONICAL_PATHOGENS),
        "missing_files": [],
        "errors": [],
        "details": {},
    }

    master_path = get_authoritative_master_path()
    if not master_path.exists():
        results["missing_files"].append(str(master_path))
        results["status"] = "FAIL"
        results["errors"].append(f"Master dataset missing at {master_path}")
        return results

    try:
        df_master = pd.read_csv(master_path)
    except Exception as e:
        results["errors"].append(f"Failed to read master dataset: {e}")
        results["status"] = "FAIL"
        return results

    if len(df_master) != SPECIFICATION.total_master_records:
        results["errors"].append(
            f"Expected {SPECIFICATION.total_master_records} total records, got {len(df_master)}"
        )
        results["status"] = "FAIL"

    if len(df_master.columns) != SPECIFICATION.expected_features_count:
        results["errors"].append(
            f"Expected {SPECIFICATION.expected_features_count} columns, got {len(df_master.columns)}"
        )
        results["status"] = "FAIL"

    total_nulls = int(df_master.isna().sum().sum())
    if total_nulls != 0:
        results["errors"].append(f"Master dataset contains {total_nulls} missing cells (expected 0)")
        results["status"] = "FAIL"

    total_rows = 0
    for slug in CANONICAL_PATHOGENS:
        sub_df = df_master[df_master[SPECIFICATION.pathogen_column] == slug]
        rows = len(sub_df)
        cols = len(sub_df.columns)
        nulls = int(sub_df.isna().sum().sum())
        total_rows += rows

        if rows != SPECIFICATION.expected_months:
            results["errors"].append(f"{slug}: expected {SPECIFICATION.expected_months} rows, got {rows}")
            results["status"] = "FAIL"

        results["details"][slug] = {
            "rows": rows,
            "columns": cols,
            "nulls": nulls,
            "target_sum": int(sub_df[SPECIFICATION.target_column].sum())
            if SPECIFICATION.target_column in sub_df.columns
            else 0,
        }

    results["total_series_rows"] = total_rows
    results["total_series_nulls"] = total_nulls
    return results

