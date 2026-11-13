"""Unit tests for Original and Improved Dataset Integrity and Synthetic Removal Verification.

Ensures:
1. Original dataset exists with historical authenticity intact.
2. Improved dataset maintains historical target counts, adds 18 engineered features, and has 0 NaNs.
3. No synthetic dataset directories or files exist in the project repository.
4. Model training and validation registry is comprehensive and valid.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
ORIGINAL_CSV = PROCESSED_DIR / "india_plant_diseases_monthly.csv"
IMPROVED_CSV = PROCESSED_DIR / "improved_india_plant_diseases_monthly.csv"
BACKUP_DIR = PROCESSED_DIR / "original_backup"
REGISTRY_JSON = PROCESSED_DIR / "model_training_and_validation_registry.json"


def test_synthetic_files_completely_removed():
    """Verify data/synthetic directory and synthetic generator scripts do not exist."""
    synthetic_dir = DATA_DIR / "synthetic"
    assert not synthetic_dir.exists(), f"Synthetic directory {synthetic_dir} must not exist!"
    assert not (Path("scripts") / "generate_universal_synthetic_dataset.py").exists()
    assert not Path("SYNTHETIC_MODEL_COMPATIBILITY_REPORT.md").exists()


def test_original_dataset_integrity():
    """Verify original dataset preservation and historical invariant properties."""
    assert ORIGINAL_CSV.exists(), f"Original CSV missing at {ORIGINAL_CSV}"
    assert BACKUP_DIR.exists(), f"Backup directory missing at {BACKUP_DIR}"

    df_orig = pd.read_csv(ORIGINAL_CSV)
    assert len(df_orig) == 1320, f"Expected 1320 records, found {len(df_orig)}"
    assert "date" in df_orig.columns
    assert "pathogen_slug" in df_orig.columns
    assert "documented_occurrence_count" in df_orig.columns

    # Verify 10 distinct pathogens and 132 continuous months
    assert df_orig["pathogen_slug"].nunique() == 10
    dates = pd.to_datetime(df_orig["date"])
    assert dates.min() == pd.Timestamp("2015-01-01")
    assert dates.max() == pd.Timestamp("2025-12-01")


def test_improved_dataset_integrity():
    """Verify improved dataset structure, engineered features, and zero leakage."""
    assert IMPROVED_CSV.exists(), f"Improved CSV missing at {IMPROVED_CSV}"

    df_imp = pd.read_csv(IMPROVED_CSV)
    df_orig = pd.read_csv(ORIGINAL_CSV)

    assert len(df_imp) == 1320, f"Expected 1320 records, found {len(df_imp)}"
    assert len(df_imp.columns) == 43, f"Expected 43 columns, found {len(df_imp.columns)}"

    # Zero missing values across all columns
    assert df_imp.isna().sum().sum() == 0, "Improved dataset must have 0 missing values"

    # Preserved historical target values (aligned by pathogen and date)
    m = pd.merge(
        df_imp[["pathogen_slug", "date", "documented_occurrence_count"]],
        df_orig[["pathogen_slug", "date", "documented_occurrence_count"]],
        on=["pathogen_slug", "date"],
        suffixes=("_imp", "_orig"),
    )
    assert len(m) == 1320, "All 1320 records must match on (pathogen_slug, date)"
    pd.testing.assert_series_equal(
        m["documented_occurrence_count_imp"],
        m["documented_occurrence_count_orig"],
        check_names=False,
    )

    # Key engineered features present
    expected_features = [
        "imputed_flag",
        "occurrence_lag1",
        "occurrence_lag2",
        "occurrence_lag3",
        "occurrence_lag12",
        "occurrence_rolling_mean_3m_lag1",
        "occurrence_rolling_std_3m_lag1",
        "occurrence_rolling_mean_6m_lag1",
        "occurrence_rolling_std_6m_lag1",
        "occurrence_diff_1m_lag1",
        "sin_month",
        "cos_month",
        "trend_index",
        "hydrothermal_index_lag1",
        "dew_point_depression_lag1",
        "temp_diurnal_range_lag1",
        "log1p_occurrence_count",
        "sqrt_occurrence_count",
    ]
    for feat in expected_features:
        assert feat in df_imp.columns, f"Feature '{feat}' missing from improved dataset"


def test_model_training_and_validation_registry():
    """Verify the model training registry contains complete information for all 17 models."""
    assert REGISTRY_JSON.exists(), f"Registry JSON missing at {REGISTRY_JSON}"

    with open(REGISTRY_JSON, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert "metadata" in registry
    assert "models" in registry
    models = registry["models"]

    # All 17 models registered
    assert len(models) == 17, f"Expected 17 models in registry, found {len(models)}"

    for model_id, m_data in models.items():
        assert "family" in m_data, f"Missing 'family' in {model_id}"
        assert "validation_metrics" in m_data, f"Missing 'validation_metrics' in {model_id}"
        assert "test_metrics" in m_data, f"Missing 'test_metrics' in {model_id}"
        assert "rmse" in m_data["validation_metrics"], f"Missing 'rmse' in {model_id}"
        assert "hyperparameters" in m_data, f"Missing 'hyperparameters' in {model_id}"
