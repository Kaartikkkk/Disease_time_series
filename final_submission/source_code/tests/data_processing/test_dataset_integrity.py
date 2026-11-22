"""Unit tests for Master Dataset Integrity, Schema Validation, and Single Source of Truth."""

import json
from pathlib import Path
import pandas as pd
import pytest

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MASTER_CSV = DATA_DIR / "disease_timeseries_master.csv"
REGISTRY_JSON = PROCESSED_DIR / "model_training_and_validation_registry.json"


def test_synthetic_files_completely_removed():
    """Verify data/synthetic directory and synthetic generator scripts do not exist."""
    synthetic_dir = DATA_DIR / "synthetic"
    assert not synthetic_dir.exists(), f"Synthetic directory {synthetic_dir} must not exist!"
    assert not (Path("scripts") / "generate_universal_synthetic_dataset.py").exists()
    assert not Path("SYNTHETIC_MODEL_COMPATIBILITY_REPORT.md").exists()


def test_master_dataset_integrity():
    """Verify master dataset structure, dimensions, engineered features, and zero missing values."""
    assert MASTER_CSV.exists(), f"Master CSV missing at {MASTER_CSV}"

    df = pd.read_csv(MASTER_CSV)
    assert len(df) == 1320, f"Expected 1320 records, found {len(df)}"
    assert len(df.columns) == 43, f"Expected 43 columns, found {len(df.columns)}"

    # Zero missing values across all columns
    assert df.isna().sum().sum() == 0, "Master dataset must have 0 missing values"

    # Core target invariants
    assert "documented_occurrence_count" in df.columns
    assert "date" in df.columns
    assert "pathogen_slug" in df.columns
    assert (df["documented_occurrence_count"] >= 0).all(), "Target variable must be non-negative"

    # 10 distinct pathogens and 132 months per pathogen
    assert df["pathogen_slug"].nunique() == 10
    counts = df["pathogen_slug"].value_counts()
    for slug, count in counts.items():
        assert count == 132, f"Pathogen {slug} has {count} records (expected 132)"

    # Temporal range invariant
    dates = pd.to_datetime(df["date"])
    assert dates.min() == pd.Timestamp("2015-01-01")
    assert dates.max() == pd.Timestamp("2025-12-01")

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
        assert feat in df.columns, f"Feature '{feat}' missing from master dataset"


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


def test_dataset_service_master_integrity():
    """Verify that dataset service loads directly from the master dataset."""
    from src.data.dataset_service import load_authoritative_master_dataset, load_authoritative_series, verify_dataset_integrity
    m_df = load_authoritative_master_dataset()
    assert len(m_df) == 1320
    assert len(m_df.columns) == 43

    s_df = load_authoritative_series("fusarium_oxysporum")
    assert s_df is not None
    assert len(s_df) == 132

    res = verify_dataset_integrity()
    assert res["status"] == "PASS", f"Dataset integrity failed: {res['errors']}"
    assert res["pathogens_checked"] == 10
    assert res["total_series_rows"] == 1320
    assert res["total_series_nulls"] == 0

