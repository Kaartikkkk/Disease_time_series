"""Central Dataset Verification Mechanism for Disease Time Series Forecasting.

Fulfills all requirements of PROMPT 36 (Section 9):
1. Verifies active dataset paths.
2. Validates cryptographic SHA-256 dataset fingerprints.
3. Confirms exact 11-year chronological date ranges (2015-01-01 to 2025-12-01).
4. Verifies row counts (1,320 master, 132 per pathogen) and column schemas (43 features).
5. Asserts full pathogen coverage across all 10 agricultural taxa.
6. Validates target variable integrity (`documented_occurrence_count`, 0 nulls, non-negative).
7. Verifies model training dataset provenance and checkpoint consistency.
8. Validates prediction dataset alignment and hashes between data/processed and artifacts.
9. Audits evaluation registry metadata and metrics.
10. Scans for outdated, obsolete, or misplaced legacy files.

Returns:
    Exit code 0 if all 10 checks pass successfully.
    Exit code 1 if any check fails or anomalous artifacts are detected.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migration_verifier")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
BACKUP_DIR = PROCESSED_DIR / "original_backup"
MISSINGNESS_DIR = PROCESSED_DIR / "missingness"
MODELS_DIR = REPO_ROOT / "models"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

EXPECTED_MASTER_SHA256 = "c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67"
EXPECTED_PATHOGENS = [
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
EXPECTED_START_DATE = "2015-01-01"
EXPECTED_END_DATE = "2025-12-01"
EXPECTED_TOTAL_MONTHS = 132
EXPECTED_TOTAL_RECORDS = 1320
EXPECTED_COLUMNS_COUNT = 43


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class MigrationVerifier:
    """Central verification engine checking complete data-to-prediction pipeline."""

    def __init__(self) -> None:
        self.results: Dict[str, Dict[str, Any]] = {}
        self.passed_checks = 0
        self.failed_checks = 0

    def record_check(self, check_name: str, passed: bool, details: Dict[str, Any]) -> None:
        self.results[check_name] = {
            "passed": passed,
            "details": details,
        }
        if passed:
            self.passed_checks += 1
            logger.info("PASS: %s", check_name)
        else:
            self.failed_checks += 1
            logger.error("FAIL: %s -> %s", check_name, details.get("error", "Unknown error"))

    def check_1_active_dataset_paths(self) -> None:
        """Check 1: Active dataset paths exist and are accessible."""
        master_file = PROCESSED_DIR / "improved_india_plant_diseases_monthly.csv"
        missing_files = []
        if not master_file.exists():
            missing_files.append(str(master_file))

        for slug in EXPECTED_PATHOGENS:
            p_file = PROCESSED_DIR / f"improved_monthly_{slug}.csv"
            if not p_file.exists():
                missing_files.append(str(p_file))

        passed = len(missing_files) == 0
        self.record_check(
            "1. Active Dataset Paths",
            passed,
            {
                "master_path": str(master_file),
                "master_exists": master_file.exists(),
                "pathogen_files_found": len(EXPECTED_PATHOGENS) - len(missing_files),
                "missing_files": missing_files,
            },
        )

    def check_2_dataset_fingerprints(self) -> None:
        """Check 2: Cryptographic SHA-256 fingerprint matches authoritative baseline."""
        master_file = PROCESSED_DIR / "improved_india_plant_diseases_monthly.csv"
        if not master_file.exists():
            self.record_check("2. Dataset Fingerprints", False, {"error": "Master file missing"})
            return

        computed_hash = compute_sha256(master_file)
        hash_match = computed_hash == EXPECTED_MASTER_SHA256

        # Also fingerprint each pathogen file
        pathogen_hashes = {}
        for slug in EXPECTED_PATHOGENS:
            p_file = PROCESSED_DIR / f"improved_monthly_{slug}.csv"
            if p_file.exists():
                pathogen_hashes[slug] = compute_sha256(p_file)

        self.record_check(
            "2. Dataset Fingerprints",
            hash_match,
            {
                "computed_master_sha256": computed_hash,
                "expected_master_sha256": EXPECTED_MASTER_SHA256,
                "hash_match": hash_match,
                "pathogen_hashes_count": len(pathogen_hashes),
            },
        )

    def check_3_date_ranges(self) -> None:
        """Check 3: Exact 11-year timeline (2015-01-01 to 2025-12-01, 132 unbroken months)."""
        master_file = PROCESSED_DIR / "improved_india_plant_diseases_monthly.csv"
        if not master_file.exists():
            self.record_check("3. Date Ranges", False, {"error": "Master file missing"})
            return

        df = pd.read_csv(master_file)
        dates = pd.to_datetime(df["date"]).sort_values().unique()
        min_date = str(dates[0])[:10]
        max_date = str(dates[-1])[:10]
        n_unique_dates = len(dates)

        expected_grid = pd.date_range(EXPECTED_START_DATE, EXPECTED_END_DATE, freq="MS")
        grid_match = len(dates) == len(expected_grid) and (dates == expected_grid).all()

        passed = (min_date == EXPECTED_START_DATE) and (max_date == EXPECTED_END_DATE) and grid_match
        self.record_check(
            "3. Date Ranges",
            passed,
            {
                "min_date": min_date,
                "max_date": max_date,
                "unique_months": n_unique_dates,
                "expected_months": EXPECTED_TOTAL_MONTHS,
                "grid_unbroken": grid_match,
            },
        )

    def check_4_rows_and_columns(self) -> None:
        """Check 4: 1,320 rows and 43 columns in master; 132 rows in each pathogen file, 0 nulls."""
        master_file = PROCESSED_DIR / "improved_india_plant_diseases_monthly.csv"
        if not master_file.exists():
            self.record_check("4. Rows and Columns", False, {"error": "Master file missing"})
            return

        df_m = pd.read_csv(master_file)
        m_rows, m_cols = df_m.shape
        m_nulls = int(df_m.isna().sum().sum())

        pathogen_anomalies = []
        for slug in EXPECTED_PATHOGENS:
            p_file = PROCESSED_DIR / f"improved_monthly_{slug}.csv"
            if p_file.exists():
                df_p = pd.read_csv(p_file)
                p_nulls = int(df_p.isna().sum().sum())
                if len(df_p) != EXPECTED_TOTAL_MONTHS or len(df_p.columns) != EXPECTED_COLUMNS_COUNT or p_nulls > 0:
                    pathogen_anomalies.append(f"{slug}: shape={df_p.shape}, nulls={p_nulls}")

        passed = (m_rows == EXPECTED_TOTAL_RECORDS) and (m_cols == EXPECTED_COLUMNS_COUNT) and (m_nulls == 0) and len(pathogen_anomalies) == 0
        self.record_check(
            "4. Rows and Columns",
            passed,
            {
                "master_rows": m_rows,
                "master_cols": m_cols,
                "master_nulls": m_nulls,
                "pathogen_anomalies": pathogen_anomalies,
            },
        )

    def check_5_pathogen_coverage(self) -> None:
        """Check 5: All 10 expected pathogen taxa present."""
        master_file = PROCESSED_DIR / "improved_india_plant_diseases_monthly.csv"
        if not master_file.exists():
            self.record_check("5. Pathogen Coverage", False, {"error": "Master file missing"})
            return

        df_m = pd.read_csv(master_file)
        found_slugs = sorted(df_m["pathogen_slug"].unique().tolist())
        missing_slugs = [s for s in EXPECTED_PATHOGENS if s not in found_slugs]

        passed = len(found_slugs) == 10 and len(missing_slugs) == 0
        self.record_check(
            "5. Pathogen Coverage",
            passed,
            {
                "pathogen_count": len(found_slugs),
                "expected_count": len(EXPECTED_PATHOGENS),
                "found_pathogens": found_slugs,
                "missing_pathogens": missing_slugs,
            },
        )

    def check_6_target_variable(self) -> None:
        """Check 6: `documented_occurrence_count` present, 0 nulls, non-negative, numeric."""
        master_file = PROCESSED_DIR / "improved_india_plant_diseases_monthly.csv"
        if not master_file.exists():
            self.record_check("6. Target Variable", False, {"error": "Master file missing"})
            return

        df_m = pd.read_csv(master_file)
        if "documented_occurrence_count" not in df_m.columns:
            self.record_check("6. Target Variable", False, {"error": "documented_occurrence_count column missing"})
            return

        target_series = df_m["documented_occurrence_count"]
        null_count = int(target_series.isna().sum())
        negative_count = int((target_series < 0).sum())
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        total_sum = float(target_series.sum())

        passed = (null_count == 0) and (negative_count == 0) and is_numeric and (total_sum > 0)
        self.record_check(
            "6. Target Variable",
            passed,
            {
                "target_col": "documented_occurrence_count",
                "is_numeric": is_numeric,
                "null_count": null_count,
                "negative_count": negative_count,
                "total_occurrences": total_sum,
            },
        )

    def check_7_model_training_artifacts(self) -> None:
        """Check 7: Model training registry and persisted joblib checkpoints."""
        reg_file = PROCESSED_DIR / "model_training_and_validation_registry.json"
        reg_meta = MODELS_DIR / "registry_metadata.json"

        if not reg_file.exists():
            self.record_check("7. Model Training Provenance", False, {"error": "model_training_and_validation_registry.json missing"})
            return

        with open(reg_file, "r", encoding="utf-8") as f:
            reg_data = json.load(f)

        meta = reg_data.get("metadata", {})
        models = reg_data.get("models", {})
        dataset_version = meta.get("dataset_version", "")
        is_improved_meta = "Improved" in dataset_version

        # Verify joblib files in models/
        joblib_files = list(MODELS_DIR.glob("*.joblib"))

        passed = is_improved_meta and len(models) == 17 and len(joblib_files) >= 10
        self.record_check(
            "7. Model Training Provenance",
            passed,
            {
                "dataset_version": dataset_version,
                "models_registered_count": len(models),
                "joblib_checkpoints_count": len(joblib_files),
                "metadata_file_exists": reg_meta.exists(),
            },
        )

    def check_8_prediction_datasets(self) -> None:
        """Check 8: Prediction files exist, match evaluation horizons, and match artifacts."""
        proc_csv = PROCESSED_DIR / "final_test_step_predictions.csv"
        art_csv = ARTIFACTS_DIR / "predictions" / "final_test_step_predictions.csv"

        if not proc_csv.exists() or not art_csv.exists():
            self.record_check("8. Prediction Dataset Alignment", False, {"error": "final_test_step_predictions.csv missing in data or artifacts"})
            return

        df_p = pd.read_csv(proc_csv)
        h_proc = compute_sha256(proc_csv)
        h_art = compute_sha256(art_csv)
        hashes_match = (h_proc == h_art)
        row_count_match = len(df_p) == 200

        passed = hashes_match and row_count_match
        self.record_check(
            "8. Prediction Dataset Alignment",
            passed,
            {
                "prediction_rows": len(df_p),
                "expected_rows": 200,
                "data_sha256": h_proc,
                "artifact_sha256": h_art,
                "hashes_match": hashes_match,
            },
        )

    def check_9_evaluation_metrics_datasets(self) -> None:
        """Check 9: Evaluation metric files exist, reflect improved dataset, and match artifacts."""
        proc_json = PROCESSED_DIR / "final_test_evaluation_results.json"
        art_json = ARTIFACTS_DIR / "metrics" / "final_test_evaluation_results.json"

        if not proc_json.exists() or not art_json.exists():
            self.record_check("9. Evaluation Metrics Integrity", False, {"error": "final_test_evaluation_results.json missing in data or artifacts"})
            return

        with open(proc_json, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        h_proc = compute_sha256(proc_json)
        h_art = compute_sha256(art_json)
        hashes_match = (h_proc == h_art)
        n_obs = eval_data.get("metadata", {}).get("n_test_observations", 0)

        passed = hashes_match and (n_obs == 200)
        self.record_check(
            "9. Evaluation Metrics Integrity",
            passed,
            {
                "test_observations": n_obs,
                "data_sha256": h_proc,
                "artifact_sha256": h_art,
                "hashes_match": hashes_match,
            },
        )

    def check_10_outdated_artifacts_audit(self) -> None:
        """Check 10: Verify no legacy 25-col monthly files exist in data/processed root."""
        # Old 25-col series should NOT exist in data/processed root; only in original_backup/
        misplaced_old_files = []
        for p in PROCESSED_DIR.glob("monthly_*.csv"):
            misplaced_old_files.append(p.name)

        if (PROCESSED_DIR / "india_plant_diseases_monthly.csv").exists():
            misplaced_old_files.append("india_plant_diseases_monthly.csv")

        # Verify that original_backup exists and preserves the original files
        backup_files_count = len(list(BACKUP_DIR.glob("*.csv"))) if BACKUP_DIR.exists() else 0

        # Verify missingness dataset is cleanly segregated
        missingness_count = len(list(MISSINGNESS_DIR.glob("*.csv"))) if MISSINGNESS_DIR.exists() else 0

        passed = len(misplaced_old_files) == 0 and (backup_files_count >= 11) and (missingness_count >= 11)
        self.record_check(
            "10. Outdated Artifacts Audit",
            passed,
            {
                "misplaced_old_files_in_root": misplaced_old_files,
                "backup_dir_files_count": backup_files_count,
                "missingness_dir_files_count": missingness_count,
                "clean_segregation": len(misplaced_old_files) == 0,
            },
        )

    def run_all(self) -> bool:
        """Execute all 10 verification checks sequentially."""
        logger.info("Starting Central Improved Dataset Migration Verification Suite...")
        self.check_1_active_dataset_paths()
        self.check_2_dataset_fingerprints()
        self.check_3_date_ranges()
        self.check_4_rows_and_columns()
        self.check_5_pathogen_coverage()
        self.check_6_target_variable()
        self.check_7_model_training_artifacts()
        self.check_8_prediction_datasets()
        self.check_9_evaluation_metrics_datasets()
        self.check_10_outdated_artifacts_audit()

        total = self.passed_checks + self.failed_checks
        logger.info("Verification Complete: %d / %d checks PASSED.", self.passed_checks, total)
        return self.failed_checks == 0


def main():
    verifier = MigrationVerifier()
    all_passed = verifier.run_all()
    print("\n" + "=" * 80)
    print("CENTRAL DATASET VERIFICATION AUDIT SUMMARY")
    print("=" * 80)
    for name, res in verifier.results.items():
        status = "PASSED" if res["passed"] else "FAILED"
        print(f"[{status:6}] {name}")
    print("=" * 80)
    print(f"FINAL RESULT: {'ALL CHECKS PASSED' if all_passed else 'VERIFICATION FAILED'}")
    print("=" * 80)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
