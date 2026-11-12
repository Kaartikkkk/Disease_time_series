"""Tests for Model Comparison UI data integrity, Phase 3 tables, and dashboard loaders."""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.multi_origin_ui import load_multi_origin_benchmark_data
from src.model_policy import get_assigned_model, load_model_selection_policy


def test_final_model_comparison_results_json_structure():
    """Verify that FINAL_MODEL_COMPARISON_RESULTS.json contains all 5 Phase 3 tables."""
    results_path = Path("data/processed/FINAL_MODEL_COMPARISON_RESULTS.json")
    assert results_path.exists(), "FINAL_MODEL_COMPARISON_RESULTS.json must exist in data/processed/"

    with open(results_path, "r") as f:
        data = json.load(f)

    # Check top-level metadata
    assert "metadata" in data
    assert data["metadata"]["n_pathogens"] == 10
    assert data["metadata"]["n_forecast_points"] == 240

    # Table A: Overall Comparison (14 models)
    assert "table_a_overall_comparison" in data
    table_a = data["table_a_overall_comparison"]
    assert len(table_a) == 14, f"Expected 14 models in Table A, found {len(table_a)}"
    assert table_a[0]["model_name"].startswith("Hurdle XGBoost")
    assert table_a[0]["rmse"] == 5.982

    # Table B: Classical vs Modern Head-to-Head (8 metrics)
    assert "table_b_classical_vs_modern" in data
    table_b = data["table_b_classical_vs_modern"]
    assert len(table_b) == 8, f"Expected 8 metrics in Table B, found {len(table_b)}"
    
    # Table C: Per-Pathogen Comparison (10 pathogens)
    assert "table_c_per_pathogen_comparison" in data
    table_c = data["table_c_per_pathogen_comparison"]
    assert len(table_c) == 10, f"Expected 10 pathogens in Table C, found {len(table_c)}"

    # Table D: Forecast Stability Across 4 Origins (14 models)
    assert "table_d_forecast_stability" in data
    table_d = data["table_d_forecast_stability"]
    assert len(table_d) == 14, f"Expected 14 models in Table D, found {len(table_d)}"
    for row in table_d:
        assert "origin_1" in row and "origin_2" in row and "origin_3" in row and "origin_4" in row

    # Table E: Outbreak Detection (14 models)
    assert "table_e_outbreak_detection" in data
    table_e = data["table_e_outbreak_detection"]
    assert len(table_e) == 14, f"Expected 14 models in Table E, found {len(table_e)}"


def test_load_multi_origin_data_loader():
    """Verify that load_multi_origin_benchmark_data runs and returns expected keys."""
    data = load_multi_origin_benchmark_data()
    assert data is not None, "load_multi_origin_benchmark_data should not return None"
    assert "overall_rankings" in data
    assert "per_pathogen_summary" in data
    assert "raw_evaluations" in data

    # Check that overall_rankings has all 14 models
    assert len(data["overall_rankings"]) == 14
    # Champion check
    assert data["overall_rankings"][0]["model_name"].startswith("Hurdle XGBoost")
    assert data["overall_rankings"][0]["pooled_rmse"] == pytest.approx(5.982, rel=1e-3)


def test_head_to_head_key_metrics():
    """Verify key Phase 3 classical vs modern values."""
    with open("data/processed/FINAL_MODEL_COMPARISON_RESULTS.json", "r") as f:
        data = json.load(f)

    b_dict = {r["metric"]: r for r in data["table_b_classical_vs_modern"]}
    
    # Pooled RMSE: Hurdle XGBoost (5.982) beats MA(1) (6.243)
    rmse_row = b_dict["Pooled RMSE"]
    assert rmse_row["best_modern_val"] < rmse_row["best_classical_val"]
    assert rmse_row["winner"].startswith("Modern")

    # Macro RMSE: Global RF (4.081) beats MA(1) (4.280)
    macro_row = b_dict["Macro RMSE"]
    assert macro_row["best_modern_val"] < macro_row["best_classical_val"]
    assert macro_row["winner"].startswith("Modern")

    # Outbreak Recall: Hurdle XGBoost (0.862) beats Holt-Winters (0.756)
    recall_row = b_dict["Outbreak Detection Recall"]
    assert recall_row["best_modern_val"] > recall_row["best_classical_val"]
    assert recall_row["winner"].startswith("Modern")


def test_model_policy_alignment_with_comparison_results():
    """Verify that model selection policy reflects validated per-pathogen winners."""
    policy = load_model_selection_policy()
    assert policy["policy_metadata"]["default_global_champion"] == "hurdle_xgboost"

    # Check that Stripe Rust selects Seasonal Naive (RMSE 0.000)
    stripe_rust_assignment = get_assigned_model("puccinia_striiformis")
    assert "Seasonal Naive" in stripe_rust_assignment["model_display_name"]
    assert stripe_rust_assignment["assigned_model_id"] == "seasonal_naive"

    # Check that Rice Blast selects Global Random Forest
    rice_blast_assignment = get_assigned_model("pyricularia_oryzae")
    assert "Global Random Forest" in rice_blast_assignment["model_display_name"]
    assert rice_blast_assignment["assigned_model_id"] == "global_random_forest"


def test_model_inventory_counts():
    """Verify exact counts of registered (17), retired (2), and retained (15) models."""
    from src.model_registry import MODEL_SPECS
    assert len(MODEL_SPECS) == 17, f"Expected 17 registered models, found {len(MODEL_SPECS)}"
    
    retired_models = {"gradient_boosting", "arimax"}
    assert all(m in MODEL_SPECS for m in retired_models)
    
    retained_models = set(MODEL_SPECS.keys()) - retired_models
    assert len(retained_models) == 15


def test_evaluation_count_and_overlap_math():
    """Verify 240 prediction records = 200 unique pathogen-months + 40 repeated steps."""
    data = load_multi_origin_benchmark_data()
    raw = data["raw_evaluations"]
    assert len(raw) == 40  # 10 pathogens * 4 origins
    
    records = []
    for chunk in raw:
        slug = chunk["pathogen_slug"]
        for dt in chunk["dates"]:
            records.append((slug, dt))
            
    assert len(records) == 240
    unique_pairs = set(records)
    assert len(unique_pairs) == 200  # 10 pathogens * 20 unique months
    assert len(records) - len(unique_pairs) == 40  # 40 repeated predictions from 4-month overlap


def test_stripe_rust_seasonal_naive_zero_rmse_audit():
    """Verify that Stripe Rust (Puccinia striiformis) has exactly 1 positive in May 2023,
    which matches May 2022 (lag 12), producing zero error for Seasonal Naive."""
    data = load_multi_origin_benchmark_data()
    raw = data["raw_evaluations"]
    ps_chunks = [c for c in raw if c["pathogen_slug"] == "puccinia_striiformis"]
    
    all_actuals = []
    all_sn_preds = []
    for chunk in ps_chunks:
        all_actuals.extend(chunk["actuals"])
        all_sn_preds.extend(chunk["forecasts"]["Seasonal Naive (Baseline)"])
        
    assert len(all_actuals) == 24
    # Exactly 1 positive in all 24 evaluation steps
    assert sum(1 for y in all_actuals if y > 0) == 1
    # Perfect match between predictions and actuals
    for y, p in zip(all_actuals, all_sn_preds):
        assert y == p
