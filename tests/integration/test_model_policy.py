"""Unit tests for the Model Selection Policy module and frozen configurations."""

import pytest
import pandas as pd
from src.model_policy import (
    load_model_selection_policy,
    get_assigned_model,
    get_all_policy_assignments,
    get_hurdle_strategy_config,
)
from src.model_registry import get_available_models


def test_load_model_selection_policy():
    """Verify policy configuration loads and contains all required metadata."""
    policy = load_model_selection_policy()
    assert "policy_metadata" in policy
    assert "pathogen_assignments" in policy

    meta = policy["policy_metadata"]
    assert meta["status"] == "frozen_for_test"
    assert meta["default_global_champion"] == "hurdle_xgboost"
    assert meta["macro_average_champion"] == "global_random_forest"
    assert "expected_count" in meta["hurdle_prediction_strategy"]["default_operational"]


def test_pathogen_assignments_coverage():
    """Verify all 10 agricultural pathogens are assigned an optimal model."""
    expected_pathogens = [
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
    policy = load_model_selection_policy()
    assignments = policy["pathogen_assignments"]

    for slug in expected_pathogens:
        assert slug in assignments, f"Pathogen {slug} missing from policy assignments"
        info = assignments[slug]
        assert "assigned_model_id" in info
        assert "model_display_name" in info
        assert "validation_mean_rmse" in info
        assert "rationale" in info


def test_assigned_model_lookups():
    """Verify exact model assignments match frozen policy."""
    assert get_assigned_model("pyricularia_oryzae")["assigned_model_id"] == "global_random_forest"
    assert get_assigned_model("alternaria_solani")["assigned_model_id"] == "lstm"
    assert get_assigned_model("fusarium_oxysporum")["assigned_model_id"] == "gru"
    assert get_assigned_model("puccinia_striiformis")["assigned_model_id"] == "seasonal_naive"
    assert get_assigned_model("ralstonia_solanacearum")["assigned_model_id"] == "hurdle_xgboost"
    assert get_assigned_model("xanthomonas_oryzae")["assigned_model_id"] == "ma"


def test_fallback_for_unknown_pathogen():
    """Verify unknown pathogens fall back to the default global champion."""
    res = get_assigned_model("nonexistent_pathogen_xyz")
    assert res["assigned_model_id"] == "hurdle_xgboost"
    assert "Global Two-Stage ML" in res["model_family"]


def test_all_assigned_models_exist_in_registry():
    """Verify every assigned model ID in the policy is registered and executable."""
    avail_ids = {spec.model_id for spec in get_available_models()}
    policy = load_model_selection_policy()
    for slug, info in policy["pathogen_assignments"].items():
        m_id = info["assigned_model_id"]
        assert m_id in avail_ids, f"Model ID '{m_id}' for '{slug}' not found in model registry!"


def test_get_all_policy_assignments_df():
    """Verify the policy assignments DataFrame is correctly formatted."""
    df = get_all_policy_assignments()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "Pathogen Slug" in df.columns
    assert "Assigned Model" in df.columns
    assert "Val Mean RMSE" in df.columns

