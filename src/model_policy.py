"""Model Selection Policy Module.

Provides programmatic access to the frozen pathogen-specific model selection policy,
hurdle prediction strategies, and experiment metadata established through multi-origin
validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

POLICY_PATH = Path("config/model_selection_policy.json")


def load_model_selection_policy() -> Dict[str, Any]:
    """Load the frozen model selection policy from configuration."""
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Model selection policy configuration not found at {POLICY_PATH}")
    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_assigned_model(pathogen_slug: str) -> Dict[str, Any]:
    """Retrieve the assigned production model specification for a given pathogen.

    Args:
        pathogen_slug: Identifier for the pathogen (e.g. 'pyricularia_oryzae').

    Returns:
        Dictionary containing model ID, display name, family, validation metrics, and selection rationale.
    """
    policy = load_model_selection_policy()
    assignments = policy.get("pathogen_assignments", {})
    if pathogen_slug not in assignments:
        default_model = policy.get("policy_metadata", {}).get("default_global_champion", "hurdle_xgboost")
        return {
            "assigned_model_id": default_model,
            "model_display_name": "Hurdle XGBoost (Classifier + Regressor)",
            "model_family": "Global Two-Stage ML",
            "rationale": f"Pathogen '{pathogen_slug}' not explicitly listed; falling back to global multi-origin benchmark champion.",
        }
    return assignments[pathogen_slug]


def get_all_policy_assignments() -> pd.DataFrame:
    """Return a DataFrame summarizing the frozen model assignments across all 10 pathogens."""
    policy = load_model_selection_policy()
    assignments = policy.get("pathogen_assignments", {})

    rows: List[Dict[str, Any]] = []
    for slug, info in assignments.items():
        rows.append({
            "Pathogen Slug": slug,
            "Common Name": info.get("common_name", slug),
            "Assigned Model": info.get("model_display_name", info.get("assigned_model_id")),
            "Model ID": info.get("assigned_model_id"),
            "Architecture Family": info.get("model_family"),
            "Val Mean RMSE": info.get("validation_mean_rmse"),
            "Val Origin Std": info.get("validation_origin_std"),
            "Val Mean MAE": info.get("validation_mean_mae"),
            "Outbreak F1": info.get("outbreak_f1"),
            "Selection Rationale": info.get("rationale"),
        })
    return pd.DataFrame(rows)


def get_hurdle_strategy_config() -> Dict[str, Any]:
    """Retrieve the frozen Hurdle prediction configuration."""
    policy = load_model_selection_policy()
    return policy.get("policy_metadata", {}).get("hurdle_prediction_strategy", {})

