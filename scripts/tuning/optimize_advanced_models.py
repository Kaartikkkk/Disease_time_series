"""Advanced Model Performance Optimization & Validation Experiments.

Evaluates hyperparameter and structural refinements on the validation multi-origin benchmark:
  - 4 Chronological Origins (Sep 2022 to Apr 2024; 240 steps)
  - 10 Indian plant pathogen series
  - Strictly preserves the final test set (May 2024 to Dec 2025) untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.base import ForecastContext
from src.models.global_models import (
    DEFAULT_GLOBAL_RF_PARAMS,
    DEFAULT_GLOBAL_XGB_PARAMS,
    DEFAULT_HURDLE_CLF_PARAMS,
    DEFAULT_HURDLE_REG_PARAMS,
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_hurdle_xgboost,
)
from src.models.gru_model import DEFAULT_GRU_CONFIG, forecast_gru
from src.models.prophet_model import DEFAULT_PROPHET_PARAMS, forecast_prophet

IMPROVED_DATA_PATH = REPO_ROOT / "data" / "disease_timeseries_master.csv"
DATA_PATH = IMPROVED_DATA_PATH if IMPROVED_DATA_PATH.exists() else REPO_ROOT / "data" / "processed" / "india_plant_diseases_monthly.csv"

ORIGINS = [
    ("Origin 1", 92, 6, "2022-09 to 2023-02"),
    ("Origin 2", 98, 6, "2023-03 to 2023-08"),
    ("Origin 3", 104, 6, "2023-09 to 2024-02"),
    ("Origin 4", 106, 6, "2023-11 to 2024-04"),
]


def evaluate_candidate_configuration(
    model_family: str,
    config: Dict[str, Any],
    df: pd.DataFrame,
    all_slugs: List[str],
) -> Dict[str, Any]:
    """Run 240-step validation evaluation for a specific candidate configuration."""
    all_actuals = []
    all_preds = []

    for orig_name, cutoff_idx, horizon, date_range in ORIGINS:
        for slug in all_slugs:
            df_p = df[df["pathogen_slug"] == slug].sort_values("date").reset_index(drop=True)
            train_slice = df_p.iloc[:cutoff_idx].copy().reset_index(drop=True)
            eval_slice = df_p.iloc[cutoff_idx:cutoff_idx + horizon].copy().reset_index(drop=True)

            actuals = eval_slice["documented_occurrence_count"].values
            all_actuals.extend(actuals)

            if model_family == "global_rf":
                preds, _, _, _, _ = forecast_global_random_forest(
                    train_slice, horizon=horizon, target_col="documented_occurrence_count", rf_params=config
                )
            elif model_family == "global_xgb":
                preds, _, _, _, _ = forecast_global_xgboost(
                    train_slice, horizon=horizon, target_col="documented_occurrence_count", xgb_params=config
                )
            elif model_family == "hurdle_xgb":
                clf_p = config.get("clf_params")
                reg_p = config.get("reg_params")
                preds, _, _, _, _ = forecast_hurdle_xgboost(
                    train_slice, horizon=horizon, target_col="documented_occurrence_count",
                    clf_params=clf_p, reg_params=reg_p
                )
            elif model_family == "gru":
                preds, _, _, _, _ = forecast_gru(
                    train_slice, horizon=horizon, target_col="documented_occurrence_count", gru_config=config
                )
            elif model_family == "prophet":
                preds, _, _, _ = forecast_prophet(
                    train_slice, horizon=horizon, target_col="documented_occurrence_count", prophet_params=config
                )
            else:
                raise ValueError(f"Unknown family: {model_family}")

            all_preds.extend(np.maximum(0.0, preds))

    yt = np.array(all_actuals, dtype=float)
    yp = np.array(all_preds, dtype=float)

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    
    b_true = (yt >= 1.0).astype(int)
    b_pred = (yp >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(b_true, b_pred, average="binary", zero_division=0)
    
    zeros_mask = (b_true == 0)
    far = float(np.sum((b_pred == 1) & zeros_mask) / np.sum(zeros_mask) * 100.0) if np.sum(zeros_mask) > 0 else 0.0

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "precision": round(float(p), 3),
        "recall": round(float(r), 3),
        "f1": round(float(f1), 3),
        "far": round(far, 1),
    }


def run_all_optimization_experiments():
    df = pd.read_csv(DATA_PATH)
    all_slugs = sorted(df["pathogen_slug"].unique())
    results: Dict[str, Any] = {}

    print("=== 1. OPTIMIZING GLOBAL RANDOM FOREST ===")
    rf_candidates = [
        ("Baseline (depth=4, leaf=4, n=120)", DEFAULT_GLOBAL_RF_PARAMS),
        ("Candidate A (depth=5, leaf=3, n=140, max_features=0.75)", {
            "n_estimators": 140, "max_depth": 5, "min_samples_leaf": 3, "max_features": 0.75, "random_state": 42, "n_jobs": 1
        }),
        ("Candidate B (depth=5, leaf=2, n=160, max_features=0.7)", {
            "n_estimators": 160, "max_depth": 5, "min_samples_leaf": 2, "max_features": 0.7, "random_state": 42, "n_jobs": 1
        }),
        ("Candidate C (depth=4, leaf=2, n=150, max_features=0.8)", {
            "n_estimators": 150, "max_depth": 4, "min_samples_leaf": 2, "max_features": 0.8, "random_state": 42, "n_jobs": 1
        }),
    ]
    rf_res = {}
    for label, cfg in rf_candidates:
        t0 = time.perf_counter()
        met = evaluate_candidate_configuration("global_rf", cfg, df, all_slugs)
        dt = time.perf_counter() - t0
        print(f"  {label} -> RMSE: {met['rmse']:.3f} | MAE: {met['mae']:.3f} | F1: {met['f1']:.3f} | FAR: {met['far']:.1f}% ({dt:.1f}s)")
        rf_res[label] = {**met, "config": cfg}
    results["global_random_forest"] = rf_res

    print("\n=== 2. OPTIMIZING GLOBAL XGBOOST ===")
    xgb_candidates = [
        ("Baseline (lr=0.03, depth=3, n=110, subsample=0.85)", DEFAULT_GLOBAL_XGB_PARAMS),
        ("Candidate A (lr=0.025, depth=4, n=130, subsample=0.8, colsample=0.8, reg_alpha=0.4, reg_lambda=1.5)", {
            "n_estimators": 130, "max_depth": 4, "learning_rate": 0.025, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.4, "reg_lambda": 1.5, "random_state": 42, "n_jobs": 1
        }),
        ("Candidate B (lr=0.02, depth=3, n=150, subsample=0.85, colsample=0.85, min_child_weight=2)", {
            "n_estimators": 150, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.85,
            "min_child_weight": 2, "reg_alpha": 0.3, "reg_lambda": 1.2, "random_state": 42, "n_jobs": 1
        }),
    ]
    xgb_res = {}
    for label, cfg in xgb_candidates:
        t0 = time.perf_counter()
        met = evaluate_candidate_configuration("global_xgb", cfg, df, all_slugs)
        dt = time.perf_counter() - t0
        print(f"  {label} -> RMSE: {met['rmse']:.3f} | MAE: {met['mae']:.3f} | F1: {met['f1']:.3f} | FAR: {met['far']:.1f}% ({dt:.1f}s)")
        xgb_res[label] = {**met, "config": cfg}
    results["global_xgboost"] = xgb_res

    print("\n=== 3. OPTIMIZING HURDLE XGBOOST ===")
    hurdle_candidates = [
        ("Baseline Hurdle (clf_depth=3, reg_depth=3, n=80)", {
            "clf_params": DEFAULT_HURDLE_CLF_PARAMS, "reg_params": DEFAULT_HURDLE_REG_PARAMS
        }),
        ("Candidate A (calibrated reg_lr=0.04, reg_depth=4, reg_lambda=2.0, clf_n=100)", {
            "clf_params": {**DEFAULT_HURDLE_CLF_PARAMS, "n_estimators": 100, "learning_rate": 0.04, "max_depth": 3},
            "reg_params": {**DEFAULT_HURDLE_REG_PARAMS, "n_estimators": 100, "learning_rate": 0.04, "max_depth": 4, "reg_lambda": 2.0}
        }),
        ("Candidate B (reg_alpha=0.5, reg_lambda=2.5, clf_subsample=0.8)", {
            "clf_params": {**DEFAULT_HURDLE_CLF_PARAMS, "subsample": 0.8, "colsample_bytree": 0.8},
            "reg_params": {**DEFAULT_HURDLE_REG_PARAMS, "max_depth": 3, "reg_alpha": 0.5, "reg_lambda": 2.5}
        }),
    ]
    hurdle_res = {}
    for label, cfg in hurdle_candidates:
        t0 = time.perf_counter()
        met = evaluate_candidate_configuration("hurdle_xgb", cfg, df, all_slugs)
        dt = time.perf_counter() - t0
        print(f"  {label} -> RMSE: {met['rmse']:.3f} | MAE: {met['mae']:.3f} | F1: {met['f1']:.3f} | FAR: {met['far']:.1f}% ({dt:.1f}s)")
        hurdle_res[label] = {**met, "config": cfg}
    results["hurdle_xgboost"] = hurdle_res

    print("\n=== 4. OPTIMIZING PROPHET ===")
    prophet_candidates = [
        ("Baseline (cps=0.05, sps=0.1, additive)", DEFAULT_PROPHET_PARAMS),
        ("Candidate A (cps=0.08, sps=0.2, additive)", {
            **DEFAULT_PROPHET_PARAMS, "changepoint_prior_scale": 0.08, "seasonality_prior_scale": 0.2
        }),
        ("Candidate B (cps=0.03, sps=0.1, additive)", {
            **DEFAULT_PROPHET_PARAMS, "changepoint_prior_scale": 0.03, "seasonality_prior_scale": 0.1
        }),
    ]
    prophet_res = {}
    for label, cfg in prophet_candidates:
        t0 = time.perf_counter()
        met = evaluate_candidate_configuration("prophet", cfg, df, all_slugs)
        dt = time.perf_counter() - t0
        print(f"  {label} -> RMSE: {met['rmse']:.3f} | MAE: {met['mae']:.3f} | F1: {met['f1']:.3f} | FAR: {met['far']:.1f}% ({dt:.1f}s)")
        prophet_res[label] = {**met, "config": cfg}
    results["prophet"] = prophet_res

    out_file = REPO_ROOT / "data" / "processed" / "advanced_model_optimization_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nOptimization experiments saved to {out_file}!")


if __name__ == "__main__":
    run_all_optimization_experiments()
