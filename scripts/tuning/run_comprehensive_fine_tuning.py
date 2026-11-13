"""Comprehensive Fine-Tuning and Performance Optimization Suite for All 8 Advanced Forecasting Models.

Evaluates hyperparameter and feature variants across:
1. Local Random Forest
2. Global Random Forest
3. Local XGBoost
4. Global XGBoost
5. Hurdle XGBoost
6. LSTM
7. GRU
8. Prophet

Strict Chronological Protocol:
- Strictly confined to the 4 validation origins (Sep 2022 to Apr 2024, 240 steps).
- Final test period (May 2024 to Dec 2025; 200 records) is permanently locked.
- Zero temporal leakage, fixed random seeds (42).
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.global_models import (
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_hurdle_xgboost,
)
from src.models.gru_model import forecast_gru
from src.models.lstm_model import forecast_lstm
from src.models.prophet_model import forecast_prophet
from src.models.random_forest import forecast_random_forest
from src.models.xgboost_model import forecast_xgboost

IMPROVED_DATA_PATH = Path("data/processed/improved_india_plant_diseases_monthly.csv")
DATA_PATH = IMPROVED_DATA_PATH if IMPROVED_DATA_PATH.exists() else Path("data/processed/india_plant_diseases_monthly.csv")
OUTPUT_JSON_PATH = Path("reports/experiments/ADVANCED_MODEL_OPTIMIZATION_RESULTS.json")

VALIDATION_ORIGINS = [
    {
        "origin_name": "Origin 1",
        "train_end": "2022-08-01",
        "eval_start": "2022-09-01",
        "eval_end": "2023-02-01",
        "horizon": 6,
    },
    {
        "origin_name": "Origin 2",
        "train_end": "2023-02-01",
        "eval_start": "2023-03-01",
        "eval_end": "2023-08-01",
        "horizon": 6,
    },
    {
        "origin_name": "Origin 3",
        "train_end": "2023-08-01",
        "eval_start": "2023-09-01",
        "eval_end": "2024-02-01",
        "horizon": 6,
    },
    {
        "origin_name": "Origin 4",
        "train_end": "2023-10-01",
        "eval_start": "2023-11-01",
        "eval_end": "2024-04-01",
        "horizon": 6,
    },
]


def load_dataset() -> Tuple[pd.DataFrame, List[str]]:
    """Load cleaned monthly panel dataset."""
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["pathogen_slug", "date"]).reset_index(drop=True)
    all_slugs = sorted(df["pathogen_slug"].unique().tolist())
    return df, all_slugs


def evaluate_model_configuration(
    model_family: str,
    config: Dict[str, Any],
    df: pd.DataFrame,
    all_slugs: List[str],
) -> Dict[str, Any]:
    """Run full 4-origin walk-forward evaluation (240 steps) for a single model configuration."""
    t_start = time.perf_counter()

    all_actuals: List[float] = []
    all_preds: List[float] = []
    per_pathogen_results: Dict[str, Dict[str, float]] = {s: {"acts": [], "preds": []} for s in all_slugs}
    per_origin_results: Dict[str, Dict[str, float]] = {o["origin_name"]: {"acts": [], "preds": []} for o in VALIDATION_ORIGINS}

    for origin in VALIDATION_ORIGINS:
        o_name = origin["origin_name"]
        t_end = pd.to_datetime(origin["train_end"])
        e_start = pd.to_datetime(origin["eval_start"])
        e_end = pd.to_datetime(origin["eval_end"])
        horizon = origin["horizon"]

        for slug in all_slugs:
            p_df = df[df["pathogen_slug"] == slug].copy().sort_values("date").reset_index(drop=True)
            train_slice = p_df[p_df["date"] <= t_end].copy().reset_index(drop=True)
            eval_slice = p_df[(p_df["date"] >= e_start) & (p_df["date"] <= e_end)].copy().reset_index(drop=True)

            actuals = eval_slice["documented_occurrence_count"].astype(float).tolist()
            all_actuals.extend(actuals)
            per_pathogen_results[slug]["acts"].extend(actuals)
            per_origin_results[o_name]["acts"].extend(actuals)

            # Generate forecast based on model family
            if model_family == "local_rf":
                preds_arr, _, _, _ = forecast_random_forest(
                    train_slice, horizon=horizon, rf_params=config.get("rf_params"), lags=config.get("lags")
                )
            elif model_family == "local_xgboost":
                preds_arr, _, _, _, _ = forecast_xgboost(
                    train_slice, horizon=horizon, xgb_params=config.get("xgb_params"), lags=config.get("lags")
                )
            elif model_family == "global_rf":
                preds_arr, _, _, _, _ = forecast_global_random_forest(
                    train_slice, horizon=horizon, rf_params=config.get("rf_params")
                )
            elif model_family == "global_xgboost":
                preds_arr, _, _, _, _ = forecast_global_xgboost(
                    train_slice, horizon=horizon, xgb_params=config.get("xgb_params")
                )
            elif model_family == "hurdle_xgboost":
                preds_arr, _, _, _, _ = forecast_hurdle_xgboost(
                    train_slice,
                    horizon=horizon,
                    clf_params=config.get("clf_params"),
                    reg_params=config.get("reg_params"),
                )
            elif model_family == "lstm":
                preds_arr, _, _, _, _ = forecast_lstm(
                    train_slice, horizon=horizon, lstm_config=config.get("lstm_config")
                )
            elif model_family == "gru":
                preds_arr, _, _, _, _ = forecast_gru(
                    train_slice, horizon=horizon, gru_config=config.get("gru_config")
                )
            elif model_family == "prophet":
                preds_arr, _, _, _ = forecast_prophet(
                    train_slice, horizon=horizon, prophet_params=config.get("prophet_params")
                )
            else:
                raise ValueError(f"Unknown family: {model_family}")

            clamped_preds = np.maximum(0.0, preds_arr).tolist()
            all_preds.extend(clamped_preds)
            per_pathogen_results[slug]["preds"].extend(clamped_preds)
            per_origin_results[o_name]["preds"].extend(clamped_preds)

    total_time = round(time.perf_counter() - t_start, 3)

    yt = np.array(all_actuals, dtype=float)
    yp = np.array(all_preds, dtype=float)

    pooled_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    pooled_mae = float(np.mean(np.abs(yt - yp)))

    # Per-pathogen aggregation for Macro metrics
    p_rmses = []
    p_maes = []
    pathogen_metrics = {}
    for slug, res in per_pathogen_results.items():
        y_act = np.array(res["acts"], dtype=float)
        y_prd = np.array(res["preds"], dtype=float)
        p_rmse = float(np.sqrt(np.mean((y_act - y_prd) ** 2)))
        p_mae = float(np.mean(np.abs(y_act - y_prd)))
        p_rmses.append(p_rmse)
        p_maes.append(p_mae)
        pathogen_metrics[slug] = {"rmse": round(p_rmse, 3), "mae": round(p_mae, 3)}

    macro_rmse = float(np.mean(p_rmses))
    macro_mae = float(np.mean(p_maes))

    # Per-origin breakdown
    origin_metrics = {}
    for o_name, res in per_origin_results.items():
        y_act = np.array(res["acts"], dtype=float)
        y_prd = np.array(res["preds"], dtype=float)
        o_rmse = float(np.sqrt(np.mean((y_act - y_prd) ** 2)))
        o_mae = float(np.mean(np.abs(y_act - y_prd)))
        origin_metrics[o_name] = {"rmse": round(o_rmse, 3), "mae": round(o_mae, 3)}

    # Outbreak detection (threshold y >= 1.0)
    b_true = (yt >= 1.0).astype(int)
    b_pred = (yp >= 1.0).astype(int)

    tp = int(np.sum((b_true == 1) & (b_pred == 1)))
    fp = int(np.sum((b_true == 0) & (b_pred == 1)))
    fn = int(np.sum((b_true == 1) & (b_pred == 0)))
    tn = int(np.sum((b_true == 0) & (b_pred == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    far = float(fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0

    # Active outbreak vs dormancy MAE
    pos_mask = b_true == 1
    zero_mask = b_true == 0
    mae_pos = float(np.mean(np.abs(yt[pos_mask] - yp[pos_mask]))) if np.sum(pos_mask) > 0 else 0.0
    mae_zero = float(np.mean(np.abs(yt[zero_mask] - yp[zero_mask]))) if np.sum(zero_mask) > 0 else 0.0

    # sMAPE
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    valid_mask = denom > 1e-6
    smape = float(np.mean(np.abs(yt[valid_mask] - yp[valid_mask]) / denom[valid_mask]) * 100.0) if np.sum(valid_mask) > 0 else 0.0

    return {
        "pooled_rmse": round(pooled_rmse, 3),
        "macro_rmse": round(macro_rmse, 3),
        "pooled_mae": round(pooled_mae, 3),
        "macro_mae": round(macro_mae, 3),
        "smape": round(smape, 2),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "far_pct": round(far, 1),
        "mae_pos": round(mae_pos, 3),
        "mae_zero": round(mae_zero, 3),
        "runtime_sec": total_time,
        "per_pathogen": pathogen_metrics,
        "per_origin": origin_metrics,
        "config": config,
    }


def build_experiment_suite() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Define targeted search grid for all 8 advanced models."""
    return {
        # 1. Local Random Forest
        "local_rf": {
            "Baseline (depth=3, leaf=2, n=100, lags=[1,2,3,12])": {
                "rf_params": {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 2, "random_state": 42},
                "lags": [1, 2, 3, 12],
            },
            "Candidate A (depth=4, leaf=3, n=150, lags=[1,2,3,12])": {
                "rf_params": {"n_estimators": 150, "max_depth": 4, "min_samples_leaf": 3, "max_features": "sqrt", "random_state": 42},
                "lags": [1, 2, 3, 12],
            },
            "Candidate B (depth=5, leaf=4, n=180, max_feat=0.7)": {
                "rf_params": {"n_estimators": 180, "max_depth": 5, "min_samples_leaf": 4, "max_features": 0.7, "random_state": 42},
                "lags": [1, 2, 3, 6, 12],
            },
            "Candidate C (depth=3, leaf=3, n=200, lags=[1,2,3,6,12])": {
                "rf_params": {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 3, "random_state": 42},
                "lags": [1, 2, 3, 6, 12],
            },
        },
        # 2. Global Random Forest
        "global_rf": {
            "Baseline (depth=4, leaf=4, n=120, max_feat=0.8)": {
                "rf_params": {"n_estimators": 120, "max_depth": 4, "min_samples_leaf": 4, "max_features": 0.8, "random_state": 42, "n_jobs": 1},
            },
            "Candidate A (depth=5, leaf=3, n=140, max_feat=0.75)": {
                "rf_params": {"n_estimators": 140, "max_depth": 5, "min_samples_leaf": 3, "max_features": 0.75, "random_state": 42, "n_jobs": 1},
            },
            "Candidate B (depth=4, leaf=3, n=160, max_feat=0.7)": {
                "rf_params": {"n_estimators": 160, "max_depth": 4, "min_samples_leaf": 3, "max_features": 0.7, "random_state": 42, "n_jobs": 1},
            },
            "Candidate C (depth=5, leaf=2, n=160, max_feat=0.7)": {
                "rf_params": {"n_estimators": 160, "max_depth": 5, "min_samples_leaf": 2, "max_features": 0.7, "random_state": 42, "n_jobs": 1},
            },
        },
        # 3. Local XGBoost
        "local_xgboost": {
            "Baseline (depth=3, lr=0.05, n=120, min_child=3, alpha=0.5, lambda=2.0)": {
                "xgb_params": {
                    "objective": "reg:squarederror", "n_estimators": 120, "max_depth": 3, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.5, "reg_lambda": 2.0,
                    "random_state": 42, "n_jobs": 1,
                },
                "lags": [1, 2, 3, 6, 12],
            },
            "Candidate A (depth=3, lr=0.03, n=140, min_child=4, alpha=0.8, lambda=3.0)": {
                "xgb_params": {
                    "objective": "reg:squarederror", "n_estimators": 140, "max_depth": 3, "learning_rate": 0.03,
                    "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 4, "reg_alpha": 0.8, "reg_lambda": 3.0,
                    "random_state": 42, "n_jobs": 1,
                },
                "lags": [1, 2, 3, 6, 12],
            },
            "Candidate B (depth=4, lr=0.025, n=150, alpha=0.6, lambda=2.5)": {
                "xgb_params": {
                    "objective": "reg:squarederror", "n_estimators": 150, "max_depth": 4, "learning_rate": 0.025,
                    "subsample": 0.75, "colsample_bytree": 0.75, "min_child_weight": 3, "reg_alpha": 0.6, "reg_lambda": 2.5,
                    "random_state": 42, "n_jobs": 1,
                },
                "lags": [1, 2, 3, 6, 12],
            },
        },
        # 4. Global XGBoost
        "global_xgboost": {
            "Baseline (depth=3, lr=0.03, n=110, subsample=0.85, alpha=0.3, lambda=1.0)": {
                "xgb_params": {
                    "n_estimators": 110, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85,
                    "reg_alpha": 0.3, "reg_lambda": 1.0, "random_state": 42, "n_jobs": 1,
                },
            },
            "Candidate A (depth=4, lr=0.025, n=130, subsample=0.80, alpha=0.4, lambda=1.5)": {
                "xgb_params": {
                    "n_estimators": 130, "max_depth": 4, "learning_rate": 0.025, "subsample": 0.80, "colsample_bytree": 0.80,
                    "reg_alpha": 0.4, "reg_lambda": 1.5, "random_state": 42, "n_jobs": 1,
                },
            },
            "Candidate B (depth=3, lr=0.02, n=140, subsample=0.85, min_child=3, alpha=0.5, lambda=2.5)": {
                "xgb_params": {
                    "n_estimators": 140, "max_depth": 3, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.85,
                    "min_child_weight": 3, "reg_alpha": 0.5, "reg_lambda": 2.5, "random_state": 42, "n_jobs": 1,
                },
            },
        },
        # 5. Hurdle XGBoost
        "hurdle_xgboost": {
            "Baseline (clf_depth=3, reg_depth=3, n=80, alpha=0.2, lambda=1.2)": {
                "clf_params": {
                    "n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
                    "eval_metric": "logloss", "random_state": 42, "n_jobs": 1,
                },
                "reg_params": {
                    "n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
                    "reg_alpha": 0.2, "reg_lambda": 1.2, "random_state": 42, "n_jobs": 1,
                },
            },
            "Candidate A (clf_lr=0.04, reg_depth=4, reg_lr=0.04, lambda=2.0)": {
                "clf_params": {
                    "n_estimators": 100, "max_depth": 3, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85,
                    "eval_metric": "logloss", "random_state": 42, "n_jobs": 1,
                },
                "reg_params": {
                    "n_estimators": 100, "max_depth": 4, "learning_rate": 0.04, "subsample": 0.85, "colsample_bytree": 0.85,
                    "reg_alpha": 0.2, "reg_lambda": 2.0, "random_state": 42, "n_jobs": 1,
                },
            },
            "Candidate B (clf_sub=0.8, reg_alpha=0.5, reg_lambda=2.5)": {
                "clf_params": {
                    "n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80,
                    "eval_metric": "logloss", "random_state": 42, "n_jobs": 1,
                },
                "reg_params": {
                    "n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85,
                    "reg_alpha": 0.5, "reg_lambda": 2.5, "random_state": 42, "n_jobs": 1,
                },
            },
        },
        # 6. LSTM
        "lstm": {
            "Baseline (lookback=12, units=16, dropout=0.1, lr=0.005, batch=8)": {
                "lstm_config": {
                    "lookback": 12, "units": 16, "dropout": 0.1, "learning_rate": 0.005,
                    "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42,
                }
            },
            "Candidate A (lookback=6, units=16, dropout=0.1, lr=0.005, batch=8)": {
                "lstm_config": {
                    "lookback": 6, "units": 16, "dropout": 0.1, "learning_rate": 0.005,
                    "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42,
                }
            },
            "Candidate B (lookback=12, units=8, dropout=0.15, lr=0.003, batch=16)": {
                "lstm_config": {
                    "lookback": 12, "units": 8, "dropout": 0.15, "learning_rate": 0.003,
                    "batch_size": 16, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42,
                }
            },
        },
        # 7. GRU
        "gru": {
            "Baseline (lookback=12, units=16, dropout=0.1, lr=0.005, batch=8)": {
                "gru_config": {
                    "lookback": 12, "units": 16, "dropout": 0.1, "learning_rate": 0.005,
                    "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42,
                }
            },
            "Candidate A (lookback=6, units=16, dropout=0.1, lr=0.005, batch=8)": {
                "gru_config": {
                    "lookback": 6, "units": 16, "dropout": 0.1, "learning_rate": 0.005,
                    "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42,
                }
            },
            "Candidate B (lookback=12, units=8, dropout=0.1, lr=0.003, batch=8)": {
                "gru_config": {
                    "lookback": 12, "units": 8, "dropout": 0.1, "learning_rate": 0.003,
                    "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42,
                }
            },
        },
        # 8. Prophet
        "prophet": {
            "Baseline (cps=0.05, sps=0.10, additive)": {
                "prophet_params": {
                    "yearly_seasonality": True, "weekly_seasonality": False, "daily_seasonality": False,
                    "seasonality_mode": "additive", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 0.10,
                    "interval_width": 0.95,
                }
            },
            "Candidate A (cps=0.08, sps=0.20, additive)": {
                "prophet_params": {
                    "yearly_seasonality": True, "weekly_seasonality": False, "daily_seasonality": False,
                    "seasonality_mode": "additive", "changepoint_prior_scale": 0.08, "seasonality_prior_scale": 0.20,
                    "interval_width": 0.95,
                }
            },
            "Candidate B (cps=0.03, sps=0.10, additive)": {
                "prophet_params": {
                    "yearly_seasonality": True, "weekly_seasonality": False, "daily_seasonality": False,
                    "seasonality_mode": "additive", "changepoint_prior_scale": 0.03, "seasonality_prior_scale": 0.10,
                    "interval_width": 0.95,
                }
            },
        },
    }


def run_all_fine_tuning() -> None:
    """Execute fine-tuning experiments across all 8 advanced models."""
    print("=" * 80)
    print("STARTING COMPREHENSIVE FINE-TUNING SUITE (ALL 8 ADVANCED MODELS)")
    print("Protocols: 4 Validation Origins · 10 Pathogens · 240 Steps · Zero Test Leakage")
    print("=" * 80)

    df, all_slugs = load_dataset()
    experiments = build_experiment_suite()
    all_results: Dict[str, Any] = {}
    summary_comparisons: List[Dict[str, Any]] = []

    # Reference classical baseline: Seasonal Naive = 6.648, MA(1) = 6.243
    CLASSICAL_MA1_RMSE = 6.243
    CLASSICAL_MA1_MAE = 2.446

    for fam_idx, (model_fam, variants) in enumerate(experiments.items(), start=1):
        print(f"\n[{fam_idx}/8] Benchmarking Model Family: {model_fam.upper()}")
        all_results[model_fam] = {}

        baseline_rmse = None
        baseline_mae = None
        best_cfg_name = None
        best_cfg_rmse = 999.0
        best_cfg_met = None

        for var_name, cfg in variants.items():
            print(f"  --> Running {var_name} ...", end="", flush=True)
            met = evaluate_model_configuration(model_fam, cfg, df, all_slugs)
            all_results[model_fam][var_name] = met
            print(f" Done in {met['runtime_sec']:.1f}s | Pooled RMSE: {met['pooled_rmse']:.3f} | Macro: {met['macro_rmse']:.3f} | MAE: {met['pooled_mae']:.3f} | F1: {met['f1']:.3f} | FAR: {met['far_pct']:.1f}%")

            if "Baseline" in var_name:
                baseline_rmse = met["pooled_rmse"]
                baseline_mae = met["pooled_mae"]

            # Select champion configuration per family based on lowest Pooled RMSE
            if met["pooled_rmse"] < best_cfg_rmse:
                best_cfg_rmse = met["pooled_rmse"]
                best_cfg_name = var_name
                best_cfg_met = met

        # Calculate Before-vs-After improvement
        opt_rmse = best_cfg_met["pooled_rmse"]
        opt_mae = best_cfg_met["pooled_mae"]
        rmse_gain = ((baseline_rmse - opt_rmse) / baseline_rmse) * 100.0 if baseline_rmse else 0.0
        mae_gain = ((baseline_mae - opt_mae) / baseline_mae) * 100.0 if baseline_mae else 0.0
        vs_classical_gain = ((CLASSICAL_MA1_RMSE - opt_rmse) / CLASSICAL_MA1_RMSE) * 100.0

        summary_comparisons.append({
            "model_family": model_fam,
            "selected_configuration": best_cfg_name,
            "baseline_rmse": baseline_rmse,
            "optimized_rmse": opt_rmse,
            "rmse_improvement_pct": round(rmse_gain, 2),
            "baseline_mae": baseline_mae,
            "optimized_mae": opt_mae,
            "mae_improvement_pct": round(mae_gain, 2),
            "macro_rmse": best_cfg_met["macro_rmse"],
            "macro_mae": best_cfg_met["macro_mae"],
            "outbreak_f1": best_cfg_met["f1"],
            "outbreak_recall": best_cfg_met["recall"],
            "false_alarm_pct": best_cfg_met["far_pct"],
            "vs_best_classical_rmse_gain_pct": round(vs_classical_gain, 2),
            "runtime_sec": best_cfg_met["runtime_sec"],
        })

    payload = {
        "metadata": {
            "evaluation_scale": "240 walk-forward steps (10 pathogens x 4 origins x 6 horizons)",
            "validation_period": "September 2022 to April 2024",
            "test_set_isolation": "May 2024 to December 2025 preserved untouched",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_tuned": list(experiments.keys()),
        },
        "summary_comparisons": summary_comparisons,
        "detailed_experiments": all_results,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SUCCESS: Comprehensive optimization results written to {OUTPUT_JSON_PATH}")
    print("=" * 80)

    # Print summary table
    df_sum = pd.DataFrame(summary_comparisons)
    print(df_sum[["model_family", "baseline_rmse", "optimized_rmse", "rmse_improvement_pct", "baseline_mae", "optimized_mae", "outbreak_f1", "vs_best_classical_rmse_gain_pct"]].to_string(index=False))


if __name__ == "__main__":
    run_all_fine_tuning()
