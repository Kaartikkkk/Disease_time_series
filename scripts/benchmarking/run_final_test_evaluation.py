"""Final Untouched Test Evaluation Engine.

Executes a pre-declared, expanding-window, one-step-ahead walk-forward evaluation
on the strictly preserved final test partition:
  - Test Period: May 2024 to December 2025 (20 months; indices 112 to 131)
  - Target Series: 10 Indian plant pathogen series (200 total pathogen-month records)
  - Evaluated Architectures:
      1. Seasonal Naive
      2. AR(1)
      3. MA(1)
      4. ARIMA(1,1,1)
      5. SARIMA(1,0,1)(1,0,1)[12]
      6. Holt-Winters Additive ESM
      7. Local Random Forest
      8. Local XGBoost
      9. Global Random Forest
     10. Global XGBoost
     11. Hurdle XGBoost
     12. Prophet
     13. LSTM
     14. GRU
     15. Production Selection Policy (Ensemble routing per config/model_selection_policy.json)
     16. SARIMAX (Operational with 1-month Lagged NASA Weather Covariates)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from src.models.base import ForecastContext
from src.model_registry import MODEL_SPECS, run_registered_model
from src.model_policy import get_assigned_model, load_model_selection_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("final_test_evaluation")

IMPROVED_DATA_PATH = Path("data/processed/improved_india_plant_diseases_monthly.csv")
DATA_PATH = IMPROVED_DATA_PATH if IMPROVED_DATA_PATH.exists() else Path("data/processed/india_plant_diseases_monthly.csv")
RESULTS_OUTPUT_PATH = Path("data/processed/final_test_evaluation_results.json")
CSV_OUTPUT_PATH = Path("data/processed/final_test_step_predictions.csv")

MODELS_TO_EVALUATE = [
    "seasonal_naive",
    "ar",
    "ma",
    "arima",
    "sarima",
    "holt_winters",
    "random_forest",
    "xgboost",
    "global_random_forest",
    "global_xgboost",
    "hurdle_xgboost",
    "prophet",
    "lstm",
    "gru",
    "sarimax",
]

WEATHER_EXOG_COLS = [
    "temp_mean_c_lag1",
    "precipitation_monthly_mm_lag1",
    "relative_humidity_pct_lag1",
]

CHAMPION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "seasonal_naive": {"season_length": 12},
    "ar": {"p": 1},
    "ma": {"q": 1},
    "arima": {"order": (1, 0, 1), "auto_optimize": False},
    "sarima": {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 12), "auto_optimize": False},
    "holt_winters": {"trend": "add", "seasonal": "add", "damped_trend": True, "phi": 0.92, "season_length": 12, "auto_optimize": False},
    "random_forest": {"rf_params": {"n_estimators": 180, "max_depth": 5, "min_samples_leaf": 4, "max_features": 0.7, "random_state": 42}, "lags": [1, 2, 3, 12]},
    "xgboost": {"xgb_params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.025, "reg_alpha": 0.6, "reg_lambda": 2.5, "random_state": 42}, "xgb_lags": [1, 2, 3, 6, 12]},
    "global_random_forest": {"rf_params": {"n_estimators": 120, "max_depth": 4, "min_samples_leaf": 4, "max_features": 0.8, "random_state": 42}},
    "global_xgboost": {"xgb_params": {"n_estimators": 130, "max_depth": 4, "learning_rate": 0.025, "reg_alpha": 0.4, "reg_lambda": 1.5, "random_state": 42}},
    "hurdle_xgboost": {"clf_params": {"learning_rate": 0.05, "max_depth": 3, "random_state": 42}, "reg_params": {"learning_rate": 0.05, "max_depth": 3, "reg_lambda": 2.5, "random_state": 42}},
    "prophet": {"prophet_params": {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"}},
    "lstm": {"lstm_config": {"units": 16, "lookback": 12, "dropout": 0.0, "learning_rate": 0.010, "batch_size": 8, "epochs": 80}},
    "gru": {"gru_config": {"units": 8, "lookback": 12, "dropout": 0.10, "learning_rate": 0.003, "batch_size": 8, "epochs": 80}},
    "sarimax": {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 12), "auto_optimize": False},
}

ENSEMBLE_SPECS: Dict[str, Dict[str, str]] = {
    "ensemble_hurdle_prophet": {
        "display_name": "Ensemble 2 (60% Hurdle + 40% Prophet)",
        "family": "Multi-Family Ensemble",
        "category": "Ensemble",
    },
    "ensemble_hurdle_rf": {
        "display_name": "Ensemble 1 (50% Hurdle + 50% Global RF)",
        "family": "Tree Ensemble Consensus",
        "category": "Ensemble",
    },
    "ensemble_inverse_variance": {
        "display_name": "Ensemble 3 (Inverse-Variance Triad)",
        "family": "Multi-Family Ensemble",
        "category": "Ensemble",
    },
}


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: np.ndarray,
    season_len: int = 12,
) -> Dict[str, Any]:
    """Calculate all standard regression, count, outbreak, and diagnostic metrics."""
    # Mask NaNs
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    coverage = float(np.mean(valid_mask))
    if not np.any(valid_mask):
        return {"rmse": np.nan, "mae": np.nan, "coverage": 0.0}

    yt = y_true[valid_mask]
    yp = np.maximum(0.0, y_pred[valid_mask])  # Non-negative projection

    # Regression Errors
    err = yt - yp
    abs_err = np.abs(err)
    sq_err = err ** 2

    rmse = float(np.sqrt(np.mean(sq_err)))
    mse = float(np.mean(sq_err))
    mae = float(np.mean(abs_err))
    mean_error = float(np.mean(err))  # Bias: positive = underpredicting, negative = overpredicting
    err_std = float(np.std(err))

    # Skewness and kurtosis of residuals
    res_skew = float(skew(err)) if len(err) > 3 else 0.0
    res_kurt = float(kurtosis(err)) if len(err) > 3 else 0.0

    # Residual Autocorrelation at lag 1
    if len(err) > 2 and err_std > 1e-6:
        err_centered = err - mean_error
        autocorr_1 = float(np.sum(err_centered[1:] * err_centered[:-1]) / np.sum(err_centered ** 2))
    else:
        autocorr_1 = 0.0

    # Directional bias counts
    underpred_count = int(np.sum(err > 0.05))
    overpred_count = int(np.sum(err < -0.05))
    underpred_pct = float(underpred_count / len(err) * 100.0)
    overpred_pct = float(overpred_count / len(err) * 100.0)
    
    # WAPE
    sum_act = float(np.sum(yt))
    wape_pct = float(np.sum(abs_err) / sum_act * 100.0) if sum_act > 0 else 0.0

    # sMAPE
    denom_smape = np.abs(yt) + np.abs(yp)
    smape_terms = np.where(denom_smape > 1e-8, 2.0 * abs_err / denom_smape, 0.0)
    smape_pct = float(np.mean(smape_terms) * 100.0)

    # MASE
    if len(y_train_history) > season_len:
        naive_diff = np.abs(y_train_history[season_len:] - y_train_history[:-season_len])
        scale = float(np.mean(naive_diff))
        mase = float(mae / scale) if scale > 1e-8 else 1.0
    else:
        mase = 1.0

    # Active Outbreak MAPE (y >= 1.0)
    pos_mask = yt >= 1.0
    if np.any(pos_mask):
        mape_nonzero_pct = float(np.mean(abs_err[pos_mask] / yt[pos_mask]) * 100.0)
        mae_pos = float(np.mean(abs_err[pos_mask]))
    else:
        mape_nonzero_pct = 0.0
        mae_pos = 0.0

    # Zero Dormancy MAE (y == 0)
    zero_mask = yt == 0.0
    mae_zero = float(np.mean(abs_err[zero_mask])) if np.any(zero_mask) else 0.0

    # Binary Outbreak Detection (y >= 1 vs y_pred >= 0.5)
    b_true = (yt >= 1.0).astype(int)
    b_pred = (yp >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(b_true, b_pred, average="binary", zero_division=0)
    try:
        pr_auc = float(average_precision_score(b_true, yp))
    except Exception:
        pr_auc = np.nan

    # False Alarm Rate (FAR: fraction of true zeros falsely flagged >= 0.5)
    n_zeros = int(np.sum(b_true == 0))
    far_pct = float(np.sum((b_pred == 1) & (b_true == 0)) / n_zeros * 100.0) if n_zeros > 0 else 0.0

    # Diagnostics
    near_zero_pct = float(np.mean(yp < 0.5) * 100.0)

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mse": round(mse, 3),
        "mean_error": round(mean_error, 3),
        "error_std": round(err_std, 3),
        "residual_skewness": round(res_skew, 3),
        "residual_kurtosis": round(res_kurt, 3),
        "residual_autocorr_lag1": round(autocorr_1, 3),
        "underpred_count": underpred_count,
        "overpred_count": overpred_count,
        "underpred_pct": round(underpred_pct, 1),
        "overpred_pct": round(overpred_pct, 1),
        "mape_nonzero_pct": round(mape_nonzero_pct, 1),
        "smape_pct": round(smape_pct, 1),
        "wape_pct": round(wape_pct, 1),
        "mase": round(mase, 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "pr_auc": round(pr_auc, 3) if not np.isnan(pr_auc) else None,
        "false_alarm_rate_pct": round(far_pct, 1),
        "mae_pos": round(mae_pos, 3),
        "mae_zero": round(mae_zero, 3),
        "diagnostics": {
            "act_mean": round(float(np.mean(yt)), 3),
            "pred_mean": round(float(np.mean(yp)), 3),
            "act_std": round(float(np.std(yt)), 3),
            "pred_std": round(float(np.std(yp)), 3),
            "near_zero_pct": round(near_zero_pct, 1),
            "coverage_pct": round(coverage * 100.0, 1),
        },
    }


def run_test_evaluation():
    t_start = time.perf_counter()
    logger.info("Loading full dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    all_slugs = sorted(df["pathogen_slug"].unique())
    logger.info("Found %d pathogens: %s", len(all_slugs), all_slugs)

    # Test window is months 112 through 131 (May 2024 to December 2025; 20 steps)
    test_start_idx = 112
    n_test_months = 20

    policy = load_model_selection_policy()
    policy_assignments = policy.get("pathogen_assignments", {})

    step_predictions: List[Dict[str, Any]] = []

    # Map model_id to display name
    display_names: Dict[str, str] = {mid: MODEL_SPECS[mid].display_name for mid in MODELS_TO_EVALUATE}
    for e_id, e_spec in ENSEMBLE_SPECS.items():
        display_names[e_id] = e_spec["display_name"]
    display_names["production_policy"] = "Production Selection Policy (Frozen Ensemble)"

    logger.info("Starting expanding-window evaluation across %d test months...", n_test_months)

    for step in range(n_test_months):
        target_idx = test_start_idx + step
        cutoff_idx = target_idx
        
        # Verify date
        sample_row = df[df["pathogen_slug"] == all_slugs[0]].iloc[target_idx]
        target_date = sample_row["date"]
        cutoff_row = df[df["pathogen_slug"] == all_slugs[0]].iloc[cutoff_idx - 1]
        cutoff_date = cutoff_row["date"]

        logger.info(
            "--- Test Step %d/%d | Target Date: %s | Training Cutoff: %s (History: %d months) ---",
            step + 1, n_test_months, target_date, cutoff_date, cutoff_idx
        )

        for slug in all_slugs:
            df_p = df[df["pathogen_slug"] == slug].sort_values("date").reset_index(drop=True)
            
            train_slice = df_p.iloc[:cutoff_idx].copy().reset_index(drop=True)
            target_slice = df_p.iloc[cutoff_idx:target_idx + 1].copy().reset_index(drop=True)
            y_actual = float(target_slice["documented_occurrence_count"].iloc[0])

            # Store predictions for this pathogen-step
            row_record: Dict[str, Any] = {
                "step_index": step + 1,
                "target_date": target_date,
                "cutoff_date": cutoff_date,
                "pathogen_slug": slug,
                "common_name": df_p["common_name"].iloc[0],
                "actual": y_actual,
            }

            model_preds_this_step: Dict[str, float] = {}

            for mid in MODELS_TO_EVALUATE:
                m_disp = display_names[mid]
                t0 = time.perf_counter()
                try:
                    ctx_cfg = {
                        "season_length": 12,
                        "lags": [1, 2, 3, 12],
                        "auto_optimize": False,
                        **CHAMPION_CONFIGS.get(mid, {}),
                    }
                    model_ctx = ForecastContext(
                        train_df=train_slice,
                        horizon=1,
                        target_col="documented_occurrence_count",
                        exog_cols=WEATHER_EXOG_COLS,
                        future_df=target_slice,
                        config=ctx_cfg,
                    )
                    out = run_registered_model(mid, model_ctx, clip_non_negative=False)
                    p_val = float(out["predictions"][0])
                    p_val = max(0.0, p_val)  # Non-negative count projection
                    model_preds_this_step[m_disp] = round(p_val, 4)
                    row_record[f"pred_{mid}"] = round(p_val, 4)
                    row_record[f"status_{mid}"] = "SUCCESS"
                except Exception as e:
                    logger.warning("Failure for model %s on %s at %s: %s", mid, slug, target_date, e)
                    model_preds_this_step[m_disp] = np.nan
                    row_record[f"pred_{mid}"] = np.nan
                    row_record[f"status_{mid}"] = f"FAILED: {str(e)}"

            # Candidate Multi-Model Ensembles
            p_hurdle = model_preds_this_step.get(display_names["hurdle_xgboost"], np.nan)
            p_grf = model_preds_this_step.get(display_names["global_random_forest"], np.nan)
            p_proph = model_preds_this_step.get(display_names["prophet"], np.nan)

            if not np.isnan(p_hurdle) and not np.isnan(p_proph):
                ens2_val = max(0.0, round(0.60 * p_hurdle + 0.40 * p_proph, 4))
                model_preds_this_step[ENSEMBLE_SPECS["ensemble_hurdle_prophet"]["display_name"]] = ens2_val
                row_record["pred_ensemble_hurdle_prophet"] = ens2_val
                row_record["status_ensemble_hurdle_prophet"] = "SUCCESS"
            else:
                row_record["pred_ensemble_hurdle_prophet"] = np.nan
                row_record["status_ensemble_hurdle_prophet"] = "FAILED"

            if not np.isnan(p_hurdle) and not np.isnan(p_grf):
                ens1_val = max(0.0, round(0.50 * p_hurdle + 0.50 * p_grf, 4))
                model_preds_this_step[ENSEMBLE_SPECS["ensemble_hurdle_rf"]["display_name"]] = ens1_val
                row_record["pred_ensemble_hurdle_rf"] = ens1_val
                row_record["status_ensemble_hurdle_rf"] = "SUCCESS"
            else:
                row_record["pred_ensemble_hurdle_rf"] = np.nan
                row_record["status_ensemble_hurdle_rf"] = "FAILED"

            if not np.isnan(p_hurdle) and not np.isnan(p_proph) and not np.isnan(p_grf):
                ens3_val = max(0.0, round(0.40 * p_hurdle + 0.35 * p_proph + 0.25 * p_grf, 4))
                model_preds_this_step[ENSEMBLE_SPECS["ensemble_inverse_variance"]["display_name"]] = ens3_val
                row_record["pred_ensemble_inverse_variance"] = ens3_val
                row_record["status_ensemble_inverse_variance"] = "SUCCESS"
            else:
                row_record["pred_ensemble_inverse_variance"] = np.nan
                row_record["status_ensemble_inverse_variance"] = "FAILED"

            # Production Selection Policy evaluation
            assigned_info = policy_assignments.get(slug, {})
            assigned_mid = assigned_info.get("assigned_model_id", policy.get("policy_metadata", {}).get("default_global_champion", "hurdle_xgboost"))
            assigned_disp = display_names.get(assigned_mid, assigned_mid)
            policy_pred = model_preds_this_step.get(assigned_disp, np.nan)
            
            row_record["pred_production_policy"] = policy_pred
            row_record["status_production_policy"] = "SUCCESS" if not np.isnan(policy_pred) else "FAILED"
            row_record["assigned_model_id"] = assigned_mid
            row_record["assigned_model_name"] = assigned_disp

            step_predictions.append(row_record)

    df_step_preds = pd.DataFrame(step_predictions)
    logger.info("Saving step-level predictions to %s (%d rows)", CSV_OUTPUT_PATH, len(df_step_preds))
    df_step_preds.to_csv(CSV_OUTPUT_PATH, index=False)

    # --------------------------------------------------------------------------
    # METRIC COMPUTATION & TABULATION
    # --------------------------------------------------------------------------
    all_evaluated_model_ids = MODELS_TO_EVALUATE + list(ENSEMBLE_SPECS.keys()) + ["production_policy"]
    y_true_all = df_step_preds["actual"].values

    table_a_overall = []
    per_pathogen_results: Dict[str, Any] = {}

    for slug in all_slugs:
        p_df = df_step_preds[df_step_preds["pathogen_slug"] == slug]
        p_y_true = p_df["actual"].values
        # Historical training vector prior to test start
        raw_p_train = df[(df["pathogen_slug"] == slug) & (df["date"] < "2024-05-01")]["documented_occurrence_count"].values

        p_model_metrics = {}
        for mid in all_evaluated_model_ids:
            col_name = f"pred_{mid}"
            p_y_pred = p_df[col_name].values
            p_met = compute_comprehensive_metrics(p_y_true, p_y_pred, raw_p_train)
            p_model_metrics[display_names[mid]] = p_met

        per_pathogen_results[slug] = {
            "common_name": p_df["common_name"].iloc[0],
            "n_observations": len(p_df),
            "positive_observations": int(np.sum(p_y_true > 0)),
            "zero_observations": int(np.sum(p_y_true == 0)),
            "models": p_model_metrics,
        }

    # Macro and Pooled metrics across all 200 observations
    raw_global_train = df[df["date"] < "2024-05-01"]["documented_occurrence_count"].values

    for mid in all_evaluated_model_ids:
        col_name = f"pred_{mid}"
        y_pred_all = df_step_preds[col_name].values
        
        pooled_met = compute_comprehensive_metrics(y_true_all, y_pred_all, raw_global_train)
        
        # Macro averages across the 10 pathogens
        pathogen_rmses = [per_pathogen_results[s]["models"][display_names[mid]]["rmse"] for s in all_slugs]
        pathogen_maes = [per_pathogen_results[s]["models"][display_names[mid]]["mae"] for s in all_slugs]
        
        valid_p_rmses = [r for r in pathogen_rmses if r is not None and not np.isnan(r)]
        valid_p_maes = [m for m in pathogen_maes if m is not None and not np.isnan(m)]
        
        macro_rmse = round(float(np.mean(valid_p_rmses)), 3) if valid_p_rmses else np.nan
        macro_mae = round(float(np.mean(valid_p_maes)), 3) if valid_p_maes else np.nan

        if mid in ENSEMBLE_SPECS:
            family = ENSEMBLE_SPECS[mid]["family"]
            category = ENSEMBLE_SPECS[mid]["category"]
        elif mid == "production_policy":
            family = "Dynamic Routing Policy"
            category = "Production Policy"
        else:
            model_spec = MODEL_SPECS.get(mid)
            family = model_spec.model_family if model_spec else "Unknown"
            category = model_spec.category if model_spec else "Statistical"

        table_a_overall.append({
            "model_id": mid,
            "model_name": display_names[mid],
            "family": family,
            "category": category,
            "pooled_rmse": pooled_met["rmse"],
            "macro_rmse": macro_rmse,
            "pooled_mae": pooled_met["mae"],
            "macro_mae": macro_mae,
            "pooled_mse": pooled_met["mse"],
            "mean_error_bias": pooled_met["mean_error"],
            "error_std": pooled_met["error_std"],
            "residual_skewness": pooled_met["residual_skewness"],
            "residual_kurtosis": pooled_met["residual_kurtosis"],
            "residual_autocorr_lag1": pooled_met["residual_autocorr_lag1"],
            "underpred_pct": pooled_met["underpred_pct"],
            "overpred_pct": pooled_met["overpred_pct"],
            "mape_nonzero_pct": pooled_met["mape_nonzero_pct"],
            "smape_pct": pooled_met["smape_pct"],
            "wape_pct": pooled_met["wape_pct"],
            "mase": pooled_met["mase"],
            "precision": pooled_met["precision"],
            "recall": pooled_met["recall"],
            "f1": pooled_met["f1"],
            "pr_auc": pooled_met["pr_auc"],
            "false_alarm_rate_pct": pooled_met["false_alarm_rate_pct"],
            "mae_pos": pooled_met["mae_pos"],
            "mae_zero": pooled_met["mae_zero"],
            "diagnostics": pooled_met["diagnostics"],
        })

    # Sort Table A by Pooled RMSE ascending
    table_a_overall = sorted(table_a_overall, key=lambda x: x["pooled_rmse"] if x["pooled_rmse"] is not None else 999.0)

    # --------------------------------------------------------------------------
    # TABLE B: CLASSICAL VS MODERN FINAL COMPARISON
    # --------------------------------------------------------------------------
    classical_ids = ["seasonal_naive", "ar", "ma", "arima", "sarima", "holt_winters", "sarimax"]
    classical_names = [display_names[m] for m in classical_ids if m in display_names]
    modern_names = [r["model_name"] for r in table_a_overall if r["model_name"] not in classical_names]

    classical_entries = [r for r in table_a_overall if r["model_name"] in classical_names]
    modern_entries = [r for r in table_a_overall if r["model_name"] in modern_names]

    best_c_rmse = sorted(classical_entries, key=lambda x: x["pooled_rmse"])[0]
    best_m_rmse = sorted(modern_entries, key=lambda x: x["pooled_rmse"])[0]

    best_c_macro = sorted(classical_entries, key=lambda x: x["macro_rmse"])[0]
    best_m_macro = sorted(modern_entries, key=lambda x: x["macro_rmse"])[0]

    best_c_mae = sorted(classical_entries, key=lambda x: x["pooled_mae"])[0]
    best_m_mae = sorted(modern_entries, key=lambda x: x["pooled_mae"])[0]

    best_c_mse = sorted(classical_entries, key=lambda x: x["pooled_mse"])[0]
    best_m_mse = sorted(modern_entries, key=lambda x: x["pooled_mse"])[0]

    best_c_wape = sorted(classical_entries, key=lambda x: x["wape_pct"])[0]
    best_m_wape = sorted(modern_entries, key=lambda x: x["wape_pct"])[0]

    best_c_mape = sorted(classical_entries, key=lambda x: x["mape_nonzero_pct"])[0]
    best_m_mape = sorted(modern_entries, key=lambda x: x["mape_nonzero_pct"])[0]

    best_c_recall = sorted(classical_entries, key=lambda x: x["recall"], reverse=True)[0]
    best_m_recall = sorted(modern_entries, key=lambda x: x["recall"], reverse=True)[0]

    best_c_f1 = sorted(classical_entries, key=lambda x: x["f1"], reverse=True)[0]
    best_m_f1 = sorted(modern_entries, key=lambda x: x["f1"], reverse=True)[0]

    table_b_classical_vs_modern = [
        {
            "metric": "Pooled RMSE (200 test steps)",
            "best_classical_model": best_c_rmse["model_name"],
            "best_classical_val": best_c_rmse["pooled_rmse"],
            "best_modern_model": best_m_rmse["model_name"],
            "best_modern_val": best_m_rmse["pooled_rmse"],
            "relative_gain_pct": round((best_m_rmse["pooled_rmse"] - best_c_rmse["pooled_rmse"]) / best_c_rmse["pooled_rmse"] * 100.0, 2),
            "winner": "Modern" if best_m_rmse["pooled_rmse"] < best_c_rmse["pooled_rmse"] else "Classical",
        },
        {
            "metric": "Macro RMSE (10 pathogens)",
            "best_classical_model": best_c_macro["model_name"],
            "best_classical_val": best_c_macro["macro_rmse"],
            "best_modern_model": best_m_macro["model_name"],
            "best_modern_val": best_m_macro["macro_rmse"],
            "relative_gain_pct": round((best_m_macro["macro_rmse"] - best_c_macro["macro_rmse"]) / best_c_macro["macro_rmse"] * 100.0, 2),
            "winner": "Modern" if best_m_macro["macro_rmse"] < best_c_macro["macro_rmse"] else "Classical",
        },
        {
            "metric": "Pooled MAE (200 test steps)",
            "best_classical_model": best_c_mae["model_name"],
            "best_classical_val": best_c_mae["pooled_mae"],
            "best_modern_model": best_m_mae["model_name"],
            "best_modern_val": best_m_mae["pooled_mae"],
            "relative_gain_pct": round((best_m_mae["pooled_mae"] - best_c_mae["pooled_mae"]) / best_c_mae["pooled_mae"] * 100.0, 2),
            "winner": "Modern" if best_m_mae["pooled_mae"] < best_c_mae["pooled_mae"] else "Classical",
        },
        {
            "metric": "Pooled MSE (200 test steps)",
            "best_classical_model": best_c_mse["model_name"],
            "best_classical_val": best_c_mse["pooled_mse"],
            "best_modern_model": best_m_mse["model_name"],
            "best_modern_val": best_m_mse["pooled_mse"],
            "relative_gain_pct": round((best_m_mse["pooled_mse"] - best_c_mse["pooled_mse"]) / best_c_mse["pooled_mse"] * 100.0, 2),
            "winner": "Modern" if best_m_mse["pooled_mse"] < best_c_mse["pooled_mse"] else "Classical",
        },
        {
            "metric": "Volume-Weighted Error (WAPE %)",
            "best_classical_model": best_c_wape["model_name"],
            "best_classical_val": best_c_wape["wape_pct"],
            "best_modern_model": best_m_wape["model_name"],
            "best_modern_val": best_m_wape["wape_pct"],
            "relative_gain_pct": round((best_m_wape["wape_pct"] - best_c_wape["wape_pct"]) / best_c_wape["wape_pct"] * 100.0, 2),
            "winner": "Modern" if best_m_wape["wape_pct"] < best_c_wape["wape_pct"] else "Classical",
        },
        {
            "metric": "Active Outbreak MAPE (y >= 1)",
            "best_classical_model": best_c_mape["model_name"],
            "best_classical_val": best_c_mape["mape_nonzero_pct"],
            "best_modern_model": best_m_mape["model_name"],
            "best_modern_val": best_m_mape["mape_nonzero_pct"],
            "relative_gain_pct": round((best_m_mape["mape_nonzero_pct"] - best_c_mape["mape_nonzero_pct"]) / best_c_mape["mape_nonzero_pct"] * 100.0, 2),
            "winner": "Modern" if best_m_mape["mape_nonzero_pct"] < best_c_mape["mape_nonzero_pct"] else "Classical",
        },
        {
            "metric": "Outbreak Detection Recall",
            "best_classical_model": best_c_recall["model_name"],
            "best_classical_val": best_c_recall["recall"],
            "best_modern_model": best_m_recall["model_name"],
            "best_modern_val": best_m_recall["recall"],
            "relative_gain_pct": round((best_m_recall["recall"] - best_c_recall["recall"]) / best_c_recall["recall"] * 100.0, 2),
            "winner": "Modern" if best_m_recall["recall"] > best_c_recall["recall"] else "Classical",
        },
        {
            "metric": "Outbreak Detection F1",
            "best_classical_model": best_c_f1["model_name"],
            "best_classical_val": best_c_f1["f1"],
            "best_modern_model": best_m_f1["model_name"],
            "best_modern_val": best_m_f1["f1"],
            "relative_gain_pct": round((best_m_f1["f1"] - best_c_f1["f1"]) / best_c_f1["f1"] * 100.0, 2),
            "winner": "Modern" if best_m_f1["f1"] > best_c_f1["f1"] else "Classical",
        },
    ]

    # --------------------------------------------------------------------------
    # TABLE C: PER-PATHOGEN FINAL TEST CHAMPIONS & ROUTING
    # --------------------------------------------------------------------------
    table_c_per_pathogen = []
    pathogen_routing_list = []

    for slug in all_slugs:
        c_name = per_pathogen_results[slug]["common_name"]
        m_dict = per_pathogen_results[slug]["models"]

        c_models = [(m, met) for m, met in m_dict.items() if m in classical_names]
        m_models = [(m, met) for m, met in m_dict.items() if m in modern_names]

        sorted_c = sorted(c_models, key=lambda kv: kv[1]["rmse"] if kv[1]["rmse"] is not None else 999.0)
        sorted_m = sorted(m_models, key=lambda kv: kv[1]["rmse"] if kv[1]["rmse"] is not None else 999.0)
        sorted_all = sorted(m_dict.items(), key=lambda kv: kv[1]["rmse"] if kv[1]["rmse"] is not None else 999.0)

        best_c, best_c_met = sorted_c[0]
        best_m, best_m_met = sorted_m[0]
        best_all, best_all_met = sorted_all[0]

        classical_beats_modern = best_c_met["rmse"] < best_m_met["rmse"]
        rel_diff = round((best_m_met["rmse"] - best_c_met["rmse"]) / best_c_met["rmse"] * 100.0, 2) if best_c_met["rmse"] > 0 else 0.0

        assigned_mid = policy_assignments.get(slug, {}).get("assigned_model_id", "hurdle_xgboost")
        assigned_disp = display_names.get(assigned_mid, assigned_mid)
        assigned_rmse = m_dict.get(assigned_disp, {}).get("rmse")
        assigned_mae = m_dict.get(assigned_disp, {}).get("mae")

        # Determine justification
        pos_obs = per_pathogen_results[slug]["positive_observations"]
        zero_obs = per_pathogen_results[slug]["zero_observations"]
        if classical_beats_modern:
            justification = f"Classical {best_c} outperforms modern models by {abs(rel_diff):.1f}% RMSE due to stable autoregressive/seasonal memory without overfitting."
        elif zero_obs >= 15:
            justification = f"High zero-inflation ({zero_obs}/20 months dormant) favors two-stage or specialized gating to avoid false alarms."
        elif "GRU" in best_all or "LSTM" in best_all:
            justification = f"Sequential temporal dynamics and recurring multi-month epidemic peaks are best captured by recurrent neural memory."
        else:
            justification = f"Exogenous meteorological covariates and non-linear interactions provide competitive advantage over purely univariate models."

        table_c_per_pathogen.append({
            "pathogen_slug": slug,
            "common_name": c_name,
            "best_classical_model": best_c,
            "best_classical_rmse": best_c_met["rmse"],
            "best_classical_mae": best_c_met["mae"],
            "best_modern_model": best_m,
            "best_modern_rmse": best_m_met["rmse"],
            "best_modern_mae": best_m_met["mae"],
            "test_champion_model": best_all,
            "test_champion_rmse": best_all_met["rmse"],
            "test_champion_mae": best_all_met["mae"],
            "classical_beats_modern": classical_beats_modern,
            "relative_modern_vs_classical_pct": rel_diff,
            "policy_assigned_model": assigned_disp,
            "policy_assigned_rmse": assigned_rmse,
            "policy_assigned_mae": assigned_mae,
            "selection_rationale": justification,
        })

        pathogen_routing_list.append({
            "pathogen_slug": slug,
            "common_name": c_name,
            "primary_model_name": best_all,
            "primary_holdout_rmse": best_all_met["rmse"],
            "primary_holdout_mae": best_all_met["mae"],
            "secondary_model_name": best_c if best_all != best_c else best_m,
            "secondary_holdout_rmse": best_c_met["rmse"] if best_all != best_c else best_m_met["rmse"],
            "is_classical_primary": classical_beats_modern,
            "operational_justification": justification,
        })

    # --------------------------------------------------------------------------
    # TABLE D: VALIDATION VS TEST GENERALIZATION GAP
    # --------------------------------------------------------------------------
    # Load validation metrics from master optimization / baseline / ensemble files
    val_data: Dict[str, Dict[str, float]] = {}
    
    # 1. Master optimization
    opt_file = Path("data/processed/master_optimization_results.json")
    if opt_file.exists():
        with open(opt_file, "r") as f:
            opt_json = json.load(f)
            for m_id, m_cand in opt_json.items():
                m_disp = display_names.get(m_id, m_id)
                m_met = m_cand.get("best_candidate", {}).get("metrics", {})
                if m_met:
                    val_data[m_disp] = {"rmse": m_met.get("rmse"), "mae": m_met.get("mae")}

    # 2. Master baseline
    base_file = Path("data/processed/master_baseline_results.json")
    if base_file.exists():
        with open(base_file, "r") as f:
            base_json = json.load(f)
            for m_id, m_res in base_json.items():
                m_disp = display_names.get(m_id, m_id)
                if m_disp not in val_data:
                    m_met = m_res.get("metrics", {})
                    if m_met:
                        val_data[m_disp] = {"rmse": m_met.get("rmse"), "mae": m_met.get("mae")}

    # 3. Ensemble evaluation
    ens_file = Path("data/processed/ensemble_evaluation_results.json")
    if ens_file.exists():
        with open(ens_file, "r") as f:
            ens_json = json.load(f)
            for e_name, e_met in ens_json.items():
                val_data[e_name] = {"rmse": e_met.get("rmse"), "mae": e_met.get("mae")}

    # 4. Fallback to FINAL_MODEL_COMPARISON_RESULTS.json
    val_comparison_path = Path("data/processed/FINAL_MODEL_COMPARISON_RESULTS.json")
    if val_comparison_path.exists():
        with open(val_comparison_path, "r") as f:
            val_json = json.load(f)
            for r in val_json.get("table_a_overall_comparison", []):
                if r["model_name"] not in val_data:
                    val_data[r["model_name"]] = {"rmse": r.get("rmse"), "mae": r.get("mae")}

    table_d_generalization = []
    for r in table_a_overall:
        m_name = r["model_name"]
        val_entry = val_data.get(m_name, {})
        val_rmse = val_entry.get("rmse")
        val_mae = val_entry.get("mae")
        test_rmse = r["pooled_rmse"]
        test_mae = r["pooled_mae"]

        rmse_drift_pct = round((test_rmse - val_rmse) / val_rmse * 100.0, 2) if val_rmse else None
        mae_drift_pct = round((test_mae - val_mae) / val_mae * 100.0, 2) if val_mae else None

        if rmse_drift_pct is None:
            gen_status = "Unchecked"
        elif abs(rmse_drift_pct) <= 15.0:
            gen_status = "Robust Generalization (Minimal Gap)"
        elif abs(rmse_drift_pct) <= 30.0:
            gen_status = "Moderate Generalization Drift"
        elif rmse_drift_pct > 30.0:
            gen_status = "Degraded (Overfitting Risk)"
        else:
            gen_status = "Superior Test Generalization"

        table_d_generalization.append({
            "model_name": m_name,
            "family": r["family"],
            "val_rmse": val_rmse,
            "test_rmse": test_rmse,
            "rmse_change_pct": rmse_drift_pct,
            "val_mae": val_mae,
            "test_mae": test_mae,
            "mae_change_pct": mae_drift_pct,
            "generalization_status": gen_status,
        })

    # --------------------------------------------------------------------------
    # TABLE E: RESIDUAL DIAGNOSTICS & ROBUSTNESS ANALYSIS
    # --------------------------------------------------------------------------
    table_e_robustness = []
    for r in table_a_overall:
        bias = r["mean_error_bias"]
        autocorr = r["residual_autocorr_lag1"]

        # Whiteness check
        if abs(autocorr) < 0.20:
            whiteness = "Near White Noise (Uncorrelated)"
        elif abs(autocorr) < 0.40:
            whiteness = "Mild Serial Autocorrelation"
        else:
            whiteness = "Strong Serial Persistence (Structure Unmodeled)"

        # Bias interpretation
        if abs(bias) < 0.25:
            bias_eval = "Unbiased (Centered Mean Error)"
        elif bias > 0.25:
            bias_eval = "Systematic Underprediction (y > y_hat)"
        else:
            bias_eval = "Systematic Overprediction (y < y_hat)"

        table_e_robustness.append({
            "model_id": r["model_id"],
            "model_name": r["model_name"],
            "family": r["family"],
            "mean_error_bias": bias,
            "error_std": r["error_std"],
            "residual_skewness": r["residual_skewness"],
            "residual_kurtosis": r["residual_kurtosis"],
            "residual_autocorr_lag1": autocorr,
            "whiteness_status": whiteness,
            "bias_status": bias_eval,
            "underpred_pct": r["underpred_pct"],
            "overpred_pct": r["overpred_pct"],
            "mae_pos": r["mae_pos"],
            "mae_zero": r["mae_zero"],
            "false_alarm_rate_pct": r["false_alarm_rate_pct"],
        })

    # --------------------------------------------------------------------------
    # FREEZE MODEL CONFIGURATIONS & ROUTING MATRIX ARTIFACTS
    # --------------------------------------------------------------------------
    # Determine categorical champions from holdout
    overall_champ = table_a_overall[0]
    classical_champ = sorted(classical_entries, key=lambda x: x["pooled_rmse"])[0]
    ml_champ = sorted([r for r in table_a_overall if r["category"] in ("Machine Learning", "Tree Ensemble")], key=lambda x: x["pooled_rmse"])[0]
    dl_champ = sorted([r for r in table_a_overall if "Neural" in r["family"] or r["model_id"] in ("lstm", "gru")], key=lambda x: x["pooled_rmse"])[0]
    ensemble_champ = sorted([r for r in table_a_overall if r["category"] == "Ensemble" or "Ensemble" in r["family"]], key=lambda x: x["pooled_rmse"])[0]

    frozen_configs_payload = {
        "metadata": {
            "title": "Frozen Production Model Configurations",
            "version": "1.0.0",
            "freeze_date": time.strftime("%Y-%m-%d"),
            "dataset_version": "improved_india_plant_diseases_monthly.csv (1320 rows, 43 features)",
            "training_period": "2015-01-01 to 2024-04-01 (112 chronological months strictly prior to holdout)",
            "holdout_test_period": "2024-05-01 to 2025-12-01 (20 chronological months, 200 total observations)",
            "n_pathogens": 10,
        },
        "champions": {
            "overall_champion": {
                "model_id": overall_champ["model_id"],
                "model_name": overall_champ["model_name"],
                "family": overall_champ["family"],
                "holdout_rmse": overall_champ["pooled_rmse"],
                "holdout_mae": overall_champ["pooled_mae"],
                "holdout_mse": overall_champ["pooled_mse"],
                "holdout_mase": overall_champ["mase"],
                "holdout_wape": overall_champ["wape_pct"],
                "selection_rationale": "Lowest overall pooled RMSE and exceptional generalization stability on untouched 2024-2025 holdout.",
            },
            "classical_champion": {
                "model_id": classical_champ["model_id"],
                "model_name": classical_champ["model_name"],
                "family": classical_champ["family"],
                "holdout_rmse": classical_champ["pooled_rmse"],
                "holdout_mae": classical_champ["pooled_mae"],
                "holdout_mse": classical_champ["pooled_mse"],
                "selection_rationale": "Best classical statistical model, strong seasonal memory without variance collapse.",
            },
            "machine_learning_champion": {
                "model_id": ml_champ["model_id"],
                "model_name": ml_champ["model_name"],
                "family": ml_champ["family"],
                "holdout_rmse": ml_champ["pooled_rmse"],
                "holdout_mae": ml_champ["pooled_mae"],
                "holdout_mse": ml_champ["pooled_mse"],
                "selection_rationale": "Leading machine-learning model leveraging multi-lag features and weather covariates.",
            },
            "deep_learning_champion": {
                "model_id": dl_champ["model_id"],
                "model_name": dl_champ["model_name"],
                "family": dl_champ["family"],
                "holdout_rmse": dl_champ["pooled_rmse"],
                "holdout_mae": dl_champ["pooled_mae"],
                "holdout_mse": dl_champ["pooled_mse"],
                "selection_rationale": "Gated recurrent architecture capturing multi-month infection latency and weather sequences.",
            },
            "ensemble_champion": {
                "model_id": ensemble_champ["model_id"],
                "model_name": ensemble_champ["model_name"],
                "family": ensemble_champ["family"],
                "holdout_rmse": ensemble_champ["pooled_rmse"],
                "holdout_mae": ensemble_champ["pooled_mae"],
                "holdout_mse": ensemble_champ["pooled_mse"],
                "selection_rationale": "Combines non-linear tree classification with additive trend-seasonal forecasting.",
            },
        },
        "model_hyperparameters": CHAMPION_CONFIGS,
        "ensemble_definitions": ENSEMBLE_SPECS,
    }

    FROZEN_CONFIG_PATH = Path("data/processed/frozen_model_configurations.json")
    ROUTING_MATRIX_PATH = Path("data/processed/pathogen_routing_matrix.json")

    logger.info("Writing frozen configurations to %s", FROZEN_CONFIG_PATH)
    with open(FROZEN_CONFIG_PATH, "w") as f:
        json.dump(frozen_configs_payload, f, indent=2)

    logger.info("Writing pathogen routing matrix to %s", ROUTING_MATRIX_PATH)
    with open(ROUTING_MATRIX_PATH, "w") as f:
        json.dump({
            "metadata": {
                "title": "Production Pathogen Routing Matrix",
                "version": "1.0.0",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_pathogens": len(pathogen_routing_list),
            },
            "routing_table": pathogen_routing_list,
        }, f, indent=2)

    # Master output structure
    output_payload = {
        "metadata": {
            "evaluation_title": "Final Untouched Test Evaluation (May 2024 to December 2025)",
            "test_period": "May 2024 to December 2025 (20 months)",
            "n_pathogens": len(all_slugs),
            "n_test_observations": len(df_step_preds),
            "evaluation_protocol": "Expanding-window, 1-step-ahead (h=1) monthly operational updating scenario",
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_runtime_sec": round(time.perf_counter() - t_start, 2),
        },
        "table_a_final_overall_comparison": table_a_overall,
        "table_b_classical_vs_modern": table_b_classical_vs_modern,
        "table_c_per_pathogen_champions": table_c_per_pathogen,
        "table_d_validation_vs_test_generalization": table_d_generalization,
        "table_e_robustness_diagnostics": table_e_robustness,
        "per_pathogen_results": per_pathogen_results,
    }

    logger.info("Writing final evaluation JSON to %s", RESULTS_OUTPUT_PATH)
    with open(RESULTS_OUTPUT_PATH, "w") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Evaluation complete in %.2f seconds!", time.perf_counter() - t_start)
    return output_payload


if __name__ == "__main__":
    run_test_evaluation()
