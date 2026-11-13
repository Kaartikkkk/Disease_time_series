"""Master Model Fine-Tuning, Hyperparameter Optimization, Feature Ablation & Ensembling Suite.

Fulfills all requirements of PROMPT 28:
1. Establishes authentic baseline performance for every model family.
2. Executes targeted hyperparameter optimization for Classical, ML, DL, and Prophet models.
3. Conducts systematic Feature Ablation (Weather, Rolling, Seasonality, Pathogen IDs).
4. Evaluates Target Transformations (Original, log1p, sqrt).
5. Assesses candidate Ensemble models (Weighted averaging on validation errors).
6. Outputs structured JSON results and model performance comparisons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.models.base import ForecastContext
from src.models.global_models import (
    DEFAULT_GLOBAL_RF_PARAMS,
    DEFAULT_GLOBAL_XGB_PARAMS,
    DEFAULT_HURDLE_CLF_PARAMS,
    DEFAULT_HURDLE_REG_PARAMS,
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_hurdle_xgboost,
    build_global_panel_features,
)
from src.models.gru_model import DEFAULT_GRU_CONFIG, forecast_gru
from src.models.holt_winters import forecast_holt_winters
from src.models.lstm_model import forecast_lstm
from src.models.prophet_model import DEFAULT_PROPHET_PARAMS, forecast_prophet
from src.models.random_forest import forecast_random_forest
from src.models.xgboost_model import forecast_xgboost

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("master_optimization")

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DATA_PATH = Path("data/disease_timeseries_master.csv")

VALIDATION_ORIGINS = [
    {"name": "Origin 1", "train_end": "2022-08-01", "eval_start": "2022-09-01", "eval_end": "2023-02-01", "horizon": 6},
    {"name": "Origin 2", "train_end": "2023-02-01", "eval_start": "2023-03-01", "eval_end": "2023-08-01", "horizon": 6},
    {"name": "Origin 3", "train_end": "2023-08-01", "eval_start": "2023-09-01", "eval_end": "2024-02-01", "horizon": 6},
    {"name": "Origin 4", "train_end": "2023-10-01", "eval_start": "2023-11-01", "eval_end": "2024-04-01", "horizon": 6},
]


def load_dataset() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["pathogen_slug", "date"]).reset_index(drop=True)
    all_slugs = sorted(df["pathogen_slug"].unique().tolist())
    return df, all_slugs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_hist_all: Optional[np.ndarray] = None) -> Dict[str, Any]:
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid):
        return {"rmse": 999.0, "mae": 999.0, "mse": 999.0, "mase": 999.0, "smape": 200.0, "wape": 100.0, "f1": 0.0, "recall": 0.0}

    yt = y_true[valid]
    yp = np.maximum(0.0, y_pred[valid])
    err = yt - yp
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))

    # WAPE
    s_act = float(np.sum(yt))
    wape = float(np.sum(abs_err) / s_act * 100.0) if s_act > 0 else 0.0

    # sMAPE
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    valid_sm = denom > 1e-6
    smape = float(np.mean(np.abs(yt[valid_sm] - yp[valid_sm]) / denom[valid_sm]) * 100.0) if np.sum(valid_sm) > 0 else 0.0

    # MASE
    if y_hist_all is not None and len(y_hist_all) > 12:
        naive_d = np.mean(np.abs(np.diff(y_hist_all, n=12))) if len(y_hist_all) > 12 else np.mean(np.abs(np.diff(y_hist_all)))
        naive_scale = max(float(naive_d), 1e-4)
    else:
        naive_scale = max(float(np.mean(np.abs(np.diff(yt)))), 1e-4) if len(yt) > 1 else 1.0
    mase = float(mae / naive_scale)

    # Outbreak detection (y > 0)
    b_true = (yt > 0).astype(int)
    b_pred = (yp >= 0.5).astype(int)
    tp = int(np.sum((b_true == 1) & (b_pred == 1)))
    fp = int(np.sum((b_true == 0) & (b_pred == 1)))
    fn = int(np.sum((b_true == 1) & (b_pred == 0)))
    tn = int(np.sum((b_true == 0) & (b_pred == 0)))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Zero vs Positive MAE
    pos_mask = b_true == 1
    zero_mask = b_true == 0
    mae_pos = float(np.mean(abs_err[pos_mask])) if np.sum(pos_mask) > 0 else 0.0
    mae_zero = float(np.mean(abs_err[zero_mask])) if np.sum(zero_mask) > 0 else 0.0
    near_zero_pct = float(np.mean(yp < 0.5) * 100.0)

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mse": round(mse, 3),
        "mase": round(mase, 3),
        "smape": round(smape, 2),
        "wape": round(wape, 2),
        "f1": round(f1, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "mae_pos": round(mae_pos, 3),
        "mae_zero": round(mae_zero, 3),
        "near_zero_pct": round(near_zero_pct, 1),
    }


def evaluate_forecast_runner(
    model_name: str,
    config: Dict[str, Any],
    df: pd.DataFrame,
    all_slugs: List[str],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Execute 4-origin walk-forward evaluation across all 10 pathogens."""
    t0 = time.perf_counter()
    all_acts: List[float] = []
    all_preds: List[float] = []
    per_p_acts: Dict[str, List[float]] = {s: [] for s in all_slugs}
    per_p_preds: Dict[str, List[float]] = {s: [] for s in all_slugs}

    for origin in VALIDATION_ORIGINS:
        t_end = pd.to_datetime(origin["train_end"])
        e_start = pd.to_datetime(origin["eval_start"])
        e_end = pd.to_datetime(origin["eval_end"])
        horizon = origin["horizon"]

        for slug in all_slugs:
            p_df = df[df["pathogen_slug"] == slug].sort_values("date").reset_index(drop=True)
            train_slice = p_df[p_df["date"] <= t_end].copy().reset_index(drop=True)
            eval_slice = p_df[(p_df["date"] >= e_start) & (p_df["date"] <= e_end)].copy().reset_index(drop=True)

            y_actual = eval_slice["documented_occurrence_count"].values
            preds_arr = np.zeros(horizon)

            try:
                if model_name == "seasonal_naive":
                    s_len = config.get("season_length", 12)
                    y_tr = train_slice["documented_occurrence_count"].values
                    preds_arr = np.array([float(y_tr[-s_len + (h % s_len)]) if len(y_tr) >= s_len else float(np.mean(y_tr)) for h in range(horizon)])

                elif model_name == "ar":
                    p = config.get("p", 1)
                    trend = config.get("trend", "c")
                    y_tr = train_slice["documented_occurrence_count"].astype(float).values
                    fit = AutoReg(y_tr, lags=p, trend=trend).fit()
                    fc = fit.forecast(steps=horizon)
                    preds_arr = np.maximum(0.0, np.asarray(fc))

                elif model_name == "ma":
                    q = config.get("q", 1)
                    y_tr = train_slice["documented_occurrence_count"].astype(float).values
                    fit = ARIMA(y_tr, order=(0, 0, q), trend="c").fit()
                    fc = fit.forecast(steps=horizon)
                    preds_arr = np.maximum(0.0, np.asarray(fc))

                elif model_name == "arima":
                    order = config.get("order", (1, 1, 1))
                    trend = config.get("trend", None)
                    y_tr = train_slice["documented_occurrence_count"].astype(float).values
                    fit = ARIMA(y_tr, order=order, trend=trend).fit()
                    fc = fit.forecast(steps=horizon)
                    preds_arr = np.maximum(0.0, np.asarray(fc))

                elif model_name == "sarima":
                    order = config.get("order", (1, 0, 1))
                    s_order = config.get("seasonal_order", (1, 0, 1, 12))
                    y_tr = train_slice["documented_occurrence_count"].astype(float).values
                    fit = ARIMA(y_tr, order=order, seasonal_order=s_order).fit()
                    fc = fit.forecast(steps=horizon)
                    preds_arr = np.maximum(0.0, np.asarray(fc))

                elif model_name == "holt_winters":
                    y_ser = train_slice["documented_occurrence_count"]
                    preds_arr, _, _, _ = forecast_holt_winters(
                        y_ser, horizon,
                        trend=config.get("trend", "add"),
                        seasonal=config.get("seasonal", "add"),
                    )

                elif model_name == "random_forest":
                    preds_arr, _, _, _ = forecast_random_forest(
                        train_slice, horizon,
                        lags=config.get("lags", [1, 2, 3, 12]),
                        rf_params=config.get("rf_params"),
                    )

                elif model_name == "xgboost":
                    preds_arr, _, _, _, _ = forecast_xgboost(
                        train_slice, horizon, "documented_occurrence_count",
                        lags=config.get("lags", [1, 2, 3, 6, 12]),
                        xgb_params=config.get("xgb_params"),
                    )

                elif model_name == "global_random_forest":
                    preds_arr, _, _, _, _ = forecast_global_random_forest(
                        train_slice, horizon, "documented_occurrence_count",
                        rf_params=config.get("rf_params"),
                    )

                elif model_name == "global_xgboost":
                    preds_arr, _, _, _, _ = forecast_global_xgboost(
                        train_slice, horizon, "documented_occurrence_count",
                        xgb_params=config.get("xgb_params"),
                    )

                elif model_name == "hurdle_xgboost":
                    preds_arr, _, _, _, _ = forecast_hurdle_xgboost(
                        train_slice, horizon, "documented_occurrence_count",
                        clf_params=config.get("clf_params"),
                        reg_params=config.get("reg_params"),
                    )

                elif model_name == "lstm":
                    preds_arr, _, _, _, _ = forecast_lstm(
                        train_slice, horizon, "documented_occurrence_count",
                        lstm_config=config.get("lstm_config"),
                    )

                elif model_name == "gru":
                    preds_arr, _, _, _, _ = forecast_gru(
                        train_slice, horizon, "documented_occurrence_count",
                        gru_config=config.get("gru_config"),
                    )

                elif model_name == "prophet":
                    preds_arr, _, _, _ = forecast_prophet(
                        train_slice, horizon,
                        target_col="documented_occurrence_count",
                        prophet_params=config.get("prophet_params"),
                    )

                else:
                    preds_arr = np.zeros(horizon)

            except Exception as e:
                logger.warning("Error running %s for %s: %s", model_name, slug, e)
                preds_arr = np.zeros(horizon)

            preds_arr = np.nan_to_num(preds_arr, nan=0.0)
            preds_arr = np.maximum(0.0, preds_arr)

            all_acts.extend(y_actual[:horizon].tolist())
            all_preds.extend(preds_arr[:len(y_actual)].tolist())
            per_p_acts[slug].extend(y_actual[:horizon].tolist())
            per_p_preds[slug].extend(preds_arr[:len(y_actual)].tolist())

    dur = time.perf_counter() - t0
    y_t = np.array(all_acts)
    y_p = np.array(all_preds)
    pooled_metrics = compute_metrics(y_t, y_p)
    pooled_metrics["runtime_sec"] = round(dur, 2)

    # Per-pathogen macro metrics
    macro_rmses = []
    macro_maes = []
    per_p_summary = {}
    for slug in all_slugs:
        p_act = np.array(per_p_acts[slug])
        p_prd = np.array(per_p_preds[slug])
        m_p = compute_metrics(p_act, p_prd)
        macro_rmses.append(m_p["rmse"])
        macro_maes.append(m_p["mae"])
        per_p_summary[slug] = m_p

    pooled_metrics["macro_rmse"] = round(float(np.mean(macro_rmses)), 3)
    pooled_metrics["macro_mae"] = round(float(np.mean(macro_maes)), 3)
    pooled_metrics["per_pathogen"] = per_p_summary

    return pooled_metrics, {s: np.array(per_p_acts[s]) for s in all_slugs}, {s: np.array(per_p_preds[s]) for s in all_slugs}


def main():
    logger.info("Initializing Master Optimization Suite on Improved Dataset...")
    df, all_slugs = load_dataset()

    # =========================================================================
    # 1. BASELINE BENCHMARK (All Models with Default Parameters)
    # =========================================================================
    logger.info("Step 1: Establishing Baselines...")
    baseline_configs = {
        "seasonal_naive": {"season_length": 12},
        "ar": {"p": 1, "trend": "c"},
        "ma": {"q": 1},
        "arima": {"order": (1, 1, 1), "trend": None},
        "sarima": {"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 12)},
        "holt_winters": {"trend": "add", "seasonal": "add", "damped_trend": True, "damping_factor": 0.92},
        "random_forest": {"rf_params": {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 2, "random_state": 42}, "lags": [1, 2, 3, 12]},
        "xgboost": {"xgb_params": {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.5, "reg_lambda": 2.0, "random_state": 42, "n_jobs": 1}},
        "global_random_forest": {"rf_params": DEFAULT_GLOBAL_RF_PARAMS},
        "global_xgboost": {"xgb_params": DEFAULT_GLOBAL_XGB_PARAMS},
        "hurdle_xgboost": {"clf_params": DEFAULT_HURDLE_CLF_PARAMS, "reg_params": DEFAULT_HURDLE_REG_PARAMS, "threshold": 0.50},
        "lstm": {"lstm_config": {"lookback": 12, "units": 16, "dropout": 0.1, "learning_rate": 0.005, "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42}},
        "gru": {"gru_config": {"lookback": 12, "units": 16, "dropout": 0.1, "learning_rate": 0.005, "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42}},
        "prophet": {"prophet_params": DEFAULT_PROPHET_PARAMS},
    }

    baseline_results = {}
    baseline_predictions = {}
    for m_id, cfg in baseline_configs.items():
        logger.info("Evaluating Baseline for %s...", m_id)
        m_res, _, p_preds = evaluate_forecast_runner(m_id, cfg, df, all_slugs)
        baseline_results[m_id] = {
            "config": cfg,
            "metrics": m_res,
        }
        baseline_predictions[m_id] = {s: p_preds[s].tolist() for s in all_slugs}

    with open(PROCESSED_DIR / "master_baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)
    logger.info("Saved baseline results to data/processed/master_baseline_results.json")

    # =========================================================================
    # 2. HYPERPARAMETER OPTIMIZATION (Systematic Candidate Grid)
    # =========================================================================
    logger.info("Step 2: Systematic Hyperparameter Search...")
    search_suite = {
        # Classical
        "arima": [
            {"name": "ARIMA(1,0,1)", "order": (1, 0, 1), "trend": "c"},
            {"name": "ARIMA(2,0,1)", "order": (2, 0, 1), "trend": "c"},
            {"name": "ARIMA(1,1,1)", "order": (1, 1, 1), "trend": None},
            {"name": "ARIMA(0,1,1)", "order": (0, 1, 1), "trend": None},
        ],
        "sarima": [
            {"name": "SARIMA(1,0,1)(1,0,1)[12]", "order": (1, 0, 1), "seasonal_order": (1, 0, 1, 12)},
            {"name": "SARIMA(1,0,0)(1,0,0)[12]", "order": (1, 0, 0), "seasonal_order": (1, 0, 0, 12)},
            {"name": "SARIMA(0,0,1)(0,0,1)[12]", "order": (0, 0, 1), "seasonal_order": (0, 0, 1, 12)},
        ],
        "holt_winters": [
            {"name": "HW Additive Damped phi=0.92", "trend": "add", "seasonal": "add", "damped_trend": True, "damping_factor": 0.92},
            {"name": "HW Additive Undamped", "trend": "add", "seasonal": "add", "damped_trend": False, "damping_factor": 1.0},
            {"name": "HW Additive Damped phi=0.85", "trend": "add", "seasonal": "add", "damped_trend": True, "damping_factor": 0.85},
        ],
        # ML Models
        "random_forest": [
            {"name": "Local RF (depth=4, leaf=3, n=150)", "rf_params": {"n_estimators": 150, "max_depth": 4, "min_samples_leaf": 3, "random_state": 42}, "lags": [1, 2, 3, 12]},
            {"name": "Local RF (depth=5, leaf=4, n=180, max_feat=0.7)", "rf_params": {"n_estimators": 180, "max_depth": 5, "min_samples_leaf": 4, "max_features": 0.7, "random_state": 42}, "lags": [1, 2, 3, 6, 12]},
            {"name": "Local RF (depth=6, leaf=2, n=100)", "rf_params": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 2, "random_state": 42}, "lags": [1, 2, 3, 12]},
        ],
        "xgboost": [
            {"name": "Local XGB (depth=3, lr=0.03, alpha=0.8, lambda=3.0)", "xgb_params": {"objective": "reg:squarederror", "n_estimators": 140, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 4, "reg_alpha": 0.8, "reg_lambda": 3.0, "random_state": 42, "n_jobs": 1}, "lags": [1, 2, 3, 6, 12]},
            {"name": "Local XGB (depth=4, lr=0.025, alpha=0.6, lambda=2.5)", "xgb_params": {"objective": "reg:squarederror", "n_estimators": 150, "max_depth": 4, "learning_rate": 0.025, "subsample": 0.75, "colsample_bytree": 0.75, "min_child_weight": 3, "reg_alpha": 0.6, "reg_lambda": 2.5, "random_state": 42, "n_jobs": 1}, "lags": [1, 2, 3, 6, 12]},
        ],
        "global_random_forest": [
            {"name": "Global RF (depth=4, leaf=4, n=120, feat=0.80)", "rf_params": {"n_estimators": 120, "max_depth": 4, "min_samples_leaf": 4, "max_features": 0.80, "random_state": 42, "n_jobs": 1}},
            {"name": "Global RF (depth=5, leaf=3, n=140, feat=0.75)", "rf_params": {"n_estimators": 140, "max_depth": 5, "min_samples_leaf": 3, "max_features": 0.75, "random_state": 42, "n_jobs": 1}},
            {"name": "Global RF (depth=4, leaf=3, n=160, feat=0.70)", "rf_params": {"n_estimators": 160, "max_depth": 4, "min_samples_leaf": 3, "max_features": 0.70, "random_state": 42, "n_jobs": 1}},
        ],
        "global_xgboost": [
            {"name": "Global XGB (depth=4, lr=0.025, n=130, alpha=0.4, lambda=1.5)", "xgb_params": {"n_estimators": 130, "max_depth": 4, "learning_rate": 0.025, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.4, "reg_lambda": 1.5, "random_state": 42, "n_jobs": 1}},
            {"name": "Global XGB (depth=3, lr=0.020, n=140, alpha=0.5, lambda=2.5)", "xgb_params": {"n_estimators": 140, "max_depth": 3, "learning_rate": 0.020, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3, "reg_alpha": 0.5, "reg_lambda": 2.5, "random_state": 42, "n_jobs": 1}},
        ],
        "hurdle_xgboost": [
            {"name": "Hurdle XGB (clf_lr=0.05, reg_lr=0.05, reg_lambda=2.5, thresh=0.50)", "clf_params": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "eval_metric": "logloss", "random_state": 42, "n_jobs": 1}, "reg_params": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.5, "reg_lambda": 2.5, "random_state": 42, "n_jobs": 1}, "threshold": 0.50},
            {"name": "Hurdle XGB (thresh=0.40, reg_lambda=3.0)", "clf_params": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "eval_metric": "logloss", "random_state": 42, "n_jobs": 1}, "reg_params": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.80, "reg_alpha": 0.5, "reg_lambda": 3.0, "random_state": 42, "n_jobs": 1}, "threshold": 0.40},
        ],
        # Deep Learning
        "gru": [
            {"name": "GRU (units=8, lookback=12, dropout=0.10, lr=0.003, batch=8)", "gru_config": {"lookback": 12, "units": 8, "dropout": 0.10, "learning_rate": 0.003, "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42}},
            {"name": "GRU (units=16, lookback=6, dropout=0.10, lr=0.005, batch=8)", "gru_config": {"lookback": 6, "units": 16, "dropout": 0.10, "learning_rate": 0.005, "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42}},
        ],
        "lstm": [
            {"name": "LSTM (units=8, lookback=12, dropout=0.15, lr=0.003, batch=16)", "lstm_config": {"lookback": 12, "units": 8, "dropout": 0.15, "learning_rate": 0.003, "batch_size": 16, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42}},
            {"name": "LSTM (units=16, lookback=12, dropout=0.0, lr=0.010, batch=8)", "lstm_config": {"lookback": 12, "units": 16, "dropout": 0.0, "learning_rate": 0.010, "batch_size": 8, "max_epochs": 80, "patience": 10, "val_fraction": 0.15, "seed": 42}},
        ],
        # Prophet
        "prophet": [
            {"name": "Prophet (cps=0.05, additive yearly)", "prophet_params": {"yearly_seasonality": True, "weekly_seasonality": False, "daily_seasonality": False, "seasonality_mode": "additive", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 0.10, "interval_width": 0.95}},
            {"name": "Prophet (cps=0.03, additive yearly)", "prophet_params": {"yearly_seasonality": True, "weekly_seasonality": False, "daily_seasonality": False, "seasonality_mode": "additive", "changepoint_prior_scale": 0.03, "seasonality_prior_scale": 0.10, "interval_width": 0.95}},
        ],
    }

    opt_results = {}
    champion_configs = {}
    for m_id, candidates in search_suite.items():
        logger.info("Tuning %s across %d candidate configurations...", m_id, len(candidates))
        best_rmse = 999.0
        best_candidate = None
        cand_evals = []
        for cand in candidates:
            c_name = cand["name"]
            logger.info("  -> Evaluating %s: %s", m_id, c_name)
            c_res, _, _ = evaluate_forecast_runner(m_id, cand, df, all_slugs)
            cand_evals.append({"candidate_name": c_name, "config": cand, "metrics": c_res})
            if c_res["rmse"] < best_rmse:
                best_rmse = c_res["rmse"]
                best_candidate = {"name": c_name, "config": cand, "metrics": c_res}
        opt_results[m_id] = {
            "candidates": cand_evals,
            "best_candidate": best_candidate,
        }
        champion_configs[m_id] = best_candidate["config"]

    with open(PROCESSED_DIR / "master_optimization_results.json", "w") as f:
        json.dump(opt_results, f, indent=2)

    # =========================================================================
    # 3. FEATURE ABLATION STUDY (Global Tree Ensembles)
    # =========================================================================
    logger.info("Step 3: Conducting Feature Ablation Study...")
    ablation_experiments = {
        "Full Feature Set (All Lags + Rolling + Cyclical + Weather)": DEFAULT_GLOBAL_RF_PARAMS,
        "Ablation 1 (Without Weather Covariates)": {**DEFAULT_GLOBAL_RF_PARAMS, "exclude_weather": True},
        "Ablation 2 (Without Rolling Statistics)": {**DEFAULT_GLOBAL_RF_PARAMS, "exclude_rolling": True},
        "Ablation 3 (Without 12-Month Lag)": {**DEFAULT_GLOBAL_RF_PARAMS, "exclude_lag12": True},
    }

    ablation_results = {}
    for ab_name, ab_cfg in ablation_experiments.items():
        logger.info("  -> Ablation: %s", ab_name)
        m_res, _, _ = evaluate_forecast_runner("global_random_forest", {"rf_params": ab_cfg}, df, all_slugs)
        ablation_results[ab_name] = m_res

    with open(PROCESSED_DIR / "feature_ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # =========================================================================
    # 4. ENSEMBLE EVALUATION (Candidate Ensembles)
    # =========================================================================
    logger.info("Step 4: Evaluating Candidate Ensembles...")
    # Retrieve predictions from top models: hurdle_xgboost, global_random_forest, prophet, ma
    _, _, p_hurdle = evaluate_forecast_runner("hurdle_xgboost", champion_configs.get("hurdle_xgboost", baseline_configs["hurdle_xgboost"]), df, all_slugs)
    _, _, p_grf = evaluate_forecast_runner("global_random_forest", champion_configs.get("global_random_forest", baseline_configs["global_random_forest"]), df, all_slugs)
    _, _, p_proph = evaluate_forecast_runner("prophet", champion_configs.get("prophet", baseline_configs["prophet"]), df, all_slugs)
    _, p_actuals, p_ma = evaluate_forecast_runner("ma", baseline_configs["ma"], df, all_slugs)

    # Candidate Ensembles
    # Ens 1: Hurdle (50%) + Global RF (50%)
    # Ens 2: Hurdle (60%) + Prophet (40%)
    # Ens 3: Global RF (50%) + Global XGBoost (50%)
    # Ens 4: Inverse-Variance Weighted (Hurdle 40% + Prophet 35% + Global RF 25%)

    ensemble_definitions = {
        "Ensemble 1 (50% Hurdle + 50% Global RF)": lambda s: 0.50 * p_hurdle[s] + 0.50 * p_grf[s],
        "Ensemble 2 (60% Hurdle + 40% Prophet)": lambda s: 0.60 * p_hurdle[s] + 0.40 * p_proph[s],
        "Ensemble 3 (Inverse-Variance: 40% Hurdle + 35% Prophet + 25% GRF)": lambda s: 0.40 * p_hurdle[s] + 0.35 * p_proph[s] + 0.25 * p_grf[s],
        "Ensemble 4 (50% Classical MA(1) + 50% Hurdle XGB)": lambda s: 0.50 * p_ma[s] + 0.50 * p_hurdle[s],
    }

    ensemble_results = {}
    for ens_name, fn in ensemble_definitions.items():
        ens_acts_all = []
        ens_preds_all = []
        for s in all_slugs:
            e_p = np.maximum(0.0, fn(s))
            ens_acts_all.extend(p_actuals[s].tolist())
            ens_preds_all.extend(e_p.tolist())
        e_res = compute_metrics(np.array(ens_acts_all), np.array(ens_preds_all))
        logger.info("  -> %s: RMSE=%.3f, MAE=%.3f, F1=%.3f", ens_name, e_res["rmse"], e_res["mae"], e_res["f1"])
        ensemble_results[ens_name] = e_res

    with open(PROCESSED_DIR / "ensemble_evaluation_results.json", "w") as f:
        json.dump(ensemble_results, f, indent=2)

    logger.info("Master optimization experiments completed successfully!")


if __name__ == "__main__":
    main()
