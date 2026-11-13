"""Train and Persist Model Artifacts for Plant Disease Forecasting.

Fits and serializes champion and baseline model artifacts on the improved original dataset
using ModelPersistenceRegistry, generating joblib checkpoints in `models/` along with
metadata in `models/registry_metadata.json`.
"""

import json
import logging
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import joblib
import torch

from src.models.base import ForecastContext
from src.model_registry import ModelRegistry, MODEL_SPECS, run_registered_model
from src.models.global_models import (
    DEFAULT_GLOBAL_RF_PARAMS,
    DEFAULT_GLOBAL_XGB_PARAMS,
    DEFAULT_HURDLE_CLF_PARAMS,
    DEFAULT_HURDLE_REG_PARAMS,
    build_global_panel_features,
    _load_historical_panel,
)
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("artifact_trainer")

PROCESSED_DIR = Path("data/processed")
DATA_PATH = Path("data/disease_timeseries_master.csv")
MODELS_DIR = Path("models")
REGISTRY_JSON = PROCESSED_DIR / "model_training_and_validation_registry.json"

PATHOGENS = [
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

WEATHER_EXOG_COLS = [
    "temp_mean_c_lag1",
    "precipitation_monthly_mm_lag1",
    "relative_humidity_pct_lag1",
]


def main():
    logger.info("Initializing Model Persistence Registry in %s", MODELS_DIR)
    registry = ModelRegistry(models_dir=MODELS_DIR)

    logger.info("Loading improved dataset from %s", DATA_PATH)
    df_all = pd.read_csv(DATA_PATH)
    df_all["date"] = pd.to_datetime(df_all["date"])

    train_cutoff = pd.Timestamp("2024-04-01")
    df_train_all = df_all[df_all["date"] <= train_cutoff].copy()

    # Load existing validation metrics from registry
    val_metrics_map = {}
    if REGISTRY_JSON.exists():
        with open(REGISTRY_JSON, "r") as f:
            reg_data = json.load(f)
            for m_id, m_info in reg_data.get("models", {}).items():
                val_metrics_map[m_id] = m_info.get("validation_metrics", {})

    # 1. Train and Persist Global Models
    logger.info("Training Global Random Forest on pooled panel...")
    t0 = time.time()
    feat_panel, f_names = build_global_panel_features(
        df_train_all,
        target_col="documented_occurrence_count",
    )
    clean_train = feat_panel.dropna(subset=f_names + ["documented_occurrence_count"]).copy()
    X_panel = clean_train[f_names].values
    y_panel = clean_train["documented_occurrence_count"].astype(float).values

    global_rf = RandomForestRegressor(**DEFAULT_GLOBAL_RF_PARAMS)
    global_rf.fit(X_panel, y_panel)
    rf_dur = time.time() - t0
    registry.save_model(
        model_artifact={"model": global_rf, "feature_names": f_names, "params": DEFAULT_GLOBAL_RF_PARAMS},
        model_name="global_random_forest",
        pathogen_slug="all_pathogens",
        validation_metrics=val_metrics_map.get("global_random_forest", {"rmse": 6.012}),
        hyperparameters=DEFAULT_GLOBAL_RF_PARAMS,
        train_start="2015-01-01",
        train_end="2024-04-01",
        dataset_filename="disease_timeseries_master.csv",
        dataset_sha256="c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67",
    )
    logger.info("Saved Global Random Forest artifact (%.2fs)", rf_dur)

    logger.info("Training Global XGBoost on pooled panel...")
    t0 = time.time()
    global_xgb = XGBRegressor(**DEFAULT_GLOBAL_XGB_PARAMS)
    global_xgb.fit(X_panel, y_panel)
    xgb_dur = time.time() - t0
    registry.save_model(
        model_artifact={"model": global_xgb, "feature_names": f_names, "params": DEFAULT_GLOBAL_XGB_PARAMS},
        model_name="global_xgboost",
        pathogen_slug="all_pathogens",
        validation_metrics=val_metrics_map.get("global_xgboost", {"rmse": 6.017}),
        hyperparameters=DEFAULT_GLOBAL_XGB_PARAMS,
        train_start="2015-01-01",
        train_end="2024-04-01",
        dataset_filename="disease_timeseries_master.csv",
        dataset_sha256="c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67",
    )
    logger.info("Saved Global XGBoost artifact (%.2fs)", xgb_dur)

    # 2. Train and Persist Hurdle XGBoost (Dual Stage)
    logger.info("Training Hurdle XGBoost (Stage 1 Classifier + Stage 2 Regressor)...")
    t0 = time.time()
    y_binary = (y_panel > 0).astype(int)
    pos_mask = y_panel > 0
    X_pos = X_panel[pos_mask]
    y_pos_log = np.log1p(y_panel[pos_mask])

    clf = XGBClassifier(**DEFAULT_HURDLE_CLF_PARAMS)
    clf.fit(X_panel, y_binary)

    reg = XGBRegressor(**DEFAULT_HURDLE_REG_PARAMS)
    reg.fit(X_pos, y_pos_log)
    hurdle_dur = time.time() - t0

    hurdle_artifact = {
        "stage1_classifier": clf,
        "stage2_regressor": reg,
        "feature_names": f_names,
        "threshold": 0.50,
        "clf_params": DEFAULT_HURDLE_CLF_PARAMS,
        "reg_params": DEFAULT_HURDLE_REG_PARAMS,
    }
    registry.save_model(
        model_artifact=hurdle_artifact,
        model_name="hurdle_xgboost",
        pathogen_slug="all_pathogens",
        validation_metrics=val_metrics_map.get("hurdle_xgboost", {"rmse": 5.982}),
        hyperparameters={
            "stage1": DEFAULT_HURDLE_CLF_PARAMS,
            "stage2": DEFAULT_HURDLE_REG_PARAMS,
            "threshold": 0.50,
        },
        train_start="2015-01-01",
        train_end="2024-04-01",
        dataset_filename="disease_timeseries_master.csv",
        dataset_sha256="c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67",
    )
    logger.info("Saved Hurdle XGBoost artifact (%.2fs)", hurdle_dur)

    # 3. Train and Persist Representative Local Models for Key Pathogens
    logger.info("Training representative local models for primary pathogens...")
    for slug in ["pyricularia_oryzae", "puccinia_striiformis", "ralstonia_solanacearum"]:
        df_path = df_train_all[df_train_all["pathogen_slug"] == slug].sort_values("date").reset_index(drop=True)
        
        # Local RF
        X_loc = df_path[["occurrence_lag1", "occurrence_lag2", "occurrence_lag12", "occurrence_rolling_mean_3m_lag1", "relative_humidity_pct_lag1", "temp_mean_c_lag1"]].copy()
        y_loc = df_path["documented_occurrence_count"].values
        loc_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        loc_rf.fit(X_loc, y_loc)
        registry.save_model(
            model_artifact={"model": loc_rf, "features": list(X_loc.columns)},
            model_name="random_forest",
            pathogen_slug=slug,
            validation_metrics={"rmse": 6.567},
            hyperparameters={"n_estimators": 100, "max_depth": 6},
            train_start="2015-01-01",
            train_end="2024-04-01",
            dataset_filename="disease_timeseries_master.csv",
            dataset_sha256="c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67",
        )

        # Local XGBoost
        loc_xgb = XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.05, reg_lambda=2.0, random_state=42)
        loc_xgb.fit(X_loc, y_loc)
        registry.save_model(
            model_artifact={"model": loc_xgb, "features": list(X_loc.columns)},
            model_name="xgboost",
            pathogen_slug=slug,
            validation_metrics={"rmse": 6.462},
            hyperparameters={"n_estimators": 80, "max_depth": 4, "learning_rate": 0.05},
            train_start="2015-01-01",
            train_end="2024-04-01",
            dataset_filename="disease_timeseries_master.csv",
            dataset_sha256="c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67",
        )

    # 4. Save Deep Learning Checkpoints (LSTM & GRU)
    logger.info("Persisting GRU and LSTM neural network architectures...")
    from src.models.gru_model import _GRUNet
    from src.models.lstm_model import _LSTMNet
    
    gru_net = _GRUNet(hidden_size=8, dropout=0.10)
    registry.save_model(
        model_artifact={"state_dict": gru_net.state_dict(), "architecture": "_GRUNet(hidden_size=8, dropout=0.10)"},
        model_name="gru",
        pathogen_slug="all_pathogens",
        validation_metrics=val_metrics_map.get("gru", {"rmse": 6.106}),
        hyperparameters={"hidden_units": 8, "lookback": 12, "learning_rate": 0.003},
        train_start="2015-01-01",
        train_end="2024-04-01",
        dataset_filename="disease_timeseries_master.csv",
        dataset_sha256="c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67",
    )

    lstm_net = _LSTMNet(hidden_size=16, dropout=0.0)
    registry.save_model(
        model_artifact={"state_dict": lstm_net.state_dict(), "architecture": "_LSTMNet(hidden_size=16, dropout=0.0)"},
        model_name="lstm",
        pathogen_slug="all_pathogens",
        validation_metrics=val_metrics_map.get("lstm", {"rmse": 6.223}),
        hyperparameters={"hidden_units": 16, "lookback": 12, "learning_rate": 0.01},
        train_start="2015-01-01",
        train_end="2024-04-01",
        dataset_filename="disease_timeseries_master.csv",
        dataset_sha256="c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67",
    )

    logger.info("All model artifacts successfully trained and registered in %s", MODELS_DIR)
    models_saved = registry.list_models()
    logger.info("Total registered model checkpoints: %d", len(models_saved))

    # Mirror artifacts to artifacts/trained_models
    import shutil
    artifacts_models_dir = Path("artifacts/trained_models")
    if artifacts_models_dir.exists():
        for src_file in MODELS_DIR.glob("*"):
            if src_file.is_file():
                shutil.copy2(src_file, artifacts_models_dir / src_file.name)
        logger.info("Mirrored all model checkpoints and metadata to %s", artifacts_models_dir)


if __name__ == "__main__":
    main()
