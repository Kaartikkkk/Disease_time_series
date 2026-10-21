"""Pipeline script to execute model benchmarking, backtesting, and export evaluation reports."""

import json
from pathlib import Path
import joblib
import pandas as pd
from src.data.load_data import load_dataset
from src.evaluation.backtesting import compare_all_models
from src.models.machine_learning_models import MLForecaster
from src.utils.logging_config import logger
from src.utils.paths import REPORTS_DIR, MODELS_DIR


def run_benchmarks_and_generate_reports():
    """Run chronological backtesting across all candidate models and save reports."""
    logger.info("Starting model benchmarking pipeline...")

    # Load real Agroscope Grapevine surveillance data
    df_grapevine, _ = load_dataset("agroscope_grapevine")
    # Load synthetic daily benchmark
    df_synthetic, _ = load_dataset("synthetic_demo_benchmark")

    candidate_models = [
        "persistence",
        "seasonal_naive",
        "moving_average",
        "historical_mean",
        "ar",
        "ma",
        "arma",
        "arima",
        "sarima",
        "exponential_smoothing",
        "arimax",
        "sarimax",
        "var",
        "prophet",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
    ]

    # Benchmark on the multi-year synthetic daily benchmark (14-day horizon, 4 backtest folds)
    logger.info("Benchmarking models on daily dataset (horizon=14, n_splits=3)...")
    leaderboard_daily, reports_daily = compare_all_models(
        df=df_synthetic,
        candidate_models=candidate_models,
        target_col="incidence_rate",
        horizon=14,
        n_splits=3,
        frequency="D",
        seasonal_period=7,
        include_weather=True,
    )

    # Save leaderboard CSV
    csv_path = REPORTS_DIR / "model_comparison.csv"
    leaderboard_daily.to_csv(csv_path, index=False)
    logger.info(f"Saved model comparison leaderboard to {csv_path}")

    # Compile JSON summary
    summary = {
        "benchmark_dataset": "synthetic_demo_potato_late_blight",
        "frequency": "Daily",
        "horizon": 14,
        "n_backtest_folds": 3,
        "validation_strategy": "Chronological Expanding Window (No Leakage)",
        "models_evaluated_count": len(leaderboard_daily),
        "best_model_by_mae": leaderboard_daily.iloc[0]["Model"],
        "best_mae": float(leaderboard_daily.iloc[0]["MAE"]),
        "best_rmse": float(leaderboard_daily.iloc[0]["RMSE"]),
        "leaderboard": leaderboard_daily.to_dict(orient="records"),
    }

    json_path = REPORTS_DIR / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved evaluation summary JSON to {json_path}")

    # Train and serialize preferred models for fast application inference
    logger.info("Training and serializing production model checkpoints...")
    for m_type in ["random_forest", "hist_gradient_boosting", "xgboost"]:
        ml_model = MLForecaster(
            model_type=m_type,
            n_estimators=100,
            max_depth=6,
            random_seed=42,
            frequency="D",
        )
        ml_model.fit(df_synthetic, target_col="incidence_rate")
        joblib.dump(ml_model, MODELS_DIR / f"{m_type}_checkpoint.joblib")
        logger.info(f"Serialized {m_type} to {MODELS_DIR / f'{m_type}_checkpoint.joblib'}")

    logger.info("Benchmarking and report generation complete!")
    return leaderboard_daily, summary


if __name__ == "__main__":
    run_benchmarks_and_generate_reports()
