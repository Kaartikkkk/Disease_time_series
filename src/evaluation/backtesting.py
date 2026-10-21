"""Rolling-origin and expanding-window chronological backtesting framework."""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from src.evaluation.metrics import calculate_metrics, MetricScores
from src.models.machine_learning_models import MLForecaster
from src.models.advanced_classical_models import (
    ARForecaster,
    MAForecaster,
    ARMAForecaster,
    ARIMAXForecaster,
    SARIMAXForecaster,
    VARForecaster,
    ProphetForecaster,
    AutoTSForecaster,
)
from src.models.model_registry import get_model, list_available_models
from src.utils.logging_config import logger


@dataclass
class BacktestFoldResult:
    fold_idx: int
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: MetricScores


@dataclass
class ModelEvaluationReport:
    model_name: str
    target_column: str
    frequency: str
    horizon: int
    n_folds: int
    mean_mae: float
    std_mae: float
    mean_rmse: float
    mean_mape: float
    mean_smape: float
    mean_r2: float
    mean_mase: Optional[float]
    total_fit_time_seconds: float
    fold_results: List[BacktestFoldResult]
    all_y_true: np.ndarray
    all_y_pred: np.ndarray
    all_dates: List[pd.Timestamp]


def backtest_single_model(
    model_name: str,
    df: pd.DataFrame,
    target_col: str = "incidence_rate",
    horizon: int = 14,
    n_splits: int = 3,
    backtest_type: str = "expanding",  # 'expanding' or 'rolling'
    frequency: str = "D",
    seasonal_period: int = 7,
    include_weather: bool = True,
    random_seed: int = 42,
) -> ModelEvaluationReport:
    """Execute leakage-free chronological backtesting for a model across multiple folds.

    Expanding-window protocol:
    Fold 0: Train [0 : T0], Test [T0 : T0 + H]
    Fold 1: Train [0 : T0 + H], Test [T0 + H : T0 + 2H]
    Fold 2: Train [0 : T0 + 2H], Test [T0 + 2H : T0 + 3H]
    """
    clean_df = df.sort_values("date").dropna(subset=[target_col]).reset_index(drop=True)
    n = len(clean_df)

    # Minimum points needed: at least 20 points for training + n_splits * horizon
    total_test_points = n_splits * horizon
    if n <= total_test_points + 14:
        # Fall back to 1 split or smaller horizon if series is short
        n_splits = max(1, (n - 14) // horizon)
        total_test_points = n_splits * horizon

    if n_splits < 1:
        raise ValueError(
            f"Series length {n} is too short to evaluate horizon={horizon} with backtesting."
        )

    t0 = n - total_test_points
    fold_results: List[BacktestFoldResult] = []

    all_true_list = []
    all_pred_list = []
    all_dates_list = []

    start_time = time.time()

    for fold in range(n_splits):
        train_end_idx = t0 + fold * horizon
        test_end_idx = train_end_idx + horizon

        if backtest_type == "rolling":
            # Fixed rolling window length equal to initial t0
            train_start_idx = max(0, train_end_idx - t0)
        else:
            # Expanding window from index 0
            train_start_idx = 0

        train_fold = clean_df.iloc[train_start_idx:train_end_idx].copy()
        test_fold = clean_df.iloc[train_end_idx:test_end_idx].copy()

        if test_fold.empty:
            break

        y_true = test_fold[target_col].values
        actual_h = len(y_true)

        # Instantiate fresh model to prevent state leakage between folds
        model = get_model(
            model_name=model_name,
            seasonal_period=seasonal_period,
            frequency=frequency,
            random_seed=random_seed,
            include_weather=include_weather,
        )

        df_fit_models = (
            MLForecaster,
            ARIMAXForecaster,
            SARIMAXForecaster,
            VARForecaster,
            ProphetForecaster,
            AutoTSForecaster,
        )

        try:
            if isinstance(model, df_fit_models):
                if isinstance(model, (ProphetForecaster, AutoTSForecaster)):
                    model.fit(train_fold, target_col=target_col, frequency=frequency)
                else:
                    model.fit(train_fold, target_col=target_col)
            else:
                model.fit(train_fold[target_col])

            y_pred = model.predict(horizon=actual_h)
        except Exception as e:
            logger.warning(f"Fold {fold} failed for {model_name}: {e}. Using persistence fallback.")
            last_val = train_fold[target_col].iloc[-1]
            y_pred = np.full(actual_h, fill_value=last_val)

        metrics = calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_train_history=train_fold[target_col].values,
            seasonal_period=seasonal_period,
        )

        fold_res = BacktestFoldResult(
            fold_idx=fold,
            train_end_date=pd.to_datetime(train_fold["date"].iloc[-1]),
            test_start_date=pd.to_datetime(test_fold["date"].iloc[0]),
            test_end_date=pd.to_datetime(test_fold["date"].iloc[-1]),
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
        )
        fold_results.append(fold_res)

        all_true_list.extend(y_true)
        all_pred_list.extend(y_pred)
        all_dates_list.extend(test_fold["date"].tolist())

    total_time = time.time() - start_time

    # Aggregate metric scores
    mae_list = [f.metrics.mae for f in fold_results]
    rmse_list = [f.metrics.rmse for f in fold_results]
    mape_list = [f.metrics.mape for f in fold_results if not np.isnan(f.metrics.mape)]
    smape_list = [f.metrics.smape for f in fold_results]
    r2_list = [f.metrics.r2 for f in fold_results]
    mase_list = [f.metrics.mase for f in fold_results if f.metrics.mase is not None]

    report = ModelEvaluationReport(
        model_name=model_name,
        target_column=target_col,
        frequency=frequency,
        horizon=horizon,
        n_folds=len(fold_results),
        mean_mae=round(float(np.mean(mae_list)), 4),
        std_mae=round(float(np.std(mae_list)), 4),
        mean_rmse=round(float(np.mean(rmse_list)), 4),
        mean_mape=round(float(np.mean(mape_list)), 2) if mape_list else 0.0,
        mean_smape=round(float(np.mean(smape_list)), 2),
        mean_r2=round(float(np.mean(r2_list)), 4),
        mean_mase=round(float(np.mean(mase_list)), 4) if mase_list else None,
        total_fit_time_seconds=round(total_time, 2),
        fold_results=fold_results,
        all_y_true=np.array(all_true_list),
        all_y_pred=np.array(all_pred_list),
        all_dates=all_dates_list,
    )
    return report


def compare_all_models(
    df: pd.DataFrame,
    candidate_models: Optional[List[str]] = None,
    target_col: str = "incidence_rate",
    horizon: int = 14,
    n_splits: int = 3,
    frequency: str = "D",
    seasonal_period: int = 7,
    include_weather: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, ModelEvaluationReport]]:
    """Evaluate and benchmark multiple models on identical chronological windows.

    Returns:
    - leaderboard_df: Ranked comparison table with MAE, RMSE, sMAPE, R², MASE, Time
    - reports_dict: Full detailed evaluation reports for each candidate model
    """
    if candidate_models is None:
        candidate_models = [
            "persistence",
            "seasonal_naive",
            "moving_average",
            "historical_mean",
            "arima",
            "sarima",
            "exponential_smoothing",
            "random_forest",
            "hist_gradient_boosting",
            "xgboost",
        ]

    reports = {}
    rows = []

    for m in candidate_models:
        try:
            rep = backtest_single_model(
                model_name=m,
                df=df,
                target_col=target_col,
                horizon=horizon,
                n_splits=n_splits,
                frequency=frequency,
                seasonal_period=seasonal_period,
                include_weather=include_weather,
                random_seed=random_seed,
            )
            reports[m] = rep
            rows.append({
                "Model": m,
                "MAE": rep.mean_mae,
                "MAE (Std)": rep.std_mae,
                "RMSE": rep.mean_rmse,
                "MAPE (%)": rep.mean_mape,
                "sMAPE (%)": rep.mean_smape,
                "R²": rep.mean_r2,
                "MASE": rep.mean_mase if rep.mean_mase is not None else np.nan,
                "Fit Time (s)": rep.total_fit_time_seconds,
            })
        except Exception as e:
            logger.warning(f"Could not evaluate model {m}: {e}")

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        # Sort leaderboard by MAE (lower is better)
        leaderboard = leaderboard.sort_values(by="MAE", ascending=True).reset_index(drop=True)
        leaderboard["Rank"] = range(1, len(leaderboard) + 1)
        cols = ["Rank", "Model", "MAE", "MAE (Std)", "RMSE", "MAPE (%)", "sMAPE (%)", "R²", "MASE", "Fit Time (s)"]
        leaderboard = leaderboard[cols]

    return leaderboard, reports
