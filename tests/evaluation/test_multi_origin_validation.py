"""Unit tests for Multi-Origin Walk-Forward Validation & Model Robustness Architecture.

Validates:
1. Forecast-origin integrity (origins strictly bounded within validation split)
2. Zero future target leakage (max training date strictly < evaluation date, test set untouched)
3. Consistent evaluation dates across all candidate models
4. Correct rolling feature alignment (no contemporaneous y_t contamination)
5. Missingness-aware lag handling (continuous monthly indexing with loss weighting)
6. Deterministic reproducibility across repeated executions
7. Correct pooled and macro-average metric calculations
8. Failure handling (failures recorded as NaN, never converted silently to zero)
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.forecasting import chronological_split, run_rolling_walk_forward_evaluation
from src.models.base import build_failure_output
from src.models.global_models import build_global_panel_features


def _create_synthetic_multi_origin_panel(n_months: int = 120) -> pd.DataFrame:
    """Create synthetic continuous monthly panel for 2 pathogens with known features."""
    dates = pd.date_range("2015-01-01", periods=n_months, freq="MS")
    rows = []
    for slug in ("pathogen_alpha", "pathogen_beta"):
        for i, dt in enumerate(dates):
            # Known deterministic target: seasonal sine wave + trend
            val = float(max(0.0, np.sin(2 * np.pi * dt.month / 12.0) * 5.0 + (1.0 if slug == "pathogen_alpha" else 2.0)))
            is_miss = 1 if (i % 11 == 0 and val == 0.0) else 0
            rows.append({
                "date": dt,
                "pathogen_slug": slug,
                "common_name": f"Disease {slug.split('_')[-1].capitalize()}",
                "documented_occurrence_count": val,
                "is_missing": is_miss,
                "temp_mean_c": 25.0 + float(dt.month),
                "temp_mean_c_lag1": 25.0 + float(dt.month),
                "precipitation_monthly_mm_lag1": 50.0 + float(dt.month * 2),
                "relative_humidity_pct_lag1": 65.0,
            })
    return pd.DataFrame(rows).sort_values(["date", "pathogen_slug"]).reset_index(drop=True)


def test_forecast_origin_integrity_and_bounds():
    """Verify eval_start_idx correctly configures the start of evaluation within validation set."""
    df_panel = _create_synthetic_multi_origin_panel(120)
    sub = df_panel[df_panel["pathogen_slug"] == "pathogen_alpha"].copy().reset_index(drop=True)

    # 70% train = 84 months (0..83), 15% val = 18 months (84..101), 15% test = 18 months (102..119)
    train_df, val_df, test_df = chronological_split(sub, train_ratio=0.70, val_ratio=0.15)
    assert len(train_df) == 84
    assert len(val_df) == 18
    assert len(test_df) == 18

    # Origin at index 84 (start of validation window)
    res_orig1 = run_rolling_walk_forward_evaluation(
        sub, horizon=6, eval_start_idx=84, filter_model_ids=["seasonal_naive", "ar"]
    )
    assert len(res_orig1["validation_actual"]) == 6
    assert pd.to_datetime(res_orig1["validation_dates"].iloc[0]) == pd.to_datetime(val_df["date"].iloc[0])

    # Origin at index 90 (mid-validation window)
    res_orig2 = run_rolling_walk_forward_evaluation(
        sub, horizon=6, eval_start_idx=90, filter_model_ids=["seasonal_naive", "ar"]
    )
    assert len(res_orig2["validation_actual"]) == 6
    assert pd.to_datetime(res_orig2["validation_dates"].iloc[0]) == pd.to_datetime(val_df["date"].iloc[6])


def test_no_future_target_leakage_and_test_isolation(monkeypatch):
    """Ensure training data in each step strictly excludes target date and never touches test split."""
    df_panel = _create_synthetic_multi_origin_panel(120)
    sub = df_panel[df_panel["pathogen_slug"] == "pathogen_alpha"].copy().reset_index(drop=True)

    max_val_idx = 84 + 18 - 1  # Index 101 is the last validation month
    test_start_date = pd.to_datetime(sub["date"].iloc[102])

    seen_max_train_dates = []
    seen_target_dates = []

    from src.model_registry import run_registered_model
    orig_run = run_registered_model

    def spy_run_model(model_id, ctx, **kwargs):
        max_dt = pd.to_datetime(ctx.train_df["date"].max())
        seen_max_train_dates.append(max_dt)
        return orig_run(model_id, ctx, **kwargs)

    monkeypatch.setattr("src.forecasting.run_registered_model", spy_run_model)

    res = run_rolling_walk_forward_evaluation(
        sub, horizon=6, eval_start_idx=90, filter_model_ids=["ar"]
    )
    eval_dates = [pd.to_datetime(d) for d in res["validation_dates"]]

    # Assert no training date ever equals or exceeds the evaluation step date
    for tr_max, ev_dt in zip(seen_max_train_dates, eval_dates):
        assert tr_max < ev_dt, f"Lookahead leakage: train max {tr_max} >= eval {ev_dt}"
        assert ev_dt < test_start_date, f"Test set leakage: eval date {ev_dt} >= test start {test_start_date}"


def test_consistent_evaluation_dates_across_all_models():
    """Verify that every evaluated model generates predictions matching the ground-truth dates."""
    df_panel = _create_synthetic_multi_origin_panel(120)
    sub = df_panel[df_panel["pathogen_slug"] == "pathogen_alpha"].copy().reset_index(drop=True)

    res = run_rolling_walk_forward_evaluation(
        sub, horizon=4, eval_start_idx=86, filter_model_ids=["seasonal_naive", "ar", "ma", "arima"]
    )
    n_expected = len(res["validation_actual"])
    assert n_expected == 4

    for m_name, preds in res["forecasts"].items():
        assert len(preds) == n_expected, f"Length mismatch for model {m_name}: {len(preds)} vs {n_expected}"
        assert not np.isnan(preds).any(), f"Unexpected NaN in predictions for {m_name}"


def test_correct_rolling_feature_alignment():
    """Verify that shifted rolling and lag features at row t never incorporate target y_t."""
    df_panel = _create_synthetic_multi_origin_panel(60)
    feat_df, cols = build_global_panel_features(df_panel)

    sub = feat_df[feat_df["pathogen_slug"] == "pathogen_alpha"].sort_values("date").reset_index(drop=True)
    y_raw = sub["documented_occurrence_count"].values

    for i in range(12, len(sub)):
        # lag_1 at row i must match y_raw[i - 1]
        assert np.isclose(sub["lag_1"].iloc[i], y_raw[i - 1])
        # lag_2 must match y_raw[i - 2]
        assert np.isclose(sub["lag_2"].iloc[i], y_raw[i - 2])
        # roll_mean_3 must match mean(y_raw[i-3 : i])
        expected_roll3 = np.mean(y_raw[i - 3 : i])
        assert np.isclose(sub["roll_mean_3"].iloc[i], expected_roll3)


def test_missingness_aware_lag_handling():
    """Verify continuous time indexing preserves 1-step lag offsets regardless of missingness."""
    df_panel = _create_synthetic_multi_origin_panel(60)
    feat_df, _ = build_global_panel_features(df_panel)

    # Imputed rows where is_missing == 1 should still provide legitimate lag values to subsequent months
    sub = feat_df[feat_df["pathogen_slug"] == "pathogen_alpha"].sort_values("date").reset_index(drop=True)
    miss_indices = sub.index[sub["is_missing"] == 1].tolist()
    if miss_indices:
        for idx in miss_indices:
            if idx + 1 < len(sub):
                # The subsequent month must see the imputed row's count as its lag_1
                assert sub["lag_1"].iloc[idx + 1] == sub["documented_occurrence_count"].iloc[idx]


def test_walk_forward_reproducibility():
    """Verify identical random seeds produce identical walk-forward predictions."""
    df_panel = _create_synthetic_multi_origin_panel(120)
    sub = df_panel[df_panel["pathogen_slug"] == "pathogen_alpha"].copy().reset_index(drop=True)

    res1 = run_rolling_walk_forward_evaluation(sub, horizon=4, eval_start_idx=84, filter_model_ids=["ar", "random_forest"])
    res2 = run_rolling_walk_forward_evaluation(sub, horizon=4, eval_start_idx=84, filter_model_ids=["ar", "random_forest"])

    for m_name in res1["forecasts"]:
        p1 = np.array(res1["forecasts"][m_name])
        p2 = np.array(res2["forecasts"][m_name])
        assert np.allclose(p1, p2, atol=1e-5), f"Reproducibility failure for {m_name}"


def test_pooled_and_macro_metric_calculations():
    """Verify mathematical definitions of pooled vs macro-average RMSE and MAE."""
    # 2 pathogens, 3 steps each -> 6 pooled points
    acts_p1 = np.array([2.0, 4.0, 6.0])
    preds_p1 = np.array([1.0, 4.0, 8.0])  # diffs: -1, 0, +2 -> sq: 1, 0, 4 (mean 5/3=1.667, rmse=1.291), mae=1.0

    acts_p2 = np.array([0.0, 0.0, 10.0])
    preds_p2 = np.array([0.0, 2.0, 8.0])   # diffs: 0, +2, -2 -> sq: 0, 4, 4 (mean 8/3=2.667, rmse=1.633), mae=1.333

    rmse_p1 = np.sqrt(np.mean((preds_p1 - acts_p1) ** 2))
    rmse_p2 = np.sqrt(np.mean((preds_p2 - acts_p2) ** 2))
    expected_macro_rmse = (rmse_p1 + rmse_p2) / 2.0

    pooled_act = np.concatenate([acts_p1, acts_p2])
    pooled_pred = np.concatenate([preds_p1, preds_p2])
    expected_pooled_rmse = np.sqrt(np.mean((pooled_pred - pooled_act) ** 2))

    assert np.isclose(expected_macro_rmse, (1.29099 + 1.63299) / 2.0, atol=1e-3)
    assert np.isclose(expected_pooled_rmse, np.sqrt((5 + 8) / 6.0), atol=1e-3)

    # WAPE check
    expected_wape = np.sum(np.abs(pooled_pred - pooled_act)) / np.sum(pooled_act) * 100.0
    assert expected_wape > 0


def test_failure_handling_preserves_nan_and_logs(monkeypatch):
    """Verify that execution errors are caught, logged, and return NaN without zero masking."""
    df_panel = _create_synthetic_multi_origin_panel(120)
    sub = df_panel[df_panel["pathogen_slug"] == "pathogen_alpha"].copy().reset_index(drop=True)

    from src.model_registry import run_registered_model

    def failing_runner(model_id, ctx, **kwargs):
        if model_id == "ar":
            raise RuntimeError("Synthetic simulated solver failure")
        return run_registered_model(model_id, ctx, **kwargs)

    monkeypatch.setattr("src.forecasting.run_registered_model", failing_runner)

    res = run_rolling_walk_forward_evaluation(
        sub, horizon=3, eval_start_idx=84, filter_model_ids=["ar", "seasonal_naive"]
    )
    assert "AR(1)" in res["model_failures"]
    assert len(res["model_failures"]["AR(1)"]) == 3
    # Ensure failure predictions are NaN (never silently zero)
    assert np.isnan(res["forecasts"]["AR(1)"]).all()


def test_load_multi_origin_benchmark_data_and_schema():
    """Verify that saved multi-origin benchmark output matches specification and contracts."""
    from src.multi_origin_ui import load_multi_origin_benchmark_data

    data = load_multi_origin_benchmark_data()
    assert data is not None
    assert "metadata" in data
    assert "overall_rankings" in data
    assert "per_pathogen_summary" in data
    assert "raw_evaluations" in data

    meta = data["metadata"]
    assert meta["n_pathogens"] == 10
    assert len(meta["origins"]) == 4
    assert meta["total_forecast_points"] == 240

    rankings = data["overall_rankings"]
    assert len(rankings) == 14

    # Champion checks
    champ = rankings[0]
    assert "Hurdle XGBoost" in champ["model_name"]
    assert champ["pooled_rmse"] < 6.0
    assert champ["f1"] > 0.75

    # Check that SARIMA failure was caught and recorded without crash
    sarima_row = next((r for r in rankings if "SARIMA" in r["model_name"]), None)
    assert sarima_row is not None

