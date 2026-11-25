"""Stand-alone Verification Script for VARIMA and AutoTS on Rice Blast (Pyricularia oryzae).

Produces exact verification metrics on:
1. 6-Month Validation Window (Sep 2022 - Feb 2023)
2. 20-Month Untouched Test Split (May 2024 - Dec 2025)
"""

import pandas as pd
import numpy as np
from src.forecasting import chronological_split, compute_forecast_metrics
from src.models.classical.varima_model import forecast_varima
from src.models.machine_learning.autots_model import forecast_autots

def main():
    print("=" * 80)
    print("EMPIRICAL RUN AUDIT: VARIMA & AUTOTS ON RICE BLAST (Pyricularia oryzae)")
    print("=" * 80)

    df = pd.read_csv("data/disease_timeseries_master.csv")
    sub = df[df["pathogen_slug"] == "pyricularia_oryzae"].sort_values("date").reset_index(drop=True)

    # Chronological Split (70% train = 92 mo, 15% val = 20 mo, 15% test = 20 mo)
    train_df, val_df, test_df = chronological_split(sub, 0.70, 0.15)
    y_train = train_df["documented_occurrence_count"].values
    y_val_6 = val_df["documented_occurrence_count"].values[:6]

    print(f"Dataset Total: {len(sub)} months (Jan 2015 - Dec 2025)")
    print(f"Training Split: {len(train_df)} months (Jan 2015 - Aug 2022)")
    print(f"Validation Split: {len(val_df)} months (Sep 2022 - Apr 2024)")
    print(f"Test Split: {len(test_df)} months (May 2024 - Dec 2025)")
    print("-" * 80)

    # --------------------------------------------------------------------------
    # 1. VARIMA(1,1,1) Evaluation
    # --------------------------------------------------------------------------
    print("\n>>> RUNNING VARIMA(1,1,1)...")
    v_pred_val, v_lb_val, v_ub_val, _, v_diag_val = forecast_varima(
        train_df, horizon=6,
        target_col="documented_occurrence_count",
        exog_cols=["temp_mean_c", "relative_humidity_pct"],
        p=1, d=1, q=1
    )
    v_met_val = compute_forecast_metrics(y_val_6, v_pred_val, v_lb_val, v_ub_val, y_train=y_train, model_name="VARIMA(1,1,1)")

    train_plus_val = pd.concat([train_df, val_df]).reset_index(drop=True)
    y_train_test = train_plus_val["documented_occurrence_count"].values
    y_test_20 = test_df["documented_occurrence_count"].values

    v_pred_test, v_lb_test, v_ub_test, _, v_diag_test = forecast_varima(
        train_plus_val, horizon=20,
        target_col="documented_occurrence_count",
        exog_cols=["temp_mean_c", "relative_humidity_pct"],
        p=1, d=1, q=1
    )
    v_met_test = compute_forecast_metrics(y_test_20, v_pred_test, v_lb_test, v_ub_test, y_train=y_train_test, model_name="VARIMA(1,1,1)")

    print("\n[VARIMA Results]")
    print(f"6-Month Validation Horizon : RMSE = {v_met_val['rmse']:.3f} | MAE = {v_met_val['mae']:.3f} | sMAPE = {v_met_val['smape']:.1f}% | WAPE = {v_met_val['wape']:.1f}% | MASE = {v_met_val['mase']:.3f}")
    print(f"20-Month Test Set Horizon  : RMSE = {v_met_test['rmse']:.3f} | MAE = {v_met_test['mae']:.3f} | sMAPE = {v_met_test['smape']:.1f}% | WAPE = {v_met_test['wape']:.1f}% | MASE = {v_met_test['mase']:.3f}")
    print(f"Parameters: {v_diag_val['parameters']} (AIC = {v_diag_val['aic']:.1f})")

    # --------------------------------------------------------------------------
    # 2. AutoTS Evaluation
    # --------------------------------------------------------------------------
    print("\n>>> RUNNING AutoTS (superfast genetic search)...")
    at_pred_val, at_lb_val, at_ub_val, at_meta_val = forecast_autots(
        train_df, horizon=6, target_col="documented_occurrence_count", model_list="superfast", max_generations=2
    )
    at_met_val = compute_forecast_metrics(y_val_6, at_pred_val, at_lb_val, at_ub_val, y_train=y_train, model_name="AutoTS")

    at_pred_test, at_lb_test, at_ub_test, at_meta_test = forecast_autots(
        train_plus_val, horizon=20, target_col="documented_occurrence_count", model_list="superfast", max_generations=2
    )
    at_met_test = compute_forecast_metrics(y_test_20, at_pred_test, at_lb_test, at_ub_test, y_train=y_train_test, model_name="AutoTS")

    print("\n[AutoTS Results]")
    print(f"6-Month Validation Horizon : RMSE = {at_met_val['rmse']:.3f} | MAE = {at_met_val['mae']:.3f} | sMAPE = {at_met_val['smape']:.1f}% | WAPE = {at_met_val['wape']:.1f}% | MASE = {at_met_val['mase']:.3f}")
    print(f"  Best Model Chosen: {at_meta_val['best_model_name']}")
    print(f"20-Month Test Set Horizon  : RMSE = {at_met_test['rmse']:.3f} | MAE = {at_met_test['mae']:.3f} | sMAPE = {at_met_test['smape']:.1f}% | WAPE = {at_met_test['wape']:.1f}% | MASE = {at_met_test['mase']:.3f}")
    print(f"  Best Model Chosen: {at_meta_test['best_model_name']}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE — Numbers verified from actual empirical run.")
    print("=" * 80)

if __name__ == "__main__":
    main()
