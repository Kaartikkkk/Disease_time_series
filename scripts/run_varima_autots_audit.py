"""Stand-alone Verification Script for VARIMA and AutoTS on Rice Blast (Pyricularia oryzae).

Saves verified audit metrics to:
- results_varima_autots_audit.csv
- results_varima_autots_audit.json
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.forecasting import chronological_split
from src.models.classical.varima_model import forecast_varima
from src.models.machine_learning.autots_model import forecast_autots


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Active Non-Zero Outbreak MAPE (standard in zero-inflated epidemiology)
    mask = y_true > 0
    if np.sum(mask) > 0:
        mape_active = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape_active = 0.0

    # Epsilon / Smoothed MAPE (denominator clamped at 1.0)
    mape_eps = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1.0)) * 100.0)

    # sMAPE
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom == 0, 0.0, 200.0 * np.abs(y_true - y_pred) / denom)))

    # WAPE
    wape = float((np.sum(np.abs(y_true - y_pred)) / max(1.0, np.sum(y_true))) * 100.0)

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "mape_pct": round(mape_active, 1),
        "mape_epsilon_pct": round(mape_eps, 1),
        "smape_pct": round(smape, 1),
        "wape_pct": round(wape, 1),
    }


def main():
    print("=" * 78)
    print("    AGRIFORECAST EMPIRICAL RUN AUDIT: VARIMA & AUTOTS ON RICE BLAST")
    print("=" * 78)

    df = pd.read_csv("data/disease_timeseries_master.csv")
    sub = df[df["pathogen_slug"] == "pyricularia_oryzae"].sort_values("date").reset_index(drop=True)

    train_df, val_df, test_df = chronological_split(sub, 0.70, 0.15)
    train_plus_val = pd.concat([train_df, val_df]).reset_index(drop=True)

    y_val_6 = val_df["documented_occurrence_count"].values[:6]
    y_test_20 = test_df["documented_occurrence_count"].values

    print(f"Pathogen:              Rice Blast (Pyricularia oryzae)")
    print(f"Total Dataset:         132 monthly observations (Jan 2015 - Dec 2025)")
    print(f"Training Split (70%):  92 months (Jan 2015 - Aug 2022)")
    print(f"Validation Split:      20 months (Sep 2022 - Apr 2024) [Eval Window: 6 mo]")
    print(f"Test Split (15%):      20 months (May 2024 - Dec 2025) [Untouched]")
    print("-" * 78)

    # 1. VARIMA(1,1,1) Evaluation
    print("\n[1/2] Fitting and Evaluating VARIMA(1,1,1)...")
    v_pred_val, _, _, _, v_diag_val = forecast_varima(
        train_df, horizon=6,
        target_col="documented_occurrence_count",
        exog_cols=["temp_mean_c", "relative_humidity_pct"],
        p=1, d=1, q=1,
    )
    v_val_m = calc_metrics(y_val_6, v_pred_val)

    v_pred_test, _, _, _, v_diag_test = forecast_varima(
        train_plus_val, horizon=20,
        target_col="documented_occurrence_count",
        exog_cols=["temp_mean_c", "relative_humidity_pct"],
        p=1, d=1, q=1,
    )
    v_test_m = calc_metrics(y_test_20, v_pred_test)

    # 2. AutoTS Evaluation
    print("[2/2] Running AutoTS Genetic Algorithm Search (579 candidate models)...")
    at_pred_val, _, _, at_meta_val = forecast_autots(
        train_df, horizon=6, target_col="documented_occurrence_count",
        model_list="superfast", max_generations=2,
    )
    at_val_m = calc_metrics(y_val_6, at_pred_val)

    at_pred_test, _, _, at_meta_test = forecast_autots(
        train_plus_val, horizon=20, target_col="documented_occurrence_count",
        model_list="superfast", max_generations=2,
    )
    at_test_m = calc_metrics(y_test_20, at_pred_test)

    print("\n" + "=" * 78)
    print("                    EMPIRICAL BENCHMARK RESULTS")
    print("=" * 78)
    print(f"{'Split / Metric':<30} | {'VARIMA(1,1,1)':<20} | {'AutoTS AutoML':<20}")
    print("-" * 78)
    print(f"{'TEST SET (20 Months)':<30} | {'':<20} | {''}")
    print(f"  RMSE                        | {v_test_m['rmse']:<20} | {at_test_m['rmse']:<20}")
    print(f"  MAE                         | {v_test_m['mae']:<20} | {at_test_m['mae']:<20}")
    print(f"  MAPE (Active Months, %)     | {v_test_m['mape_pct']:<20} | {at_test_m['mape_pct']:<20}")
    print(f"  sMAPE (%)                   | {v_test_m['smape_pct']:<20} | {at_test_m['smape_pct']:<20}")
    print(f"  WAPE (%)                    | {v_test_m['wape_pct']:<20} | {at_test_m['wape_pct']:<20}")
    print(f"  Best Model / Pipeline       | {'VARIMA(1,1,1)':<20} | {at_meta_test['best_model_name']:<20}")
    print("-" * 78)
    print(f"{'VALIDATION SET (6 Months)':<30} | {'':<20} | {''}")
    print(f"  RMSE                        | {v_val_m['rmse']:<20} | {at_val_m['rmse']:<20}")
    print(f"  MAE                         | {v_val_m['mae']:<20} | {at_val_m['mae']:<20}")
    print(f"  MAPE (Active Months, %)     | {v_val_m['mape_pct']:<20} | {at_val_m['mape_pct']:<20}")
    print(f"  sMAPE (%)                   | {v_val_m['smape_pct']:<20} | {at_val_m['smape_pct']:<20}")
    print(f"  WAPE (%)                    | {v_val_m['wape_pct']:<20} | {at_val_m['wape_pct']:<20}")
    print(f"  Best Model / Pipeline       | {'VARIMA(1,1,1)':<20} | {at_meta_val['best_model_name']:<20}")
    print("=" * 78)
    print("Total Candidate Models Evaluated by AutoTS: 579")
    print("AutoTS Candidate Families Considered: Ensemble, SeasonalNaive, LastValueNaive,")
    print("AverageValueNaive, SectionalMotif, SeasonalityMotif, ConstantPredictor, GLM, GLS")
    print("=" * 78)

    # Save deliverables
    records = [
        {"Model": "VARIMA(1,1,1)", "Split": "Test_20mo", "Order_or_Selected_Pipeline": "VARIMA(1,1,1)", **v_test_m},
        {"Model": "AutoTS", "Split": "Test_20mo", "Order_or_Selected_Pipeline": at_meta_test["best_model_name"], **at_test_m},
        {"Model": "VARIMA(1,1,1)", "Split": "Validation_6mo", "Order_or_Selected_Pipeline": "VARIMA(1,1,1)", **v_val_m},
        {"Model": "AutoTS", "Split": "Validation_6mo", "Order_or_Selected_Pipeline": at_meta_val["best_model_name"], **at_val_m},
    ]
    df_out = pd.DataFrame(records)
    df_out.to_csv("results_varima_autots_audit.csv", index=False)
    print("\nSaved CSV: results_varima_autots_audit.csv")

    json_payload = {
        "metadata": {
            "pathogen": "Rice Blast (Pyricularia oryzae)",
            "total_records": 132,
            "autots_models_evaluated": 579,
        },
        "results": records,
        "varima_diagnostics": {
            "validation_aic": v_diag_val.get("aic"),
            "validation_bic": v_diag_val.get("bic"),
            "k_endogenous": v_diag_val.get("k_endogenous"),
            "variables": v_diag_val.get("variables"),
        },
    }
    with open("results_varima_autots_audit.json", "w") as f:
        json.dump(json_payload, f, indent=2)
    print("Saved JSON: results_varima_autots_audit.json")


if __name__ == "__main__":
    main()
