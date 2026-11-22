"""Generate publication-ready diagnostic and comparative figures for the Final Test Evaluation.

Generates 7 core figures saved to `reports/figures/final_test/`:
  1. `1_actual_vs_predicted_timelines.png`
  2. `2_classical_vs_modern_rmse_comparison.png`
  3. `3_classical_vs_modern_mae_comparison.png`
  4. `4_per_pathogen_rmse_heatmap.png`
  5. `5_validation_vs_test_generalization.png`
  6. `6_outbreak_f1_vs_false_alarm_rate.png`
  7. `7_residual_diagnostic_distributions.png`
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_JSON_PATH = REPO_ROOT / "data" / "processed" / "final_test_evaluation_results.json"
PREDICTIONS_CSV_PATH = REPO_ROOT / "data" / "processed" / "final_test_step_predictions.csv"
OUTPUT_DIR = REPO_ROOT / "reports" / "figures" / "final_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font="DejaVu Sans")
PALETTE_FAMILY = {
    "Classical Statistical": "#6B7280",
    "Modern Machine Learning": "#166534",
    "Specialized (Hurdle ML)": "#102C26",
    "Specialized (Additive Curve)": "#D97706",
    "Deep Learning": "#2563EB",
    "Ensemble / Policy": "#9333EA",
}


def generate_all_figures():
    if not RESULTS_JSON_PATH.exists() or not PREDICTIONS_CSV_PATH.exists():
        print(f"Results files not found: {RESULTS_JSON_PATH} or {PREDICTIONS_CSV_PATH}")
        return

    with open(RESULTS_JSON_PATH, "r") as f:
        res = json.load(f)

    df_preds = pd.read_csv(PREDICTIONS_CSV_PATH)
    table_a = pd.DataFrame(res["table_a_final_overall_comparison"])
    table_d = pd.DataFrame(res["table_d_validation_vs_test_generalization"])
    per_p = res["per_pathogen_results"]

    # --------------------------------------------------------------------------
    # 1. ACTUAL VS PREDICTED TIMELINES (10 Pathogens)
    # --------------------------------------------------------------------------
    print("Generating Figure 1: Actual vs Predicted Timelines...")
    fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=True)
    axes = axes.flatten()

    slugs = sorted(df_preds["pathogen_slug"].unique())
    for idx, slug in enumerate(slugs):
        ax = axes[idx]
        sub = df_preds[df_preds["pathogen_slug"] == slug].sort_values("target_date")
        dates = pd.to_datetime(sub["target_date"])

        ax.plot(dates, sub["actual"], label="Actual Ground Truth", color="#102C26", linewidth=2.5, marker="o", markersize=5)
        
        # Policy prediction
        if "pred_production_policy" in sub.columns:
            ax.plot(dates, sub["pred_production_policy"], label="Production Policy", color="#9333EA", linewidth=2.0, linestyle="--", marker="s", markersize=4)

        # Hurdle XGBoost
        if "pred_hurdle_xgboost" in sub.columns:
            ax.plot(dates, sub["pred_hurdle_xgboost"], label="Hurdle XGBoost", color="#166534", linewidth=1.5, linestyle=":", marker="^", markersize=4)

        # Baseline MA(1)
        if "pred_ma" in sub.columns:
            ax.plot(dates, sub["pred_ma"], label="MA(1) (Classical)", color="#6B7280", linewidth=1.2, linestyle="-.")

        c_name = sub["common_name"].iloc[0]
        ax.set_title(f"{c_name} ({slug})", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Occurrence Count", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Final Test Evaluation: Expanding-Window Predictions vs Actuals (May 2024 – Dec 2025)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "1_actual_vs_predicted_timelines.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path}")

    # --------------------------------------------------------------------------
    # 2. CLASSICAL VS MODERN RMSE COMPARISON
    # --------------------------------------------------------------------------
    print("Generating Figure 2: RMSE Comparison Bar Chart...")
    plt.figure(figsize=(12, 6))
    sorted_a = table_a.sort_values("pooled_rmse")
    
    colors = [
        "#9333EA" if "Policy" in r["model_name"] else
        "#102C26" if "Hurdle" in r["model_name"] else
        "#166534" if any(k in r["model_name"] for k in ["Random Forest", "XGBoost"]) and "Hurdle" not in r["model_name"] else
        "#2563EB" if any(k in r["model_name"] for k in ["LSTM", "GRU"]) else
        "#D97706" if "Prophet" in r["model_name"] else "#6B7280"
        for _, r in sorted_a.iterrows()
    ]

    bars = plt.barh(sorted_a["model_name"], sorted_a["pooled_rmse"], color=colors, edgecolor="black", linewidth=0.5)
    plt.xlabel("Pooled Out-of-Sample RMSE (200 test steps) — Lower is Better", fontsize=11, fontweight="bold")
    plt.title("Final Test Period: Out-of-Sample Pooled RMSE Across All Architectures", fontsize=13, fontweight="bold", pad=12)
    plt.gca().invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.08, bar.get_y() + bar.get_height() / 2.0, f"{w:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "2_classical_vs_modern_rmse_comparison.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}")

    # --------------------------------------------------------------------------
    # 3. CLASSICAL VS MODERN MAE COMPARISON
    # --------------------------------------------------------------------------
    print("Generating Figure 3: MAE Comparison Bar Chart...")
    plt.figure(figsize=(12, 6))
    sorted_mae = table_a.sort_values("pooled_mae")
    
    colors_mae = [
        "#9333EA" if "Policy" in r["model_name"] else
        "#102C26" if "Hurdle" in r["model_name"] else
        "#166534" if any(k in r["model_name"] for k in ["Random Forest", "XGBoost"]) and "Hurdle" not in r["model_name"] else
        "#2563EB" if any(k in r["model_name"] for k in ["LSTM", "GRU"]) else
        "#D97706" if "Prophet" in r["model_name"] else "#6B7280"
        for _, r in sorted_mae.iterrows()
    ]

    bars = plt.barh(sorted_mae["model_name"], sorted_mae["pooled_mae"], color=colors_mae, edgecolor="black", linewidth=0.5)
    plt.xlabel("Pooled Out-of-Sample MAE (200 test steps) — Lower is Better", fontsize=11, fontweight="bold")
    plt.title("Final Test Period: Out-of-Sample Pooled MAE Across All Architectures", fontsize=13, fontweight="bold", pad=12)
    plt.gca().invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.04, bar.get_y() + bar.get_height() / 2.0, f"{w:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig3_path = OUTPUT_DIR / "3_classical_vs_modern_mae_comparison.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}")

    # --------------------------------------------------------------------------
    # 4. PER-PATHOGEN RMSE HEATMAP (10 Pathogens x Models)
    # --------------------------------------------------------------------------
    print("Generating Figure 4: Per-Pathogen Heatmap...")
    matrix_data = []
    pathogen_labels = []
    model_cols = [r["model_name"] for r in table_a.sort_values("pooled_rmse").to_dict(orient="records")]

    for slug, p_data in sorted(per_p.items()):
        pathogen_labels.append(p_data["common_name"])
        row = []
        for m in model_cols:
            met = p_data["models"].get(m, {})
            row.append(met.get("rmse", np.nan))
        matrix_data.append(row)

    df_matrix = pd.DataFrame(matrix_data, index=pathogen_labels, columns=model_cols)

    plt.figure(figsize=(18, 9))
    sns.heatmap(df_matrix, annot=True, fmt=".2f", cmap="YlGnBu_r", cbar_kws={"label": "Out-of-Sample RMSE"}, linewidths=0.5)
    plt.title("Per-Pathogen Test RMSE Matrix: Classical vs. Modern Forecasters", fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Forecasting Model Architecture", fontsize=11, fontweight="bold")
    plt.ylabel("Target Crop Pathogen", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    fig4_path = OUTPUT_DIR / "4_per_pathogen_rmse_heatmap.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig4_path}")

    # --------------------------------------------------------------------------
    # 5. VALIDATION VS TEST GENERALIZATION CHART
    # --------------------------------------------------------------------------
    print("Generating Figure 5: Generalization Scatter Plot...")
    if not table_d.empty and "val_rmse" in table_d.columns and table_d["val_rmse"].notnull().any():
        plt.figure(figsize=(9, 8))
        sub_d = table_d.dropna(subset=["val_rmse", "test_rmse"]).copy()

        plt.scatter(sub_d["val_rmse"], sub_d["test_rmse"], s=100, c="#166534", edgecolors="black", linewidth=1.0, zorder=4)

        # Plot y=x parity line
        min_v = min(sub_d["val_rmse"].min(), sub_d["test_rmse"].min()) * 0.9
        max_v = max(sub_d["val_rmse"].max(), sub_d["test_rmse"].max()) * 1.1
        plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Parity (Val RMSE = Test RMSE)")

        for _, r in sub_d.iterrows():
            plt.annotate(
                r["model_name"].split("(")[0].strip(),
                (r["val_rmse"], r["test_rmse"]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=8,
            )

        plt.xlabel("Validation Window RMSE (Multi-Origin Walk-Forward)", fontsize=11, fontweight="bold")
        plt.ylabel("Final Test Window RMSE (Expanding 1-Step Walk-Forward)", fontsize=11, fontweight="bold")
        plt.title("Model Generalization: Validation RMSE vs. Final Untouched Test RMSE", fontsize=13, fontweight="bold", pad=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left")
        plt.tight_layout()
        fig5_path = OUTPUT_DIR / "5_validation_vs_test_generalization.png"
        plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig5_path}")

    # --------------------------------------------------------------------------
    # 6. OUTBREAK F1 VS FALSE ALARM RATE
    # --------------------------------------------------------------------------
    print("Generating Figure 6: Outbreak F1 vs False Alarm Rate...")
    plt.figure(figsize=(10, 7))
    plt.scatter(
        table_a["false_alarm_rate_pct"],
        table_a["f1"],
        s=[max(40, (12 - r["pooled_rmse"]) * 20) for _, r in table_a.iterrows()],
        c=table_a["f1"],
        cmap="viridis",
        edgecolors="black",
        linewidth=1.0,
        zorder=4,
    )

    for _, r in table_a.iterrows():
        plt.annotate(
            r["model_name"].split("(")[0].strip(),
            (r["false_alarm_rate_pct"], r["f1"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    plt.xlabel("False Alarm Rate (%) — Lower is Better", fontsize=11, fontweight="bold")
    plt.ylabel("Outbreak Detection F1 Score (y >= 1) — Higher is Better", fontsize=11, fontweight="bold")
    plt.title("Outbreak Alerting Trade-off: Outbreak F1 vs. False Alarm Rate on Final Test Set", fontsize=13, fontweight="bold", pad=12)
    plt.axvline(x=30.0, color="#DC2626", linestyle=":", alpha=0.5, label="High FAR Threshold (30%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig6_path = OUTPUT_DIR / "6_outbreak_f1_vs_false_alarm_rate.png"
    plt.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig6_path}")

    # --------------------------------------------------------------------------
    # 7. RESIDUAL DIAGNOSTICS DISTRIBUTION
    # --------------------------------------------------------------------------
    print("Generating Figure 7: Residual Diagnostics...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calculate residuals for Policy and Baseline
    res_policy = df_preds["actual"] - df_preds["pred_production_policy"]
    res_ma = df_preds["actual"] - df_preds["pred_ma"]
    res_hurdle = df_preds["actual"] - df_preds["pred_hurdle_xgboost"]

    sns.kdeplot(res_policy, ax=axes[0], label="Production Policy", color="#9333EA", linewidth=2.0)
    sns.kdeplot(res_hurdle, ax=axes[0], label="Hurdle XGBoost", color="#102C26", linewidth=1.8, linestyle="--")
    sns.kdeplot(res_ma, ax=axes[0], label="MA(1) (Classical)", color="#6B7280", linewidth=1.5, linestyle=":")
    
    axes[0].set_title("Forecast Error Distribution: Actual - Predicted", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Residual Error (Occurrences)", fontsize=10)
    axes[0].set_ylabel("Kernel Density", fontsize=10)
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[0].legend(loc="upper right", fontsize=9)

    # Boxplot of absolute errors
    df_box = pd.DataFrame({
        "Production Policy": np.abs(res_policy),
        "Hurdle XGBoost": np.abs(res_hurdle),
        "MA(1)": np.abs(res_ma),
        "Seasonal Naive": np.abs(df_preds["actual"] - df_preds["pred_seasonal_naive"]),
    })
    sns.boxplot(data=df_box, ax=axes[1], palette=["#9333EA", "#102C26", "#6B7280", "#DC2626"])
    axes[1].set_title("Absolute Error Distribution by Architecture", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Absolute Error (|Actual - Pred|)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Final Test Residual Diagnostics (May 2024 to Dec 2025)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig7_path = OUTPUT_DIR / "7_residual_diagnostic_distributions.png"
    plt.savefig(fig7_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig7_path}")
    print("All 7 figures generated successfully!")


if __name__ == "__main__":
    generate_all_figures()

