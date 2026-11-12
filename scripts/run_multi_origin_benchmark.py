"""Multi-Origin Walk-Forward Validation Engine for 10 Agricultural Pathogens.

Executes expanding-window walk-forward validation across 4 chronological forecast origins
within the validation period (September 2022 to April 2024):
  - Origin 1: Cutoff 2022-08-01 -> Eval 2022-09 to 2023-02 (6 months)
  - Origin 2: Cutoff 2023-02-01 -> Eval 2023-03 to 2023-08 (6 months)
  - Origin 3: Cutoff 2023-08-01 -> Eval 2023-09 to 2024-02 (6 months)
  - Origin 4: Cutoff 2023-10-01 -> Eval 2023-11 to 2024-04 (6 months)

Evaluates 14 models:
  Seasonal Naive, AR(1), MA(1), ARIMA(1,1,1), SARIMA, Holt-Winters,
  Local Random Forest, Local XGBoost, Global Random Forest, Global XGBoost,
  Hurdle XGBoost, LSTM, GRU, Prophet.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from src.forecasting import run_rolling_walk_forward_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_origin_benchmark")

DATA_PATH = Path("data/processed/india_plant_diseases_monthly.csv")
OUTPUT_PATH = Path("data/processed/multi_origin_validation_results.json")

ORIGINS: List[Tuple[str, int, int, str]] = [
    ("Origin 1", 92, 6, "2022-09 to 2023-02"),
    ("Origin 2", 98, 6, "2023-03 to 2023-08"),
    ("Origin 3", 104, 6, "2023-09 to 2024-02"),
    ("Origin 4", 106, 6, "2023-11 to 2024-04"),
]

MODELS_TO_BENCHMARK = [
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
    "lstm",
    "gru",
    "prophet",
]


def moving_block_bootstrap_ci(
    diffs: np.ndarray,
    block_length: int = 3,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate moving block bootstrap confidence interval for paired differences."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n < block_length * 2:
        mean_val = float(np.mean(diffs))
        return mean_val, mean_val, mean_val

    n_blocks = int(np.ceil(n / block_length))
    # Construct all overlapping blocks
    blocks = [diffs[i : i + block_length] for i in range(n - block_length + 1)]
    boot_means = []
    for _ in range(n_boot):
        chosen_indices = rng.integers(0, len(blocks), size=n_blocks)
        sample = np.concatenate([blocks[idx] for idx in chosen_indices])[:n]
        boot_means.append(np.mean(sample))

    point_est = float(np.mean(diffs))
    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return point_est, low, high


def run_benchmark():
    t_start = time.perf_counter()
    logger.info("Loading master dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    all_slugs = sorted(df["pathogen_slug"].unique())
    logger.info("Found %d pathogens: %s", len(all_slugs), all_slugs)

    # Master storage structure
    raw_evaluations: List[Dict[str, Any]] = []

    # Run multi-origin evaluation
    total_runs = len(all_slugs) * len(ORIGINS)
    current_run = 0

    for slug in all_slugs:
        sub_df = df[df["pathogen_slug"] == slug].sort_values("date").reset_index(drop=True)
        common_name = sub_df["common_name"].iloc[0] if "common_name" in sub_df.columns else slug

        for origin_name, start_idx, horizon, date_range in ORIGINS:
            current_run += 1
            t_orig = time.perf_counter()
            logger.info(
                "[%d/%d] Pathogen: %s (%s) | %s (%s)",
                current_run,
                total_runs,
                slug,
                common_name,
                origin_name,
                date_range,
            )

            res = run_rolling_walk_forward_evaluation(
                sub_df,
                horizon=horizon,
                eval_start_idx=start_idx,
                filter_model_ids=MODELS_TO_BENCHMARK,
            )

            actuals = [float(v) for v in res["validation_actual"]]
            dates = [d.strftime("%Y-%m") for d in res["validation_dates"]]

            record = {
                "pathogen_slug": slug,
                "common_name": common_name,
                "origin_name": origin_name,
                "date_range": date_range,
                "start_idx": start_idx,
                "horizon": horizon,
                "dates": dates,
                "actuals": actuals,
                "forecasts": {k: [float(x) for x in v] for k, v in res["forecasts"].items()},
                "model_failures": res.get("model_failures", {}),
                "metrics": {m["model_name"]: m for m in res["evaluation_metrics"]},
                "elapsed_sec": round(time.perf_counter() - t_orig, 2),
            }
            raw_evaluations.append(record)

    logger.info("All walk-forward runs completed in %.2f s. Computing aggregate metrics...", time.perf_counter() - t_start)

    # --------------------------------------------------------------------------
    # Aggregation & Detailed Metric Computation
    # --------------------------------------------------------------------------
    model_names = set()
    for rec in raw_evaluations:
        model_names.update(rec["forecasts"].keys())
    model_names = sorted(model_names)

    # 1. Overall Pooled & Macro Metrics across all (Pathogen x Origin x Step)
    overall_metrics = {}
    model_pooled_preds = {m: [] for m in model_names}
    model_pooled_acts = {m: [] for m in model_names}
    per_pathogen_records = {slug: {m: {"act": [], "pred": []} for m in model_names} for slug in all_slugs}
    per_origin_records = {oname: {m: {"act": [], "pred": []} for m in model_names} for oname, _, _, _ in ORIGINS}

    for rec in raw_evaluations:
        slug = rec["pathogen_slug"]
        oname = rec["origin_name"]
        acts = rec["actuals"]
        for mname in model_names:
            if mname in rec["forecasts"]:
                preds = rec["forecasts"][mname]
                model_pooled_preds[mname].extend(preds)
                model_pooled_acts[mname].extend(acts)
                per_pathogen_records[slug][mname]["act"].extend(acts)
                per_pathogen_records[slug][mname]["pred"].extend(preds)
                per_origin_records[oname][mname]["act"].extend(acts)
                per_origin_records[oname][mname]["pred"].extend(preds)

    # Baseline for improvement calculations
    baseline_name = "Seasonal Naive (Baseline)"

    for mname in model_names:
        y_act = np.array(model_pooled_acts[mname], dtype=float)
        y_pred = np.array(model_pooled_preds[mname], dtype=float)

        if len(y_act) == 0:
            continue

        pooled_rmse = float(np.sqrt(np.mean((y_pred - y_act) ** 2)))
        pooled_mae = float(np.mean(np.abs(y_pred - y_act)))
        sum_act = float(np.sum(y_act))
        wape = float(np.sum(np.abs(y_pred - y_act)) / sum_act * 100.0) if sum_act > 0 else 0.0

        # Positive and zero MAE
        pos_mask = y_act > 0.0
        zero_mask = y_act == 0.0
        mae_pos = float(np.mean(np.abs(y_pred[pos_mask] - y_act[pos_mask]))) if pos_mask.sum() > 0 else 0.0
        mae_zero = float(np.mean(np.abs(y_pred[zero_mask]))) if zero_mask.sum() > 0 else 0.0

        # Outbreak detection (threshold y_pred >= 0.5 vs y_act > 0)
        bin_act = (y_act > 0.0).astype(int)
        bin_pred = (y_pred >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(bin_act, bin_pred, average="binary", zero_division=0)
        # False alarm rate: false positives / actual negatives
        n_neg = int(zero_mask.sum())
        fp = int(((bin_pred == 1) & (bin_act == 0)).sum())
        far = float(fp / n_neg * 100.0) if n_neg > 0 else 0.0

        # PR-AUC
        try:
            pr_auc = float(average_precision_score(bin_act, y_pred))
        except Exception:
            pr_auc = float("nan")

        # Diagnostics
        diag = {
            "act_mean": float(np.mean(y_act)),
            "pred_mean": float(np.mean(y_pred)),
            "act_std": float(np.std(y_act)),
            "pred_std": float(np.std(y_pred)),
            "near_zero_pct": float(np.mean(y_pred < 0.1) * 100.0),
        }

        # Macro averages across pathogens
        path_rmses = []
        path_maes = []
        for slug in all_slugs:
            p_act = np.array(per_pathogen_records[slug][mname]["act"], dtype=float)
            p_pred = np.array(per_pathogen_records[slug][mname]["pred"], dtype=float)
            if len(p_act) > 0:
                path_rmses.append(np.sqrt(np.mean((p_pred - p_act) ** 2)))
                path_maes.append(np.mean(np.abs(p_pred - p_act)))

        macro_rmse = float(np.mean(path_rmses)) if path_rmses else float("nan")
        macro_mae = float(np.mean(path_maes)) if path_maes else float("nan")

        overall_metrics[mname] = {
            "model_name": mname,
            "pooled_rmse": round(pooled_rmse, 3),
            "pooled_mae": round(pooled_mae, 3),
            "macro_rmse": round(macro_rmse, 3),
            "macro_mae": round(macro_mae, 3),
            "wape_pct": round(wape, 1),
            "mae_pos": round(mae_pos, 3),
            "mae_zero": round(mae_zero, 3),
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "f1": round(float(f1), 3),
            "pr_auc": round(pr_auc, 3),
            "false_alarm_rate_pct": round(far, 1),
            "diagnostics": diag,
        }

    # 2. Robustness: Wins across Pathogens & Origins
    # Origin Wins
    origin_wins = {m: 0 for m in model_names}
    for oname, _, _, _ in ORIGINS:
        best_m = None
        best_rmse = float("inf")
        for mname in model_names:
            o_act = np.array(per_origin_records[oname][mname]["act"], dtype=float)
            o_pred = np.array(per_origin_records[oname][mname]["pred"], dtype=float)
            if len(o_act) > 0:
                rmse = float(np.sqrt(np.mean((o_pred - o_act) ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_m = mname
        if best_m:
            origin_wins[best_m] += 1

    # Pathogen Wins & Top 3
    pathogen_first_place = {m: 0 for m in model_names}
    pathogen_top3 = {m: 0 for m in model_names}
    per_pathogen_summary = {}

    for slug in all_slugs:
        slug_scores = []
        c_name = df[df["pathogen_slug"] == slug]["common_name"].iloc[0]
        per_pathogen_summary[slug] = {"common_name": c_name, "models": {}}

        for mname in model_names:
            p_act = np.array(per_pathogen_records[slug][mname]["act"], dtype=float)
            p_pred = np.array(per_pathogen_records[slug][mname]["pred"], dtype=float)
            if len(p_act) > 0:
                rmse = float(np.sqrt(np.mean((p_pred - p_act) ** 2)))
                mae = float(np.mean(np.abs(p_pred - p_act)))
                slug_scores.append((mname, rmse, mae))
                per_pathogen_summary[slug]["models"][mname] = {"rmse": round(rmse, 3), "mae": round(mae, 3)}

        slug_scores.sort(key=lambda x: x[1])
        if slug_scores:
            pathogen_first_place[slug_scores[0][0]] += 1
            for top_m, _, _ in slug_scores[:3]:
                pathogen_top3[top_m] += 1

    # 3. Block-Bootstrap Paired Differences vs Seasonal Naive & AR(1)
    bootstrap_results = {}
    base_acts = np.array(model_pooled_acts[baseline_name], dtype=float)
    base_preds = np.array(model_pooled_preds[baseline_name], dtype=float)
    base_abs_err = np.abs(base_preds - base_acts)

    ar_name = "AR(1)"
    ar_acts = np.array(model_pooled_acts[ar_name], dtype=float) if ar_name in model_pooled_acts else base_acts
    ar_preds = np.array(model_pooled_preds[ar_name], dtype=float) if ar_name in model_pooled_preds else base_preds
    ar_abs_err = np.abs(ar_preds - ar_acts)

    for mname in model_names:
        m_acts = np.array(model_pooled_acts[mname], dtype=float)
        m_preds = np.array(model_pooled_preds[mname], dtype=float)
        m_abs_err = np.abs(m_preds - m_acts)

        # Improvement vs Seasonal Naive: (MAE_base - MAE_m)
        diff_vs_naive = base_abs_err - m_abs_err
        pt_naive, low_naive, high_naive = moving_block_bootstrap_ci(diff_vs_naive, block_length=3, n_boot=1000)

        # Improvement vs AR(1): (MAE_ar - MAE_m)
        diff_vs_ar = ar_abs_err - m_abs_err
        pt_ar, low_ar, high_ar = moving_block_bootstrap_ci(diff_vs_ar, block_length=3, n_boot=1000)

        bootstrap_results[mname] = {
            "vs_naive_mean_mae_reduction": round(pt_naive, 3),
            "vs_naive_95_ci": [round(low_naive, 3), round(high_naive, 3)],
            "vs_naive_stat_sig": bool(low_naive > 0.0),
            "vs_ar_mean_mae_reduction": round(pt_ar, 3),
            "vs_ar_95_ci": [round(low_ar, 3), round(high_ar, 3)],
            "vs_ar_stat_sig": bool(low_ar > 0.0),
        }

    # Add robustness counts to overall metrics
    for mname in model_names:
        if mname in overall_metrics:
            overall_metrics[mname]["pathogen_first_place_count"] = pathogen_first_place[mname]
            overall_metrics[mname]["pathogen_top3_count"] = pathogen_top3[mname]
            overall_metrics[mname]["origin_wins_count"] = origin_wins[mname]
            overall_metrics[mname]["bootstrap_ci"] = bootstrap_results.get(mname, {})

    # Final combined payload
    payload = {
        "metadata": {
            "validation_period": "September 2022 to April 2024 (20 months)",
            "n_pathogens": len(all_slugs),
            "origins": ORIGINS,
            "models_evaluated": list(model_names),
            "total_forecast_points": len(model_pooled_acts.get(baseline_name, [])),
            "total_execution_runtime_sec": round(time.perf_counter() - t_start, 2),
        },
        "overall_rankings": sorted(list(overall_metrics.values()), key=lambda x: x["pooled_rmse"]),
        "per_pathogen_summary": per_pathogen_summary,
        "raw_evaluations": raw_evaluations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Successfully saved multi-origin benchmark results to %s", OUTPUT_PATH)
    return payload


if __name__ == "__main__":
    run_benchmark()

