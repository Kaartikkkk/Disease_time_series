"""Batch audit and empirical evaluation of ARCH and GARCH models across all 10 plant pathogens.

Outputs comprehensive JSON metrics into data/processed/arch_garch_evaluation_results.json
and artifacts/metrics/arch_garch_evaluation_results.json.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.models.volatility.arch_garch_model import (
    ARCHGARCHModel,
    VOLATILITY_SPECS,
    fit_and_compare_all_specifications,
    perform_arch_lm_test,
    run_volatility_walk_forward_evaluation,
)

DATA_PATH = Path("data/disease_timeseries_master.csv")
OUT_JSON = Path("data/processed/arch_garch_evaluation_results.json")
ARTIFACT_JSON = Path("artifacts/metrics/arch_garch_evaluation_results.json")


def main():
    print(f"Loading authoritative dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    pathogens = sorted(df["pathogen_slug"].unique())
    print(f"Found {len(pathogens)} pathogens.")

    audit_results = {
        "metadata": {
            "dataset": "disease_timeseries_master.csv",
            "total_pathogens": len(pathogens),
            "date_range": [str(df["date"].min()), str(df["date"].max())],
            "target_col": "documented_occurrence_count",
        },
        "pathogen_suitability": {},
        "model_specifications": {},
        "walk_forward_benchmarks": {},
    }

    for slug in pathogens:
        sub_df = df[df["pathogen_slug"] == slug].sort_values("date")
        y = sub_df["documented_occurrence_count"].values.astype(float)
        diff_y = np.diff(y)
        zero_pct = float(np.mean(y == 0) * 100.0)

        # 1. Statistical Suitability Diagnostics
        adf_p = float(adfuller(y)[1])
        try:
            kpss_p = float(kpss(y, regression="c", nlags="auto")[1])
        except Exception:
            kpss_p = np.nan

        lm_raw = perform_arch_lm_test(y, lags=5)
        lm_diff = perform_arch_lm_test(diff_y, lags=5)

        # Fit ARIMA(1,0,1) for residual ARCH-LM test
        try:
            from statsmodels.tsa.arima.model import ARIMA
            ar_fit = ARIMA(y, order=(1, 0, 1)).fit()
            resid = ar_fit.resid
            lm_resid = perform_arch_lm_test(resid, lags=5)
        except Exception:
            lm_resid = {"lm_stat": np.nan, "lm_pvalue": np.nan, "has_arch_effects": False}

        # Suitability Assessment
        if lm_diff["has_arch_effects"] and (lm_resid["has_arch_effects"] or zero_pct < 80.0):
            suitability = "Strong ARCH Effects (Highly Suitable for Volatility Modelling)"
        elif lm_diff["has_arch_effects"]:
            suitability = "Moderate ARCH Effects (Suitable on Differenced Series)"
        else:
            suitability = "Weak / Low Volatility Clustering"

        audit_results["pathogen_suitability"][slug] = {
            "zero_percentage": round(zero_pct, 1),
            "mean": round(float(np.mean(y)), 2),
            "std": round(float(np.std(y)), 2),
            "adf_pvalue": round(adf_p, 4),
            "kpss_pvalue": round(kpss_p, 4),
            "arch_lm_raw_p": round(lm_raw["lm_pvalue"], 4),
            "arch_lm_diff_p": round(lm_diff["lm_pvalue"], 4),
            "arch_lm_resid_p": round(lm_resid["lm_pvalue"], 4),
            "suitability_verdict": suitability,
        }

        # 2. Fit All 7 Specifications (Approach A & Approach B)
        specs_a = fit_and_compare_all_specifications(y, dist="StudentsT", approach="approach_a")
        specs_b = fit_and_compare_all_specifications(y, dist="StudentsT", approach="approach_b")

        audit_results["model_specifications"][slug] = {
            "approach_a_residuals": {
                name: {
                    "p": res.p,
                    "q": res.q,
                    "log_likelihood": round(res.log_likelihood, 2),
                    "aic": round(res.aic, 2),
                    "bic": round(res.bic, 2),
                    "arch_lm_pvalue": round(res.arch_lm_pvalue, 4),
                    "converged": res.converged,
                    "params": {k: round(v, 4) for k, v in res.params.items()},
                }
                for name, res in specs_a.items()
            },
            "approach_b_diff": {
                name: {
                    "p": res.p,
                    "q": res.q,
                    "log_likelihood": round(res.log_likelihood, 2),
                    "aic": round(res.aic, 2),
                    "bic": round(res.bic, 2),
                    "arch_lm_pvalue": round(res.arch_lm_pvalue, 4),
                    "converged": res.converged,
                    "params": {k: round(v, 4) for k, v in res.params.items()},
                }
                for name, res in specs_b.items()
            },
        }

        # 3. Walk-Forward Validation on Representative Specification GARCH(1,1)
        try:
            wf = run_volatility_walk_forward_evaluation(
                y,
                spec_name="garch_1_1",
                p=1,
                q=1,
                dist="StudentsT",
                approach="approach_a",
                horizon=1,
                min_train_len=90,
            )
            audit_results["walk_forward_benchmarks"][slug] = {
                "n_eval_points": wf["n_predictions"],
                "rmse": round(wf["point_metrics"]["rmse"], 3),
                "mae": round(wf["point_metrics"]["mae"], 3),
                "wape": round(wf["point_metrics"]["wape"], 3),
                "mase": round(wf["point_metrics"]["mase"], 3),
                "rmsve": round(wf["volatility_metrics"]["rmsve"], 3),
                "mave": round(wf["volatility_metrics"]["mave"], 3),
                "qlike": round(wf["volatility_metrics"]["qlike"], 3),
            }
        except Exception as exc:
            print(f"Walk-forward failed for {slug}: {exc}")

        print(f"Completed audit for {slug:26s} | Zero%: {zero_pct:4.1f}% | {suitability}")

    # Write output
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nSaved audit results to {OUT_JSON}")

    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_JSON, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Mirrored audit results to {ARTIFACT_JSON}")


if __name__ == "__main__":
    main()
