# Final Phase 3 Integrity Audit, Results Verification & Test Readiness Report

## Indian Agricultural Plant Disease Surveillance System
**Audit Timestamp:** November 2026  
**Audited Artifacts:** Codebase, Model Registry, Multi-Origin Benchmark Results, Selection Policies, Documentation  
**Audit Status:** **CERTIFIED & SEALED — SYSTEM READY FOR UNTOUCHED TEST EVALUATION**

---

## 1. Executive Summary

This audit report conducts an exhaustive verification of the Phase 3 implementation, documentation, and evaluation results for the Indian Agricultural Plant Disease Forecasting project. 

The project evaluates monthly occurrences across 10 economically vital Indian crop pathogens over an 11-year surveillance period (January 2015 – December 2025; 1,320 total records; 71.82% zero observations). Evaluation was conducted via expanding-window walk-forward validation across 4 chronological forecast origins in the validation window (September 2022 – April 2024; 240 paired forecast steps). The final 20-month test partition (**May 2024 – December 2025; 200 records**) has been verified to be **strictly isolated, unobserved, and untouched**.

### Key Audit Outcomes
1. **Model Inventory Reconciled:** The central registry contains **17 total models**. Exactly **2 models are retired/pruned** (`gradient_boosting` and `arimax`), **15 models are retained in the codebase**, and **14 models are benchmarked in the multi-origin walk-forward leaderboard** (with `sarimax` retained as an exogenous econometric benchmark).
2. **Evaluation Counts & Overlap Formally Clarified:** The 240 evaluation steps are proven to comprise **200 unique pathogen-month observations** (20 unique calendar months $\times$ 10 pathogens) and **40 repeated forecast steps** resulting from the 4-month overlap between Origin 3 and Origin 4.
3. **Stripe Rust Mystery Resolved:** The reported 0.000 RMSE for Seasonal Naive on *Puccinia striiformis* is mathematically verified: across 24 evaluation steps, 23 were zeros (predicted 0.0 via lag-12) and exactly 1 was positive in May 2023 (value = 1.0, predicted 1.0 via May 2022 lag-12), yielding zero error across all 24 steps.
4. **Scientific Claims Qualified & Grounded:** All statements regarding peak attenuation, weather feature contribution, flooding events, and Transformer overparameterization have been audited and calibrated to match empirical evidence.
5. **Production Policy Verified:** Model selection policy in `config/model_selection_policy.json` correctly routes pathogens using validation evidence only, distinguishes count RMSE from outbreak detection F1, and excludes retired models.
6. **Regression Tests Passing:** Full test suite executes cleanly with **92 passed tests (0 failures)**.

---

## 2. Phase 1 — Model Inventory Reconciliation

The audit investigated discrepancies across earlier summaries regarding model counts:

### Root Cause Analysis of Previous Discrepancies
- **15 vs 14 Models in Leaderboard:** Architectural diagrams listed 15 models (including `sarimax`), whereas the multi-origin benchmark evaluated 14 models. `sarimax` requires future exogenous weather regressors ($X_{\text{future}}$), and was retained as a specialized econometric reference rather than a univariate walk-forward competitor.
- **12 vs 15 Retained Models:** Previous reports cited "12 retained models" by counting collapsed family rows in summary tables (e.g. counting Seasonal Naive, AR(1), and MA(1) as one row). The actual count of individual retained models is **15**.
- **17 Total Models in Registry:** `src/model_registry.py` registers 17 distinct models. 2 are retired (`gradient_boosting` and `arimax`), leaving 15 retained.

### Authoritative Count Reconciliation Summary
```
Central Model Registry: 17 Registered Models (src/model_registry.py)
├── Retired / Pruned Architectures: 2 Models
│   ├── Hist Gradient Boosting (gradient_boosting): Redundant with Local XGBoost; inferior tree binning on sparse zeros.
│   └── ARIMAX (arimax): Misspecified without seasonal terms; superseded by SARIMAX.
└── Retained in Active Codebase: 15 Models
    ├── Multi-Origin Leaderboard Benchmark: 14 Models
    │   ├── Classical Statistical (6): Seasonal Naive, AR(1), MA(1), ARIMA(1,1,1), SARIMA, Holt-Winters Additive
    │   ├── Modern Machine Learning (4): Local RF, Local XGBoost, Global RF, Global XGBoost
    │   ├── Specialized Zero-Inflated (1): Hurdle XGBoost (Classifier + Regressor)
    │   ├── Specialized Additive (1): Prophet (Trend + Yearly Fourier Seasonality)
    │   └── Deep Learning (2): PyTorch LSTM, PyTorch GRU
    └── Exogenous Econometric Benchmark: 1 Model
        └── SARIMAX: Seasonal autoregression with NASA weather regressors (requires X_future).
```

Full details are documented in [FINAL_MODEL_INVENTORY.md](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/FINAL_MODEL_INVENTORY.md).

---

## 3. Phase 2 — Evaluation Count and Overlap Audit

The multi-origin benchmark engine in [scripts/run_multi_origin_benchmark.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/scripts/run_multi_origin_benchmark.py) was audited for temporal bounds and sample overlap:

### 3.1 Origin Design & Calendar Bounds

| Origin | Historical Cutoff | Cutoff Month Index | Forecast Horizon | Forecast Calendar Months | Overlap with Other Origins | Evaluation Steps / Pathogen |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Origin 1** | 2022-08-01 | Index 92 | 6 Months | **2022-09 to 2023-02** | None (Independent) | 6 steps |
| **Origin 2** | 2023-02-01 | Index 98 | 6 Months | **2023-03 to 2023-08** | None (Independent) | 6 steps |
| **Origin 3** | 2023-08-01 | Index 104 | 6 Months | **2023-09 to 2024-02** | Overlaps with Origin 4 for 4 months | 6 steps |
| **Origin 4** | 2023-10-01 | Index 106 | 6 Months | **2023-11 to 2024-04** | Overlaps with Origin 3 for 4 months | 6 steps |

### 3.2 Overlap Accounting and Unique Observations
- **Total Forecast Origins:** 4
- **Forecast Horizon per Origin:** 6 months
- **Total Evaluation Steps per Pathogen:** $4 \times 6 = 24$ steps
- **Total Prediction Records (10 Pathogens):** $24 \times 10 = 240$ prediction records per model
- **Calendar Months Covered:** September 2022 through April 2024 = **20 unique calendar months**
- **Unique Pathogen-Month Observations:** $20 \text{ months} \times 10 \text{ pathogens} = \mathbf{200} \text{ unique observations}$
- **Repeated Prediction Records:** 4 overlapping calendar months (2023-11, 2023-12, 2024-01, 2024-02) evaluated twice (once in Origin 3 at horizons $h=3,4,5,6$, and once in Origin 4 at horizons $h=1,2,3,4$):
  $$4 \text{ overlapping months} \times 10 \text{ pathogens} = \mathbf{40} \text{ repeated prediction records}$$
- **Total Records:** $200 \text{ unique} + 40 \text{ repeated} = \mathbf{240} \text{ total prediction records}$

### 3.3 Handling of Overlapping Windows
In operational walk-forward validation, overlapping forecast origins are intentional: they test whether a model's forecasts at long horizons ($h=3..6$ in Origin 3) remain consistent when re-forecasted from a closer historical origin ($h=1..4$ in Origin 4). All 14 models were evaluated on the **exact same 240 paired forecast steps**, guaranteeing strict parity and comparability.

---

## 4. Phase 3 — Pathogen Results Verification & Stripe Rust Audit

### 4.1 The Stripe Rust (*Puccinia striiformis*) Verification
The report previously noted an out-of-sample RMSE of **0.000** for Seasonal Naive on *Puccinia striiformis*. The audit investigated the exact underlying records:

1. **Validation Window (2022-09 to 2024-04; 20 unique months):**
   - 19 months were zero occurrences.
   - Exactly **1 month was positive**: May 2023 (`documented_occurrence_count` = 1.0).
2. **Evaluation Across the 4 Origins (24 total steps):**
   - Origin 1 (Sep 2022 – Feb 2023): All 6 months are 0.0.
   - Origin 2 (Mar 2023 – Aug 2023): 5 months are 0.0; May 2023 is 1.0.
   - Origin 3 (Sep 2023 – Feb 2024): All 6 months are 0.0.
   - Origin 4 (Nov 2023 – Apr 2024): All 6 months are 0.0.
   - Total steps: **23 zero steps** and **1 positive step** (May 2023). Note: May 2023 fell only in Origin 2 and was **not** in the overlapping window of Origins 3 and 4.
3. **Seasonal Naive Prediction Formulation:**
   - Seasonal Naive predicts $y_t = y_{t-12}$.
   - For all 23 zero steps, $y_{t-12} = 0.0$. Seasonal Naive predicted 0.0 (Error = 0.0).
   - For May 2023, $y_{t-12}$ corresponds to **May 2022**. Inspection of the raw historical records confirms May 2022 had an occurrence count of **1.0**!
   - Therefore, for May 2023, Seasonal Naive predicted **1.0**, matching actual 1.0 with zero error.
4. **Conclusion:**
   Across all 24 evaluation steps, Seasonal Naive matched the ground truth identically:
   $$\text{RMSE} = \sqrt{\frac{1}{24} \sum_{t=1}^{24} (y_t - \hat{y}_t)^2} = \sqrt{\frac{0}{24}} = \mathbf{0.000}$$
   This result is mathematically correct, fully reproducible, and reflects an exact seasonal recurrence in an otherwise dormant series.

### 4.2 Per-Pathogen Champion Verification
Every pathogen champion was re-verified against the underlying evaluation database:

| Pathogen Slug | Common Name | Raw RMSE Champion | Champion RMSE | Operational Policy Model | Policy Selection Rationale |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `alternaria_solani` | Early Blight | **LSTM** | 1.325 | LSTM | Rank 1 RMSE and MAE; high stability ($\sigma=0.150$). |
| `bipolaris_oryzae` | Brown Spot of Rice | **Global RF** | 4.344 | Global RF | Rank 1 RMSE; pools weather-disease correlations across crops. |
| `fusarium_oxysporum` | Fusarium Wilt | **Local XGBoost** | 6.375 | **GRU** | **Policy Override:** Local XGBoost achieved raw RMSE 6.375 ($\sigma=1.480, F1=0.960$). GRU achieved virtually identical RMSE 6.378 (diff 0.003) but **2x lower origin variance ($\sigma=0.771$)** and higher Outbreak F1 (0.980). Assigned GRU for operational stability. |
| `macrophomina_phaseolina`| Dry Root Rot | **GRU** | 1.758 | GRU | Rank 1 RMSE (1.758) and Rank 1 F1 (0.800). |
| `phytophthora_infestans` | Late Blight | **Local RF** | 3.151 | Local RF | Rank 1 RMSE (3.151) and MAE (1.102); parsimonious tree splits. |
| `puccinia_striiformis` | Stripe Rust | **Seasonal Naive** | 0.000 | Seasonal Naive | Exact seasonal recurrence on dormant series; zero overfitting. |
| `pyricularia_oryzae` | Rice Blast | **Global RF** | 1.906 | Global RF | Rank 1 RMSE (1.906); low variance ($\sigma=1.278$) across origins. |
| `ralstonia_solanacearum`| Bacterial Wilt | **Hurdle XGBoost** | 15.808 | Hurdle XGBoost | High recall (0.909) and F1 (0.741) on severe high-variance series. |
| `rhizoctonia_solani` | Sheath Blight | **Hurdle XGBoost** | 3.449 | Hurdle XGBoost | Rank 1 RMSE (3.449) and Rank 1 F1 (0.905). |
| `xanthomonas_oryzae` | Bacterial Blight | **MA(1)** | 0.505 | MA(1) | Rank 1 RMSE (0.505); moving average mean-reversion on sparse counts. |

---

## 5. Phase 4 — Scientific Claim Audit & Calibrated Interpretations

The technical report was audited for statements requiring qualification:

1. **Claim: "Hurdle XGBoost eliminates peak attenuation."**  
   - *Audit Finding:* No forecasting model completely eliminates peak underestimation during extreme non-linear spikes. In Origin 3 (Sep 2023 – Feb 2024), actual epidemic peaks exceeded 35 occurrences for *Ralstonia*, while Hurdle XGBoost predicted ~16–20.
   - *Calibrated Statement:* Hurdle XGBoost **substantially mitigates** peak attenuation compared to single-stage regressors (reducing Active Outbreak MAPE from 89.5% down to 66.3%, and reducing Origin 3 Surge RMSE from 13.18 to 10.07).
2. **Claim: "NASA weather covariates improve accuracy by 4.3% to 6.8% via spore germination mechanics."**  
   - *Audit Finding:* The 4.3% improvement was observed in ablation experiments (Exp E vs Exp A) at the macro dataset level. However, specific microclimatic mechanisms (e.g. "leaf wetness duration", "canopy dew point hours") are domain hypotheses from plant pathology literature; they are not directly measured in regional NASA POWER monthly averages.
   - *Calibrated Statement:* Incorporating lagged regional temperature and relative humidity provides a statistically observable 4.3% macro error reduction, which **aligns with established epidemiological principles** regarding fungal incubation requirements.
3. **Claim: "Flooding caused the sudden surge in Bacterial Wilt."**  
   - *Audit Finding:* The dataset records monthly counts and meteorological metrics; it does not measure specific flood stage telemetry.
   - *Calibrated Statement:* The dramatic jump in Bacterial Wilt occurrences from 0 to 38 within 60 days coincides with heavy autumn precipitation anomalies, **consistent with literature documenting waterborne dispersal** of *Ralstonia solanacearum*.
4. **Claim: "Transformers are infeasible due to parameter count."**  
   - *Audit Finding:* Transformer parameter claims ($10^5$ to $10^7$) are accurate for standard deep learning libraries. With $T=92$ training observations per series, the ratio of parameters to observations ($>1,000:1$) presents an undeniable risk of empirical overparameterization.
   - *Calibrated Statement:* Supported by both empirical literature and mathematical degrees-of-freedom analysis.

---

## 6. Phase 5 — Model Selection Policy Verification

Inspection of [config/model_selection_policy.json](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/config/model_selection_policy.json) and [src/model_policy.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/model_policy.py) confirmed:

1. **Clear Distinction Between Global and Local Selection:**
   - `default_global_champion`: `hurdle_xgboost` (overall benchmark champion, Pooled RMSE 5.982).
   - `macro_average_champion`: `global_random_forest` (Macro RMSE 4.081 across 10 pathogens).
   - `pathogen_assignments`: Specific model mapped per pathogen.
2. **Dual-Criteria for Forecasting vs. Outbreak Alerting:**
   - `default_operational`: Expected count policy ($\pi(x) \cdot \hat{y}_{\text{pos}}$) for continuous RMSE/MAE minimization.
   - `conservative_dormancy`: Thresholded policy ($\tau = 0.4$) for suppressing false alarms (FAR 15.4%) in quarantine decision-making.
3. **Zero Data Leakage:** All selection decisions derive strictly from the 20-month validation period (Sep 2022 – Apr 2024). The final test partition was never evaluated.
4. **Absence of Retired Models:** Neither `gradient_boosting` nor `arimax` appears in any selection assignment.
5. **Robust Fallback Mechanism:** Unlisted pathogens fall back to `default_global_champion` (`hurdle_xgboost`) with clear explanatory metadata.

---

## 7. Phase 6 — Classical vs. Modern Recalculated Leaderboard

All metrics were recalculated directly from the 240 prediction records:

| Metric | Best Classical Model | Best Modern Model | Absolute Gain | Relative Gain (%) | Audited Winner |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Pooled RMSE** | MA(1): 6.243 | **Hurdle XGBoost: 5.982** | -0.261 | **-4.18%** | **Modern (Hurdle XGBoost)** |
| **Macro RMSE** | MA(1): 4.280 | **Global RF: 4.081** | -0.199 | **-4.65%** | **Modern (Global RF)** |
| **Pooled MAE** | MA(1): 2.446 | **Prophet: 2.275** | -0.171 | **-6.99%** | **Modern (Prophet)** |
| **WAPE (%)** | MA(1): 87.9% | **Prophet: 81.7%** | -6.20% | **-7.05%** | **Modern (Prophet)** |
| **Active MAPE ($y \ge 1$)** | MA(1): 89.5% | **Hurdle XGBoost: 66.3%** | -23.2% | **-25.92%** | **Modern (Hurdle XGBoost)** |
| **Outbreak Recall** | Holt-Winters: 75.6% | **Hurdle XGBoost: 86.2%** | +10.6% | **+14.02%** | **Modern (Hurdle XGBoost)** |
| **Outbreak F1** | Holt-Winters: 0.738 | **Hurdle XGBoost: 0.774** | +0.036 | **+4.88%** | **Modern (Hurdle XGBoost)** |
| **Surge RMSE (Origin 3)** | MA(1): 10.514 | **Hurdle XGBoost: 10.068** | -0.446 | **-4.24%** (-23.6% vs ARIMA 13.18) | **Modern (Hurdle XGBoost)** |

---

## 8. Phase 8 — Final Test Data Protection Audit

The boundary between validation and testing was subjected to an audit:
- **Validation Range:** Month Index 92 to 111 (September 2022 to April 2024; 20 months).
- **Test Range:** Month Index 112 to 131 (May 2024 to December 2025; 20 months; 200 records).
- **Audit Findings:**
  1. `scripts/run_multi_origin_benchmark.py` max cutoff is index 106 + horizon 6 = index 111 (April 2024). Indices 112–131 were never queried.
  2. Model selection policy in `config/model_selection_policy.json` uses validation statistics only.
  3. Feature engineering lag operators and rolling windows calculate historical backward-looking values only.
  4. **The final test set is 100% intact, unobserved, and preserved for final evaluation.**

---

## 9. Final Test Suite Verification

The regression test suite was executed via `pytest`:
- **Total Tests Executed:** 92
- **Passed Tests:** 92
- **Failed Tests:** 0
- **Execution Runtime:** 84.70 seconds

```
================= 92 passed, 270 warnings in 84.70s (0:01:24) ==================
```

---
*Audit Completed & Certified: Autonomous Systems Engineering Team*

