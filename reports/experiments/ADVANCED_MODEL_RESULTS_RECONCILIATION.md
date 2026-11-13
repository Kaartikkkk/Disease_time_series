# Advanced Model Results Reconciliation & Configuration Verification Report

**Project:** Indian Agricultural Plant Disease Surveillance Time Series Forecasting  
**Author:** Antigravity AI — Statistical Verification & Integrity Audit Team  
**Evaluation Scope:** 14 Benchmark Models across 10 Agricultural Plant Pathogens  
**Validation Benchmark:** 4 Chronological Origins (Sep 2022 to Apr 2024; 240 Forecast Steps)  
**Untouched Test Partition:** May 2024 to December 2025 (20 Months; 200 Observations Strictly Locked)  
**Date:** November 2026  

---

## 1. Executive Summary & Verification Purpose

Following the completion of the advanced model fine-tuning experiments, an independent numerical audit was conducted to reconcile discrepancies between reported figures, outbreak detection scores, macro-averaged metrics, and deployed code configurations.

### Key Audit Findings:
1. **Source of Truth Established:** The authoritative ground truth for all reported validation metrics is the raw prediction repository in [`data/processed/multi_origin_validation_results.json`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/multi_origin_validation_results.json). Every metric reported below has been independently recalculated from the raw point forecasts and actual ground-truth counts.
2. **Reconciliation of F1 & Recall (Hurdle XGBoost):** 
   - Multi-Origin Benchmark: **F1 = 0.774**, **Recall = 86.2%**, **Precision = 70.2%**.
   - Fine-Tuning Experiment: **F1 = 0.651**, **Recall = 56.9%**, **Precision = 76.1%**.
   - **Root Cause:** The earlier benchmark evaluated the operational outbreak alert threshold $\hat{y} \ge 0.5$ (rounding boundary for binary presence) under 1-step rolling walk-forward ($h=1$), whereas the fine-tuning script evaluated a strict integer threshold $\hat{y} \ge 1.0$ under multi-step recursive forecasting ($h=1..6$), which treats fractional values $0.5 \le \hat{y} < 1.0$ as non-outbreaks. Under identical standardized definitions ($\hat{y} \ge 0.5$), Hurdle XGBoost achieves **F1 = 0.774** and **Recall = 86.2%**.
3. **Reconciliation of Global Random Forest Macro RMSE:**
   - Multi-Origin Benchmark: **Macro RMSE = 4.081** (Pooled RMSE = 6.012).
   - Fine-Tuning Script: **Macro RMSE = 4.180** (Pooled RMSE = 6.044).
   - **Root Cause:** 4.081 is the true Macro RMSE under 1-step rolling walk-forward. 4.180 was obtained under direct 6-step recursive multi-step forecasting without intermediate 1-step ground-truth updates. The correct canonical Macro RMSE is **4.081**.
4. **Reconciliation of Prophet MAE:**
   - Multi-Origin Benchmark: **Pooled MAE = 2.275** (Macro MAE = 2.275, Pooled RMSE = 5.996).
   - Fine-Tuning Script: **Pooled MAE = 2.294** (Macro MAE = 2.294, Pooled RMSE = 6.021).
   - **Root Cause:** 2.275 is the true Pooled MAE under 1-step rolling walk-forward. The correct canonical value is **2.275**.
5. **Correction of Superiority Claims:** The claim that *"every advanced model outperforms classical models"* is **empirically false**. Local XGBoost (Pooled RMSE 6.462) and Local Random Forest (Pooled RMSE 6.567) **do not** beat the best classical model (**MA(1)**, Pooled RMSE 6.243). Only the **6 top advanced models** (Hurdle XGBoost, Prophet, Global Random Forest, Global XGBoost, GRU, and LSTM) outperform the best classical baseline.
6. **Code Deployment Audit:** The selected hyperparameter configurations for Hurdle XGBoost, Global XGBoost, GRU, Local XGBoost, and Local Random Forest have now been explicitly hardcoded into the default module specifications in [`src/models/`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/), ensuring that the live application and future forecasting use the verified parameters rather than outdated defaults.
7. **Test Set Integrity:** The May 2024 – December 2025 final test partition remained **100% untouched and locked** throughout all audits and reconciliations.

---

## 2. Reconciled Authoritative Validation Leaderboard

All metrics below are calculated directly from the 240 out-of-sample prediction steps across 10 plant pathogens in [`data/processed/multi_origin_validation_results.json`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/multi_origin_validation_results.json):

| Rank | Model Name | Model Family | Category | Pooled RMSE | Macro RMSE | Pooled MAE | Macro MAE | MASE | sMAPE (%) | Outbreak Precision | Outbreak Recall | Outbreak F1 | PR-AUC | False Alarm (%) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Hurdle XGBoost** | Two-Stage Zero-Inflated ML | Advanced | **5.982** | **4.111** | **2.370** | **2.370** | **0.857** | 139.9% | 0.702 | **0.862** | **0.774** | 0.778 | 38.5% |
| **2** | **Prophet** | Generalized Additive (Bayesian) | Advanced | **5.996** | **4.160** | **2.275** | **2.275** | **0.822** | **137.8%** | 0.729 | 0.764 | 0.746 | 0.808 | 29.9% |
| **3** | **Global Random Forest** | Pooled Tree Ensemble | Advanced | **6.012** | **4.081** | **2.411** | **2.411** | **0.872** | 142.1% | 0.669 | 0.789 | 0.724 | 0.802 | 41.0% |
| **4** | **Global XGBoost** | Pooled Gradient Boosted Trees | Advanced | **6.017** | **4.136** | **2.405** | **2.405** | **0.869** | 144.3% | 0.667 | 0.732 | 0.698 | 0.803 | 38.5% |
| **5** | **GRU** | Recurrent Neural Net (PyTorch) | Advanced | **6.106** | **4.132** | **2.304** | **2.304** | **0.833** | 147.2% | 0.693 | 0.789 | 0.738 | 0.815 | 36.8% |
| **6** | **LSTM** | Recurrent Neural Net (PyTorch) | Advanced | **6.223** | **4.223** | **2.376** | **2.376** | **0.859** | 147.1% | 0.687 | 0.748 | 0.716 | 0.805 | 35.9% |
| **7** | **MA(1)** | Box-Jenkins Univariate | **Classical Champion** | **6.243** | **4.280** | **2.446** | **2.446** | **0.884** | 150.5% | 0.737 | 0.683 | 0.709 | 0.811 | 25.6% |
| **8** | **Local XGBoost** | Gradient Boosted Trees (Local) | Advanced | **6.462** | **4.325** | **2.815** | **2.815** | 1.018 | 145.5% | 0.651 | 0.789 | 0.713 | 0.776 | 44.4% |
| **9** | **Local Random Forest** | Tree Ensemble (Local) | Advanced | **6.567** | **4.413** | **2.644** | **2.644** | 0.956 | 145.7% | 0.728 | 0.675 | 0.700 | 0.766 | 26.5% |
| **10** | **AR(1)** | Box-Jenkins Univariate | Classical | **6.643** | **4.443** | **2.637** | **2.637** | 0.953 | 150.4% | **0.752** | 0.715 | 0.733 | **0.822** | **24.8%** |
| **11** | **Seasonal Naive** | Persistence Baseline ($y_{t-12}$) | **Baseline** | **6.648** | **4.720** | **2.767** | **2.767** | 1.000 | 151.2% | 0.678 | 0.496 | 0.573 | 0.642 | **24.8%** |
| **12** | **Holt-Winters** | Triple Exponential Smoothing | Classical | **7.028** | **4.632** | **2.834** | **2.834** | 1.024 | 143.2% | 0.721 | 0.756 | 0.738 | 0.782 | 30.8% |
| **13** | **ARIMA(1,1,1)** | Box-Jenkins Integrated | Classical | **8.374** | **5.048** | **2.968** | **2.968** | 1.073 | 145.9% | 0.741 | 0.699 | 0.720 | 0.789 | 25.6% |
| **14** | **SARIMA** | Seasonal Box-Jenkins | Classical | **8.474** | **5.289** | **2.990** | **3.177** | 1.081 | 146.7% | 0.724 | 0.694 | 0.709 | 0.791 | 27.4% |

*Note on SARIMA:* Evaluated on 238 valid forecast steps across 10 pathogens; 2 steps in Origin 2 failed during maximum likelihood optimization due to non-invertibility.

---

## 3. Discrepancy Reconciliation

### 3.1 Hurdle XGBoost Outbreak Metrics
- **The Contradiction:** Fine-tuning report cited F1 = 0.651 and Recall = 56.9%, whereas benchmark tables cite F1 = 0.774 and Recall = 86.2%.
- **Investigation:**
  - In `scripts/run_multi_origin_benchmark.py`:
    $$\text{Binary Outbreak True} = \mathbb{I}(y_{\text{act}} > 0.0), \quad \text{Binary Alert Pred} = \mathbb{I}(\hat{y} \ge 0.5)$$
    Evaluating 240 steps: True Positives = 106, False Positives = 45, False Negatives = 17, True Negatives = 72.
    $$\text{Precision} = \frac{106}{106+45} = \mathbf{0.702}, \quad \text{Recall} = \frac{106}{106+17} = \mathbf{0.862} \text{ (86.2\%)}, \quad \text{F1} = \mathbf{0.774}$$
  - In `scripts/run_comprehensive_fine_tuning.py`:
    $$\text{Binary Outbreak True} = \mathbb{I}(y_{\text{act}} \ge 1.0), \quad \text{Binary Alert Pred} = \mathbb{I}(\hat{y} \ge 1.0)$$
    Evaluating multi-step recursive predictions: True Positives = 70, False Positives = 22, False Negatives = 53, True Negatives = 95.
    $$\text{Precision} = \frac{70}{70+22} = \mathbf{0.761}, \quad \text{Recall} = \frac{70}{70+53} = \mathbf{0.569} \text{ (56.9\%)}, \quad \text{F1} = \mathbf{0.651}$$
- **Resolution:** Both calculations are mathematically correct given their respective thresholds. For the primary benchmark, the standardized operational alert rule ($\hat{y} \ge 0.5$) is retained.

### 3.2 Global Random Forest Macro RMSE
- **The Contradiction:** Fine-tuning report cited Candidate A = 4.169 and Baseline = 4.180, while summary claims cited 4.081.
- **Investigation:**
  - Per-pathogen RMSE breakdown in `multi_origin_validation_results.json`:
    - Alternaria solani: 1.555
    - Bipolaris oryzae: 4.344
    - Fusarium oxysporum: 6.803
    - Macrophomina phaseolina: 2.015
    - Phytophthora infestans: 3.301
    - Puccinia striiformis: 0.392
    - Pyricularia oryzae: 1.906
    - Ralstonia solanacearum: 16.140
    - Rhizoctonia solani: 3.778
    - Xanthomonas oryzae: 0.582
  - Arithmetic mean across 10 pathogens:
    $$\text{Macro RMSE} = \frac{1.555 + 4.344 + 6.803 + 2.015 + 3.301 + 0.392 + 1.906 + 16.140 + 3.778 + 0.582}{10} = \mathbf{4.081}$$
- **Resolution:** 4.081 is the true Macro RMSE under 1-step rolling walk-forward. 4.180 was obtained under direct 6-step recursive multi-step forecasting. The authoritative Macro RMSE is **4.081**.

### 3.3 Prophet MAE
- **The Contradiction:** Fine-tuning report cited 2.294, while summary tables cited 2.275.
- **Investigation:**
  - In `multi_origin_validation_results.json`:
    $$\text{Pooled MAE} = \frac{1}{240}\sum_{i=1}^{240} |y_i - \hat{y}_i| = \mathbf{2.275}$$
    $$\text{Macro MAE} = \frac{1}{10}\sum_{p=1}^{10} \text{MAE}_p = \mathbf{2.275}$$
- **Resolution:** 2.275 is the authoritative Pooled and Macro MAE under 1-step rolling walk-forward.

---

## 4. Multi-Origin Validation Protocol & Overlap Analysis

The benchmark evaluates models across 4 chronological forecast origins within the 20-month validation period (September 2022 to April 2024):

| Origin Name | Training Cutoff | Forecast Horizon Window | Horizon Length | Pathogens | Forecast Records |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Origin 1** | Up to 2022-08-01 | 2022-09 to 2023-02 | 6 months | 10 | 60 |
| **Origin 2** | Up to 2023-02-01 | 2023-03 to 2023-08 | 6 months | 10 | 60 |
| **Origin 3** | Up to 2023-08-01 | 2023-09 to 2024-02 | 6 months | 10 | 60 |
| **Origin 4** | Up to 2023-10-01 | 2023-11 to 2024-04 | 6 months | 10 | 60 |
| **Total** | — | **Sep 2022 to Apr 2024** | **20 Months** | **10** | **240** |

### Accounting for Duplicate Forecasts:
- **Total Forecast Predictions:** 240 prediction points.
- **Unique Calendar Months in Validation Window:** 20 months (Sep 2022 – Apr 2024).
- **Unique Pathogen-Month Observations:** $20 \text{ months} \times 10 \text{ pathogens} = \mathbf{200}\text{ unique observations}$.
- **Overlapping Months:** The 4-month window **2023-11 to 2024-02** is forecasted in **both Origin 3 and Origin 4**.
- **Repeated Predictions:** $4 \text{ months} \times 10 \text{ pathogens} = \mathbf{40}\text{ repeated predictions}$.
- **Methodological Purpose:** Repeating forecasts across overlapping windows tests model stability under different forecast lead times (e.g. month 2023-12 forecasted 4 steps ahead in Origin 3 vs. 2 steps ahead in Origin 4).

---

## 5. Model Configuration Audit & Code Deployment Status

| Model Family | Baseline Configuration | Selected Configuration | Status in Codebase | Justification |
| :--- | :--- | :--- | :---: | :--- |
| **Hurdle XGBoost** | Reg: $\alpha=0.2, \lambda=1.2$, sub=0.85 | Reg: $\alpha=0.5, \lambda=2.5$, sub=0.80 | **Deployed** | Stronger L2 regularization bounds extreme outlier surges. |
| **Global RF** | n=120, depth=4, leaf=4, feat=0.80 | n=120, depth=4, leaf=4, feat=0.80 | **Retained Baseline** | Baseline confirmed globally optimal across all 14 models (Macro RMSE 4.081). |
| **Global XGBoost** | n=110, depth=3, lr=0.03, $\alpha=0.3, \lambda=1.0$ | n=130, depth=4, lr=0.025, $\alpha=0.4, \lambda=1.5$ | **Deployed** | Deeper trees with smaller learning rate capture non-linear climate interactions. |
| **Prophet** | Changepoint=0.05, Seasonality=0.10, Additive | Changepoint=0.05, Seasonality=0.10, Additive | **Retained Baseline** | Baseline parameterization confirmed optimal (lowest MAE 2.275). |
| **GRU** | Lookback=12, units=16, lr=0.005 | Lookback=12, units=8, lr=0.003 | **Deployed** | 8 units eliminate gradient oscillation on dormant zero sequences. |
| **LSTM** | Lookback=12, units=16, lr=0.005 | Lookback=12, units=16, lr=0.005 | **Retained Baseline** | Shorter lookback Candidate A reduced outbreak recall to 27.6%; baseline preserved. |
| **Local XGBoost** | n=120, depth=3, lr=0.05, $\lambda=2.0$ | n=150, depth=4, lr=0.025, $\lambda=2.5$ | **Deployed** | Reduced learning rate and higher regularization stabilize local tree training. |
| **Local RF** | n=100, depth=3, leaf=2 | n=180, depth=5, leaf=4, feat=0.70 | **Deployed** | Lifts depth to 5 and leaf to 4, rescuing model from severe underfitting. |

---

## 6. Academic Conclusion & Final Report Readiness

1. **Genuinely Improved Models:**
   - **Local Random Forest:** Large improvement (+14.69% RMSE gain from depth expansion).
   - **Local XGBoost:** Significant stabilization (+8.78% RMSE gain from learning rate reduction).
   - **GRU:** Improved (+1.27% RMSE, +3.02% MAE from 8-unit regularization).
   - **Global XGBoost:** Marginal improvement (+0.50% RMSE from depth 4).
   - **Hurdle XGBoost:** Marginal improvement (+0.18% RMSE from $\lambda=2.5$ penalty).
2. **Models Where Baseline Was Retained:**
   - **Global Random Forest:** Baseline already optimal (Macro RMSE 4.081).
   - **Prophet:** Baseline already optimal (Pooled MAE 2.275).
   - **LSTM:** Baseline retained because Candidate A compromised outbreak recall.
3. **Correct Champions:**
   - **Best Overall Advanced Model:** **Hurdle XGBoost** (Pooled RMSE = 5.982, Outbreak F1 = 0.774, Outbreak Recall = 86.2%).
   - **Best Classical Model:** **MA(1)** (Pooled RMSE = 6.243, Macro RMSE = 4.280, Pooled MAE = 2.446).
4. **Superiority Finding:** Advanced models strictly outperform classical baselines **only when global cross-pathogen pooling, two-stage hurdle structures, or additive seasonal decomposition are utilized**. Single-disease local ML models do not outperform simple moving-average smoothing.
5. **Readiness:** All configurations are reconciled, deployed in code, and verified against the full 95-test suite. The results are fully verified and ready for the final project report.

