# Final Model Inventory & Architectural Reconciliation

## Indian Agricultural Plant Disease Surveillance System
**Audit Date:** November 2026  
**Document Version:** 1.0.0 (Authoritative Post-Phase 3 Reconciliation)  
**Scope:** Complete accounting of all forecasting models registered, evaluated, retained, and retired across the project.

---

## 1. Executive Reconciliation of Model Counts

During Phase 3 development and reporting, differing model counts appeared across various summaries (e.g. 15 in architectural trees, 14 in the multi-origin leaderboard, 12 in reduction tables, 17 in the registry). This document provides the single authoritative reconciliation:

### The Definitive Count Hierarchy
1. **Total Registered Models in Central Registry (`src/model_registry.py`): 17 Models**  
   Every model possesses a registered `ModelSpec`, a standardized adapter function in `src/models/adapters.py`, and passes the central `ModelOutput` contract validator.
2. **Retired / Pruned Models from Operational Deployment: 2 Models**  
   - `gradient_boosting` (Local HistGradientBoosting)
   - `arimax` (Non-seasonal ARIMAX)
3. **Retained Models in Active Codebase: 15 Models**  
   15 models are retained as operational production candidates or research econometric benchmarks.
4. **Models Evaluated in the Multi-Origin Validation Benchmark: 14 Models**  
   The standardized multi-origin walk-forward engine (`scripts/run_multi_origin_benchmark.py`) evaluates 14 models across all 10 pathogens. The 15th retained model (`sarimax`) requires aligned future exogenous weather series ($X_{\text{future}}$), and is retained as an exogenous econometric benchmark rather than a univariate walk-forward competitor.
5. **Resolution of the "12 Models" Reference:**  
   Earlier drafts referred to "12 retained models" by condensing model families into 11/12 table rows (e.g., grouping Seasonal Naive, AR(1), and MA(1) as one classical row; grouping LSTM and GRU as one deep learning row). The actual individual count of retained models is **15**.

---

## 2. Authoritative Master Model Inventory Table

| # | Model ID | Display Name | Model Family | Category | Implementation File | Multi-Origin Benchmark Status | Operational Status | Technical Rationale for Retention or Retirement |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `seasonal_naive` | Seasonal Naive (Baseline) | Persistence Baseline | Classical | [src/models/statistical.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/statistical.py) | **Evaluated (240 steps)** | **Retained** | Essential zero-parameter baseline ($s=12$). Achieves 0.000 RMSE on *Puccinia striiformis* by matching annual May pulses. |
| **2** | `ar` | AR(1) | Box-Jenkins Univariate | Statistical | [src/models/statistical.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/statistical.py) | **Evaluated (240 steps)** | **Retained** | Sub-millisecond first-order linear autoregression; benchmark for baseline persistence. |
| **3** | `ma` | MA(1) | Box-Jenkins Univariate | Statistical | [src/models/statistical.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/statistical.py) | **Evaluated (240 steps)** | **Retained** | Top-performing classical model across quiescent periods (Macro RMSE 4.280); quick shock reversion. |
| **4** | `arima` | ARIMA(1,1,1) | Box-Jenkins Integrated | Statistical | [src/models/statistical.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/statistical.py) | **Evaluated (240 steps)** | **Retained** | Standard econometric reference. Captures integrated non-stationary dynamics ($d=1$). |
| **5** | `sarima` | SARIMA(1,0,1)(1,0,1)[12] | Seasonal Box-Jenkins | Statistical | [src/models/sarima_model.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/sarima_model.py) | **Evaluated (238 steps)** | **Retained** | Captures 12-month agricultural seasonality. Handled 2 convergence retries in Origin 3/4 gracefully. |
| **6** | `holt_winters` | Holt-Winters Additive ESM | Exponential Smoothing | Classical | [src/models/statistical.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/statistical.py) | **Evaluated (240 steps)** | **Retained** | Classical benchmark with highest outbreak recall (75.6%) among classical methods. |
| **7** | `arimax` | ARIMAX (Weather Regressors) | Exogenous Econometric | Statistical | [src/models/statistical.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/statistical.py) | Omitted (Requires $X_{\text{fut}}$) | **Retired** | **Pruned:** Non-seasonal formulation is mathematically misspecified for annual crop cycles; strictly superseded by SARIMAX. |
| **8** | `sarimax` | SARIMAX (Seasonal + Weather) | Exogenous Econometric | Statistical | [src/models/sarima_model.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/sarima_model.py) | Omitted from Leaderboard | **Retained** | Advanced econometric benchmark integrating NASA weather covariates into seasonal state space. |
| **9** | `gradient_boosting` | Hist Gradient Boosting | Tree Ensemble ML | Machine Learning | [src/models/ml_models.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/ml_models.py) | Omitted (Pre-screened) | **Retired** | **Pruned:** Redundant with Local XGBoost; histogram binning causes higher quantization error on 71.8% zero counts. |
| **10** | `random_forest` | Random Forest (Autoregressive) | Tree Ensemble ML | Machine Learning | [src/models/ml_models.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/ml_models.py) | **Evaluated (240 steps)** | **Retained** | Pathogen-specific non-linear tree bagging. Winning model for *Phytophthora infestans* (RMSE 3.151). |
| **11** | `xgboost` | XGBoost (Autoregressive) | Gradient Boosted Trees | Machine Learning | [src/models/xgboost_model.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/xgboost_model.py) | **Evaluated (240 steps)** | **Retained** | Local regularized gradient boosting. Lowest empirical RMSE for *Fusarium oxysporum* (6.375). |
| **12** | `global_random_forest` | Global Random Forest | Pooled Tree Ensemble | Machine Learning | [src/models/global_models.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/global_models.py) | **Evaluated (240 steps)** | **Retained** | **Macro RMSE Champion (4.081).** Multi-pathogen panel pooling ($N=920$) enables cross-crop transfer learning. |
| **13** | `global_xgboost` | Global XGBoost | Pooled Gradient Boosted Trees | Machine Learning | [src/models/global_models.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/global_models.py) | **Evaluated (240 steps)** | **Retained** | Pooled gradient boosting with cross-pathogen lag summaries (Pooled RMSE 6.017). |
| **14** | `hurdle_xgboost` | Hurdle XGBoost (Classifier + Regressor) | Two-Stage Zero-Inflated ML | Machine Learning | [src/models/hurdle_model.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/hurdle_model.py) | **Evaluated (240 steps)** | **Retained** | **Overall Pooled Champion (RMSE 5.982, Outbreak F1 0.774).** Separates zero-dormancy from active count intensity. |
| **15** | `prophet` | Prophet (Trend + Seasonality) | Generalized Additive | Statistical | [src/models/prophet_model.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/prophet_model.py) | **Evaluated (240 steps)** | **Retained** | **Absolute Error Champion (Pooled MAE 2.275, WAPE 81.7%).** Robust trend changepoints and Fourier harmonics. |
| **16** | `lstm` | LSTM (Sequence Neural Net) | Recurrent Neural Net | Deep Learning | [src/models/lstm_model.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/lstm_model.py) | **Evaluated (240 steps)** | **Retained** | Continuous sequential hidden state ($H=32, L=6$). Winning model for *Alternaria solani* (RMSE 1.325). |
| **17** | `gru` | GRU (Gated Recurrent Unit) | Recurrent Neural Net | Deep Learning | [src/models/gru_model.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/models/gru_model.py) | **Evaluated (240 steps)** | **Retained** | Parsimonious recurrent gating. Winning model for *Macrophomina phaseolina* (RMSE 1.758); elite F1 on Fusarium (0.980). |

---

## 3. Summary of Operational State

- **Active Central Registry:** All 17 models are formally registered in [src/model_registry.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/model_registry.py).
- **Multi-Origin Leaderboard:** 14 models benchmarked across all 10 pathogens over 240 steps (September 2022 to April 2024).
- **Production Routing Policy:** Enforced via [config/model_selection_policy.json](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/config/model_selection_policy.json) and [src/model_policy.py](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/model_policy.py).
- **Retired Models:** 2 models (`gradient_boosting`, `arimax`) are preserved in code for architectural reference but excluded from active production deployment and selection policies.
- **Untouched Test Set:** May 2024 to December 2025 (200 records) is strictly preserved for final verification.

