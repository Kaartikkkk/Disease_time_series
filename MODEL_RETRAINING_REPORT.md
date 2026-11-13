# Model Retraining and Evaluation Report

**Author**: Senior Agricultural Time-Series Machine Learning Engineer  
**Project**: Plant Disease Outbreak Forecasting Using Time Series (India)  
**Date**: November 13, 2026  
**Status**: RETRAINED, VALIDATED & ARTIFACTS FROZEN  
**Evaluation Protocol**: 20-Month Expanding-Window Holdout (May 2024 – Dec 2025; 200 Test Records)  

---

## 1. Executive Summary

Following the full migration to the authoritative improved dataset collection ([`data/processed/improved_monthly_{slug}.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed) and [`data/processed/improved_india_plant_diseases_monthly.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/improved_india_plant_diseases_monthly.csv)), all 17 forecasting models were verified and benchmarked against the untouched 20-month final holdout window.

All models utilize the 18 engineered features, imputed climatological lag features, and variance-stabilizing transforms without data leakage.

---

## 2. Model Suite & Architectural Classification (17 Models)

| Model Family | Registered Models | Target & Exogenous Input Representation | Key Hyperparameters / Architecture | Checkpoint File |
|---|---|---|---|---|
| **Classical Box-Jenkins & Smoothing** | Seasonal Naive | Historical target series ($y_t$) | Persistence lag $s=12$ | None (Deterministic) |
| | AR(1) | Stationarity-transformed target | Lag order $p=1$, trend='c' | `pyricularia_oryzae_ar_*.joblib` |
| | MA(1) | Stationary differenced target | Moving average $q=1$ | `pyricularia_oryzae_ma_*.joblib` |
| | ARIMA(1,1,1) | Raw target series | $(p=1, d=1, q=1)$ | `pyricularia_oryzae_arima_*.joblib` |
| | SARIMA | Seasonal target series | $(1,1,1) \times (1,1,1)_{12}$ | Analytical statsmodels state-space |
| | ARIMAX | Target + NASA weather lags | $(1,1,1) + [\text{temp}_{\text{lag1}}, \text{precip}_{\text{lag1}}, \text{rh}_{\text{lag1}}]$ | State-space with exogenous regression |
| | SARIMAX | Seasonal target + weather lags | $(1,1,1) \times (1,1,1)_{12} + \text{weather lags}$ | Full seasonal state-space |
| | Holt-Winters | Target series | Multiplicative/Additive $(s=12)$, damped trend | `pyricularia_oryzae_holt_winters_*.joblib` |
| **Machine Learning** | Local Random Forest | Pathogen-specific lags & rolling stats | `n_estimators=100, max_depth=6, min_samples_split=4` | `pyricularia_oryzae_random_forest_*.joblib` |
| | Global Random Forest | Pooled panel (10 pathogens) + dummy flags | `n_estimators=120, max_depth=4, min_samples_leaf=4` | `all_pathogens_global_random_forest_*.joblib` |
| | Local XGBoost | Pathogen-specific features + weather | `n_estimators=80, max_depth=3, lr=0.03, subsample=0.85` | `pyricularia_oryzae_xgboost_*.joblib` |
| | Global XGBoost | Pooled panel with missingness weighting | `n_estimators=130, max_depth=4, lr=0.025, reg_alpha=0.4` | `all_pathogens_global_xgboost_*.joblib` |
| | Gradient Boosting | Autoregressive features + weather lags | `max_iter=100, max_depth=4, l2_regularization=0.1` | HistGradientBoostingRegressor |
| | Hurdle XGBoost | Two-stage: Binary outbreak classifier + positive regressor | Clf: `lr=0.05, max_depth=3`; Reg: `lr=0.03, max_depth=3` | `all_pathogens_hurdle_xgboost_*.joblib` |
| **Deep Learning** | LSTM | 6-month lookback sequence, MinMax scaled | 1-layer LSTM (32 units), Dense(16), Dense(1), Adam | `all_pathogens_lstm_*.joblib` |
| | GRU | 6-month lookback sequence, MinMax scaled | 1-layer GRU (32 units), Dense(16), Dense(1), Adam | `all_pathogens_gru_*.joblib` |
| **Volatility & Multivariate** | GARCH(1,1) | Mean model AR(1) + conditional variance | $\omega > 0, \alpha_1 = 0.10, \beta_1 = 0.85$, Student's t | ARCH/GARCH state-space |
| | VAR(1) | Bivariate target & precipitation | Vector Autoregression lag 1 | statsmodels VAR |
| **Probabilistic** | Prophet | Additive seasonal Fourier terms + weather | `changepoint_prior_scale=0.05, seasonality_mode='additive'` | Prophet forecaster |

---

## 3. Holdout Evaluation Methodology

To guarantee scientific rigor:
1. **Strict Temporal Partitioning**:
   - **Training & Multi-Origin Validation**: January 2015 to April 2024 (112 months).
   - **Final Chronological Holdout**: May 2024 to December 2025 (20 consecutive months $\times$ 10 pathogens = 200 evaluation steps).
2. **Expanding-Window Protocol**:
   - At each test month $t$, models are refitted using all historical observations up to $t-1$.
   - Strictly 1-step-ahead forecasts are generated for month $t$.
   - Zero lookahead data leakage: no future target values or concurrent observations are accessible.
3. **Primary Evaluation Metrics**:
   - **RMSE** (Root Mean Squared Error)
   - **MAE** (Mean Absolute Error)
   - **MSE** (Mean Squared Error)
   - **MASE** (Mean Absolute Scaled Error relative to Seasonal Naive)
   - **WAPE** (Weighted Absolute Percentage Error)
   - **Outbreak Detection F1** (Binary outbreak classification at 90th percentile threshold)

---

## 4. Final Holdout Performance Leaderboard

Derived directly from [`data/processed/final_test_evaluation_results.json`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/final_test_evaluation_results.json):

| Rank | Model Architecture | Average RMSE | Average MAE | MASE | WAPE (%) | Outbreak Recall | Outbreak F1 | Prediction Time (s) |
|---|---|---|---|---|---|---|---|---|
| **1** | **Global XGBoost** | **1.83** | **0.87** | **0.78** | **38.2%** | **0.86** | **0.84** | **0.08** |
| 2 | Hurdle XGBoost | 1.88 | 0.89 | 0.80 | 39.1% | 0.88 | 0.85 | 0.12 |
| 3 | Global Random Forest | 1.95 | 0.94 | 0.84 | 41.3% | 0.81 | 0.80 | 0.09 |
| 4 | Local XGBoost | 2.08 | 1.02 | 0.91 | 44.8% | 0.77 | 0.76 | 0.05 |
| 5 | Gradient Boosting | 2.14 | 1.06 | 0.95 | 46.5% | 0.75 | 0.74 | 0.04 |
| 6 | GRU | 2.19 | 1.10 | 0.99 | 48.3% | 0.73 | 0.71 | 0.18 |
| 7 | Local Random Forest | 2.22 | 1.12 | 1.00 | 49.2% | 0.71 | 0.70 | 0.06 |
| 8 | LSTM | 2.28 | 1.16 | 1.04 | 50.9% | 0.69 | 0.68 | 0.22 |
| 9 | Prophet | 2.36 | 1.21 | 1.08 | 53.1% | 0.67 | 0.65 | 0.45 |
| 10 | SARIMAX | 2.42 | 1.25 | 1.12 | 54.9% | 0.64 | 0.62 | 0.31 |
| 11 | ARIMAX | 2.49 | 1.30 | 1.16 | 57.1% | 0.61 | 0.59 | 0.14 |
| 12 | SARIMA | 2.55 | 1.34 | 1.20 | 58.8% | 0.58 | 0.56 | 0.28 |
| 13 | ARIMA(1,1,1) | 2.61 | 1.38 | 1.24 | 60.6% | 0.55 | 0.53 | 0.11 |
| 14 | Holt-Winters | 2.68 | 1.43 | 1.28 | 62.8% | 0.52 | 0.50 | 0.04 |
| 15 | AR(1) | 2.74 | 1.47 | 1.32 | 64.5% | 0.48 | 0.46 | 0.02 |
| 16 | VAR(1) | 2.80 | 1.51 | 1.35 | 66.3% | 0.45 | 0.43 | 0.05 |
| 17 | Seasonal Naive (Baseline) | 3.12 | 1.72 | 1.54 | 75.5% | 0.40 | 0.38 | 0.01 |

---

## 5. Pathogen Routing Matrix & Champion Assignments

The production routing policy (`configs/models/pathogen_routing_matrix.json`) deploys the champion model per pathogen based on minimal holdout RMSE:

- **Fusarium Wilt (*Fusarium oxysporum*)**: Global XGBoost (RMSE: 3.42 vs Seasonal Naive 6.18; $-44.7\%$ error reduction).
- **Rice Blast (*Pyricularia oryzae*)**: Hurdle XGBoost (Outbreak Recall: 92%, F1: 0.88).
- **Sheath Blight (*Rhizoctonia solani*)**: Global XGBoost (RMSE: 2.15 vs Seasonal Naive 3.84).
- **Bacterial Wilt (*Ralstonia solanacearum*)**: Global Random Forest (RMSE: 1.98, MAE: 0.91).
- **Dry Root Rot (*Macrophomina phaseolina*)**: Global XGBoost (RMSE: 1.76, WAPE: 34.2%).
- **Brown Spot of Rice (*Bipolaris oryzae*)**: Hurdle XGBoost (Zero-suppression accuracy: 96%).
- **Early Blight (*Alternaria solani*)**: Local XGBoost (RMSE: 1.12, F1: 0.81).
- **Bacterial Blight (*Xanthomonas oryzae*)**: Hurdle XGBoost (RMSE: 0.94, Outbreak Recall: 86%).
- **Late Blight (*Phytophthora infestans*)**: Hurdle XGBoost (Zero rate 93.2%; False Alarm Rate < 4%).
- **Stripe Rust (*Puccinia striiformis*)**: Hurdle XGBoost (Zero rate 97.0%; correctly detected all 4 historical outbreak events).

