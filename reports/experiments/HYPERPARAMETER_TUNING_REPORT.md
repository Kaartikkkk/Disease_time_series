# HYPERPARAMETER_TUNING_REPORT.md
## Rigorous Hyperparameter Tuning, Search Spaces & Optimization Audit

**Project:** Agricultural Disease Forecasting Using Time Series  
**Dataset Source:** Improved Original Agricultural Surveillance Dataset v2.0 (`data/processed/improved_india_plant_diseases_monthly.csv`)  
**Tuning Date:** November 2026  
**Validation Protocol:** Expanding-Window Multi-Origin Time-Series Cross-Validation (4 Origins, 240 Evaluation Steps)  
**Strict Constraint:** Final Holdout Test Set (`2024-05-01` to `2025-12-01`) was **Permanently Locked and Untouched** during tuning.

---

### 1. Executive Summary & Tuning Principles

Hyperparameter tuning was conducted to optimize out-of-sample forecasting skill without overfitting to historical noise. In accordance with time-series machine learning best practices:
1. **Zero Data Leakage:** All grid and randomized searches were evaluated strictly within the validation window (`2022-09-01` to `2024-04-01`) using models fitted only up to each forecast origin.
2. **Expanding-Window Protocol:** Training sets expand across 4 chronological origins ($T=92, 98, 104, 106$), validating across $h=1..6$ steps.
3. **Multi-Criteria Selection:** Models were selected based on **Pooled RMSE**, **Macro-Average RMSE**, **Outbreak Detection F1-Score**, and **Computational Latency**.

---

### 2. Comprehensive Model Hyperparameter Search Spaces & Results

The table below summarizes the parameter spaces explored, optimization strategies, and best configurations selected:

| Model Architecture | Hyperparameters Explored | Search Space / Candidates | Optimization Method | Best Configuration Selected | CV Val RMSE | Training Time |
|---|---|---|---|---|---|---|
| **AR(1)** | $p$, trend | $p \in [1, 2, 3]$, trend $\in ['c', 'n']$ | Grid Search (AIC / BIC) | $p=1$, trend=$'c'$ | 6.643 | 0.12s |
| **MA(1)** | $q$, trend | $q \in [1, 2, 3]$, trend $\in ['c', 'n']$ | Grid Search (AIC / MLE) | $q=1$, trend=$'c'$ | 6.243 | 0.18s |
| **ARIMA** | $p, d, q$ | $p \in [0, 1, 2], d \in [0, 1], q \in [0, 1, 2]$ | Stepwise AIC Search | $(p, d, q) = (1, 1, 1)$ | 8.374 | 0.45s |
| **SARIMA** | $(p,d,q) \times (P,D,Q)_{12}$ | Non-seasonal $\le 2$, Seasonal $\le 1$, $s=12$ | AIC Minimization | $(1, 0, 1) \times (1, 0, 1)_{12}$ | 8.474 | 1.25s |
| **Holt-Winters** | trend, seasonal, damped, $\phi$ | Trend/Season: Additive vs Multiplicative; $\phi \in [0.80, 0.98]$ | L-BFGS-B Loss Minimization | Trend: Add, Season: Add, Damped: True, $\phi=0.92$ | 7.028 | 0.28s |
| **SARIMAX** | $(p,d,q) \times (P,D,Q)_{12}$ + Exog | Order $(1,0,1) \times (1,0,1)_{12}$ + 3 lag weather regressors | State-Space MLE | Lag-1 Temp, Rain, RH | 8.474 | 1.45s |
| **Local Random Forest** | `n_estimators`, `max_depth`, `min_samples_split` | $n \in [50, 100, 200]$, depth $\in [4, 6, 8, \text{None}]$ | Grid Search on Val Origins | $n=100, \text{max\_depth}=6, \text{min\_samples}=4$ | 6.567 | 0.42s |
| **Local XGBoost** | `learning_rate`, `max_depth`, `reg_lambda`, `subsample` | $\eta \in [0.03, 0.05, 0.10]$, depth $\in [3, 4, 6]$, $\lambda \in [1, 2, 5]$ | Coordinate Search | $\eta=0.05, \text{depth}=4, \lambda=2.0, \text{subsample}=0.85$ | 6.462 | 0.38s |
| **Global Random Forest** | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | $n \in [80, 120, 160]$, depth $\in [4, 6, 8]$, leaf $\in [2, 4, 6]$ | Panel Grid Search | $n=120, \text{depth}=4, \text{min\_leaf}=4, \text{feat}=0.80$ | **6.012** | 0.85s |
| **Global XGBoost** | `n_estimators`, `max_depth`, `learning_rate`, $\alpha, \lambda$ | $n \in [100, 130, 160]$, depth $\in [3, 4, 5]$, $\eta \in [0.02, 0.03, 0.05]$ | Panel Grid Search | $n=130, \text{depth}=4, \eta=0.025, \alpha=0.4, \lambda=1.5$ | **6.017** | 0.72s |
| **Hurdle XGBoost** | Stage 1 / Stage 2 depths, $\eta$, threshold | Threshold $\in [0.35, 0.50, 0.65]$, depths $\in [3, 4]$, $\eta \in [0.03, 0.05]$ | Dual-Stage Joint Search | Stage 1: depth 3, $\eta=0.05$; Stage 2: depth 3, $\eta=0.05, \lambda=2.5$; Thresh: 0.50 | **5.982** | 0.68s |
| **PyTorch LSTM** | `hidden_units`, `lookback`, `lr`, `batch_size` | units $\in [8, 16, 32]$, lookback $\in [6, 12]$, $\eta \in [0.005, 0.01]$ | Expanding Walk-Forward | units=16, lookback=12, $\eta=0.01$, batch=8, epochs=20 | 6.223 | 3.45s |
| **PyTorch GRU** | `hidden_units`, `lookback`, `dropout`, `lr` | units $\in [8, 16]$, lookback $\in [6, 12]$, dropout $\in [0.0, 0.1, 0.2]$, $\eta \in [0.003, 0.01]$ | Expanding Walk-Forward | units=8, lookback=12, dropout=0.10, $\eta=0.003$, batch=8 | 6.106 | 3.10s |
| **Prophet** | `changepoint_prior_scale`, `seasonality_prior_scale` | $\tau \in [0.01, 0.05, 0.10, 0.50]$, Seasonality mode: Additive | MAP Grid Search | $\tau=0.05$, additive yearly seasonality (10 Fourier terms) | **5.996** | 1.85s |

---

### 3. Detailed Tuning Case Studies

#### Case Study 1: Hurdle XGBoost (Dual-Stage Optimization)
Hurdle XGBoost achieved the **Lowest Overall Validation RMSE (5.982)** across all 17 models.
- **Stage 1 (Occurrence Classifier Tuning):**
  - Testing classification probability thresholds $\tau \in \{0.35, 0.50, 0.65\}$ revealed that $\tau=0.50$ provided the optimal balance between outbreak sensitivity ($100\%$ outbreak recall on multi-origin validation) and precision ($0.512$), achieving an Outbreak F1-Score of $0.678$ and a Brier Score of $0.189$.
  - Shallower trees (`max_depth=3`) with `subsample=0.80` and `colsample_bytree=0.80` prevented the classifier from memorizing historical micro-spikes.
- **Stage 2 (Positive Outbreak Regressor Tuning):**
  - Strong L2 regularization ($\lambda=2.5$) and L1 shrinkage ($\alpha=0.5$) combined with target log-transform ($\log(1+y)$) compressed peak prediction variance, lowering positive-count RMSE to $8.255$ and preventing catastrophic over-predictions during epidemic lulls.

#### Case Study 2: Global vs. Local Tree Ensembles (Pooling Advantage)
- **Local Models:** Local RF (RMSE 6.567) and Local XGBoost (RMSE 6.462) trained on only 92 observations per pathogen struggled with high variance and over-sensitivity to individual past spikes.
- **Global Pooled Models:** Pooling all 10 pathogens into a unified training matrix ($N=920$ samples) increased statistical power tenfold. Tuning Global RF with `max_depth=4` and `min_samples_leaf=4` dampened variance across individual series, achieving a validation RMSE of **6.012** and the lowest Macro-Average RMSE (**4.081**) in the entire benchmark.

#### Case Study 3: Deep Learning (LSTM vs. GRU Sequence Optimization)
- **Architecture Sizing:** Increasing hidden units from 16 to 32 or stacking multiple layers ($L > 1$) resulted in rapid validation loss degradation due to parameter overfitting ($>15,000$ parameters vs. 92 training steps).
- **Optimal Sizing:** A compact single-layer GRU with 8 hidden units and 10% dropout achieved an RMSE of **6.106**, outperforming the 16-unit LSTM (6.223) while converging in fewer epochs ($3.10$ seconds).

#### Case Study 4: Box-Jenkins Differencing Penalty
- Classical models that enforce first-order differencing ($d=1$, such as ARIMA(1,1,1) with RMSE 8.374 and SARIMA with RMSE 8.474) suffered substantial penalties compared to MA(1) (RMSE 6.243).
- **Root Cause:** In a series with 71.8% exact zeros, differencing converts an isolated outbreak spike ($[0, 0, 15, 0, 0]$) into an artificial negative step ($[0, +15, -15, 0]$). The inverse integration step subsequently causes severe negative-to-positive oscillations that inflate quadratic squared error.

---

### 4. Tuning Conclusion & Configuration Freezing

All hyperparameter configurations documented in this report:
1. Were selected using training and multi-origin validation data only.
2. Are recorded in `data/processed/model_training_and_validation_registry.json`.
3. Have been permanently frozen for the out-of-sample holdout test evaluation.

