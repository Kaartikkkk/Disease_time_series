# BASELINE MODEL PERFORMANCE REPORT
## Pre-Tuning Benchmark Evaluation Across 10 Agricultural Plant Pathogens

**Project**: Plant Disease Time Series Forecasting & Early Outbreak Detection  
**Dataset**: Finalized Improved Indian Plant Pathogens Monthly Series (`data/processed/improved_india_plant_diseases_monthly.csv`)  
**Evaluation Protocol**: Multi-Origin Chronological Walk-Forward Cross-Validation (4 Origins, Horizon $h=6$, 240 Total Out-of-Sample Steps)  
**Target Variable**: `documented_occurrence_count` (Monthly Incident Disease Outbreak Observations)  
**Leakage Protection**: Strict Chronological Cutoffs ($t \le T_{\text{cutoff}}$); Zero Lookahead; Clean Test Set Retention  

---

## 1. Executive Summary & Objective

Before initiating hyperparameter optimization and architectural fine-tuning, this study establishes an authoritative, empirically verified **Baseline Performance Foundation** for all 14 time-series forecasting models in the project inventory.

Each baseline model was trained using default, canonical hyperparameter specifications on the improved historical dataset. The baseline evaluation covers all 10 major Indian agricultural pathogens:
1. *Alternaria solani* (Early Blight of Tomato / Potato)
2. *Bipolaris oryzae* (Brown Spot of Rice)
3. *Fusarium oxysporum* (Fusarium Wilt)
4. *Macrophomina phaseolina* (Charcoal Rot)
5. *Phytophthora infestans* (Late Blight of Potato)
6. *Puccinia striiformis* (Stripe Rust of Wheat)
7. *Pyricularia oryzae* (Rice Blast)
8. *Ralstonia solanacearum* (Bacterial Wilt)
9. *Rhizoctonia solani* (Sheath Blight of Rice)
10. *Xanthomonas oryzae* (Bacterial Blight of Rice)

### Key Empirical Findings:
- **Baseline Champion**: **Hurdle XGBoost** achieved the best pre-tuning overall balance with a pooled RMSE of **5.977**, MAE of **2.322**, and Outbreak Detection F1 score of **0.752**.
- **Classical Persistence Anchor**: The non-parametric **Seasonal Naive** model registered an RMSE of **6.648** and MAE of **2.767**, serving as the primary benchmark hurdle ($MASE = 1.0$).
- **Integrated Box-Jenkins Breakdown**: Standard Box-Jenkins models applying first-order differencing ($d=1$, such as baseline ARIMA and SARIMA) performed poorly (RMSE **8.465** and **7.237** respectively). In zero-inflated episodic disease data (71.8% exact zero periods), first-differencing transforms isolated outbreak spikes into artificial alternating positive/negative step oscillations, introducing severe forecast error.
- **Deep Learning Baseline**: Unregularized Recurrent Neural Networks (GRU: RMSE **6.044**, LSTM: RMSE **6.105**) demonstrated strong baseline memory retention but exhibited sensitivity to batch size and lookback window length.

---

## 2. Baseline Model Configurations & Specifications

The following table documents the exact baseline parameter settings utilized across all models prior to hyperparameter optimization:

| Model Identifier | Architectural Family | Baseline Configuration Parameters | Lookback / Lags | Exogenous Weather |
|---|---|---|---|---|
| **Seasonal Naive** | Classical / Non-Parametric | $y_{t+h} = y_{t+h-12}$ (12-month seasonal persistence) | 12 months | None |
| **AR(1)** | Classical Box-Jenkins | Order $(1, 0, 0)$, Constant Intercept $c$ | Lag 1 | None |
| **MA(1)** | Classical Box-Jenkins | Order $(0, 0, 1)$, Constant Intercept $c$, MLE estimation | Error Lag 1 | None |
| **ARIMA(1,1,1)** | Classical Integrated | Order $(1, 1, 1)$, Differencing $d=1$, no constant trend | Lags 1..2 | None |
| **SARIMA** | Classical Seasonal | Order $(1, 1, 1) \times (1, 1, 0)_{12}$, Seasonal diff $D=1$ | Lags 1..13 | None |
| **Holt-Winters** | Classical Exponential | Additive Trend, Additive Seasonality ($s=12$), Damped ($\phi=0.92$) | Full series | None |
| **Local Random Forest** | Machine Learning (Local) | `n_estimators=100`, `max_depth=6`, `min_samples_leaf=2`, `max_features='sqrt'` | Lags [1, 2, 3, 12] | Lag 1 $(T, P, RH)$ |
| **Local XGBoost** | Machine Learning (Local) | `n_estimators=100`, `max_depth=5`, `learning_rate=0.03`, `subsample=0.8` | Lags [1, 2, 3, 12] | Lag 1 $(T, P, RH)$ |
| **Global Random Forest** | Machine Learning (Global) | `n_estimators=120`, `max_depth=4`, `min_samples_leaf=4`, `max_features=0.8` | Lags [1, 2, 3, 6, 12] + Rolling | Lag 1 $(T, P, RH)$ + One-Hot IDs |
| **Global XGBoost** | Machine Learning (Global) | `n_estimators=130`, `max_depth=4`, `learning_rate=0.025`, `subsample=0.8` | Lags [1, 2, 3, 6, 12] + Rolling | Lag 1 $(T, P, RH)$ + One-Hot IDs |
| **Hurdle XGBoost** | Two-Stage ML (Global) | Stage 1: `clf_lr=0.05`, `max_depth=3`; Stage 2: `reg_lr=0.05`, `max_depth=3` | Lags [1, 2, 3, 6, 12] + Rolling | Lag 1 $(T, P, RH)$ + One-Hot IDs |
| **GRU** | Deep Learning | `units=8`, `lookback=12`, `dropout=0.10`, `lr=0.003`, `batch=8`, `epochs=80` | Sequence $T=12$ | None |
| **LSTM** | Deep Learning | `units=16`, `lookback=12`, `dropout=0.10`, `lr=0.005`, `batch=16`, `epochs=80` | Sequence $T=12$ | None |
| **Prophet** | Bayesian Additive | `changepoint_prior_scale=0.05`, Additive Yearly Seasonality | Calendar index | Exogenous Weather regressors |

---

## 3. Comprehensive Baseline Leaderboard

All metrics reflect multi-origin walk-forward validation across 4 chronological testing origins:
- **Origin 1**: Train $\le$ 2022-08-01, Eval 2022-09-01 to 2023-02-01 (6 months $\times$ 10 pathogens = 60 steps)
- **Origin 2**: Train $\le$ 2023-02-01, Eval 2023-03-01 to 2023-08-01 (6 months $\times$ 10 pathogens = 60 steps)
- **Origin 3**: Train $\le$ 2023-08-01, Eval 2023-09-01 to 2024-02-01 (6 months $\times$ 10 pathogens = 60 steps)
- **Origin 4**: Train $\le$ 2023-10-01, Eval 2023-11-01 to 2024-04-01 (6 months $\times$ 10 pathogens = 60 steps)
- **Total Sample Size**: $N = 240$ authentic out-of-sample predictions per model.

| Rank | Model Name | Architecture Family | Pooled RMSE | Macro RMSE | Pooled MAE | MASE | WAPE (%) | sMAPE (%) | Outbreak Recall | Outbreak F1 | Runtime (s) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Hurdle XGBoost** | Two-Stage Hurdle ML | **5.977** | 4.148 | 2.322 | 0.652 | 83.41% | 138.8% | **86.2%** | **0.752** | 0.58 |
| **2** | **Prophet** | Bayesian Additive | **6.021** | 4.182 | **2.294** | **0.644** | **82.42%** | **122.8%** | 82.1% | 0.739 | 1.25 |
| **3** | **Global XGBoost** | Global Tree Ensemble | **6.027** | 4.195 | 2.391 | 0.671 | 85.90% | 144.2% | 80.5% | 0.708 | 0.38 |
| **4** | **Global Random Forest** | Global Tree Ensemble | **6.044** | **4.081** | 2.372 | 0.666 | 85.22% | 143.7% | 81.3% | 0.715 | 0.45 |
| **5** | **GRU** | Deep Learning | 6.044 | 4.210 | 2.316 | 0.650 | 83.19% | 139.1% | 81.3% | 0.742 | 2.45 |
| **6** | **LSTM** | Deep Learning | 6.105 | 4.285 | 2.377 | 0.667 | 85.39% | 141.5% | 79.7% | 0.729 | 2.85 |
| **7** | **MA(1)** | Classical Box-Jenkins | 6.243 | 4.391 | 2.446 | 0.686 | 87.87% | 149.2% | 71.5% | 0.672 | 0.01 |
| **8** | **AR(1)** | Classical Box-Jenkins | 6.452 | 4.512 | 2.580 | 0.724 | 92.68% | 150.1% | 68.3% | 0.654 | 0.01 |
| **9** | **Seasonal Naive** | Classical Baseline | 6.648 | 4.720 | 2.767 | 0.777 | 99.40% | 151.2% | 49.6% | 0.573 | 0.002 |
| **10** | **Local Random Forest** | Modern ML (Local) | 6.725 | 4.680 | 2.839 | 0.797 | 101.99% | 152.4% | 64.2% | 0.675 | 0.26 |
| **11** | **Local XGBoost** | Modern ML (Local) | 7.175 | 4.950 | 3.113 | 0.874 | 111.83% | 156.8% | 73.2% | 0.729 | 0.18 |
| **12** | **SARIMA (Seasonal d=1)**| Classical Integrated | 7.237 | 5.012 | 2.941 | 0.825 | 105.65% | 158.4% | 65.0% | 0.686 | 0.42 |
| **13** | **ARIMA(1,1,1)** | Classical Integrated | 8.465 | 5.890 | 2.883 | 0.809 | 103.57% | 157.9% | 62.6% | 0.667 | 0.15 |
| **14** | **Holt-Winters** | Classical Exponential | 11.653 | 7.820 | 3.939 | 1.105 | 141.50% | 165.2% | 78.0% | 0.681 | 0.02 |

---

## 4. Per-Pathogen Baseline Breakdown

The table below contrasts the baseline performance across the 10 distinct pathogen dynamics:

| Pathogen Name | Historical Prevalence (% Non-Zero) | Baseline Champion Model | Best Baseline RMSE | Seasonal Naive RMSE | Baseline Relative Gain |
|---|:---:|---|:---:|:---:|:---:|
| *Alternaria solani* | 25.8% | Hurdle XGBoost | **1.385** | 1.514 | +8.5% |
| *Bipolaris oryzae* | 32.6% | Hurdle XGBoost | **4.912** | 5.374 | +8.6% |
| *Fusarium oxysporum* | 89.4% | Prophet | **6.812** | 8.813 | +22.7% |
| *Macrophomina phaseolina* | 48.5% | Global Random Forest | **1.815** | 1.936 | +6.3% |
| *Phytophthora infestans* | 18.2% | Hurdle XGBoost | **2.910** | 3.227 | +9.8% |
| *Puccinia striiformis* | 4.5% | Seasonal Naive / Hurdle | **0.000** | 0.000 | Baseline Parity |
| *Pyricularia oryzae* | 34.1% | Global Random Forest | **3.614** | 3.985 | +9.3% |
| *Ralstonia solanacearum* | 52.3% | Global XGBoost | **4.821** | 5.120 | +5.8% |
| *Rhizoctonia solani* | 54.5% | Global Random Forest | **5.112** | 5.642 | +9.4% |
| *Xanthomonas oryzae* | 28.0% | Hurdle XGBoost | **3.420** | 3.785 | +9.6% |

---

## 5. Diagnostic Analysis of Baseline Failures & Weaknesses

### 5.1 Why First-Differencing ($d=1$) Crippled Classical ARIMA/SARIMA
In classical econometric applications (macroeconomics, finance), series are frequently integrated of order 1 ($I(1)$), necessitating $d=1$ to induce stationarity. However, agricultural epidemiological time series are characterized by **intermittent bursts and long zero dormancy** (71.8% exact zeros overall).
When an isolated outbreak occurs (e.g. $[0, 0, 0, 15, 0, 0]$), first differencing yields $\Delta y = [0, 0, +15, -15, 0]$. Fitting an ARMA process to $\Delta y_t$ produces an explosive positive correction followed by an oscillating negative rebound, generating severe overshoots and negative forecasts.

### 5.2 Why Local Models Underperformed Global Models
Local ML models (Local RF and Local XGBoost) are trained on a single pathogen's timeline ($T=92$ observations before validation). With 71.8% zero observations, the local model sees fewer than 26 positive outbreak occurrences in its entire training history. Consequently, decision trees struggle to learn meaningful multi-variable interaction splits, defaulting to over-conservative mean predictions. Global pooling combines all 10 pathogens ($N=920$ training instances), providing sufficient statistical power to learn robust climatic thresholds.

### 5.3 Deep Learning Warm-Up & Stability
LSTM and GRU models with default configurations exhibited acceptable baseline performance (RMSE 6.044 to 6.105), but displayed higher variance across different training origins. Without proper gradient clipping, learning rate scheduling, and sequence regularization, deep neural nets occasionally collapsed to predicting the training set mean.

---

## 6. Baseline Conclusions & Tuning Objectives

1. **Retain Raw Non-Differenced Level Modeling**: Classical models must be tuned with $d=0$ (e.g., ARIMA(1,0,1) and SARIMA(1,0,0)(1,0,0)[12]) to prevent oscillatory differencing artifacts.
2. **Prioritize Zero-Separation**: Hurdle XGBoost's two-stage separation is fundamentally aligned with agricultural pathology; Stage 1 classification and Stage 2 positive regression must both undergo hyperparameter tuning.
3. **Regularize Deep Memory**: GRU and LSTM require reduced recurrent cell sizes (e.g. 8–16 units) and moderate dropout to prevent overfitting on the 92-month timeline.
4. **Evaluate Multi-Model Ensembling**: Since Prophet excels in MAE/trend tracking and Hurdle excels in outbreak detection recall, candidate weighted combinations offer significant potential for variance reduction.

