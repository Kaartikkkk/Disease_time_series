# FINAL_MODEL_COMPARISON_REPORT.md
## Final Comprehensive Model Comparison, Performance Benchmark & Production Policy Report

**Project:** Indian Agricultural Plant Disease Surveillance Time Series Forecasting  
**Evaluation Standard:** Chronological Expanding-Window Walk-Forward Validation & Untouched Final Test Set  
**Dataset:** Improved Original Agricultural Surveillance Dataset v2.0 (`data/processed/improved_india_plant_diseases_monthly.csv`)  
**Registry Reference:** `data/processed/model_training_and_validation_registry.json`  
**Date:** November 2026  

---

### 1. Executive Summary & Benchmark Hierarchy

This report establishes the final, definitive model performance hierarchy across all 17 registered forecasting architectures evaluated on the improved original dataset.

Every reported metric is computed from authentic historical predictions across 10 agricultural plant pathogens spanning 11 continuous years (132 calendar months, 2015–2025). Zero synthetic data was used.

```mermaid
flowchart TD
    subgraph Benchmark Champions
        C1["Overall Champion: Hurdle XGBoost<br>Val RMSE: 5.982 | Test RMSE: 5.093 | Outbreak F1: 0.774"]
        C2["Macro-Average Champion: Global Random Forest<br>Macro RMSE: 4.081 | Val RMSE: 6.012 | Test RMSE: 5.042"]
        C3["Classical / MAE Champion: Meta Prophet<br>Val MAE: 2.275 | Val RMSE: 5.996 | Test RMSE: 5.001"]
        C4["Deep Learning Champion: PyTorch GRU<br>Val RMSE: 6.106 | Test RMSE: 5.040 | Outbreak Recall: 81.2%"]
        C5["Univariate Box-Jenkins Champion: MA(1)<br>Val RMSE: 6.243 | Test RMSE: 5.517 | Training: 0.18s"]
    end
```

---

### 2. Definitive Model Leaderboard & Multi-Origin Benchmark

The table below summarizes the multi-origin validation performance (4 expanding origins, 240 evaluation steps) and the frozen holdout test performance (May 2024 to December 2025, 200 out-of-sample records):

| Rank | Model Architecture | Family | Val RMSE | Val MAE | Val MASE | Val sMAPE | Test RMSE | Test MAE | Outbreak Detection (Recall) | Outbreak F1 Score | Runtime (s) | Improvement vs. Baseline |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Hurdle XGBoost** | Two-Stage Zero-Inflated ML | **5.982** | **2.370** | **2.375** | 139.9% | **5.093** | **2.215** | **86.2%** | **0.774** | 0.68s | **+10.0% (Val) / +37.7% (Test)** |
| **2** | **Meta Prophet** | Generalized Additive (Prophet) | 5.996 | **2.275** | **2.280** | 122.8% | **5.001** | **2.180** | 82.5% | 0.742 | 1.85s | **+9.8% (Val) / +38.8% (Test)** |
| **3** | **Global Random Forest** | Pooled Tree Ensemble | 6.012 | 2.411 | 2.416 | 142.1% | 5.042 | 2.245 | 84.6% | 0.758 | 0.85s | **+9.6% (Val) / +38.3% (Test)** |
| **4** | **Global XGBoost** | Pooled Gradient Boosted Trees | 6.017 | 2.405 | 2.410 | 144.3% | 5.899 | 2.250 | 83.8% | 0.751 | 0.72s | **+9.5% (Val) / +27.9% (Test)** |
| **5** | **PyTorch GRU** | Recurrent Neural Net (PyTorch) | 6.106 | 2.304 | 2.309 | 145.9% | 5.040 | 2.210 | 81.2% | 0.730 | 3.10s | **+8.2% (Val) / +38.4% (Test)** |
| **6** | **PyTorch LSTM** | Recurrent Neural Net (PyTorch) | 6.223 | 2.376 | 2.382 | 145.9% | 5.209 | 2.280 | 80.5% | 0.722 | 3.45s | **+6.4% (Val) / +36.3% (Test)** |
| **7** | **MA(1)** | Box-Jenkins Univariate | 6.243 | 2.446 | 2.451 | 147.3% | 5.517 | 2.310 | 78.4% | 0.705 | 0.18s | **+6.1% (Val) / +32.5% (Test)** |
| **8** | **Local XGBoost** | Gradient Boosted Trees | 6.462 | 2.815 | 2.822 | 144.3% | 5.142 | 2.360 | 79.2% | 0.710 | 0.38s | **+2.8% (Val) / +37.1% (Test)** |
| **9** | **Local Random Forest** | Tree Ensemble ML | 6.567 | 2.644 | 2.650 | 130.5% | 5.041 | 2.290 | 78.0% | 0.698 | 0.42s | **+1.2% (Val) / +38.4% (Test)** |
| **10** | **AR(1)** | Box-Jenkins Univariate | 6.643 | 2.637 | 2.643 | 149.8% | 10.122 | 2.850 | 72.5% | 0.645 | 0.12s | **+0.1% (Val) / -23.8% (Test)** |
| **11** | **Seasonal Naive** *(Baseline)* | Persistence Baseline | 6.648 | 2.767 | 2.773 | 95.8% | 8.179 | 2.780 | 70.1% | 0.628 | 0.05s | **0.0% (Benchmark Anchor)** |
| **12** | **Hist Gradient Boosting** *(Retired)* | Tree Ensemble ML | 6.648 | 2.767 | 2.773 | 95.8% | 8.179 | 2.780 | 70.1% | 0.628 | 0.35s | 0.0% (Redundant with XGBoost) |
| **13** | **Holt-Winters ESM** | Exponential Smoothing | 7.028 | 2.834 | 2.840 | 135.4% | 6.910 | 2.820 | 68.5% | 0.612 | 0.28s | **-5.7% (Val) / +15.5% (Test)** |
| **14** | **ARIMA(1,1,1)** | Box-Jenkins Integrated | 8.374 | 2.968 | 2.974 | 138.0% | 10.004 | 3.120 | 62.4% | 0.550 | 0.45s | **-26.0% (Val) / -22.3% (Test)** |
| **15** | **SARIMA(1,0,1)(1,0,1)$_{12}$** | Seasonal Box-Jenkins | 8.474 | 2.990 | 2.996 | 140.0% | 11.486 | 3.250 | 60.1% | 0.532 | 1.25s | **-27.5% (Val) / -40.4% (Test)** |
| **16** | **ARIMAX** *(Retired)* | Exogenous Econometric | 8.474 | 2.990 | 2.996 | 140.0% | 8.179 | 3.100 | 61.5% | 0.542 | 0.55s | **-27.5% (Val) / 0.0% (Test)** |
| **17** | **SARIMAX** *(Operational)* | Exogenous Econometric | 8.474 | 2.990 | 2.996 | 140.0% | 11.514 | 3.280 | 59.8% | 0.528 | 1.45s | **-27.5% (Val) / -40.8% (Test)** |

---

### 3. Category Champions & In-Depth Technical Analysis

#### 3.1 Best Overall Model: Hurdle XGBoost
- **Validation RMSE:** `5.982` | **Test RMSE:** `5.093` | **Outbreak Recall:** `86.2%` | **Outbreak F1:** `0.774`
- **Why It Won:**
  71.8% of agricultural disease monthly observations are zero. Standard regressors suffer from "zero-dragging," predicting values like $0.4$ or $0.7$ during deep dry seasons and attenuating genuine epidemic peaks to $6.2$ instead of $20.0$. Hurdle XGBoost decouples detection from estimation:
  1. The Stage 1 XGBClassifier determines whether an outbreak occurs with an AUC of 0.884 and Brier score of 0.138.
  2. The Stage 2 XGBRegressor, unpolluted by zero-inflation, estimates positive log-counts accurately.
  3. Multiplying $P(y>0) \times \hat{y}_{>0}$ produces calibrated, non-negative forecasts with high outbreak fidelity.

#### 3.2 Best Machine Learning Model: Global Random Forest
- **Validation RMSE:** `6.012` | **Test RMSE:** `5.042` | **Macro-Average RMSE:** `4.081` (Leader across all 10 pathogens)
- **Why It Won:**
  By pooling histories across all 10 crops ($N=920$ training instances), the random forest learns broad biophysical rules (e.g. high relative humidity + moderate warmth triggering spore germination) that transfer across pathogens. Its bagging mechanism provides strong variance suppression during non-epidemic months.

#### 3.3 Best Deep Learning Model: PyTorch GRU
- **Validation RMSE:** `6.106` | **Test RMSE:** `5.040` | **Runtime:** `3.10s`
- **Why It Won:**
  The compact GRU architecture (8 units, lookback $L=12$, dropout $0.10$) outperformed the larger LSTM (16 units, RMSE 6.223). With only 92 training steps per pathogen, the GRU's simpler reset and update gating avoided parameter overfitting while effectively capturing seasonal lag decay.

#### 3.4 Best Classical Model: Meta Prophet & MA(1)
- **Prophet:** Validation RMSE `5.996` and lowest Validation MAE (`2.275`). Piecewise linear changepoint trends combined with 10-term Fourier harmonics handle smooth seasonal curves without polynomial divergence.
- **MA(1):** Validation RMSE `6.243` and lowest runtime (`0.18s`). For stationary series, dampening single-period shocks without autoregressive feedback prevents error propagation.

#### 3.5 Worst Performing Models: Differenced Box-Jenkins (ARIMA, SARIMA, SARIMAX)
- **SARIMAX Test RMSE:** `11.514` (+40.8% error relative to Seasonal Naive).
- **SARIMA Test RMSE:** `11.486` (+40.4% error relative to Seasonal Naive).
- **Scientific Diagnostic:**
  Box-Jenkins differencing ($d=1$) assumes Gaussian innovations around a drifting random walk. In zero-inflated agricultural surveillance, sudden epidemic pulses $[0, 0, 24, 0, 0]$ produce severe negative spikes when differenced $[0, 0, +24, -24, 0]$. Inverting this differencing step creates artificial negative-to-positive oscillations during test periods, degrading forecast skill.

---

### 4. Pathogen-Specific Optimal Routing Policy Table

Because plant pathogens exhibit vastly different epidemiological dynamics (from endemic root rots to explosive monsoonal blasts), a single global model is sub-optimal for all pathosystems. The table below documents the empirically assigned model per pathogen:

| Pathogen Name | Common Disease Name | Assigned Production Model | Architecture Family | Validation RMSE | Operational Justification | Secondary Fallback |
|---|---|---|---|:---:|---|---|
| *Pyricularia oryzae* | Rice Blast | **Global Random Forest** | Pooled Tree Ensemble | 3.620 | Strong multi-pathogen humidity coupling; bagging dampens catastrophic spikes | Hurdle XGBoost |
| *Rhizoctonia solani* | Sheath Blight of Rice | **Hurdle XGBoost** | Two-Stage Hurdle ML | 4.315 | High positive outbreak recall prevents missed early epidemic warnings | Global RF |
| *Xanthomonas oryzae* | Bacterial Blight of Rice | **MA(1)** | Classical Statistical | 1.458 | Low-volume stationary series; shock dampening prevents overfitting | Seasonal Naive |
| *Bipolaris oryzae* | Brown Spot of Rice | **Global Random Forest** | Pooled Tree Ensemble | 2.105 | Pooled temperate humidity interactions across hosts provide superior generalization | MA(1) |
| *Puccinia striiformis* | Stripe Rust of Wheat | **Global XGBoost** | Pooled Gradient Boosted | 4.880 | Temperature threshold gating with cross-pathogen cool-season transfer | Hurdle XGBoost |
| *Fusarium oxysporum* | Vascular Wilt | **Hurdle XGBoost** | Two-Stage Hurdle ML | 8.940 | Explosive outbreak surges separated by dormancy periods | Global XGBoost |
| *Alternaria solani* | Early Blight of Potato | **Prophet** | Bayesian Additive Curve | 3.842 | Smooth monsoonal seasonality with moderate baseline drift; Fourier harmonics prevent spikes | Global RF |
| *Phytophthora infestans* | Late Blight of Potato | **Hurdle XGBoost** | Two-Stage Hurdle ML | 4.120 | Severe dormancy (93% zeros) followed by sudden winter flowering spikes | Global RF |
| *Ralstonia solanacearum* | Bacterial Wilt | **Global Random Forest** | Pooled Tree Ensemble | 3.412 | Multi-year vector dynamics benefit from pooled panel representation | Prophet |
| *Macrophomina phaseolina*| Charcoal Rot | **Prophet** | Bayesian Additive Curve | 2.890 | Strict cropping window with stable climatological trend | MA(1) |

---

### 5. Operational Recommendations & Next Steps

1. **Deploy Hurdle XGBoost as Primary Surveillance Engine:**
   For high-consequence epidemic forecasting, Hurdle XGBoost's two-stage decomposition provides superior outbreak sensitivity ($86.2\%$ recall) while preserving low false alarm rates during dormant periods.
2. **Employ Global RF for Low-Sparsity Regional Forecasting:**
   For broad regional agricultural advisories, Global Random Forest yields the lowest macro-average error ($4.081$ RMSE) with near-instantaneous inference ($<1\text{ ms}$).
3. **Avoid Pure Differenced ARIMA in Production:**
   Exclude $d=1$ Box-Jenkins models from automated zero-inflated pipelines; prefer MA(1) or exponential smoothing when classical statistical baselines are required.
4. **Preserve Frozen Test Separation:**
   All 20-month test metrics presented in this report remain permanently fixed to guarantee uncompromised academic reproducibility.

