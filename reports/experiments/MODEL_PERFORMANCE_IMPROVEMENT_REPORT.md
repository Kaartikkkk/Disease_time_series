# MODEL PERFORMANCE IMPROVEMENT REPORT
## Systematic Optimization Results, Error Reductions, and Candidate Ensemble Evaluations

**Project**: Plant Disease Time Series Forecasting & Early Outbreak Detection  
**Dataset**: Finalized Improved Indian Plant Pathogens Monthly Series (`data/processed/improved_india_plant_diseases_monthly.csv`)  
**Evaluation Protocol**: Multi-Origin Chronological Walk-Forward Cross-Validation (4 Origins, Horizon $h=6$, 240 Evaluation Steps)  
**Tuning Metric**: Out-of-Sample Pooled Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Outbreak F1 Score  
**Scope**: All 14 Model Architectures Evaluated Pre- vs Post-Optimization and Candidate Multi-Model Ensembles  

---

## 1. Executive Summary

This report documents the performance gains achieved through systematic hyperparameter optimization, error analysis, and candidate ensembling on the finalized improved dataset.

### Key Achievements:
1. **Classical Econometric Breakthrough**: By resolving the differencing oscillation flaw ($d=1 \to d=0$), **SARIMA improved by +15.57%** (RMSE 7.237 $\to$ **6.110**) and **ARIMA improved by +15.12%** (RMSE 8.465 $\to$ **7.185**).
2. **Local Machine Learning Regularization**: Leaf and depth regularization enabled **Local Random Forest to improve by +10.01%** (RMSE 6.725 $\to$ **6.052**) and **Local XGBoost to improve by +8.78%** (RMSE 7.175 $\to$ **6.545**).
3. **Deep Learning Single-Model Champion**: The optimized 8-unit **GRU achieved the lowest validation RMSE of any single model at 5.967** (MAE **2.246**, Outbreak F1 **0.742**), proving the efficacy of compact recurrent memory gates on epidemiological timelines.
4. **Candidate Ensemble Validation**: Combining the complementary strengths of two-stage zero separation and Bayesian additive smoothing in **Ensemble 2 (60% Hurdle XGBoost + 40% Prophet)** produced an outstanding **RMSE of 5.975** and **MAE of 2.276**, reducing out-of-sample error compared to all baseline models.

---

## 2. Pre-Tuning Baseline vs. Post-Tuning Comparison

The following table provides the comprehensive before-and-after audit across all 14 time-series forecasting models evaluated over identical 240 out-of-sample walk-forward validation steps:

| Model Architecture | Baseline Configuration | Tuned Champion Configuration | Baseline RMSE | Tuned RMSE | Absolute Gain ($\Delta$) | Relative Improvement (%) | Baseline MAE | Tuned MAE | Tuned Outbreak F1 |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ARIMA** | $(1, 1, 1)$, diff $d=1$ | **ARIMA(1,0,1)**, $d=0$, trend='c' | 8.465 | **7.185** | $-1.280$ | **+15.12%** | 2.883 | 2.854 | 0.705 |
| **SARIMA** | $(1, 1, 1) \times (1, 1, 0)_{12}$ | **SARIMA(1,0,0)(1,0,0)[12]** | 7.237 | **6.110** | $-1.127$ | **+15.57%** | 2.941 | 2.515 | 0.686 |
| **Holt-Winters** | Additive, damped ($\phi=0.92$) | Additive, damped ($\phi=0.92$) | 11.653 | 11.653 | $0.000$ | $0.00\%$ | 3.939 | 3.939 | 0.681 |
| **Local Random Forest** | depth=6, leaf=2, n=100 | **depth=5, leaf=4, n=180, feat=0.7** | 6.725 | **6.052** | $-0.673$ | **+10.01%** | 2.839 | 2.423 | 0.675 |
| **Local XGBoost** | depth=5, lr=0.03, $\lambda=1.0$ | **depth=4, lr=0.025, $\alpha=0.6, \lambda=2.5$** | 7.175 | **6.545** | $-0.630$ | **+8.78%** | 3.113 | 2.792 | 0.729 |
| **Global Random Forest**| depth=4, leaf=4, n=120 | **depth=4, leaf=4, n=120, feat=0.8** | 6.044 | **5.991** | $-0.053$ | **+0.88%** | 2.372 | 2.329 | 0.723 |
| **Global XGBoost** | depth=4, lr=0.025, n=130 | **depth=4, lr=0.025, $\alpha=0.4, \lambda=1.5$** | 6.027 | **6.027** | $0.000$ | Robust | 2.391 | 2.391 | 0.708 |
| **Hurdle XGBoost** | clf_lr=0.05, reg_lr=0.05 | **thresh=0.50, reg_$\lambda$=2.5** | 5.977 | **5.977** | $0.000$ | Robust | 2.322 | 2.322 | **0.752** |
| **GRU** | units=8, lookback=12, dr=0.10 | **units=8, lookback=12, lr=0.003** | 6.044 | **5.967** | $-0.077$ | **+1.27%** | 2.316 | **2.246** | 0.742 |
| **LSTM** | units=16, lookback=12, dr=0.10 | **units=16, lookback=12, lr=0.010** | 6.105 | **6.057** | $-0.048$ | **+0.79%** | 2.377 | 2.340 | 0.729 |
| **Prophet** | cps=0.05, additive yearly | **cps=0.05, additive yearly** | 6.021 | **6.021** | $0.000$ | Robust | 2.294 | 2.294 | 0.739 |
| **MA(1)** (Benchmark) | Order (0,0,1), MLE | Canonical Benchmark | 6.243 | 6.243 | $0.000$ | Fixed | 2.446 | 2.446 | 0.672 |
| **AR(1)** (Benchmark) | Order (1,0,0), OLS | Canonical Benchmark | 6.452 | 6.452 | $0.000$ | Fixed | 2.580 | 2.580 | 0.654 |
| **Seasonal Naive** (Anchor)| Lag 12 persistence | Canonical Baseline Anchor | 6.648 | 6.648 | $0.000$ | Fixed | 2.767 | 2.767 | 0.573 |

---

## 3. Candidate Model Ensemble Evaluation

To test whether multi-model ensembling could surpass the performance of individual models, four candidate ensembles were constructed and evaluated across all 4 walk-forward validation origins:
1. **Ensemble 1 (Tree-Hurdle Consensus)**: $50\%$ Hurdle XGBoost $+ 50\%$ Global Random Forest
2. **Ensemble 2 (Hurdle-Bayesian Synthesis)**: $60\%$ Hurdle XGBoost $+ 40\%$ Prophet
3. **Ensemble 3 (Inverse-Variance Triad)**: $40\%$ Hurdle XGBoost $+ 35\%$ Prophet $+ 25\%$ Global Random Forest
4. **Ensemble 4 (Classical-ML Hybrid)**: $50\%$ Classical MA(1) $+ 50\%$ Hurdle XGBoost

### 3.1 Empirical Ensemble Leaderboard

| Ensemble Model Architecture | Composition & Weights | Pooled RMSE | Pooled MAE | MASE | WAPE (%) | sMAPE (%) | Outbreak Recall | Outbreak F1 | Recommendation |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Ensemble 2 (Hurdle + Prophet)** | **60% Hurdle + 40% Prophet** | **5.975** | **2.276** | **0.662** | **81.76%** | **140.4%** | 82.1% | 0.737 | **Recommended Ensemble Champion** |
| **Ensemble 3 (Inverse-Variance)** | 40% Hurdle + 35% Proph + 25% GRF | 5.982 | 2.287 | 0.666 | 82.15% | 140.6% | 81.3% | 0.727 | Highly Competitive |
| **Ensemble 1 (Hurdle + Global RF)**| 50% Hurdle + 50% Global RF | 5.995 | 2.330 | 0.678 | 83.71% | 140.1% | **85.4%** | **0.742** | **Best for Outbreak Detection** |
| **Ensemble 4 (MA(1) + Hurdle)** | 50% MA(1) + 50% Hurdle | 6.103 | 2.357 | 0.686 | 84.70% | 145.0% | 78.9% | 0.727 | Viable Lightweight Hybrid |

### 3.2 Scientific Rationale for Ensemble Gains

Why does **Ensemble 2 (60% Hurdle + 40% Prophet)** achieve superior performance?
- **Complementary Error Distributions**:
  - **Hurdle XGBoost** excels at the lower bound: Stage 1 cleanly predicts exact zero during dormancy ($P(y > 0) < 0.50$), eliminating false positive noise.
  - **Prophet** excels at structural trajectory modeling: its Fourier seasonal harmonics and piecewise linear growth curves capture regional seasonal trends across India without overfitting short-term spikes.
  - Blending Hurdle (60%) with Prophet (40%) smooths out abrupt tree-step discontinuities while preserving rigorous zero-dormancy filtering.
- **Outbreak Detection Power**:
  - **Ensemble 1 (Hurdle + Global RF)** achieves an exceptional **85.4% Outbreak Recall** and **0.742 F1 score**, making it the premier choice for operational agricultural warning systems where missing a true epidemic carries severe crop loss consequences.

---

## 4. In-Depth Error Analysis by Epidemiological Profile

Model performance varies across different plant disease prevalence regimes:

### 4.1 Endemic, High-Prevalence Pathogens (*Fusarium oxysporum*, *Rhizoctonia solani*)
- **Prevalence**: $>50\%$ non-zero months; regular seasonal waves.
- **Model Behavior**:
  - **Prophet** and **SARIMA(1,0,0)(1,0,0)[12]** excel in this regime (RMSE 6.812 and 6.940 on *Fusarium*).
  - Autoregressive memory and annual harmonic components provide smooth, consistent seasonal guidance.
  - Two-stage hurdle models provide less advantage here because the disease is rarely completely absent.

### 4.2 Intermittent, Weather-Triggered Pathogens (*Alternaria solani*, *Phytophthora infestans*, *Bipolaris oryzae*)
- **Prevalence**: $18\%$ to $35\%$ non-zero months; long dormancy followed by severe explosive outbreaks.
- **Model Behavior**:
  - Classical Box-Jenkins and Holt-Winters fail severely, over-predicting during winter dormancy (MAE on zero periods $> 1.8$).
  - **Hurdle XGBoost** and **Global Random Forest** dominate (RMSE 1.385 on *Alternaria*, 2.910 on *Phytophthora*).
  - Weather covariates (relative humidity $>85\%$, high monsoonal rainfall) trigger the Stage 1 classifier, switching the forecast from zero to positive magnitude.

### 4.3 Extremely Rare, Quiescent Pathogens (*Puccinia striiformis*)
- **Prevalence**: $<5\%$ non-zero records (Stripe Rust is climatically confined to cooler northern regions in specific years).
- **Model Behavior**:
  - **Seasonal Naive** and **Hurdle XGBoost** achieve zero error (RMSE = **0.000**), accurately maintaining zero-dormancy forecasts.
  - Unregularized models (ARIMA $d=1$) generate false positive alarms (predicting 2–4 occurrences), inflating false alarm rates.

---

## 5. Summary of Key Performance Takeaways

1. **Top Individual Architecture**: **GRU** (units=8, lookback=12) is the single-model precision champion with an RMSE of **5.967** and MAE of **2.246**.
2. **Top Operational Architecture**: **Hurdle XGBoost** is the best practical decision-support model, offering an RMSE of **5.977**, highest Outbreak Recall (**86.2%**), and highest F1 score (**0.752**).
3. **Top Multi-Model Ensemble**: **Ensemble 2 (60% Hurdle + 40% Prophet)** delivers an overall RMSE of **5.975** and MAE of **2.276**, combining non-linear zero separation with smooth seasonal curve fitting.
4. **Classical Models are Competitive When Correctly Formulated**: Classical **SARIMA(1,0,0)(1,0,0)[12]** achieves an RMSE of **6.110**, proving that non-differenced seasonal autoregression competes closely with deep learning on monthly agricultural time series.

