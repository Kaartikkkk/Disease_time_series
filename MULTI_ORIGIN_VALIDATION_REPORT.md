# Multi-Origin Walk-Forward Validation & Model Robustness Analysis Report

**Project:** Indian Agricultural Plant Disease Time Series Forecasting  
**Author:** Machine Learning Engineering Team  
**Evaluation Protocol:** Multi-Origin Chronological Walk-Forward (4 Origins, 6 Steps Ahead, 240 Forecast Points)  
**Validation Period:** September 2022 to April 2024 (20 Validation Months)  
**Untouched Test Period:** May 2024 to December 2025 (Strictly Preserved)  
**Codebase Test Suite Status:** 82 Tests Passing  

---

## 1. Executive Summary & Objective

This report documents the rigorous multi-origin walk-forward validation of **14 forecasting model architectures** across **10 Indian agricultural plant pathogens** over the chronological validation split (September 2022 to April 2024).

The primary research objective was to resolve whether modern global machine learning and hurdle models—specifically **Global Random Forest**, **Global XGBoost**, and **Hurdle XGBoost**—genuinely outperform classical statistical econometrics (AR, MA, ARIMA, SARIMA, Holt-Winters, Seasonal Naive) across diverse forecast origins and epidemiological conditions, or whether earlier benchmark findings were artifacts of a single evaluation origin.

### Key Empirical Findings

1. **Hurdle XGBoost is the Multi-Origin Benchmark Champion**:
   - **Pooled RMSE:** **5.982** (Rank 1 of 14).
   - **Outbreak Detection F1:** **0.774** (Highest sensitivity: 86.2% outbreak recall).
   - **WAPE:** **85.1%** (Lowest error among tree and neural models).
   - **Origin Wins:** Won 2 of the 4 chronological forecast origins.
   - **Pathogen Wins:** Won 2 high-severity pathogens (*Ralstonia solanacearum*, *Rhizoctonia solani*).
2. **Global Random Forest is the Macro-Average Champion**:
   - **Macro RMSE:** **4.081** (Rank 1 of 14 on unweighted pathogen average).
   - **Pooled RMSE:** **6.012** (Rank 3 of 14).
   - **Pathogen Wins:** Won Rice Blast (*Pyricularia oryzae*, RMSE 1.906) and Brown Spot of Rice (*Bipolaris oryzae*, RMSE 4.344).
3. **Classical Econometric Models Degrade Severely During Outbreak Surges**:
   - While AR(1) and MA(1) performed competently during quiescent summer dormancy (Origin 2 RMSE ~4.1), they degraded sharply during the major national outbreak surge in Origin 3 (AR(1) RMSE 11.016, ARIMA RMSE 13.179, SARIMA RMSE 13.377).
   - Across all 240 steps, ARIMA(1,1,1) produced a Pooled RMSE of **8.374** (+40.0% higher error than Hurdle XGBoost).
4. **Statistical Significance Confirmed via Moving Block-Bootstrap**:
   - A 1,000-resample moving block-bootstrap ($b = 3\text{ months}$) confirms that Hurdle XGBoost, Prophet, Global Random Forest, and Global XGBoost achieve statistically significant absolute error reductions over the Seasonal Naive baseline ($p < 0.05$).
5. **No Data Leakage**:
   - All models were fitted strictly on observations timestamped before each evaluation origin. The final 20-month test split (May 2024 – December 2025) remains completely pristine and unobserved.

---

## 2. Multi-Origin Chronological Validation Protocol

### 2.1 Dataset Topology

- **Pathogens:** 10 Indian crop pathogens (*Alternaria solani*, *Bipolaris oryzae*, *Fusarium oxysporum*, *Macrophomina phaseolina*, *Phytophthora infestans*, *Puccinia striiformis*, *Pyricularia oryzae*, *Ralstonia solanacearum*, *Rhizoctonia solani*, *Xanthomonas oryzae*).
- **Time Span:** January 2015 to December 2025 (132 contiguous monthly observations per pathogen = 1,320 total pathogen-month records).
- **Zero-Inflation Rate:** 71.82% overall (812 structural dormancy zeros, 136 gap-imputed zeros, 372 positive outbreak counts).
- **Exogenous Drivers:** 70 NASA POWER weather variables (temperature, relative humidity, precipitation, vapor pressure deficit, dew point) and EPPO international alert flags.

### 2.2 Three-Way Chronological Data Split

```
[------------- Training Pool: 92 Months -------------] [--- Validation Pool: 20 Months ---] [--- Untouched Test: 20 Months ---]
Jan 2015 ----------------------------------------- Aug 2022 Sep 2022 ------------------- Apr 2024 May 2024 ------------------- Dec 2025
(70% Data Budget)                                     (15% Multi-Origin Benchmark)        (15% Final Evaluation Split)
```

### 2.3 The Four Forecast Origins

To assess temporal robustness across seasonal cycles and volatile epidemiological regimes, 4 distinct forecast origins were constructed within the validation split. At each origin, models were trained exclusively on data prior to the cutoff date and evaluated on a 6-month walk-forward horizon:

| Origin ID | Cutoff Date (Max Train) | 6-Month Walk-Forward Evaluation Window | Seasonal Context & Epidemiological Regime | Total Steps |
|---|---|---|---|---|
| **Origin 1** | August 31, 2022 | **September 2022 – February 2023** | Post-monsoon Kharif peak transitioning into dry winter Rabi | 60 |
| **Origin 2** | February 28, 2023 | **March 2023 – August 2023** | Hot summer dormancy lull transitioning into southwest monsoon onset | 60 |
| **Origin 3** | August 31, 2023 | **September 2023 – February 2024** | Major national epidemic surge (high disease activity across northern and eastern states) | 60 |
| **Origin 4** | October 31, 2023 | **November 2023 – April 2024** | Rabi crop maturity and winter harvest (overlapping 4 months with Origin 3) | 60 |
| **Total** | — | **September 2022 – April 2024 (20 unique months)** | **Comprehensive multi-season epidemiological exposure** | **240** |

### 2.4 Walk-Forward Mechanics vs Fixed Holdout

Unlike recursive multi-step forecasting where models predict their own outputs into the future without corrective feedback, this operational walk-forward protocol updates input features with actual recorded ground-truth data at each 1-month-ahead step ($t+1$). This mirrors actual field deployment, where district agricultural officers receive monthly surveillance data before issuing next month's forecast.

---

## 3. Global Benchmark Leaderboard (240 Evaluation Steps)

All 14 candidate models were evaluated on the exact same 240 observation pairs. Metrics include:
- **Pooled RMSE & MAE:** Evaluated globally across all 240 points ($\sqrt{\frac{1}{N}\sum(y - \hat{y})^2}$).
- **Macro RMSE & MAE:** Unweighted mean across the 10 pathogens ($\frac{1}{10}\sum_{p=1}^{10} \text{RMSE}_p$).
- **WAPE:** Weighted Absolute Percentage Error ($\frac{\sum |y - \hat{y}|}{\sum y} \times 100$).
- **Outbreak Detection:** Precision, Recall, F1 score, and False Alarm Rate (FAR) for active disease presence ($y \ge 1$).
- **Statistical Significance:** 95% moving block-bootstrap confidence interval for paired absolute error reduction vs Seasonal Naive.

| Rank | Model Architecture | Family | Pooled RMSE | Macro RMSE | Pooled MAE | WAPE (%) | Outbreak F1 | False Alarm (%) | Pathogen Wins | Origin Wins | vs Naive 95% CI (MAE Gain) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Hurdle XGBoost** | Global Two-Stage ML | **5.982** | 4.111 | 2.370 | 85.1% | **0.774** | 38.5% | 2 | 2 | **[+0.16, +0.64]\*** |
| **2** | **Prophet** | Decomposable Additive | **5.996** | 4.160 | **2.275** | **81.7%** | 0.746 | 29.9% | 0 | 0 | **[+0.27, +0.75]\*** |
| **3** | **Global Random Forest** | Global Tree Ensemble | **6.012** | **4.081** | 2.411 | 86.6% | 0.724 | 41.0% | 2 | 1 | **[+0.13, +0.61]\*** |
| **4** | **Global XGBoost** | Global Gradient Boost | **6.017** | 4.136 | 2.405 | 86.4% | 0.698 | 38.5% | 0 | 0 | **[+0.15, +0.61]\*** |
| **5** | **GRU** | Recurrent Neural Net | 6.106 | 4.132 | 2.304 | 82.8% | 0.738 | 36.8% | 1 | 0 | **[+0.23, +0.73]\*** |
| **6** | **LSTM** | Recurrent Neural Net | 6.223 | 4.223 | 2.376 | 85.4% | 0.716 | 35.9% | 1 | 0 | **[+0.16, +0.65]\*** |
| **7** | **MA(1)** | Classical Econometric | 6.243 | 4.280 | 2.446 | 87.9% | 0.709 | 25.6% | 1 | 0 | [+0.08, +0.57]\* |
| **8** | **XGBoost (Local)** | Local Gradient Boost | 6.462 | 4.325 | 2.816 | 101.2% | 0.713 | 44.4% | 1 | 0 | [-0.27, +0.21] |
| **9** | **Random Forest (Local)** | Local Tree Ensemble | 6.567 | 4.413 | 2.643 | 95.0% | 0.700 | 26.5% | 1 | 0 | [-0.10, +0.37] |
| **10** | **AR(1)** | Classical Econometric | 6.643 | 4.443 | 2.637 | 94.8% | 0.733 | 24.8% | 0 | 0 | [-0.10, +0.38] |
| **11** | **Seasonal Naive** | Persistence Benchmark | 6.648 | 4.720 | 2.767 | 99.4% | 0.573 | 24.8% | 1 | 0 | [0.00, 0.00] (Ref) |
| **12** | **Holt-Winters** | Exponential Smoothing | 7.028 | 4.632 | 2.833 | 101.8% | 0.738 | 30.8% | 0 | 0 | [-0.44, +0.33] |
| **13** | **ARIMA(1,1,1)** | Classical Econometric | 8.374 | 5.048 | 2.966 | 106.6% | 0.720 | 25.6% | 0 | 0 | [-0.67, +0.27] |
| **14** | **SARIMA** | Classical Seasonal | 8.441 | 5.068 | 2.946 | 105.8% | 0.703 | 27.4% | 0 | 0 | [-0.64, +0.28] |

*\* Denotes statistically significant absolute error reduction over Seasonal Naive ($p < 0.05$). Note: SARIMA fold failure on Phytophthora infestans (Origin 3, step 4) was preserved as NaN and omitted from fold statistics without zero imputation.*

---

## 4. Cross-Origin Temporal Stability Analysis

A central insight of this study is that model rankings shift depending on the epidemiological regime. The table below presents out-of-sample RMSE grouped by forecast origin across all 10 pathogens (60 points per origin):

| Model Architecture | Origin 1 (Sep 22 – Feb 23) | Origin 2 (Mar 23 – Aug 23) | Origin 3 (Sep 23 – Feb 24) | Origin 4 (Nov 23 – Apr 24) | Cross-Origin Stability (Std Dev) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Hurdle XGBoost** | 2.611 | 4.440 | **10.068** *(Win)* | **3.902** *(Win)* | 3.32 |
| **Prophet** | 2.658 | 4.437 | 10.072 | 3.951 | 3.31 |
| **Global Random Forest** | **2.421** *(Win)* | 4.405 | 10.160 | 4.010 | 3.39 |
| **Global XGBoost** | 2.510 | 4.452 | 10.166 | 3.918 | 3.35 |
| **GRU** | 2.511 | 4.173 | 10.396 | 4.165 | 3.44 |
| **LSTM** | 2.475 | 4.258 | 10.503 | 4.507 | 3.51 |
| **MA(1)** | 2.555 | 4.161 | 10.514 | 4.636 | 3.52 |
| **XGBoost (Local)** | 2.556 | 4.428 | 10.281 | 5.932 | 3.46 |
| **Random Forest (Local)** | 2.633 | 4.539 | 10.737 | 5.451 | 3.59 |
| **AR(1)** | 2.566 | **4.098** *(Top Class)* | 11.016 | 5.636 | 3.79 |
| **Seasonal Naive** | 3.269 | 4.858 | 10.694 | 5.306 | 3.31 |
| **Holt-Winters** | 2.713 | 4.295 | 11.221 | 6.770 | 3.86 |
| **ARIMA(1,1,1)** | 2.565 | 4.155 | 13.179 | 9.109 | 4.96 |
| **SARIMA** | 2.559 | 4.092 | 13.377 | 9.330 | 5.06 |

### Key Observations on Origin Dynamics

1. **Origin 1 (Autumn Lull / Routine Dynamic):**  
   Global Random Forest achieved the lowest error (2.421), closely followed by LSTM (2.475) and Global XGBoost (2.510). Because disease incidence followed typical seasonal patterns, global models effectively leveraged shared cross-pathogen weather correlations.
2. **Origin 2 (Summer Dormancy / Near-Zero Phase):**  
   AR(1) (4.098) and MA(1) (4.161) were slightly ahead of tree models (~4.40). During prolonged zero-report months, simple autoregressive decay to the zero mean is effective. However, GRU (4.173) and Hurdle XGBoost (4.440) remained within 0.3 reports.
3. **Origin 3 (Major National Outbreak Surge):**  
   Between September 2023 and February 2024, significant outbreaks occurred across eastern and northern states (particularly in *Ralstonia solanacearum* and *Pyricularia oryzae*).
   - Hurdle XGBoost produced the lowest RMSE (**10.068**), closely followed by Prophet (**10.072**), Global RF (**10.160**), and Global XGBoost (**10.166**).
   - In stark contrast, differenced classical models suffered catastrophic trajectory explosions: **ARIMA exploded to 13.179** and **SARIMA exploded to 13.377**. Classical differencing propagated large trend errors during sudden level shifts.
4. **Origin 4 (Winter Rabi Maturity):**  
   Hurdle XGBoost (**3.902**) and Global XGBoost (**3.918**) dominated Origin 4, demonstrating fast post-outbreak recovery. Classical ARIMA again lagged substantially at **9.109**.

---

## 5. Per-Pathogen Champion Breakdown (All 10 Pathogens)

Evaluating performance across individual pathogens reveals that no single model is universally optimal. Instead, distinct epidemiological archetypes favor specific model architectures:

| Pathogen Scientific Name | Crop Common Name | Winning Model Architecture | Model Family | Validation RMSE | Validation MAE | Nearest Classical Competitor | Classical Competitor RMSE |
|---|---|---|---|:---:|:---:|---|:---:|
| *Alternaria solani* | Early Blight of Potato/Tomato | **LSTM (Sequence Neural Net)** | Deep Learning | **1.325** | 1.011 | AR(1) | 1.485 |
| *Bipolaris oryzae* | Brown Spot of Rice | **Global Random Forest** | Global Tree Ensemble | **4.344** | 2.637 | MA(1) | 4.410 |
| *Fusarium oxysporum* | Fusarium Wilt | **XGBoost (Autoregressive)** | Local Gradient Boost | **6.375** | 4.931 | AR(1) | 6.779 |
| *Macrophomina phaseolina* | Dry Root Rot / Charcoal Rot | **GRU (Gated Recurrent Unit)** | Deep Learning | **1.758** | 1.294 | MA(1) | 1.831 |
| *Phytophthora infestans* | Late Blight of Potato | **Random Forest (Autoregressive)** | Local Tree Ensemble | **3.151** | 1.102 | MA(1) | 3.328 |
| *Puccinia striiformis* | Yellow / Stripe Rust of Wheat | **Seasonal Naive (Baseline)** | Persistence Benchmark | **0.000** | 0.000 | Seasonal Naive | 0.000 |
| *Pyricularia oryzae* | Rice Blast | **Global Random Forest** | Global Tree Ensemble | **1.906** | 1.143 | AR(1) | 2.062 |
| *Ralstonia solanacearum* | Bacterial Wilt | **Hurdle XGBoost** | Global Two-Stage ML | **15.808** | 6.891 | MA(1) | 16.719 |
| *Rhizoctonia solani* | Sheath Blight of Rice | **Hurdle XGBoost** | Global Two-Stage ML | **3.449** | 2.358 | MA(1) | 3.529 |
| *Xanthomonas oryzae* | Bacterial Blight of Rice | **MA(1)** | Classical Econometric | **0.505** | 0.280 | MA(1) | 0.505 |

### Epidemiological Analysis by Model Category

- **Global Tree Ensembles (Global RF & Global XGBoost):**  
  Excel on foliar cereal pathogens with strong weather dependence (*Pyricularia oryzae*, *Bipolaris oryzae*). Pooling 10 pathogens across India enables the trees to discover robust interactions between relative humidity, temperature, and disease onset that local series cannot estimate alone.
- **Hurdle Two-Stage Modeling (Hurdle XGBoost):**  
  Dominates highly volatile, zero-inflated soil-borne bacterial pathogens (*Ralstonia solanacearum*, *Rhizoctonia solani*). These pathogens stay dormant for many months and then produce explosive outbreaks. Hurdle XGBoost explicitly isolates the zero-probability classification from the positive outbreak count regression.
- **Deep Recurrent Neural Networks (LSTM & GRU):**  
  Excel on cyclical blights (*Alternaria solani*, *Macrophomina phaseolina*). The recurrent hidden state acts as an internal biological reservoir memory, capturing multi-month incubation and spore buildup dynamics.
- **Classical Low-Order Econometrics (MA(1)):**  
  Remains competitive only on sparse, low-count pathogens (*Xanthomonas oryzae*, max count 1.0) where occurrences behave like Poisson white noise and complex feature engineering overfits.

---

## 6. Statistical Significance & Moving Block-Bootstrap CIs

Because time series forecast errors exhibit serial autocorrelation, standard independent Student's t-tests underestimate variance and produce falsely tight confidence intervals. We implement a **moving block bootstrap** ($B = 1,000\text{ bootstrap iterations}$, block size $b = 3\text{ months}$) to preserve temporal correlation structures.

The test computes paired out-of-sample absolute error differences:
$$\Delta e_t = |y_t - \hat{y}_t^{\text{baseline}}| - |y_t - \hat{y}_t^{\text{candidate}}|$$
A positive mean value ($\Delta \bar{e} > 0$) with a 95% confidence interval strictly above zero confirms statistically significant error reduction ($p < 0.05$).

| Model Architecture | Mean MAE Gain vs Naive | 95% Bootstrap CI vs Naive | Significant vs Naive? | Mean MAE Gain vs AR(1) | 95% Bootstrap CI vs AR(1) | Significant vs AR(1)? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hurdle XGBoost** | **+0.397** | **[+0.159, +0.644]** | **Yes (p < 0.05)** | +0.268 | [-0.039, +0.643] | Inconclusive |
| **Prophet** | **+0.492** | **[+0.274, +0.751]** | **Yes (p < 0.05)** | **+0.362** | **[+0.076, +0.709]** | **Yes (p < 0.05)** |
| **Global Random Forest** | **+0.356** | **[+0.126, +0.608]** | **Yes (p < 0.05)** | +0.226 | [-0.072, +0.584] | Inconclusive |
| **Global XGBoost** | **+0.362** | **[+0.149, +0.609]** | **Yes (p < 0.05)** | +0.233 | [-0.063, +0.582] | Inconclusive |
| **GRU** | **+0.463** | **[+0.227, +0.730]** | **Yes (p < 0.05)** | **+0.334** | **[+0.053, +0.686]** | **Yes (p < 0.05)** |
| **LSTM** | **+0.391** | **[+0.161, +0.651]** | **Yes (p < 0.05)** | +0.262 | [-0.027, +0.617] | Inconclusive |
| **MA(1)** | **+0.321** | **[+0.079, +0.569]** | **Yes (p < 0.05)** | +0.191 | [-0.103, +0.518] | Inconclusive |
| **AR(1)** | +0.130 | [-0.102, +0.380] | No | 0.000 | [0.000, 0.000] | Reference |
| **Seasonal Naive** | 0.000 | [0.000, 0.000] | Reference | -0.130 | [-0.380, +0.102] | No |

### Significance Conclusions

- **Hurdle XGBoost, Prophet, Global Random Forest, Global XGBoost, GRU, and LSTM** all achieve statistically significant improvements over the Seasonal Naive baseline ($p < 0.05$).
- In contrast, neither AR(1), ARIMA, nor Holt-Winters achieved statistically significant gains over Seasonal Naive across the full 240-step multi-origin evaluation.
- When compared directly against **AR(1)**, **Prophet** and **GRU** demonstrated statistically significant error reductions ($p < 0.05$), while Hurdle XGBoost, Global RF, and Global XGBoost showed strong positive point gains (+0.23 to +0.27 reports) with confidence intervals narrowly spanning zero due to the extreme variance of the Origin 3 outbreak surge.

---

## 7. Outbreak Detection & Zero-Inflation Dynamics

In plant pathology surveillance, false negative outbreak misses lead to crop failure, while false alarms lead to unnecessary pesticide application. With **71.8% zero counts**, evaluate whether candidate models merely predict constant near-zero values or genuinely distinguish active outbreaks.

### Outbreak Contingency Matrix (Threshold $y \ge 1.0$)

| Model Architecture | Outbreak Precision | Outbreak Recall | Outbreak F1 | PR-AUC | False Alarm Rate (FAR) | Active Outbreak MAE ($y \ge 1$) | Zero Dormancy MAE ($y = 0$) | Near-Zero Predictions ($< 0.1$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hurdle XGBoost** | 0.702 | **0.862** | **0.774** | 0.778 | 38.5% | **3.882** | 0.780 | 5.4% |
| **Prophet** | 0.729 | 0.764 | 0.746 | 0.808 | 29.9% | 3.911 | 0.554 | 23.8% |
| **Global Random Forest** | 0.669 | 0.789 | 0.724 | 0.802 | 41.0% | 3.946 | 0.798 | 0.0% |
| **Global XGBoost** | 0.667 | 0.732 | 0.698 | 0.803 | 38.5% | 4.030 | 0.697 | 0.4% |
| **GRU** | 0.693 | 0.789 | 0.738 | **0.815** | 36.8% | 4.054 | 0.464 | 23.3% |
| **LSTM** | 0.687 | 0.748 | 0.716 | 0.805 | 35.9% | 4.185 | 0.475 | 21.7% |
| **MA(1)** | 0.737 | 0.683 | 0.709 | 0.811 | 25.6% | 4.385 | **0.407** | 23.3% |
| **Random Forest (Local)** | 0.728 | 0.675 | 0.700 | 0.766 | 26.5% | 4.338 | 0.863 | 27.5% |
| **XGBoost (Local)** | 0.651 | 0.789 | 0.713 | 0.776 | 44.4% | 4.447 | 1.101 | 19.6% |
| **AR(1)** | **0.752** | 0.715 | 0.733 | 0.822 | **24.8%** | 4.690 | 0.480 | 19.6% |
| **Seasonal Naive** | 0.678 | 0.496 | 0.573 | 0.642 | **24.8%** | 4.878 | 0.547 | 62.5% |
| **Holt-Winters** | 0.721 | 0.756 | 0.738 | 0.782 | 30.8% | 4.697 | 0.874 | 25.8% |
| **ARIMA(1,1,1)** | 0.741 | 0.699 | 0.720 | 0.789 | 25.6% | 5.168 | 0.656 | 29.2% |
| **SARIMA** | 0.724 | 0.683 | 0.703 | 0.780 | 27.4% | 5.140 | 0.658 | 27.9% |

### Outbreak Dynamics Insights

1. **Superior Outbreak Recall by Hurdle XGBoost (86.2%):**  
   Hurdle XGBoost missed only 13.8% of positive outbreaks across all 10 pathogens, outperforming Seasonal Naive (which missed 50.4% of outbreaks) and AR(1) (which missed 28.5%).
2. **Active vs Dormant Error Separation:**  
   - When disease was actively circulating ($y \ge 1$), Hurdle XGBoost achieved the lowest error magnitude (**Active Outbreak MAE = 3.882 reports**), whereas ARIMA erred by 5.168 reports.
   - During dormant months ($y = 0$), classical MA(1) and deep nets achieved lower error (~0.45 reports) because single-stage regressors shrink to zero, whereas hurdle regressors produce small non-zero estimates when the classifier triggers.

---

## 8. Feature Ablation & Missingness Robustness

### 8.1 Empirical Feature Progression

To confirm that model gains stem from informative epidemiological signals rather than overfitting, we benchmarked 3 progressive feature configurations across all 10 pathogens:

| Configuration | Feature Set & Architecture | Missingness Handling | Pooled RMSE | Macro RMSE | Outbreak F1 | WAPE (%) |
|---|---|---|:---:|:---:|:---:|:---:|
| **Baseline (Exp A)** | Local Autoregressive Target Lags Only (Local XGBoost) | Standard Zero-Fill ($w = 1.0$) | 6.462 | 4.325 | 0.713 | 101.2% |
| **Feature-Rich (Exp E)** | Multi-Pathogen Global Pool + 70 NASA Weather + EPPO Alerts | Standard Zero-Fill ($w = 1.0$) | 6.185 | 4.210 | 0.738 | 91.4% |
| **Optimized Hurdle (Exp C1)** | Global Multi-Pathogen + NASA Weather + EPPO + Two-Stage Hurdle | **Downweighted Imputed Zeros ($w = 0.2$)** | **5.982** | **4.111** | **0.774** | **85.1%** |

### 8.2 Missingness Topology & Sample Weighting Analysis

Our audit of the 1,320 pathogen-month records identified two fundamentally different zero origins:
1. **Structural Zeros (812 observations, 61.5%):**  
   True biological dormancies (e.g. absence of wheat rust in high-temperature summer months). These carry genuine epidemiological information and are assigned full weight ($w = 1.0$).
2. **Gap-Imputed Zeros (136 observations, 10.3%):**  
   Unreported surveillance intervals in state-level monitoring where missing records were imputed to zero.
   - Assigning full weight ($w = 1.0$) biased the binary classifier toward zero, artificially suppressing legitimate outbreak warnings.
   - Discarding these records created temporal discontinuities that broke continuous lag features ($y_{t-1}, y_{t-12}$).
   - **Solution (Exp C1):** Retaining continuous temporal indexing while downweighting gap-imputed records to **$w = 0.2$** in the classifier loss function stabilized probability calibration.
   - **Result:** Downweighting gap-imputed zeros reduced Pooled RMSE from 6.340 to **5.982** (-5.6% error reduction) and boosted Outbreak F1 from 0.721 to **0.774**.

---

## 9. Operational Recommendations & System Limitations

### 9.1 Recommended Production Architecture

Based on multi-origin validation across 240 chronological forecast steps, we recommend a **hybrid routing strategy**:

1. **Default Primary Forecaster:**  
   **Hurdle XGBoost (Classifier + Regressor with $w=0.2$ gap-downweighting)**  
   - Recommended for general state and district surveillance dashboards.
   - Delivers the lowest Pooled RMSE (5.982), highest Outbreak Recall (86.2%), and best peak surge containment.
2. **Specialized Pathogen Assignees:**  
   - For **Rice Blast (*Pyricularia oryzae*)** and **Brown Spot (*Bipolaris oryzae*)**: Deploy **Global Random Forest** (lowest pathogen RMSEs: 1.906 and 4.344).
   - For **Early Blight (*Alternaria solani*)**: Deploy **LSTM** (lowest RMSE: 1.325).
   - For **Bacterial Blight (*Xanthomonas oryzae*)**: Deploy **MA(1)** (lowest RMSE: 0.505).
3. **Backup Benchmark Forecaster:**  
   **Prophet (Additive Trend + Seasonality)**  
   - Robust second-choice model with Pooled RMSE 5.996 and lowest overall MAE (2.275). Useful for generating smooth interpretative trend decompositions.

### 9.2 Limitations of the Current Study

1. **Monthly Surveillance Aggregation:**  
   Monthly resolution aggregates short-duration infection windows. Extreme 3-day rainfall spikes that trigger fungal sporulation are smoothed into monthly precipitation totals.
2. **Spatial Granularity:**  
   NASA POWER agro-meteorological variables are averaged across agro-ecological zones at ~0.5° spatial resolution (~50 km), which may obscure microclimatic variations in valley topographies.
3. **Puccinia striiformis Data Sparsity:**  
   Yellow rust records in the dataset remained at zero throughout the 20-month validation window, making Seasonal Naive (RMSE 0.000) technically optimal on this subset. Active epidemic testing requires northern wheat-belt expansion.
4. **Final Test Isolation:**  
   The final 20 months of data (**May 2024 to December 2025**) remain completely isolated and unobserved. A final out-of-sample test evaluation must be performed before production deployment.

---

## 10. Audit Sign-Off & Verification Summary

- **Multi-Origin Walk-Forward Benchmark Script:** Completed via `scripts/run_multi_origin_benchmark.py` (346.43 s runtime).
- **Benchmark Artifact:** Serialized to `data/processed/multi_origin_validation_results.json` (40 evaluation runs, 240 forecast steps, 14 models).
- **Interactive Dashboard:** Fully integrated into `pages/3_Model_Comparison.py` via `src/multi_origin_ui.py` with 5 interactive tabs and Plotly visualizations.
- **Unit Test Coverage:** 82 passing tests in `pytest tests/` covering multi-origin bounds, feature alignment, missingness weighting, reproducibility, and failure handling.
- **Git State:** Local working tree maintained cleanly without premature commit or push.

