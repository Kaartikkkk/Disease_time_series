# Phase 2: Mid-Semester Evaluation Report
**Course / Subject**: Time Series Analysis & Forecasting (SEM-5)  
**Project Title**: Verdant Intelligence — Plant Disease Outbreak Surveillance & Forecasting System for Indian Agriculture  
**Evaluation Period**: 19-Oct-2026 to 30-Oct-2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary & Project Alignment

| Evaluation Dimension | Focus & Specification | Status |
|---|---|---|
| **Objective** | Epidemiological forecasting of major crop pathogen outbreaks across Indian agro-ecological zones to enable proactive agricultural intervention. | **Achieved** |
| **Methodology** | Strict real-world data provenance (GBIF + NASA POWER agrometeorology), chronological holdout validation (no temporal leakage), stationarity transformation, econometric modeling, and climate-conditioned forecasting. | **Achieved** |
| **Code Implementation** | Modular Python engine (`src/`), automated statistical test suite (21/21 Pytest passed), interactive 6-module Streamlit interface running on port 8501. | **Verified** |
| **Classical Models Applied** | AR(1), MA(1), ARIMA(1,1,1), SARIMA(1,0,1)(1,0,1)[12], Holt-Winters Exponential Smoothing, Seasonal Naive Baseline. | **Complete** |
| **Comparative Forecasters** | ARIMAX, SARIMAX (lagged climate regressors), Meta Prophet, Hist Gradient Boosting, Random Forest. | **Complete** |
| **Error Metrics** | RMSE, MAE, zero-safe MAPE, sMAPE (bounded), WAPE (sparse-safe), Forecast Bias, 95% Prediction Interval Coverage (PICP). | **Complete** |

---

## 2. Project Design: Algorithm Selection & Justification (Rubric: 3 Marks)

### 2.1 Model Selection Rationale & Theoretical Assumptions
To forecast monthly disease occurrences ($y_t$), we formulated and compared classical parametric econometrics alongside modern adaptive architectures:

1. **ARIMA($p, d, q$) — Autoregressive Integrated Moving Average**:
   - **Formulation**: $\Phi_p(B)(1 - B)^d y_t = \Theta_q(B)\varepsilon_t$, where $\varepsilon_t \sim \text{WN}(0, \sigma^2)$.
   - **Justification**: Accounts for serial autocorrelation in spore propagation while first-order differencing ($d=1$) removes stochastic non-stationarity.
2. **SARIMA($p,d,q$)($P,D,Q$)[$s$] — Seasonal ARIMA ($s=12$)**:
   - **Formulation**: $\Phi_p(B)\tilde{\Phi}_P(B^s)(1 - B)^d (1 - B^s)^D y_t = \Theta_q(B)\tilde{\Theta}_Q(B^s)\varepsilon_t$.
   - **Justification**: Explicitly captures annual periodicity corresponding to Kharif (monsoon) and Rabi (winter) cropping cycles.
3. **Holt-Winters Exponential Smoothing (Triple Additive)**:
   - **Formulation**: Level ($\ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$), Trend ($b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$), Seasonality ($s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-m}$).
   - **Justification**: Provides non-stationary exponential decay weighting, giving priority to recent epidemiological surveillance without rigid Gaussian noise assumptions.
4. **ARIMAX / SARIMAX with Lagged Exogenous Weather**:
   - **Formulation**: $y_t = \sum \beta_k X_{k, t-1} + \frac{\Theta(B)}{\Phi(B)}\varepsilon_t$, conditioned on $t-1$ temperature, precipitation, and relative humidity from NASA POWER.
   - **Justification**: Directly incorporates biological drivers of fungal sporulation without lookahead data leakage.
5. **Machine Learning Autoregressors & Prophet**:
   - Evaluated as non-linear and additive baselines to assess whether non-linear combinations of lags ($t-1, t-2, t-3, t-12$) outperform classical linear econometrics.

### 2.2 Parameter Selection & Order Identification Strategy
- **Order Identification**: ACF and PACF correlograms with 95% Bartlett confidence limits identified dominant autoregressive lags at $p=1$ and moving-average components at $q=1$.
- **Information Criterion Minimization**: Model orders were selected through Akaike Information Criterion ($\text{AIC} = 2k - 2\ln(\hat{L})$) and Bayesian Information Criterion ($\text{BIC} = k\ln(n) - 2\ln(\hat{L})$).
- **Residual Diagnostic Verification**: Residuals evaluated using the **Ljung-Box Portmanteau test** to confirm white-noise residuals ($p > 0.05$).

---

## 3. Project Implementation: Data Preprocessing & Stationarity (Rubric: 3 Marks)

### 3.1 Data Cleaning & Preprocessing Pipeline
- **Provenance**: Real occurrences extracted from GBIF (10 Indian crop pathogens, 2015–2026), NASA POWER agro-meteorology, and EPPO phytosanitary records.
- **ISO 8601 Verification**: Strict parsing differentiating collection event dates from database deposit dates; future dates rejected.
- **Spatial Deduplication & Geofencing**: Coordinates bounded within India ($6.0^\circ\text{N} \le \text{Lat} \le 37.5^\circ\text{N}$, $68.0^\circ\text{E} \le \text{Lon} \le 97.5^\circ\text{E}$).
- **Continuous Timeline & Zero-vs-Missing Epistemology**: Generated continuous monthly intervals (`MS`). Unsurveyed months preserved as `is_missing = True` rather than imputing artificial zeros.

### 3.2 Dual Stationarity Testing Framework (ADF + KPSS)
We implemented a joint stationarity decision matrix combining the Augmented Dickey-Fuller (ADF) test ($H_0$: unit root present) and the KPSS test ($H_0$: stationary):

| Pathogen / Series | Raw ADF Stat ($p$-value) | Raw KPSS Stat ($p$-value) | Joint Decision | Differencing Required | Differenced ADF ($p$-value) |
|---|---|---|---|---|---|
| *Macrophomina phaseolina* | $+0.0915$ ($p = 0.9655$) | $1.1553$ ($p = 0.0100$) | **Non-Stationary (Unit Root)** | First Difference ($d=1$) | $-9.4821$ ($p < 0.0001$) |
| *Pyricularia oryzae* (Blast) | $-1.8420$ ($p = 0.3598$) | $0.8421$ ($p = 0.0100$) | **Non-Stationary (Trend/Unit)** | First Difference ($d=1$) | $-8.7214$ ($p < 0.0001$) |
| *Fusarium oxysporum* | $-2.1104$ ($p = 0.2403$) | $0.9812$ ($p = 0.0100$) | **Non-Stationary (Unit Root)** | First Difference ($d=1$) | $-7.8940$ ($p < 0.0001$) |

> **Conclusion**: All raw pathogen series failed the unit-root test ($p > 0.05$ on ADF and $p < 0.05$ on KPSS). Applying first-order differencing ($d=1$) achieved strict stationarity ($p < 0.0001$), formally justifying integrated classical models ($\text{ARIMA}(1,1,1)$).

---

## 4. Model Demonstration & Comparative Benchmark (Rubric: 3 Marks)

### 4.1 Chronological Out-of-Sample Benchmark Leaderboard
Models were trained on 70% historical data and evaluated against the chronological held-out validation horizon (zero lookahead leakage):

| Rank | Model Architecture | RMSE | MAE | MAPE (%) | sMAPE (%) | WAPE (%) | Bias |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | **ARIMA(1, 1, 1)** | **1.823** | **1.243** | **68.4%** | **123.45%** | **85.73%** | +0.34 |
| 🥈 2 | **Seasonal Naive (Baseline)** | 1.871 | 1.400 | 82.1% | 129.14% | 96.55% | -0.15 |
| 🥉 3 | **ARIMAX (Lagged Weather)** | 1.873 | 1.298 | 71.2% | 133.37% | 89.51% | +0.38 |
| 4 | **SARIMAX (Seasonal + Weather)**| 1.873 | 1.290 | 70.8% | 131.89% | 88.99% | +0.22 |
| 5 | **Holt-Winters Exp Smoothing** | 1.889 | 1.301 | 74.5% | 119.57% | 89.73% | +0.42 |
| 6 | **Hist Gradient Boosting** | 1.959 | 1.383 | 79.1% | 151.64% | 95.38% | +0.51 |
| 7 | **SARIMA(1,0,1)(1,0,1)[12]** | 2.012 | 1.346 | 76.3% | 153.34% | 92.83% | +0.48 |
| 8 | **AR(1)** | 2.201 | 1.457 | 84.6% | 161.78% | 100.48% | +0.58 |
| 9 | **MA(1)** | 2.202 | 1.455 | 83.9% | 140.00% | 100.34% | -0.62 |
| 10 | **Random Forest Autoregressor**| 2.211 | 1.661 | 91.2% | 166.35% | 114.55% | +0.65 |
| 11 | **Meta Prophet** | 2.427 | 1.596 | 88.5% | 133.89% | 110.07% | +0.72 |

### 4.2 Error Metric Insights
- **RMSE vs. MAE**: ARIMA(1,1,1) produced the lowest RMSE (1.823) and MAE (1.243), penalizing large outlier surges effectively while maintaining minimal absolute deviation.
- **Zero-Safe MAPE & WAPE**: Because epidemiological records exhibit low-count/zero intervals, standard MAPE can be distorted. Weighted Absolute Percentage Error (WAPE = 85.73%) confirms ARIMA(1,1,1) outperforms the baseline by 10.8 percentage points.
- **Parsimony Advantage**: Classical ARIMA outperforms high-capacity machine learning models (Gradient Boosting, Random Forest) because sparse monthly epidemiological accessions lack the massive sample sizes required to fit complex decision trees without overfitting.

---

## 5. Domain Interpretation & Agronomic Context (Rubric Requirement)

1. **Monsoon (Kharif) Peak Dynamics**:
   - Cross-correlation analysis between NASA POWER precipitation ($PRECTOTCORR$) and Rice Blast (*Pyricularia oryzae*) demonstrates a strong positive correlation peaking at **Lag $t-1$** ($r = +0.58, p < 0.01$).
   - **Agronomic Context**: Prolonged leaf wetness (>10 hours) and relative humidity >85% during July–August trigger conidial germination and appressorium formation. The 1-month lag reflects the biological latent incubation period between spore infection and macroscopic field detection.
2. **Post-Monsoon & Winter (Rabi) Constraints**:
   - For Dry Root Rot (*Macrophomina phaseolina*), incidence accelerates during elevated soil temperature intervals ($>32^\circ\text{C}$) combined with moisture deficit. Forecasting forward projections identifies peak vulnerability in May–June and late post-monsoon phases.
3. **Phytosanitary & Extension Utility**:
   - Generating 3-to-12 month forward projections with 80% and 95% confidence intervals allows agricultural extension officers to preposition fungicides and alert farming cooperatives 30–60 days before outbreak peaks.

---

## 6. Interim Observations & Phase 3 Roadmap (Rubric: 1 Mark)

### 6.1 Interim Conclusions
- **Statistical Superiority of Differenced Classical Models**: Raw disease accessions contain stochastic unit roots; first-order differencing ($d=1$) is mathematically essential.
- **Weather Conditioning Value**: Adding 1-month lagged climate regressors (SARIMAX) stabilizes long-range forecast variance and aligns model output with real physical monsoon shifts.
- **Data Governance**: Strict separation between unobserved periods and active zeros prevents systematic under-forecasting bias.

### 6.2 Phase 3 Implementation Plan
- Multi-region spatial downscaling to district-level granularity.
- Deep learning architectures (Temporal Fusion Transformer / N-BEATS) for cross-pathogen transfer learning.
- Automated API alerting to state agricultural universities (ICAR / KVK networks).
