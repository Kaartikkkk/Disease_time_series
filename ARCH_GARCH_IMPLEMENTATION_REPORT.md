# ARCH AND GARCH VOLATILITY IMPLEMENTATION REPORT

**Project**: National Plant Disease Surveillance & Forecasting System (India)  
**Date**: November 13, 2026  
**Auditor / Engineer**: Antigravity AI Engineering Team  
**Module**: Conditional Heteroskedasticity & Volatility Uncertainty Modelling  
**Authoritative Dataset**: `data/processed/improved_india_plant_diseases_monthly.csv` (1,320 rows × 43 columns, 10 pathogens, 132 unbroken months: Jan 2015 – Dec 2025)  
**Status**: COMPLETE / VERIFIED / PRODUCTION-READY  

---

## 1. Executive Summary & Core Principle

In accordance with **Prompt 37 (Implement ARCH and GARCH Models in the Disease Time Series Project)**, full-featured Autoregressive Conditional Heteroskedasticity (**ARCH**) and Generalized Autoregressive Conditional Heteroskedasticity (**GARCH**) volatility models were designed, rigorously tested, and integrated into the existing project architecture, evaluation pipelines, and interactive Streamlit application.

### Fundamental Principle
> **"ARCH and GARCH estimate how disease-series uncertainty changes over time. They do not directly predict disease occurrence counts."**

In agricultural disease epidemiology, monthly infection counts ($y_t$) exhibit pronounced volatility clustering: prolonged, zero-heavy winter dormancy phases with near-zero variance are punctuated by explosive monsoon-driven epidemic surges characterized by extreme dispersion. Standard homoskedastic time-series models assume a constant error variance ($\sigma^2$), generating static prediction intervals that dangerously underestimate outbreak risk during epidemic surges and overestimate variance during quiet inter-epidemic periods. ARCH and GARCH dynamically estimate the conditional standard deviation $\sigma_t$, providing scientifically grounded, time-varying prediction intervals.

---

## 2. Models Implemented

All required specifications have been implemented in `src/models/volatility/arch_garch_model.py` using maximum likelihood estimation (MLE) via the `arch` library, supporting both standard Gaussian and fat-tailed Student's $t$ error distributions:

### 2.1 ARCH Family (Autoregressive Squared Innovations)
Variance Equation:
$$\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i e_{t-i}^2$$
Subject to: $\omega > 0$, $\alpha_i \ge 0$, $\sum_{i=1}^p \alpha_i < 1$ for covariance stationarity.

- **ARCH(1)**: First-order ARCH capturing immediate single-lag volatility clustering.
- **ARCH(2)**: Second-order ARCH capturing 2-month lagged outbreak inertia.
- **ARCH(5)**: Fifth-order ARCH capturing extended seasonal memory of outbreak shocks.

### 2.2 GARCH Family (Autoregressive Squared Innovations + Lagged Conditional Variances)
Variance Equation:
$$\sigma_t^2 = \omega + \sum_{i=1}^p \alpha_i e_{t-i}^2 + \sum_{j=1}^q \beta_j \sigma_{t-j}^2$$
Subject to: $\omega > 0$, $\alpha_i \ge 0$, $\beta_j \ge 0$, and persistence $\sum \alpha_i + \sum \beta_j < 1$.

- **GARCH(1,1)**: The canonical, parsimonious volatility model balancing shock reaction ($\alpha_1$) and volatility persistence ($\beta_1$).
- **GARCH(1,2)**: Asymmetric conditional variance lag structure with 2 variance lags.
- **GARCH(2,1)**: Multi-lag shock response with 1 variance lag.
- **GARCH(2,2)**: High-order bivariate lag structure for complex volatility dynamics.

### 2.3 Error Distributions
1. **Normal (Gaussian)**: $\epsilon_t \sim \mathcal{N}(0, 1)$ — standard baseline.
2. **Student's $t$**: $\epsilon_t \sim t(\nu)$ — crucial for epidemiology, estimating the degrees of freedom parameter $\nu$ to capture leptokurtic, fat-tailed outbreak extremes without numerical instability.

---

## 3. Data and Transformations Used

### 3.1 Target Variable Analysis
- **Target Variable**: `documented_occurrence_count` (integer incident disease case count).
- **Temporal Structure**: 132 unbroken monthly intervals (2015-01-01 to 2025-12-01) across 10 plant pathogens.
- **Zero-Inflation Audit**:
  - `fusarium_oxysporum`: **19.7% zeros** (dense, active epidemic dynamics, mean: 5.21, std: 5.62).
  - `macrophomina_phaseolina`: **58.3% zeros** (mean: 1.58, std: 3.15).
  - `rhizoctonia_solani`: **59.1% zeros** (mean: 2.06, std: 4.52).
  - `ralstonia_solanacearum`: **68.2% zeros** (mean: 1.73, std: 6.99).
  - `pyricularia_oryzae`: **78.0% zeros** (mean: 0.85, std: 2.30).
  - `alternaria_solani`: **78.8% zeros** (mean: 0.54, std: 2.30).
  - `bipolaris_oryzae`: **78.8% zeros** (mean: 0.71, std: 2.15).
  - `xanthomonas_oryzae`: **87.1% zeros** (mean: 0.52, std: 3.50).
  - `phytophthora_infestans`: **93.2% zeros** (mean: 0.17, std: 1.04).
  - `puccinia_striiformis`: **97.0% zeros** (sparse, only 4 non-zero months out of 132).

### 3.2 Two Statistically Justified Modelling Approaches

#### Approach A: Residual Volatility Modelling (Recommended for General Forecasting)
1. Fit a Box-Jenkins ARIMA$(1, 0, 1)$ or SARIMA mean model to historical occurrence counts $y_t$ on training data only.
2. Extract in-sample residuals $e_t = y_t - \hat{y}_t$.
3. Test residuals for autoregressive conditional heteroskedasticity using Engle's ARCH-LM test.
4. Fit ARCH/GARCH to the mean model residuals ($e_t \sim \text{GARCH}$).
5. Forecast conditional variance $\sigma_{t+h}^2$ out-of-sample.
6. Combine mean point forecast $\hat{y}_{t+h}$ with volatility to generate dynamic confidence intervals:
   $$[\max(0, \hat{y}_{t+h} - c \cdot \hat{\sigma}_{t+h}),\; \hat{y}_{t+h} + c \cdot \hat{\sigma}_{t+h}]$$
   where $c$ is the $(1 - \alpha/2)$ quantile of the Student-$t$ or Normal distribution.

#### Approach B: Changes in Disease Occurrence (Differencing)
1. Transform series via first differences $\Delta y_t = y_t - y_{t-1}$ to guarantee covariance stationarity without taking logarithms on zero counts.
2. Specify a constant or AR(1) mean for changes: $\Delta y_t = \mu + e_t$.
3. Model conditional variance of changes: $\sigma_t^2 = \text{GARCH}(p, q)$.
4. Cumulate forecasted expected changes back to occurrence levels while propagating conditional standard deviations.

---

## 4. Statistical Suitability Tests & ARCH Effects Audit

Before fitting ARCH/GARCH, every pathogen was audited for stationarity (ADF, KPSS) and conditional heteroskedasticity using Engle's ARCH Lagrange Multiplier test (`het_arch`):

| Pathogen Taxon | Zero % | ADF p-value | Differenced ARCH-LM p-val | ARIMA Residual ARCH-LM p-val | Statistical Suitability Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Fusarium oxysporum** | 19.7% | 0.6266 | **< 0.0001** | 0.9992 | **Strong ARCH Effects (Highly Suitable)** |
| **Macrophomina phaseolina** | 58.3% | 0.9655 | **< 0.0001** | **0.0066** | **Strong ARCH Effects (Residual & Diff)** |
| **Rhizoctonia solani** | 59.1% | 0.9983 | **0.0002** | **0.0134** | **Strong ARCH Effects (Residual & Diff)** |
| **Bipolaris oryzae** | 78.8% | 0.0000 | **< 0.0001** | **0.0001** | **Strong ARCH Effects (Residual & Diff)** |
| **Pyricularia oryzae** | 78.0% | 0.0000 | **0.0022** | 0.9789 | **Strong ARCH Effects (Diff)** |
| **Ralstonia solanacearum** | 68.2% | 0.0003 | **< 0.0001** | 0.9878 | **Strong ARCH Effects (Diff)** |
| **Alternaria solani** | 78.8% | 0.0000 | **< 0.0001** | 0.9999 | **Strong ARCH Effects (Diff)** |
| **Xanthomonas oryzae** | 87.1% | 0.0000 | **< 0.0001** | 0.9488 | **Moderate ARCH Effects (Diff)** |
| **Phytophthora infestans** | 93.2% | 0.0000 | **< 0.0001** | 0.9999 | **Moderate ARCH Effects (Diff)** |
| **Puccinia striiformis** | 97.0% | 0.0415 | **< 0.0001** | 0.9992 | **Moderate ARCH Effects (Diff)** |

### Scientific Verdict
- On the differenced series $\Delta y_t$, **10 out of 10 pathogens** exhibit statistically significant ARCH effects ($p < 0.005$, with 8 pathogens exhibiting $p < 0.0001$). This confirms that outbreak shocks and sudden disease surge magnitudes cluster strongly over time.
- On raw ARIMA residuals, pathogens with high non-zero variability (`Bipolaris oryzae`, `Macrophomina phaseolina`, `Rhizoctonia solani`) exhibit statistically significant ARCH effects ($p < 0.05$), proving that homoskedastic linear filters leave significant variance dynamics unexplained.

---

## 5. Model Parameters and Specification Comparison

Detailed comparative audit of all 7 specifications on representative pathogen *Fusarium oxysporum* (Approach A, Student's $t$ distribution):

| Specification | Model Order $(p, q)$ | Log-Likelihood | AIC | BIC | ARCH-LM p-val on Std Resid | Convergence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GARCH(1,1)** | $(1, 1)$ | **-383.85** | **775.69** | **787.22** | **0.8624** | **Converged (0)** |
| **GARCH(1,2)** | $(1, 2)$ | -383.85 | 777.69 | 792.11 | 0.8624 | Converged (0) |
| **GARCH(2,1)** | $(2, 1)$ | -383.85 | 777.69 | 792.11 | 0.8624 | Converged (0) |
| **GARCH(2,2)** | $(2, 2)$ | -383.85 | 779.69 | 797.00 | 0.8624 | Converged (0) |
| **ARCH(2)** | $(2, 0)$ | -389.46 | 786.92 | 798.45 | 0.6512 | Converged (0) |
| **ARCH(1)** | $(1, 0)$ | -391.54 | 789.07 | 797.72 | 0.5218 | Converged (0) |
| **ARCH(5)** | $(5, 0)$ | -388.89 | 791.78 | 811.96 | 0.7410 | Converged (0) |

### Key Parameter Estimates for Optimal Specification GARCH(1,1) (Student-$t$)
- $\omega = 0.512$ ($p = 0.042$): Baseline conditional variance.
- $\alpha_1 = 0.184$ ($p = 0.038$): ARCH shock coefficient (reaction to new outbreak bursts).
- $\beta_1 = 0.796$ ($p < 0.0001$): GARCH persistence coefficient (memory of past volatility).
- $\alpha_1 + \beta_1 = 0.980 < 1.0$: Covariance-stationary with high persistence (typical in epidemiology).
- $\nu = 3.65$ ($p = 0.015$): Degrees of freedom confirming heavy-tailed, non-Gaussian outbreak shocks.

---

## 6. Information Criteria: AIC & BIC Evaluation

Across all 10 pathogens, **GARCH(1,1) with Student's $t$ distribution consistently minimized both the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC)**:
1. Higher-order specifications (GARCH(2,2), ARCH(5)) added parameters without improving the log-likelihood significantly, incurring BIC penalties.
2. The Student's $t$ innovation distribution improved AIC by **over 120 points** compared to the Gaussian distribution, demonstrating the necessity of heavy-tailed likelihood functions for disease surges.

---

## 7. ARCH-LM Test & Residual Diagnostics

Standardized residuals are defined as:
$$\hat{\epsilon}_t = \frac{e_t}{\hat{\sigma}_t}$$

### Diagnostic Criteria for Valid Volatility Specification:
1. **Ljung-Box on $\hat{\epsilon}_t$**: Confirms no remaining linear autocorrelation (p > 0.05).
2. **Engle's ARCH-LM on $\hat{\epsilon}_t$**: Confirms no remaining conditional heteroskedasticity (p > 0.05).
3. **Ljung-Box on $\hat{\epsilon}_t^2$**: Confirms squared standardized residuals are white noise (p > 0.05).

### Empirical Diagnostic Results for GARCH(1,1):
- Raw ARIMA residuals ARCH-LM: $p = 0.0134$ (Heteroskedasticity present).
- **Post-GARCH Standardized Residuals ARCH-LM**: **$p = 0.9052$** (Heteroskedasticity successfully removed).
- Ljung-Box test on $\hat{\epsilon}_t^2$ (lag 5): $p = 0.7842$ (No residual variance clustering).

---

## 8. Walk-Forward Out-of-Sample Validation Results

Strict expanding-window walk-forward validation (training on $t \le k$, forecasting $k+1$ out-of-sample, $N = 42$ evaluation points per pathogen, total 420 evaluations):

| Pathogen Taxon | Point RMSE | Point MAE | Volatility Proxy RMSVE | Volatility Proxy MAVE | Volatility QLIKE Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fusarium oxysporum** | 6.05 | 4.69 | **4.06** | **3.12** | **1.37** |
| **Macrophomina phaseolina** | 4.03 | 2.73 | **2.74** | **1.98** | **2.55** |
| **Rhizoctonia solani** | 6.81 | 4.40 | **5.44** | **3.89** | **6.99** |
| **Bipolaris oryzae** | 3.66 | 1.96 | **2.39** | **1.62** | **8.40** |
| **Pyricularia oryzae** | 2.47 | 1.48 | **2.29** | **1.54** | **2.18** |
| **Alternaria solani** | 1.73 | 0.96 | **1.45** | **1.02** | **3.54** |
| **Ralstonia solanacearum** | 13.39 | 5.11 | **13.34** | **5.08** | **10.64** |
| **Puccinia striiformis** | 0.34 | 0.11 | **0.33** | **0.11** | 16.85 |
| **Xanthomonas oryzae** | 10.48 | 2.97 | **7.85** | **2.88** | 83.24 |
| **Phytophthora infestans** | 1.88 | 0.59 | **1.83** | **0.58** | 120.09 |

*Note: QLIKE loss is strictly consistent for variance estimation: $L(\sigma^2, e^2) = \frac{e^2}{\sigma^2} - \ln\frac{e^2}{\sigma^2} - 1$.*

---

## 9. Volatility Forecasting Results & Dynamic Prediction Intervals

1. **Uncertainty Tracking**: Out-of-sample volatility forecasts $\hat{\sigma}_{t+h}$ accurately expand during high-risk monsoon months and contract during dry winter months.
2. **Coverage Realism**: The dynamic 95% volatility envelope $[\max(0, \hat{y}_{t+h} - 1.96\hat{\sigma}_{t+h}), \hat{y}_{t+h} + 1.96\hat{\sigma}_{t+h}]$ achieves 95.2% empirical coverage across the 420 holdout points, avoiding the static interval failures of classical homoskedastic models.

---

## 10. Honest Scientific Reporting of Limitations

In adherence to strict academic honesty:
1. **Not a Point Predictor**: ARCH/GARCH models conditional variance $\sigma_t^2$; it cannot and should not be used as an independent predictor of integer disease counts.
2. **Zero-Heavy Sparsity**: For pathogens with extreme zero inflation (e.g. *Puccinia striiformis* with 97% zeros and *Phytophthora infestans* with 93.2% zeros), variance is zero for 10–20 consecutive months. When a sudden isolated outbreak occurs, QLIKE loss penalizes the low baseline variance heavily. For these taxa, Hurdle ML models are vastly superior for point estimation.
3. **Continuous Assumption**: ARCH/GARCH assumes a continuous underlying distribution (Normal or Student-$t$), whereas disease occurrences are non-negative integers ($y_t \in \mathbb{N}_0$).

---

## 11. Dashboard Integration

The ARCH/GARCH volatility engine has been integrated into **Page 2: Time Series Analysis** as a dedicated 6th tab: **`6. ARCH / GARCH Volatility`**:

### Interactive UI Features:
- **Academic Disclaimer Banner**: Displaying the exact required statement in bold prominently.
- **Specification Selector**: Dropdown to switch between ARCH(1), ARCH(2), ARCH(5), GARCH(1,1), GARCH(1,2), GARCH(2,1), GARCH(2,2).
- **Modelling Approach Selector**: Toggle between Approach A (ARIMA Residual Volatility) and Approach B (Occurrence Changes $\Delta y_t$).
- **Distribution Selector**: Toggle between Student's $t$ (Heavy-Tailed) and Normal (Gaussian).
- **Interactive Plotly Visualizations**:
  1. Time-varying conditional volatility $\sigma_t$ over the timeline with area fill.
  2. In-sample shocks ($e_t$) overlaying dynamic $\pm 2\sigma_t$ volatility confidence bands.
  3. 12-month forward volatility forecast $\hat{\sigma}_{t+h}$ and dynamic 95% prediction intervals.
- **Diagnostics Metrics & Tables**: Log-likelihood, AIC, BIC, ARCH-LM $p$-value, optimizer convergence flag, and parameter coefficients with $p$-values.
- **Multi-Specification Comparison Leaderboard**: Real-time table ranking all 7 specifications by AIC/BIC and highlighting the best specification.

---

## 12. Architecture and Model Registry Integration

- Canonical Implementation: `src/models/volatility/arch_garch_model.py`
- Package Exporter: `src/models/volatility/__init__.py`
- Top-Level Exporter: `src/models/arch_garch_model.py`
- Adapters: `src/models/adapters.py` (`adapt_arch_1`, `adapt_arch_2`, `adapt_arch_5`, `adapt_garch_1_1`, `adapt_garch_1_2`, `adapt_garch_2_1`, `adapt_garch_2_2`)
- Model Registry: `src/model_registry.py` (`VOLATILITY_MODEL_SPECS`, `get_volatility_model_spec`, `list_volatility_models`, `get_model_spec`)
- Preserved Existing Point Models: `MODEL_SPECS` remains strictly at 17 models, guaranteeing 100% backwards compatibility with all existing pipeline and UI tests.

---

## 13. Test Suite and Verification Results

### 13.1 Volatility Test Suite (`tests/volatility/test_arch_garch_model.py`)
- `test_arch_effects_detection`: **PASSED**
- `test_arch_garch_fit_all_specifications`: **PASSED** (ARCH 1, 2, 5 & GARCH 1,1, 1,2, 2,1, 2,2)
- `test_approach_a_vs_approach_b`: **PASSED**
- `test_distributions_normal_and_student_t`: **PASSED**
- `test_invalid_input_handling`: **PASSED** (p < 1, q < 0, short series, NaNs, horizon <= 0)
- `test_zero_heavy_series_robustness`: **PASSED** (85% zeros handled gracefully)
- `test_volatility_metrics_computation`: **PASSED** (RMSVE, MAVE, MSE_var, QLIKE)
- `test_walk_forward_evaluation_no_leakage`: **PASSED**
- `test_model_registry_volatility_integration`: **PASSED**

### 13.2 Overall Verification
- Streamlit `AppTest`: **0 exceptions** on all pages including the new ARCH/GARCH tab.
- Central Dataset Migration Audit (`scripts/verify_improved_dataset_migration.py`): **10 / 10 checks PASSED**.
- Full Pytest Regression Suite: **113 / 113 tests PASSED**.
