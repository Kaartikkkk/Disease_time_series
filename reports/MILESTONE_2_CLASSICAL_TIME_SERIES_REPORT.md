# Classical Time Series Analysis and Forecasting of Agricultural Plant Diseases — Milestone 2 Evaluation Report

**Course:** CSET 369 — Time Series Analysis & Forecasting  
**Author:** Kartik Lamba  
**Enrollment Number:** S24CSEU0642  
**Email:** lambakartik2006@gmail.com  
**Department:** School of Computer Science, Engineering & Technology (SCSET)  
**Institution:** Bennett University, Greater Noida, India  
**Course Coordinator:** Anurag Choubey  
**Academic Year:** 2026–2027 (Semester 5)  
**Milestone Focus:** Milestone 2 — Classical Time Series Econometrics & Statistical Baselines  
**Evaluation Rubric Alignment:** CSET 369 Mid-Semester & Project Milestones (IEEE Standard, $\le 10$ Pages)  

---

### Abstract
*Plant fungal and bacterial epidemics impose severe yield losses across Indian agro-ecological zones, compromising agricultural income and regional food security. Conventional crop protection protocols depend on late-stage reactive visual scouting, missing the optimal window for prophylactic biological or chemical interventions. Milestone 2 of the AgriForecast research project establishes a rigorous classical econometric and statistical baseline for monthly disease occurrence forecasting across 10 economically critical Indian crop pathogens. Utilizing 132 unbroken monthly observations (January 2015 – December 2025) derived from verified Global Biodiversity Information Facility (GBIF) records harmonized with NASA POWER agroclimatological reanalysis, we formulate, calibrate, and validate nine classical parametric and non-parametric time series models: Autoregressive (AR), Moving Average (MA), Autoregressive Moving Average (ARMA), Autoregressive Integrated Moving Average (ARIMA), Seasonal ARIMA (SARIMA), ARIMAX and SARIMAX with lagged microclimate covariates, Holt-Winters Triple Additive Exponential Smoothing, and Seasonal Naive persistence. Dual stationarity testing (Augmented Dickey-Fuller and KPSS) confirms that raw disease incidence series are non-stationary ($p > 0.05$ on ADF, $p < 0.01$ on KPSS) but achieve strict stationarity under first-order differencing ($d=1, p < 0.0001$). Under a leak-free chronological holdout validation protocol, parsimonious linear models demonstrated superior out-of-sample performance: ARMA achieved the lowest test MAE of 0.4844 (RMSE: 0.5854, MAPE: 59.34%), followed closely by AR(1) (MAE: 0.4856) and ARIMA(1,1,1) (MAE: 0.4953, RMSE: 0.5976). Residual diagnostics via the Ljung-Box test confirm that low-order autoregressive differencing successfully extracts linear serial correlation. However, classical models exhibit severe limitations when handling 71.8% zero-inflation and non-linear microclimatic infection thresholds, establishing a clear imperative for Milestone 3 advanced machine learning and deep learning architectures.*

**Index Terms—** Agricultural epidemiology, classical time series, Box-Jenkins ARIMA, SARIMAX, Holt-Winters, stationarity testing, Bennett University.

---

### I. Introduction

#### A. Epidemiological Background & Operational Problem
Agricultural production in the Indian subcontinent is highly vulnerable to foliar, vascular, and root-rot plant diseases driven by fungal and bacterial pathogens. In staple cereal systems, Rice Blast (*Pyricularia oryzae*) and Sheath Blight (*Rhizoctonia solani*) routinely cause 20% to 45% localized grain yield reductions under favorable weather conditions. Similarly, Stripe Rust of Wheat (*Puccinia striiformis*) threatens northern wheat bowls, while Potato Late Blight (*Phytophthora infestans*) inflicts sudden catastrophic wipeouts in sub-tropical plains.

Under conventional agricultural extension workflows, disease detection is strictly reactive. Farmers and field officers identify visible foliar chlorosis, leaf lesions, or necrotic blotches only after pathogen infection, colonization, and field sporulation are already extensive. Curative chemical applications at this stage provide suboptimal yield protection, incur high chemical costs, and exacerbate environmental and resistance risks. Early warning systems that forecast disease occurrences 1 to 6 months in advance enable state agricultural departments to pre-position fungicide inventories, mobilize biocontrol agents, and issue targeted spray advisories before exponential spore propagation commences.

#### B. Classical Time Series Paradigm
The initial milestone of time-series modeling traditionally leverages classical linear statistical econometrics founded on the Box-Jenkins methodology. Classical time-series models treat the observed epidemiological sequence as a realization of an underlying stochastic process. By decomposing the series into autoregressive persistence (memory of past infections), moving-average shocks (unobserved environmental perturbations), and seasonal periodicity (annual cropping cycles), classical models provide a transparent, mathematically rigorous foundation.

#### C. Objectives of Milestone 2
The specific technical objectives of Milestone 2 comprise:
1. Harmonize multi-source surveillance and agroclimatic datasets into an unbroken, quality-verified monthly time series.
2. Conduct diagnostic stationarity pre-testing using dual statistical tests (ADF and KPSS) to establish the necessary order of integration ($d$).
3. Formulate and fit nine classical time-series architectures spanning univariate Box-Jenkins, seasonal multiplicative models, exogenous weather regressions, exponential smoothing, and naive baselines.
4. Establish a strict out-of-sample chronological evaluation benchmark without lookahead leakage.
5. Identify the structural limitations of linear classical assumptions on zero-inflated biological series, laying the technical roadmap for Milestone 3.

---

### II. Literature Review

The application of statistical time series to biological and agricultural forecasting has evolved through foundational milestones:

1. **Box-Jenkins Autoregressive Models:** Box and Jenkins [1] formulated the unified Autoregressive Integrated Moving Average (ARIMA) methodology. In plant pathology, early studies utilized low-order ARIMA models to forecast airborne spore concentrations, establishing that spore release follows autoregressive persistence influenced by diurnal and weekly weather patterns.
2. **Seasonal Extensions & Multiplicative Decompositions:** Box, Jenkins, and Reinsel [2] extended ARIMA to seasonal formulations (SARIMA), enabling the modeling of annual agricultural cycles driven by monsoon (Kharif) and post-monsoon (Rabi) cropping seasons.
3. **Exponential Smoothing Formulations:** Holt [3] and Winters [4] developed additive and multiplicative exponential smoothing methods that dynamically update level, trend, and seasonal components without requiring formal stationarity transformations, offering computationally efficient baselines for agricultural inventory planning.
4. **Exogenous Agroclimatic Conditioning:** Madden, Hughes, and van den Bosch [5] analyzed the epidemiology of plant diseases, establishing the "disease triangle" where infection occurs only when a susceptible host, virulent pathogen, and favorable microclimate coincide. Integrating lagged meteorological regressors into ARIMAX models represents the standard classical mechanism for incorporating environmental drivers.

---

### III. Dataset Description & Preprocessing

#### A. Data Provenance & Synchronization
The authoritative dataset (`data/disease_timeseries_master.csv`) integrates verified spatial occurrence records from the Global Biodiversity Information Facility (GBIF) and European and Mediterranean Plant Protection Organization (EPPO) with surface meteorological reanalysis from the NASA Prediction of Worldwide Energy Resources (POWER) agroclimatology database.

```
+-----------------------------------------------------------------------------------------+
|                              DATASET AUDIT OVERVIEW (MILESTONE 2)                       |
+-----------------------------------------------------------------------------------------+
| Total Observations:  1,320 monthly records (132 calendar months x 10 pathogens)         |
| Temporal Coverage:   January 1, 2015 - December 31, 2025 (11 continuous annual cycles)   |
| Primary Target:      documented_occurrence_count (Integer >= 0, Mean: 1.34, Max: 74)   |
| Zero Inflation:      71.82% exact zero-count months (dry-season biological dormancy)     |
| Feature Space:       43 features (climate lags, rolling stats, cyclical month encodings)|
+-----------------------------------------------------------------------------------------+
```

#### B. Monitored Pathogen Taxa
The surveillance suite tracks 10 pathogens across staple Indian cropping systems:
- *Pyricularia oryzae* (Rice Blast) — Rice (*Oryza sativa*)
- *Bipolaris oryzae* (Brown Spot of Rice) — Rice
- *Rhizoctonia solani* (Sheath Blight of Rice) — Rice
- *Xanthomonas oryzae* (Bacterial Blight of Rice) — Rice
- *Puccinia striiformis* (Stripe / Yellow Rust of Wheat) — Wheat (*Triticum aestivum*)
- *Alternaria solani* (Early Blight) — Potato & Tomato
- *Phytophthora infestans* (Late Blight) — Potato (*Solanum tuberosum*)
- *Fusarium oxysporum* (Fusarium Wilt Complex) — Chickpea, Cotton, Tomato
- *Macrophomina phaseolina* (Dry Root Rot / Charcoal Rot) — Chickpea, Soybean, Cotton
- *Ralstonia solanacearum* (Bacterial Wilt) — Potato, Brinjal, Ginger

#### C. Preprocessing & Leakage Prevention Protocol
1. **Chronological Splitting:** The 132-month sequence is partitioned strictly forward in time:
   - *Training Split (70%):* Jan 2015 – Aug 2022 ($T = 92$ months).
   - *Validation Split (15%):* Sep 2022 – Apr 2024 ($T = 20$ months).
   - *Final Test Split (15%):* May 2024 – Dec 2025 ($T = 20$ months).
2. **Lagged Weather Conditioning:** Surface temperature ($T_{\text{mean}}$), precipitation ($P$), and relative humidity ($RH$) are strictly lagged by 1 and 2 months ($t-1, t-2$) so future weather is never leaked into training models.
3. **Scaling:** MinMaxScaler feature transformations are fitted *strictly* on training slices.

---

### IV. Exploratory Data Analysis & Stationarity Testing

#### A. Dual Stationarity Testing Framework (ADF + KPSS)
Classical Box-Jenkins models require weak stationarity (constant mean, constant variance, and autocovariance dependent solely on temporal displacement). We implemented a joint stationarity decision matrix combining the Augmented Dickey-Fuller (ADF) test ($H_0$: unit root present) and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test ($H_0$: trend-stationary):

| Pathogen / Series | Raw ADF Stat ($p$-value) | Raw KPSS Stat ($p$-value) | Joint Decision | Differencing Required | Differenced ADF ($p$-value) |
|---|---|---|---|---|---|
| *Macrophomina phaseolina* | $+0.0915$ ($p = 0.9655$) | $1.1553$ ($p = 0.0100$) | **Non-Stationary (Unit Root)** | First Difference ($d=1$) | $-9.4821$ ($p < 0.0001$) |
| *Pyricularia oryzae* (Blast) | $-1.8420$ ($p = 0.3598$) | $0.8421$ ($p = 0.0100$) | **Non-Stationary (Trend/Unit)** | First Difference ($d=1$) | $-8.7214$ ($p < 0.0001$) |
| *Fusarium oxysporum* | $-2.1104$ ($p = 0.2403$) | $0.9812$ ($p = 0.0100$) | **Non-Stationary (Unit Root)** | First Difference ($d=1$) | $-7.8940$ ($p < 0.0001$) |
| *Puccinia striiformis* | $-1.9543$ ($p = 0.3069$) | $0.7891$ ($p = 0.0100$) | **Non-Stationary (Unit Root)** | First Difference ($d=1$) | $-8.1250$ ($p < 0.0001$) |
| *Phytophthora infestans* | $-2.0411$ ($p = 0.2688$) | $0.9124$ ($p = 0.0100$) | **Non-Stationary (Unit Root)** | First Difference ($d=1$) | $-8.4512$ ($p < 0.0001$) |

**Conclusion:** All raw pathogen series fail the unit-root test ($p > 0.05$ on ADF and $p < 0.05$ on KPSS). First-order differencing ($d=1$) achieves strict stationarity ($p < 0.0001$), formally justifying integrated classical models.

#### B. Autocorrelation (ACF) and Partial Autocorrelation (PACF) Analysis
Sample correlograms with 95% Bartlett confidence bands reveal:
1. **Autoregressive Signature:** The PACF shows sharp cutoffs after lag $k=1$ or $k=2$, indicating dominant low-order autoregressive dynamics.
2. **Moving-Average Signature:** The ACF exhibits gradual exponential decay with occasional significant spikes at lag 1 and lag 12 (annual seasonal cycle).

---

### V. Classical Model Formulations & Methodology

We formulated and calibrated eight distinct classical model families:

1. **Autoregressive Model — AR($p$):**
   $$y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, \sigma^2)$$
   Calibrated with $p=1$ based on PACF truncation.

2. **Moving Average Model — MA($q$):**
   $$y_t = \mu + \varepsilon_t + \sum_{j=1}^q \theta_j \varepsilon_{t-j}$$
   Captures transient environmental shocks using past white-noise forecast errors ($q=1$).

3. **Autoregressive Moving Average — ARMA($p, q$):**
   $$y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \varepsilon_t + \sum_{j=1}^q \theta_j \varepsilon_{t-j}$$
   Combines autoregressive memory with shock absorption on stationary series.

4. **Autoregressive Integrated Moving Average — ARIMA($p, d, q$):**
   $$\Phi_p(B)(1 - B)^d y_t = \Theta_q(B)\varepsilon_t$$
   Where $B$ is the backshift operator ($B^k y_t = y_{t-k}$). Optimal specification determined as $\text{ARIMA}(1, 1, 1)$.

5. **Seasonal ARIMA — SARIMA($p,d,q$)($P,D,Q$)[$s$]:**
   $$\Phi_p(B)\tilde{\Phi}_P(B^s)(1 - B)^d (1 - B^s)^D y_t = \Theta_q(B)\tilde{\Theta}_Q(B^s)\varepsilon_t$$
   With seasonal period $s = 12$ months, capturing annual monsoon recurrence.

6. **ARIMAX and SARIMAX with Lagged Weather Regressors:**
   $$y_t = \sum_{k=1}^K \beta_k X_{k, t-1} + \frac{\Theta_q(B)}{\Phi_p(B)(1 - B)^d}\varepsilon_t$$
   Exogenous regressors $X$ include lagged surface temperature, relative humidity, and precipitation.

7. **Holt-Winters Triple Exponential Smoothing (Additive):**
   - Level: $\ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$
   - Trend: $b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$
   - Seasonality: $s_t = \gamma(y_t - \ell_t) + (1-\gamma)s_{t-m}$
   - Forecast: $\hat{y}_{t+h} = \ell_t + h b_t + s_{t+h-m}$

8. **Seasonal Naive Baseline:**
   $$\hat{y}_{t+h} = y_{t+h - 12}$$
   Standard benchmark assuming disease incidence exactly replicates the identical month from the preceding calendar year.

---

### VI. Training, Validation & Hyperparameter Selection

#### A. Model Order Selection via Information Criteria
Model orders were selected by minimizing the Akaike Information Criterion ($\text{AIC} = 2k - 2\ln\hat{L}$) and Bayesian Information Criterion ($\text{BIC} = k\ln(n) - 2\ln\hat{L}$):

| Model Candidate | $p, d, q$ | $(P, D, Q)_{12}$ | Log-Likelihood | AIC | BIC |
|---|:---:|:---:|:---:|:---:|:---:|
| AR(1) | (1, 0, 0) | - | -112.4 | 230.8 | 238.4 |
| MA(1) | (0, 0, 1) | - | -118.1 | 242.2 | 249.8 |
| ARMA(1, 1) | (1, 0, 1) | - | -108.6 | 225.2 | 235.3 |
| ARIMA(1, 1, 1) | (1, 1, 1) | - | -106.2 | 220.4 | 230.5 |
| SARIMA(1, 0, 1)(1, 0, 1) | (1, 0, 1) | (1, 0, 1) | -104.1 | 220.2 | 235.4 |
| SARIMA(1, 1, 1)(1, 1, 1) | (1, 1, 1) | (1, 1, 1) | -115.8 | 243.6 | 258.7 |

#### B. Residual Diagnostic Checks
To confirm model adequacy, standardized residuals $e_t = (y_t - \hat{y}_t) / \hat{\sigma}$ were evaluated:
- **Ljung-Box Test:** $Q = n(n+2)\sum_{k=1}^h \frac{r_k^2}{n-k} \sim \chi^2(h - p - q)$. For $\text{ARIMA}(1,1,1)$, $Q(12) = 14.21$ ($p = 0.221 > 0.05$), failing to reject the null hypothesis of white-noise residuals.
- **Normality:** Normal Q-Q plots indicated slight leptokurtosis (fat tails) due to sudden unseasonal disease surges, but no severe systematic autocorrelation remained.

---

### VII. Results & Comparative Analysis

#### A. Out-of-Sample Performance Leaderboard
All models were fitted on the training split (Jan 2015 – Aug 2022) and evaluated strictly on the out-of-sample validation/test horizons. The verified Milestone 2 comparative results are summarized below:

| Rank | Model Architecture | MAE | MAE (Std) | RMSE | MAPE (%) | sMAPE (%) | $R^2$ | MASE | Fit Time (s) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **ARMA(1, 1)** | **0.4844** | 0.0457 | **0.5854** | **59.34%** | **80.44%** | **0.1461** | **0.3918** | 0.69 |
| 2 | **AR(1)** | 0.4856 | 0.0677 | 0.5868 | 63.01% | 80.51% | 0.1052 | 0.3928 | 0.52 |
| 3 | **ARIMA(1, 1, 1)** | 0.4953 | 0.0541 | 0.5976 | 58.37% | 84.63% | 0.1315 | 0.4006 | 2.52 |
| 4 | **VAR(2)** | 0.5029 | 0.0940 | 0.6038 | 68.38% | 80.25% | 0.0158 | 0.4069 | 0.01 |
| 5 | **ARIMAX** | 0.5112 | 0.0457 | 0.6259 | 57.54% | 87.61% | 0.0601 | 0.4135 | 1.00 |
| 6 | **SARIMAX** | 0.5596 | 0.0257 | 0.7391 | 67.61% | 94.09% | -0.3021 | 0.4528 | 5.15 |
| 7 | **SARIMA** | 0.5607 | 0.0292 | 0.7320 | 64.90% | 94.09% | -0.2778 | 0.4536 | 2.01 |
| 8 | **Holt-Winters Exp. Smoothing** | 0.6406 | 0.0587 | 0.8492 | 89.27% | 109.19% | -0.7180 | 0.5182 | 0.78 |
| 9 | **Moving Average** | 0.6410 | 0.0807 | 0.7489 | 67.50% | 91.56% | -0.3602 | 0.5188 | 0.01 |
| 10 | **Meta Prophet (Baseline)** | 0.6832 | 0.1415 | 0.7762 | 75.15% | 86.29% | -0.4770 | 0.5524 | 0.96 |
| 11 | **Persistence** | 0.6845 | 0.0787 | 0.9243 | 102.29% | 108.96% | -1.0459 | 0.5539 | 0.01 |
| 12 | **MA(1)** | 0.7531 | 0.1195 | 0.9416 | 99.57% | 84.32% | -1.1147 | 0.6093 | 0.27 |
| 13 | **Seasonal Naive** | 0.8869 | 0.0781 | 1.0205 | 109.08% | 124.36% | -1.5086 | 0.7178 | 0.01 |

#### B. Discussion of Performance Dynamics
1. **Parsimony Advantage:** ARMA(1,1), AR(1), and ARIMA(1,1,1) outperformed complex seasonal architectures (SARIMA, SARIMAX) and additive Prophet. In monthly disease data characterized by high sparsity, fitting extra seasonal parameters ($P=1, Q=1, s=12$) induces parameter variance and overfitting, whereas low-order autoregression effectively captures the immediate month-to-month persistence of active fungal inocula.
2. **MASE Superiority:** All classical models achieved a Mean Absolute Scaled Error ($\text{MASE}$) below 1.0 ($\text{MASE} = 0.3918$ for ARMA), proving that classical models substantially outperform the non-seasonal one-step random walk benchmark.

---

### VIII. Agronomic & Epidemiological Insights

1. **Monsoon (Kharif) Peak Dynamics:** Cross-correlation analysis between NASA POWER precipitation ($PRECTOTCORR$) and Rice Blast (*Pyricularia oryzae*) demonstrates a strong positive correlation peaking at **Lag $t-1$** ($r = +0.58, p < 0.01$). High humidity (>85%) during July–August drives spore germination, while the 1-month lag reflects the biological latent incubation period before macroscopic symptoms appear.
2. **Winter (Rabi) Rust Dynamics:** Stripe Rust (*Puccinia striiformis*) exhibits negative correlation with temperature at Lag $t-1$ ($r = -0.42, p < 0.05$), reflecting its biological adaptation to cool winter temperatures ($10^\circ\text{C} - 18^\circ\text{C}$) in northern wheat belts.

---

### IX. Limitations of Classical Models & Roadmap for Milestone 3

Despite their mathematical transparency, classical linear models suffer from three structural shortcomings:
1. **Severe Zero-Inflation Failure:** Linear Gaussian assumptions cannot account for the fact that 71.8% of monthly records are zero. Linear models frequently predict small non-physical negative numbers during dormant dry seasons.
2. **Rigid Linearity vs. Biological Thresholds:** Fungal spore germination is non-linear: temperature and humidity below threshold levels result in zero sporulation, whereas exceeding the threshold triggers exponential disease spread. Linear regressors (ARIMAX) cannot model sharp conditional thresholds.
3. **Constant Residual Variance Assumption:** Classical ARIMA assumes homoscedastic residual variance ($\varepsilon_t \sim \text{WN}(0, \sigma^2)$). In reality, disease outbreaks exhibit massive volatility clustering during epidemics, requiring econometric volatility modeling (ARCH/GARCH).

**Milestone 3 Roadmap:** To resolve these limitations, Milestone 3 will introduce:
- Two-stage **Hurdle XGBoost** to explicitly separate zero occurrence from positive count severity.
- Deep sequence models (**PyTorch LSTM and GRU**) for capturing long-term dependencies.
- **Temporal Transformers (PatchTST)** utilizing multi-head self-attention.
- **ARCH/GARCH** volatility modeling to construct dynamic, risk-calibrated uncertainty cones.

---

### X. Conclusion

Milestone 2 successfully validated the foundational dataset and established an authoritative classical benchmark for Indian agricultural disease forecasting. Dual stationarity testing proved that differencing ($d=1$) stabilizes the stochastic mean of pathogen series. Parsimonious models (ARMA, AR, and ARIMA) proved superior among linear approaches, achieving MAE values under 0.50. These classical results establish an empirical baseline against which advanced non-linear machine learning, deep neural networks, and econometric volatility architectures will be evaluated in Milestone 3.

---

### XI. References

[1] G. E. P. Box and G. M. Jenkins, *Time Series Analysis: Forecasting and Control*. San Francisco, CA: Holden-Day, 1970.  
[2] G. E. P. Box, G. M. Jenkins, and G. C. Reinsel, *Time Series Analysis: Forecasting and Control*, 4th ed. Hoboken, NJ: John Wiley & Sons, 2008.  
[3] C. C. Holt, "Forecasting seasonals and trends by exponentially weighted moving averages," *International Journal of Forecasting*, vol. 20, no. 1, pp. 5–10, 2004 (reprinted from 1957).  
[4] P. R. Winters, "Forecasting sales by exponentially weighted moving averages," *Management Science*, vol. 6, no. 3, pp. 324–342, 1960.  
[5] L. V. Madden, G. Hughes, and F. van den Bosch, *The Study of Plant Disease Epidemics*. St. Paul, MN: APS Press, 2007.  
[6] S. J. Taylor and B. Letham, "Forecasting at scale," *The American Statistician*, vol. 72, no. 1, pp. 37–45, 2018.  
[7] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021.  
[8] R. F. Engle, "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation," *Econometrica*, vol. 50, no. 4, pp. 987–1007, 1982.  
[9] T. Bollerslev, "Generalized autoregressive conditional heteroskedasticity," *Journal of Econometrics*, vol. 31, no. 3, pp. 307–327, 1986.  
[10] J. D. Hamilton, *Time Series Analysis*. Princeton, NJ: Princeton University Press, 1994.  
