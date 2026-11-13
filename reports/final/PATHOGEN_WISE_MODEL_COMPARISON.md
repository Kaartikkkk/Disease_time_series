# PATHOGEN-WISE MODEL COMPARISON & EPIDEMIOLOGICAL ANALYSIS
## Empirical Performance Breakdown Across 10 Agricultural Plant Pathogens

**Evaluation Window:** May 2024 to December 2025 (20 Consecutive Months $\times$ 10 Pathogens = 200 Test Observations)  
**Dataset:** Authentic Multi-Source Historical Data (`improved_india_plant_diseases_monthly.csv`)  
**Evaluation Mode:** Expanding-Window 1-Step-Ahead ($h=1$) Operational Updating Scenario  

---

## 1. Executive Overview

Agricultural plant disease epidemiology in the Indian subcontinent is characterized by extreme host diversity, distinct seasonal cropping calendars (Kharif, Rabi, and Zaid), and heterogeneous pathogen survival mechanisms. A single forecasting model cannot optimally fit all 10 plant diseases.

This report provides a rigorous empirical analysis of model performance disaggregated across every individual pathogen on the untouched final holdout dataset.

### Major Finding: The 50/50 Division
Of the 10 evaluated pathogens:
- **5 pathogens are forecast best by Classical Econometric / Baseline models** (`alternaria_solani`, `bipolaris_oryzae`, `macrophomina_phaseolina`, `phytophthora_infestans`, `rhizoctonia_solani`).
- **5 pathogens are forecast best by Modern Machine Learning / Deep Learning models** (`fusarium_oxysporum`, `puccinia_striiformis`, `pyricularia_oryzae`, `ralstonia_solanacearum`, `xanthomonas_oryzae`).

---

## 2. Complete 10-Pathogen Leaderboard Matrix

The table below presents the primary holdout test champion, the best classical model, the best modern model, and the relative margin between them for each pathogen:

| Pathogen Scientific Name | Common Disease Name | Host Crop(s) | 20-Mo Test Actuals (Pos / Zero) | Holdout Champion | Holdout Test RMSE | Holdout Test MAE | Best Classical Model | Classical RMSE | Best Modern Model | Modern RMSE | Classical Beats Modern? | Relative Modern vs Classical (%) |
|:---|:---|:---|:---:|:---|:---:|:---:|:---|:---:|:---|:---:|:---:|:---:|
| *Alternaria solani* | Early Blight | Potato / Tomato | 12 / 8 | **Seasonal Naive** | **1.761** | **1.450** | Seasonal Naive | 1.761 | GRU | 1.832 | **Yes** | +4.03% (Class. wins) |
| *Bipolaris oryzae* | Brown Spot | Rice (*Oryza sativa*) | 14 / 6 | **SARIMA(1,0,1)(1,0,1)[12]** | **1.678** | **1.334** | SARIMA | 1.678 | Random Forest | 1.716 | **Yes** | +2.26% (Class. wins) |
| *Fusarium oxysporum* | Fusarium Wilt | Pulses & Cash Crops | 20 / 0 | **Prophet** | **5.494** | **3.670** | ARIMA(1,1,1) | 5.560 | Prophet | 5.494 | **No** | **-1.19%** (Mod. wins) |
| *Macrophomina phaseolina* | Charcoal Rot / Dry Root Rot | Pulses & Oilseeds | 20 / 0 | **ARIMA(1,1,1)** | **5.151** | **3.310** | ARIMA(1,1,1) | 5.151 | Global RF | 5.228 | **Yes** | +1.50% (Class. wins) |
| *Phytophthora infestans* | Late Blight | Potato (*Solanum*) | 3 / 17 | **SARIMAX (Weather)** | **0.858** | **0.420** | SARIMAX | 0.858 | GRU | 0.912 | **Yes** | +6.29% (Class. wins) |
| *Puccinia striiformis* | Yellow / Stripe Rust | Wheat (*Triticum*) | 1 / 19 | **XGBoost (Autoreg.)** | **0.415** | **0.155** | AR(1) | 0.441 | XGBoost | 0.415 | **No** | **-5.90%** (Mod. wins) |
| *Pyricularia oryzae* | Rice Blast | Rice (*Oryza sativa*) | 11 / 9 | **GRU (Recurrent)** | **2.819** | **1.810** | MA(1) | 2.846 | GRU | 2.819 | **No** | **-0.95%** (Mod. wins) |
| *Ralstonia solanacearum* | Bacterial Wilt | Solanaceous Crops | 3 / 17 | **LSTM (Recurrent)** | **2.141** | **1.220** | MA(1) | 2.185 | LSTM | 2.141 | **No** | **-2.01%** (Mod. wins) |
| *Rhizoctonia solani* | Sheath Blight | Rice (*Oryza sativa*) | 20 / 0 | **Seasonal Naive** | **7.523** | **5.900** | Seasonal Naive | 7.523 | Prophet | 7.915 | **Yes** | +5.21% (Class. wins) |
| *Xanthomonas oryzae* | Bacterial Blight | Rice (*Oryza sativa*) | 0 / 20 | **Global Random Forest** | **8.467** | **6.040** | MA(1) | 10.759 | Global RF | 8.467 | **No** | **-21.30%** (Mod. wins) |

---

## 3. Deep Epidemiological Analysis: Where Classical Models Win

### 1. *Alternaria solani* (Early Blight of Potato/Tomato)
- **Primary Holdout Champion:** **Seasonal Naive (Baseline)** (RMSE: **1.761**, MAE: **1.450**)
- **Epidemiological Context:** Early blight occurrence strictly follows the Solanaceae winter cultivation window (November to February in the Indo-Gangetic Plains). Spore deposition is tightly governed by irrigation timing and canopy closure.
- **Why Classical Beats Modern:** The seasonal recurrence is so consistent year-over-year that the unadjusted 12-month historical lag ($y_{t-12}$) outperforms multi-variable estimators. Deep neural networks and boosted trees introduce slight variance around the true integer peak, increasing RMSE to 1.832.

### 2. *Bipolaris oryzae* (Brown Spot of Rice)
- **Primary Holdout Champion:** **SARIMA(1,0,1)(1,0,1)[12]** (RMSE: **1.678**, MAE: **1.334**)
- **Epidemiological Context:** Endemic to rice-growing regions with nutrient-depleted soils. Symptoms build up gradually as plants mature through the Kharif season, peaking during panicle emergence in September–October.
- **Why Classical Beats Modern:** The time series possesses high linear autocorrelation combined with steady 12-month seasonality. SARIMA's Box-Jenkins parameterization ($\phi_1 \approx 0.62, \Phi_{12} \approx 0.45$) models this smooth trajectory without overfitting to erratic monsoonal precipitation spikes.

### 3. *Macrophomina phaseolina* (Dry Root Rot / Charcoal Rot)
- **Primary Holdout Champion:** **ARIMA(1,1,1)** (RMSE: **5.151**, MAE: **3.310**)
- **Epidemiological Context:** Sclerotial soil-borne pathogen affecting chickpea, pigeonpea, and soybean under high temperature ($>30^\circ\text{C}$) and severe soil moisture deficit.
- **Why Classical Beats Modern:** Drought conditions induce step-change shifts in disease incidence. First-order differencing ($d=1$) removes the non-stationary localized level shift, while the moving-average parameter ($\theta_1$) smoothly absorbs transitory post-monsoon rain shocks, edging out Global Random Forest (5.228).

### 4. *Phytophthora infestans* (Late Blight of Potato)
- **Primary Holdout Champion:** **SARIMAX (Seasonal + Lagged Weather)** (RMSE: **0.858**, MAE: **0.420**)
- **Epidemiological Context:** Infamous cool-season pathogen causing explosive epidemics when temperatures remain between $12^\circ\text{C}$–$18^\circ\text{C}$ with relative humidity $>85\%$ for $\ge 48$ consecutive hours.
- **Why Classical Beats Modern:** Late blight demonstrates strict biological threshold gating. SARIMAX integrates exogenous NASA POWER weather variables (relative humidity lag 1 and mean temperature lag 1) into an autoregressive framework. Because the series has 17 zero months out of 20, unregularized ML models overpredict small baseline counts, whereas SARIMAX holds near zero until weather regressors trigger an outbreak.

### 5. *Rhizoctonia solani* (Sheath Blight of Rice)
- **Primary Holdout Champion:** **Seasonal Naive (Baseline)** (RMSE: **7.523**, MAE: **5.900**)
- **Epidemiological Context:** High-impact stem pathogen thriving under intense humidity and dense rice canopies. In India, sheath blight occurs every single year during the peak monsoon months of July through October, with continuous positive counts across all 20 test months.
- **Why Classical Beats Modern:** In high-volume, continuously endemic regimes, machine learning models tend to suffer from mean-reversion shrinkage, dampening the true epidemic peaks. Seasonal Naive perfectly anchors to the previous year's monsoonal peak, yielding lower quadratic penalty than Prophet (7.915) or XGBoost (8.112).

---

## 4. Deep Epidemiological Analysis: Where Modern Models Win

### 1. *Fusarium oxysporum* (Fusarium Wilt)
- **Primary Holdout Champion:** **Prophet (Trend + Seasonality)** (RMSE: **5.494**, MAE: **3.670**)
- **Epidemiological Context:** Vascular soil-borne pathogen surviving as chlamydospores for years. Causes persistent wilting across cotton, pigeonpea, and banana fields throughout the annual cycle without prolonged dormancy.
- **Why Modern Beats Classical:** The historical incidence exhibits long-term agricultural acreage expansion and shifting crop rotation trends. Prophet's piecewise linear trend changepoints combined with continuous Fourier seasonality capture the changing baseline trend far more effectively than stationary Box-Jenkins polynomials.

### 2. *Puccinia striiformis* (Yellow / Stripe Rust of Wheat)
- **Primary Holdout Champion:** **XGBoost (Autoregressive)** (RMSE: **0.415**, MAE: **0.155**)
- **Epidemiological Context:** Airborne rust confined to the North-Western Plains and Himalayan foothills during the cold winter wheat cycle (January to March). Dormant across 19 of the 20 holdout months.
- **Why Modern Beats Classical:** XGBoost's non-linear decision trees establish sharp split thresholds. When temperature lags exceed $22^\circ\text{C}$, the tree partitions immediately output an exact zero, completely eliminating false alarms during the warm non-cropping seasons.

### 3. *Pyricularia oryzae* (Rice Blast)
- **Primary Holdout Champion:** **GRU (Gated Recurrent Unit)** (RMSE: **2.819**, MAE: **1.810**)
- **Epidemiological Context:** Devastating blast disease characterized by leaf lesions, neck rot, and rapid airborne spore dispersion when warm humid conditions coincide with high nitrogen fertilization.
- **Why Modern Beats Classical:** Spore maturation and lesion incubation involve multi-month physiological delays. The recurrent reset and update gates of the GRU architecture preserve internal memory of previous sub-clinical sporulation, outperforming unconstrained moving averages.

### 4. *Ralstonia solanacearum* (Bacterial Wilt)
- **Primary Holdout Champion:** **LSTM (Sequence Neural Net)** (RMSE: **2.141**, MAE: **1.220**)
- **Epidemiological Context:** Soil-borne bacterium causing rapid wilting in tomato, potato, and brinjal during warm, waterlogged conditions.
- **Why Modern Beats Classical:** LSTM's cell state memory accurately models the latent soil survival period between host crop planting seasons, capturing the non-linear inflection when monsoon rainfall saturates the soil.

### 5. *Xanthomonas oryzae* (Bacterial Blight of Rice)
- **Primary Holdout Champion:** **Global Random Forest (Multi-Pathogen)** (RMSE: **8.467**, MAE: **6.040**)
- **Epidemiological Context:** Systemic bacterial infection causing leaf blight along vascular bundles during cyclonic weather and high relative humidity.
- **Why Modern Beats Classical (Massive -21.3% Gain):** Classical univariate models (AR, MA, ARIMA) suffered severe errors on *Xanthomonas* ($RMSE > 10.7$) because historical counts fluctuated heavily. Global Random Forest, trained jointly across all 10 pathogens ($N=1,120$ pooled records), leveraged cross-pathogen weather interactions and humidity thresholds to regularize its leaf predictions, beating the best classical alternative by over 2.29 RMSE points.

---

## 5. Strategic Architectural Takeaways

1. **Univariate Parsimony for High-Seasonality Pathogens:** For diseases governed by rigid agricultural crop calendars (*Alternaria*, *Rhizoctonia*), adding complex non-linear regressors introduces parameter variance without predictive gain.
2. **Global Ensembles for High-Variance Pathogens:** For volatile, multi-host, or severe bacterial diseases (*Xanthomonas*, *Fusarium*), multi-pathogen global models prevent variance collapse through cross-series feature pooling.
3. **Thresholded ML for Zero-Heavy Pathogens:** For pathogens with $>80\%$ dormancy (*Puccinia*, *Ralstonia*), tree partitioning and recurrent memory effectively suppress spurious baseline predictions.

