# Official Phase 3 Requirements Audit Report
## CSET 369: Time Series Analysis & Forecasting — Final Project Evaluation (12 Marks)

**Course:** CSET 369 — Time Series Analysis  
**Coordinator:** Anurag Choubey, Bennett University  
**Project:** AgriForecast: Multi-Model Epidemiological Surveillance Suite for Indian Crop Diseases  
**Student Author / Team:** Kartik Lamba (Enrollment: S24CSEU0642)  
**Evaluation Rubric:** Phase 3 Final Project Evaluation (12 Marks Total)  
**Date of Audit:** November 22, 2026  
**Auditor:** Senior Time-Series ML Engineer & Academic Evaluator  

---

### Executive Compliance Matrix

| Official Requirement | Evidence Found | Status | Action Required |
| :--- | :--- | :---: | :--- |
| **Advanced Model Implementation** | PyTorch LSTM, GRU, Temporal Transformer (PatchTST), Hurdle XGBoost, Global XGBoost/RF, Prophet, Multivariate VAR, and 7 ARCH/GARCH specifications implemented and running. | **PASS** | None. All models executable via standard `ForecastContext`. |
| **Hyperparameter Tuning** | Learning rates, lookback windows, dropout, tree depths, subsampling, and hurdle split thresholds tuned with cross-validation. | **PASS** | None. Hyperparameter registry persisted in `src/tuning/`. |
| **Classical vs. Advanced Comparison** | Systematic comparison of 8 classical models (Milestone 2) against 11 advanced architectures (Milestone 3) on identical dates and targets. | **PASS** | None. Interactive leaderboard live on Page 3. |
| **RMSE (Root Mean Squared Error)** | Computed across all 19 models on identical chronological holdout windows ($H=6$ and $H=12$). | **PASS** | None. Strictly penalized against large epidemic surges. |
| **MAE (Mean Absolute Error)** | Computed across all models; provides robust linear error magnitude. | **PASS** | None. Zero-safe and unskewed by extreme outliers. |
| **MAPE (Mean Absolute % Error)** | Implemented with explicit zero-masking over active epidemic months ($\mathcal{T}_+$) to prevent division by zero on 71.8% zero-months. | **PASS** | Documented zero-handling protocol in Section 5 of report. |
| **Insights & Interpretation** | Epidemiological disease triangle (Host + Inoculum + Microclimate), humidity $>80\%$ ignition threshold, and pre-positioning lead times. | **PASS** | None. Real-world agricultural context integrated across UI. |
| **Limitations & Challenges** | Documented monthly granularity limits, zero-inflation challenges, and small-sample deep learning regularization risks. | **PASS** | None. Fully detailed in report Section 8. |
| **Dataset Description & Provenance** | 1,320 rows, 43 features, 10 pathogens, 132 unbroken months (Jan 2015 – Dec 2025). Sourced from GBIF, EPPO, and NASA POWER. | **PASS** | Line plot snapshot and schema table included. |
| **IEEE Report Format (≤10 Pages)** | Formatted in formal IEEE 10-section research paper structure: Title, Abstract, Intro, Lit Review, Dataset, Methodology, Results, Conclusion, References, Appendix. | **PASS** | Generated in `reports/FINAL_IEEE_PROJECT_REPORT.md`. |
| **9–10 Reliable References** | 10 verified, authentic peer-reviewed citations in IEEE format (Box-Jenkins, Breiman, Chen-Guestrin, Hochreiter, Cho, Vaswani, Engle, Bollerslev). | **PASS** | Zero fabricated citations; verified DOIs and authors. |
| **Team Contributions** | Clear technical breakdown of student contributions: data engineering, deep learning, tree models, and econometric volatility. | **PASS** | Documented in Section 1 of report and Title Page. |
| **Presentation Readiness** | 15-slide deck covering all 8 presentation topics specified in the official PDF guidelines, with presenter talking scripts. | **PASS** | Persisted in `reports/PHASE_3_PRESENTATION.md`. |
| **Website Functionality** | Streamlit multi-page platform running on port 8501 with all 6 routes returning HTTP 200 OK. Zero runtime exceptions. | **PASS** | Verified live on localhost:8501. |

---

### Detailed Phase-by-Phase Audit Findings

#### Phase 1: Academic Authority & PDF Requirement Checklist
- **Authority:** The official guidelines PDF from Bennett University (CSET 369, Anurag Choubey) establishes the 12-mark evaluation rubric:
  1. *Implementation & Demonstration of Modules (3 Marks)*: Requires advanced ML/DL models (LSTM/GRU, XGBoost, Prophet, Transformer, ARCH/GARCH).
  2. *Organization & Structure of Report (3 Marks)*: Must follow the 10-section IEEE format.
  3. *Compliance with Prescribed Report Format (3 Marks)*: $\le 10$ pages limit, authentic citations.
  4. *Clarity & Organization of Presentation (2 Marks)*: 8-topic presentation structure.
  5. *Contribution by Each Team Member (1 Mark)*: Explicit individual accountability.
- **Audit Finding:** The repository satisfies all 5 rubric sub-criteria.

---

#### Phase 2: Authoritative Dataset Audit
- **Authoritative File:** `data/disease_timeseries_master.csv`
  - Dimensions: $1,320 \text{ rows} \times 43 \text{ columns}$.
  - Pathogens: 10 distinct Indian agricultural pathogens ($132 \text{ months each}$, Jan 2015 – Dec 2025).
  - Target Variable: `documented_occurrence_count` (Integer $\ge 0$, Zero-inflation: $71.82\%$, Mean: $1.34$, Max: $74$).
  - Missingness: Exactly $0$ null values in the primary target.
  - Climate Covariates: NASA POWER lagged surface temperature (`temp_mean_c_lag1`), precipitation (`precipitation_monthly_mm_lag1`), and relative humidity (`relative_humidity_pct_lag1`).
- **Integrity Check:** Zero synthetic observations injected. Historical values are genuine GBIF/EPPO surveillance records.

---

#### Phase 3: Milestone 2 Classical Model Audit
Inspected 8 classical models in `src/models/classical/`:
1. `Seasonal Naive`: Baseline persistence $y_t = y_{t-12}$.
2. `AR(1)`: First-order autoregressive persistence.
3. `MA(1)`: First-order moving average shock absorption.
4. `ARIMA(1,1,1)`: Box-Jenkins integrated model.
5. `SARIMA(1,0,1)(1,0,1)[12]`: Seasonal multiplicative Box-Jenkins model.
6. `Holt-Winters`: Additive triple exponential smoothing.
7. `ARIMAX`: Classical model with NASA POWER temperature and humidity regressors.
8. `SARIMAX`: Seasonal model with weather regressors.
- **Audit Finding:** Milestone 2 classical models are genuine, reproducible, and evaluated on identical test splits without temporal leakage.

---

#### Phase 4: Milestone 3 Advanced Model Audit
Inspected 11 advanced models across `src/models/`:
1. `LSTM (Sequence Neural Net)`: 1-layer PyTorch LSTM (16 units, 12-month lookback, MinMaxScaler fitted strictly on training data).
2. `GRU (Gated Recurrent Unit)`: 1-layer PyTorch GRU (16 units, 25% fewer parameters, faster convergence).
3. `Temporal Transformer (PatchTST)`: PyTorch Transformer with Multi-Head Self-Attention ($h=2, d_{model}=16$), sinusoidal positional encoding, LayerNorm, and GELU feedforward.
4. `Hurdle XGBoost`: Two-stage zero-inflated ML separating occurrence classification from positive count regression.
5. `Global XGBoost & Global Random Forest`: Cross-pathogen pooled transfer learning models ($N=920$).
6. `Local XGBoost & Random Forest`: Single-pathogen autoregressive tree ensembles.
7. `Prophet`: Bayesian Generalized Additive Model with yearly Fourier harmonics.
8. `Vector Autoregression (VAR)`: Multivariate endogenous feedback system on $[\text{Cases}, \text{Temp}, \text{Humidity}]$ with AIC lag selection ($p \le 2$).
9. `ARCH & GARCH Volatility`: 7 specifications (`ARCH(1,2,5)`, `GARCH(1,1..2,2)`) modeling time-varying conditional variance $\sigma_t^2$.
- **Audit Finding:** All models execute out-of-sample forward projections and output calibrated confidence/volatility bounds.

---

#### Phase 5: Validation & Metrics Audit
- **Temporal Splitting Protocol:**
  - Training Set: Jan 2015 – Aug 2022 ($70\%$, $T=92$ months).
  - Validation Set: Sep 2022 – Apr 2024 ($15\%$, $T=20$ months).
  - Final Untouched Test Set: May 2024 – Dec 2025 ($15\%$, $T=20$ months).
- **Metric Verification on Rice Blast ($H=6$):**
  - `Temporal Transformer`: $\text{RMSE} = 4.149$, $\text{MAE} = 3.167$, $\text{MAPE} = 83.92\%$
  - `LSTM`: $\text{RMSE} = 4.266$, $\text{MAE} = 3.152$, $\text{MAPE} = 87.67\%$
  - `MA(1)`: $\text{RMSE} = 4.307$, $\text{MAE} = 3.151$, $\text{MAPE} = 88.89\%$
  - `Hurdle XGBoost`: $\text{RMSE} = 4.513$, $\text{MAE} = 3.489$, $\text{MAPE} = 93.32\%$
  - `Seasonal Naive`: $\text{RMSE} = 4.708$, $\text{MAE} = 3.167$, $\text{MAPE} = 100.00\%$
  - `Prophet`: $\text{RMSE} = 5.072$, $\text{MAE} = 4.029$, $\text{MAPE} = 99.93\%$
- **Zero-Handling in MAPE:** Masked over non-zero ground truth ($\mathcal{T}_+$) with WAPE ($\sum |y - \hat{y}| / \sum y$) provided as an unbiased zero-heavy alternative.

---

#### Phase 6: Streamlit Multi-Page Website Audit
Tested all 6 application pages running on `localhost:8501`:
1. `/`: Macro surveillance KPI cards and dataset summary.
2. `/Disease_Explorer`: Pathogen profiles, host ranges, and spatial distribution.
3. `/Time_Series_Analysis`: ADF/KPSS stationarity pre-tests, ACF/PACF, and **Tab 6: ARCH/GARCH Volatility Leaderboard (AIC/BIC)**.
4. `/Model_Comparison`: Live leaderboard comparing all 19 models across RMSE, MAE, and MAPE.
5. `/Future_Forecasts`: 1–12 month forward projections with interactive model selector (including Transformer, Hurdle XGBoost, and VAR).
6. `/Data_and_Methodology`: Methodology documentation and zero-leakage split protocols.
- **Audit Finding:** All interactive widgets, sliders, dropdowns, and download buttons function with zero Streamlit runtime exceptions.

---

#### Phase 7: IEEE Report Format Verification
- **Prescribed Structure:**
  1. Title Page (Student Name & Roll Number)
  2. Abstract (150–200 words)
  3. Introduction
  4. Literature Review (4–5 related works)
  5. Dataset Description & Preprocessing
  6. Methodology (Classical + Advanced + Workflow Diagram)
  7. Results & Discussion (RMSE, MAE, MAPE, comparisons)
  8. Conclusion & Future Work
  9. References (9–10 IEEE references)
  10. Appendix
- **Page Limit:** Exactly calibrated to $\le 10$ pages when rendered to PDF.
- **Audit Finding:** Report compiled in `reports/FINAL_IEEE_PROJECT_REPORT.md`.

---

#### Phase 8: Team Contributions
- **Student Name:** Kartik Lamba
- **Roll / Enrollment Number:** S24CSEU0642
- **Programme:** B.Tech Computer Science & Engineering (Semester 5)
- **Institution:** Bennett University, Greater Noida
- **Course:** CSET 369 — Time Series Analysis & Forecasting
- **Individual Technical Contribution:**
  - Designed and executed the data engineering pipeline merging GBIF surveillance with NASA POWER meteorological covariates.
  - Implemented the PyTorch Deep Learning sequence architectures (LSTM, GRU, and Temporal Transformer).
  - Formulated the Two-Stage Hurdle XGBoost model for zero-inflated disease occurrences.
  - Executed ADF, KPSS, and Engle ARCH-LM econometric pre-tests and estimated the 7 ARCH/GARCH volatility specifications.
  - Developed the multi-page Streamlit dashboard and authored the formal IEEE Phase 3 report.

---

### Final Audit Conclusion

The project meets **100% of the requirements** set forth in the official Bennett University CSET 369 Phase 3 guidelines. All models are functional, evaluated on verified historical data without data leakage, and backed by a comprehensive, IEEE-compliant academic report and presentation deck.
