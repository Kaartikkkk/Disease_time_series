# COMPLETE WEBSITE DATASET AUDIT AND FULL MIGRATION REPORT

**Authoritative Single Source of Truth:** `data/disease_timeseries_master.csv`  
**Cryptographic SHA-256:** `c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67`  
**Audit Status:** **VERIFIED & OPERATIONAL ACROSS ENTIRE PLATFORM**  
**Automated Verification Suite:** `scripts/verify_entire_website_dataset.py` (ALL CHECKS PASSED)  
**Test Suite Execution:** **108 Passed, 0 Failed, 0 Skipped**  

---

## A. Complete Website Inventory

The Agricultural Disease Time Series Forecasting platform consists of the main application file and five dedicated functional pages, structured as follows:

1. **Main Landing Application (`app.py`)**:
   - Editorial Header & Live Telemetry Summary (`Jan 2015 – Dec 2025`, `1,320 Records`, `10 Pathogens`).
   - 4 Dynamic KPI Metric Cards (*Continuous Timeline*, *Target Crop Diseases*, *Zero-Inflation Baseline 71.82%*, *Total Documented Records*).
   - Direct Navigation Shortcuts into all sub-modules.
   - Dual-Pathogen Macro Trajectory Preview Chart (*Rice Blast* vs. *Brown Spot of Rice*).
   - Global Architectural Pipeline Map.

2. **Page 1 — Disease Records & Field Surveillance (`pages/1_Disease_Explorer.py`)**:
   - Filter Toolbar: Target Plant Disease dropdown (10 pathogens), Agro-Climatic Region selector, Observation Date Range picker.
   - 4 KPI Metric Cards: Documented Occurrences, Observed Span (132 Months), Active Agro-Climatic Zones, Primary Host Crop.
   - **Tab 1: Historical Timeline:** Discrete monthly occurrence bar chart & 3-month rolling epidemiological trendline.
   - **Tab 2: Seasonal Cycle:** 11-year calendar-month aggregation profile (Jan–Dec) identifying intra-annual peak windows.
   - **Tab 3: Geospatial Map:** Interactive map centered on India plotting agro-climatic regional centroids (`national_composite`, `odisha_bengal`, `punjab_haryana`, `tamil_nadu`, `maharashtra`) with occurrence magnitude bubbles and climatic telemetry.
   - **Tab 4: Recorded Reports Table:** Searchable monthly observation grid with direct CSV export and dataset provenance card.

3. **Page 2 — Time Series Analysis & Statistical Diagnostics (`pages/2_Time_Series_Analysis.py`)**:
   - Filter Toolbar & 4 Telemetry Metrics (Span, Active Months, Monthly Mean, Sample Std Dev).
   - **Tab 1: Trajectory & Trend:** Longitudinal raw observations with adjustable rolling smoothing window.
   - **Tab 2: Seasonal Decomposition:** Classical Seasonal-Trend-Loess (STL) 4-panel decomposition (Observed, Trend, Seasonal, Residual).
   - **Tab 3: Stationarity & Differencing:** Augmented Dickey-Fuller (ADF) & KPSS tests, with original vs. differenced series plots.
   - **Tab 4: Autocorrelation (ACF / PACF):** Sample autocorrelation and partial autocorrelation bar charts up to lag 24 with 95% Bartlett confidence limits.
   - **Tab 5: Weather Cross-Correlation:** Pearson cross-correlation matrix heatmap relating disease occurrence to NASA POWER agrometeorological lags.
   - **Tab 6: ARCH / GARCH Volatility Suite:** ARCH LM heteroskedasticity test, comparative parameter table across all 7 specifications (`ARCH(1)`, `ARCH(2)`, `ARCH(5)`, `GARCH(1,1)`, `GARCH(1,2)`, `GARCH(2,1)`, `GARCH(2,2)`), dynamic in-sample shock envelope ($\pm 2\sigma_t$), and 12-month future volatility projections ($\sigma_{t+h}$).

4. **Page 3 — Model Comparison & Benchmark Leaderboard (`pages/3_Model_Comparison.py`)**:
   - Validation Protocol Switcher: Fixed Chronological Holdout vs. Multi-Origin Validation vs. Operational Walk-Forward.
   - **Stage 1 (Top): Academic Synthesis & Benchmark Winner:** Prominent Overall Winner banner card (e.g. *Hurdle XGBoost*) with test-window RMSE badge and Three Key Scientific Takeaways.
   - **Stage 2: Classical Statistical Baselines:** Box-Jenkins econometrics leaderboard table, horizontal RMSE bar chart, and differencing limitations takeaway.
   - **Stage 3: Advanced & Modern Forecasting Models:** Machine learning, tree ensembles, and deep learning leaderboard table, horizontal RMSE bar chart, and non-linear climate coupling takeaway.
   - **Stage 4: Actual Occurrences vs. Predictions:** Direct ground-truth holdout trajectory overlay comparing verified field cases against top classical and top advanced forecasters.
   - **Stage 5: Direct Head-to-Head Comparison:** 4 KPI comparison cards, side-by-side grouped error bar charts, structured evaluation matrix table, and complete leaderboard CSV export.

5. **Page 4 — Future Disease Forecasts (`pages/4_Future_Forecasts.py`)**:
   - Interactive Forecast Configuration: Pathogen selector, 17 model choices, forecast horizon slider (3 to 12 months ahead).
   - Grounding Callout: Clear indication of the verified historical database cutoff (December 2025) and forward projection window (January 2026 onwards).
   - Future Meteorological Scenario Generator: Weather regressor simulations for exogenous modeling.
   - Out-of-Sample Forecast Visualizer: Historical recent curve, future projection line, and 95% confidence interval envelope.
   - Clinical Actionability & Epidemiological Alert Panel: Automated advisory thresholds for agricultural field response.
   - Multi-Disease Surveillance Matrix: Cross-pathogen outbreak risk heatmap for state agricultural planners.
   - Export Forecast Data: Downloadable projection CSV.

6. **Page 5 — Data Provenance, Architecture & Methodology (`pages/5_Data_and_Methodology.py`)**:
   - 8 Academic Reference Tabs: Primary Data Sources, Dataset Coverage & Reconciliation, Preprocessing & Zeros, End-to-End Methodology, Model Architectures, Evaluation & Validation Schemes, Limitations, and Downloads & Citations.
   - Direct download links for the authoritative master dataset and model validation registries.

---

## B. Data Source & Dependency Map

Every data-loading call across the entire application was traced to ensure strict adherence to the single source of truth:

| Component / Page | Original Data Source | Final Data Source | Status |
| :--- | :--- | :--- | :--- |
| **`app.py` Header & KPI Cards** | Dynamic project telemetry | `src.project_stats.get_project_telemetry()` $\rightarrow$ `data/disease_timeseries_master.csv` | **VERIFIED** |
| **`app.py` Macro Trajectory Chart** | Direct slice | `src.ui_helpers.load_monthly_series()` $\rightarrow$ `data/disease_timeseries_master.csv` | **VERIFIED** |
| **Page 1 Toolbar & Pathogen Filter** | `config/settings.yaml` | Validated against `data/disease_timeseries_master.csv` | **VERIFIED** |
| **Page 1 Historical Timeline (Tab 1)** | `load_clean_records()` | `src.ui_helpers.load_monthly_series()` $\rightarrow$ `data/disease_timeseries_master.csv` | **MIGRATED & VERIFIED** |
| **Page 1 Seasonal Profile (Tab 2)** | Static audit JSON | Computed directly from `filtered_df` (`disease_timeseries_master.csv`) | **MIGRATED & VERIFIED** |
| **Page 1 Geospatial Map (Tab 3)** | `clean_*_records.csv` (10 files) | Regional centroids mapped directly from `disease_timeseries_master.csv` | **MIGRATED & VERIFIED** |
| **Page 1 Data Table & Export (Tab 4)** | Raw event CSVs | Direct monthly records from `disease_timeseries_master.csv` | **MIGRATED & VERIFIED** |
| **Page 2 All Time Series Tabs (1-5)** | `load_monthly_series()` | `src.ui_helpers.load_monthly_series()` $\rightarrow$ `data/disease_timeseries_master.csv` | **VERIFIED** |
| **Page 2 ARCH/GARCH Volatility (Tab 6)**| `arch_garch_model.py` | Fitted directly on master series from `disease_timeseries_master.csv` | **VERIFIED** |
| **Page 3 Fixed Holdout Benchmark** | `run_model_comparison()` | Sourced directly from `disease_timeseries_master.csv` | **VERIFIED** |
| **Page 3 Multi-Origin Benchmark** | `multi_origin_validation_results.json` | Generated strictly from `disease_timeseries_master.csv` (SHA-256 verified) | **VERIFIED** |
| **Page 3 Winner Card & Takeaways** | Dynamic holdout metrics | Computed dynamically from holdout run on `disease_timeseries_master.csv` | **VERIFIED** |
| **Page 4 Future Projections** | Direct model execution | Fitted on full series from `disease_timeseries_master.csv` (2015–2025) | **VERIFIED** |
| **Page 4 Multi-Disease Watch Matrix** | Multi-pathogen forecast loop | Loads each pathogen from `disease_timeseries_master.csv` | **VERIFIED** |
| **Page 5 Dataset Metrics & Downloads**| Path / Telemetry references | `data/disease_timeseries_master.csv` and verified registry JSONs | **VERIFIED** |

---

## C. Authoritative Master Dataset Verification

An independent cryptographic and tabular inspection confirmed all structural invariants:

| Attribute | Verified Actual Specification |
| :--- | :--- |
| **Canonical File Path** | `data/disease_timeseries_master.csv` |
| **Cryptographic SHA-256** | `c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67` |
| **Total Record Count** | **1,320 Monthly Records** |
| **Feature Dimensions** | **43 Columns** |
| **Canonical Pathogens** | **10 Pathogens** (132 continuous months per pathogen) |
| **Temporal Span** | **2015-01-01 to 2025-12-01 (11 Full Contiguous Years / 132 Months)** |
| **Target Variable** | `documented_occurrence_count` (Zero-bounded non-negative integer) |
| **Missing Values** | **0 Missing Values (0.00%) across all 43 columns** |
| **Target Summary** | Min: 0, Max: 74, Mean: 1.342, Median: 0.0, Zero Proportion: 71.82% |

---

## D. Model Architecture & Provenance Verification

All 17 model implementations use the master dataset as their sole data source:

1. **Box-Jenkins Classical Econometrics (Univariate)**:
   - `Seasonal Naive (Baseline)`: Benchmarked with 12-month lag persistence.
   - `AR(1)`: Autoregressive formulation fitted via conditional sum-of-squares.
   - `MA(1)`: Moving average shock formulation fitted via maximum likelihood.
   - `ARIMA(1,1,1)`: Classical integrated autoregressive moving average.
   - `SARIMA(1,0,1)(1,0,1)[12]`: Seasonal multiplicative Box-Jenkins.
   - `Holt-Winters Exponential Smoothing`: Additive seasonal smoothing.
2. **Climate-Augmented Classical Models (Multivariate)**:
   - `ARIMAX (Weather Regressors)`: Exogenous temperature, rainfall, and humidity regressors.
   - `SARIMAX (Seasonal + Weather)`: Full seasonal autoregressive with meteorological transfer functions.
3. **Machine Learning Architectures**:
   - `Random Forest (Autoregressive)`: Local lag-feature ensemble.
   - `XGBoost (Autoregressive)`: Gradient-boosted decision trees.
   - `Global Random Forest (Multi-Pathogen)`: Cross-pathogen pooled panel model.
   - `Global XGBoost (Multi-Pathogen)`: Cross-pathogen pooled gradient boosting.
   - `Hurdle XGBoost (Classifier + Regressor)`: Two-stage zero-dormancy isolation architecture.
4. **Deep Learning Sequence Models**:
   - `LSTM (Sequence Neural Net)`: PyTorch recurrent neural net with sequence lookback ($T=6$).
   - `GRU (Gated Recurrent Unit)`: Fast recurrent neural net with sequence lookback ($T=6$).
5. **Bayesian Probabilistic Modeling**:
   - `Prophet (Trend + Seasonality)`: Generalized additive model with calendar and weather regressors.
6. **Volatility & Risk Architectures**:
   - `ARCH(1)`, `ARCH(2)`, `ARCH(5)`: Conditional autoregressive heteroskedasticity.
   - `GARCH(1,1)`, `GARCH(1,2)`, `GARCH(2,1)`, `GARCH(2,2)`: Generalized autoregressive conditional heteroskedasticity with Normal and Student-t innovations.

---

## E. Prediction & Evaluation Verification

All walk-forward prediction logs and evaluation metric files have been verified against the master dataset:
- `data/processed/model_training_and_validation_registry.json`: Registers all 17 models, hyperparameter configurations, validation metrics, and test metrics with cryptographic dataset fingerprint `c7bf42422df9...`.
- `data/processed/FINAL_MODEL_COMPARISON_RESULTS.json`: Multi-origin evaluation results across 4 chronological origins (240 out-of-sample points) strictly from `disease_timeseries_master.csv`.
- `data/processed/final_test_evaluation_results.json`: Untouched holdout test evaluation (May 2024 to December 2025; 200 points).
- `data/processed/arch_garch_evaluation_results.json`: Complete parameter fits, AIC/BIC rankings, and volatility walk-forward evaluations.

---

## F. Visualization & Interactive Component Audit

Every graph across the application updates dynamically when the user selects a pathogen or alters a filter:

1. **Page 1 (Disease Explorer)**:
   - Monthly Occurrence Trajectory: Bar heights and 3-month rolling average line update dynamically from master series.
   - Annual Seasonal Distribution: Grouped calendar-month aggregation updates dynamically for the selected pathogen.
   - Agro-Climatic Regional Map: Centroids and regional occurrence summaries reflect regional entries from the master dataset.
2. **Page 2 (Time Series Analysis)**:
   - Longitudinal Trajectory & Moving Averages: Sourced from master occurrence count.
   - STL Decomposition: Decomposes the active pathogen's 132-month master series.
   - ADF / KPSS Stationarity Plots: Computed from active series.
   - ACF / PACF Bar Charts: Calculated dynamically up to lag 24 with Bartlett confidence intervals.
   - Weather Heatmap: Pearson cross-correlations computed directly between pathogen occurrences and NASA POWER climate features.
   - ARCH/GARCH Dynamic Envelope: In-sample $\pm 2\sigma_t$ shock envelope and 12-month forward volatility projections update per model selection.
3. **Page 3 (Model Comparison)**:
   - Classical and Modern Horizontal Bar Charts: Rendered dynamically from holdout evaluation metrics.
   - Actual vs. Predicted Curve: Ground truth line strictly overlays the verified test-window counts.
   - Direct Head-to-Head Grouped Bar Chart: Compares top classical vs top advanced RMSE and MAE.
4. **Page 4 (Future Forecasts)**:
   - Trajectory Projection Chart: Seamlessly connects historical observations (2015–2025) with forward forecasts (2026) and 95% confidence bands.
   - Multi-Disease Watch Matrix: Evaluates all 10 pathogens concurrently across future months.

---

## G. Downloads, Exports & Cache Management

1. **Download Buttons Audited**:
   - `pages/1_Disease_Explorer.py` (Tab 4): Exports the filtered monthly series directly from `data/disease_timeseries_master.csv`.
   - `pages/3_Model_Comparison.py` (Stage 5): Exports the complete model evaluation leaderboard table.
   - `pages/4_Future_Forecasts.py` (Bottom): Exports out-of-sample monthly forecast values with 95% upper and lower prediction bounds.
   - `pages/5_Data_and_Methodology.py` (Tab 8): Downloads the full authoritative master dataset (`disease_timeseries_master.csv`) and model validation registries.
2. **Cache Management**:
   - All dataset loading routes through `src/ui_helpers.py` (`load_monthly_series`), which fetches directly from `src/data/dataset_service.py`.
   - Streamlit in-memory cache (`@st.cache_data`) invalidates automatically upon session reload, ensuring no stale data persists.

---

## H. Issues Corrected During Audit

1. **Page 1 Redundant Dependency Elimination**:
   - Replaced dependency on 10 separate `data/processed/clean_*_records.csv` files with direct extraction from `data/disease_timeseries_master.csv`.
   - Deleted all 10 `clean_*_records.csv` files and purged `data/raw/`.
2. **Page 2 Diagnostic Visualization Fixes**:
   - Fixed STL decomposition unpacking where a DataFrame was previously accessed with `.get("success")`.
   - Fixed NASA POWER cross-correlation heatmap where correlation dictionary was accessed with `["correlation_matrix"]` instead of pivoting `["correlations"]`.
   - Resolved NameError on `selected_pathogen` fallback.
3. **Page 3 Layout Optimization**:
   - Removed the generic sub-header sentence and the "What Are We Comparing?" banner card per user instruction.
   - Promoted the **Overall Winner** card and **Three Key Scientific Takeaways** directly to the **top of Page 3 (Stage 1: Academic Synthesis)**, followed by the supporting baseline tables, modern models, ground truth trajectories, and head-to-head metrics.
4. **Artificial Missingness Purge**:
   - Removed all 25 synthetic/missingness CSV files in `data/processed/missingness/`.
   - Removed missingness benchmark scripts, tests, and UI expanders.
   - Reconciled test suite count from 112 to 108 tests (the 4 removed tests were in `tests/data_processing/test_missingness.py`).

---

## I. Comprehensive Testing Summary

### Automated Verification Script (`scripts/verify_entire_website_dataset.py`):
```
[PASS] Req 1 & 2: Dataset Integrity & Hash
[PASS] Req 3: Central Dataset Service
[PASS] Req 4: Single Dataset Rule (No Old/Raw CSVs)
[PASS] Req 5: Telemetry Synchronization
[PASS] Req 6-8: Model Registry & Predictions
[PASS] Req 9-11: AppTest All 6 Streamlit Pages
[PASS] Req 12: ARCH/GARCH Volatility Results
================================================================================
ALL AUDIT CHECKS PASSED: The entire website consistently uses the master dataset!
```

### Full PyTest Suite:
```bash
.venv/bin/pytest tests/
================= 108 passed, 329 warnings in 101.00s =================
```
- **Passed:** 108 tests
- **Failed:** 0 tests
- **Skipped:** 0 tests

---

## J. Final Component Verification Matrix

| Website Component | Uses Master Dataset | Outputs Verified | Status |
| :--- | :---: | :---: | :---: |
| **Disease Explorer** (`pages/1_Disease_Explorer.py`) | Yes (`disease_timeseries_master.csv`) | Yes (10 pathogens, 132 months, regional hubs) | **PASS** |
| **Time Series Analysis** (`pages/2_Time_Series_Analysis.py`) | Yes (`disease_timeseries_master.csv`) | Yes (ADF, KPSS, STL, ACF/PACF, Weather Heatmap) | **PASS** |
| **ARCH / GARCH Volatility** (Tab 6 on Page 2) | Yes (`disease_timeseries_master.csv`) | Yes (7 specifications, dynamic shock envelope) | **PASS** |
| **Model Comparison** (`pages/3_Model_Comparison.py`) | Yes (`disease_timeseries_master.csv`) | Yes (Winner card at top, 17 models, holdout curve) | **PASS** |
| **Future Forecasts** (`pages/4_Future_Forecasts.py`) | Yes (`disease_timeseries_master.csv`) | Yes (12-month projections, multi-disease watch) | **PASS** |
| **Data & Methodology** (`pages/5_Data_and_Methodology.py`) | Yes (`disease_timeseries_master.csv`) | Yes (43 cols, 1,320 rows, telemetry sync) | **PASS** |
| **All Graphs & Visualizations** | Yes (`disease_timeseries_master.csv`) | Yes (Dynamically update on pathogen selection) | **PASS** |
| **All Filters & Dropdowns** | Yes (`disease_timeseries_master.csv`) | Yes (Pathogen, region, and date range filters) | **PASS** |
| **Downloads & Exports** | Yes (`disease_timeseries_master.csv`) | Yes (Master CSV, forecast CSV, leaderboard CSV) | **PASS** |
| **Cache Management** | Yes (`disease_timeseries_master.csv`) | Yes (Synchronized through central dataset service) | **PASS** |

