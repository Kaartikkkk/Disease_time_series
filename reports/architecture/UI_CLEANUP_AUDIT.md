# UI Cleanup Audit — Comprehensive Project-Wide Interface Streamlining

**Project:** Indian Agricultural Disease Surveillance Time Series Forecasting  
**Author:** Antigravity AI  
**Scope:** Pre-Refactor Interface Audit Across All 6 Primary Application Pages  
**Target:** Eliminate Redundant Cards, Repeated Explanations, Inconsistent Dates, and Cognitive Clutter  
**Date:** November 2026  

---

## 1. Executive Audit Summary

A full visual, architectural, and component-level inspection was performed across all six primary pages of the Streamlit application:
1. `app.py` (Overview)
2. `pages/1_Disease_Explorer.py` (Disease Records & Spatial Analytics)
3. `pages/2_Time_Series_Analysis.py` (Statistical Diagnostics & Seasonality)
4. `pages/3_Model_Comparison.py` & `src/multi_origin_ui.py` (Benchmark & Leaderboard)
5. `pages/4_Future_Forecasts.py` (Predictive Surveillance & Monthly Outlook)
6. `pages/5_Data_and_Methodology.py` (Data Provenance & Scientific Methodology)

### Core Identified Inefficiencies & Clutter Patterns:
- **Redundant Explanatory Cards**: Pervasive use of `render_what_does_this_mean()` and `render_graph_explanation_card()` rendered identical, repetitive paragraphs beneath nearly every chart.
- **Card Explosion on Overview**: `app.py` contained four consecutive grids of cards (Feature Cards, Key Insights, Research Metrics, How Predictions Are Made, Micro-Climates Gallery, and Model Cards) resulting in extreme scroll fatigue before reaching any actual data.
- **Tab Fragmentation on Time Series Analysis**: Eight separate tabs (`tab_prep`, `tab_missing`, `tab_patterns`, `tab_adf_kpss`, `tab_diff`, `tab_decomp`, `tab_acf`, `tab_weather`), where three separate tabs were dedicated solely to stationarity and two to data quality.
- **Date Discrepancies**:
  - Raw GBIF accessions span up to April 2026.
  - Standardized regularized monthly forecasting series span strictly from **January 2015 to December 2025** (132 continuous months).
  - Disease Explorer reported pathogen-specific date bounds (e.g., Aug 2015 – Sep 2025) which conflicted with static text stating "2015–2026".
  - Future Forecasts lacked prominent, unambiguous grounding showing that projections originate from **December 2025** (the last verified observation date in the dataset).
- **Hardcoded Metric Instances**: Certain overview cards displayed hardcoded RMSEs or cached single-series values rather than dynamically querying verified experiment logs.
- **WAPE Metric Artifacts**: Fully purged from calculation in Prompt 18/19, but requires ongoing audit to ensure no UI tooltips or remnants persist.

---

## 2. Component-by-Component Classification Audit

### Page 0: Overview (`app.py`)

| Component / Section | Current State | Classification | Planned Action & Justification |
| :--- | :--- | :---: | :--- |
| **Panoramic Hero Banner** | Attractive visual, project title, subtitle, CTA buttons, and sparkline | **KEEP & SIMPLIFY** | Keep banner image and title. Streamline CTA buttons. Ground the Rice Blast sparkline dynamically with clear date bounds. |
| **Hero Glass Card (Rice Blast Preview)** | Shows sparkline + 2 mini metrics | **KEEP** | Provides instant visual evidence of time-series dynamics within 2 seconds. |
| **Four Feature Cards** | Links to Explorer, Time Series, Models, etc. | **SIMPLIFY** | Replace bulky cards with clean, compact navigation pill shortcuts. |
| **Key Insights Section** | 3 large cards summarizing high humidity / monsoon notice | **REMOVE** | Redundant. Insights belong in Disease Explorer and Future Forecasts; they clutter the overview. |
| **Research Metrics (4 Cards)** | Historical coverage, dated records, 10 taxa, climate telemetry | **MERGE & SIMPLIFY** | Merge into a single sleek 4-metric statistics strip anchored to dynamic project telemetry. |
| **"How Predictions Are Made" (01-04 Process Cards)** | 4 process cards explaining pipeline stages | **REMOVE** | Completely duplicates Section 4 of Data & Methodology page. |
| **"Agro-Ecological Terrains" (4 Image Cards)** | 4 photo cards of monsoon, rabi, canopy, golden field | **MOVE TO EXPANDABLE SECTION** | High visual quality, but displaces key data. Place inside a clean, collapsed expander for environmental context. |
| **"Prediction Methods Suite" (6 Model Cards)** | Cards for ARIMA, SARIMAX, Holt-Winters, etc. with hardcoded 6M RMSEs | **REMOVE** | Obsolete and redundant. Comprehensive model benchmarks live on Page 3 (Model Comparison). |
| **Preview Chart (Rice Blast & Brown Spot)** | Plotly scatter plot (2015–2025) | **KEEP & SIMPLIFY** | Serves as the primary meaningful dataset visual. Remove the bulky `render_graph_explanation_card` below it. |
| **Final CTA & Footer** | Call to action button + attribution links | **KEEP** | Clean and standard. Ensure attribution links are intact. |

---

### Page 1: Disease Records & Explorer (`pages/1_Disease_Explorer.py`)

| Component / Section | Current State | Classification | Planned Action & Justification |
| :--- | :--- | :---: | :--- |
| **Header & Live Telemetry Badge** | Title, subtitle, live sync badge | **KEEP** | Clean editorial header. |
| **Unified Filter Toolbar** | Disease select, Agro-climatic region select, Date range input | **KEEP** | Essential interactive controls. |
| **Four KPI Metric Cards** | Recorded Reports, Date Range, Mapped Locations, Host Crop | **SIMPLIFY** | Replace oversized cards with compact, responsive metrics. Ensure date range dynamically matches filtered dataset. |
| **Tab 1: Historical Timeline** | Bar chart + 3m rolling mean | **KEEP & SIMPLIFY** | Core visual. Remove redundant explanation card below it; add clean 1-line interpretation directly beneath the chart. |
| **Tab 2: Seasonal Cycle** | Monthly distribution bar chart | **KEEP & SIMPLIFY** | Clean bar chart showing Kharif vs. Rabi seasonality. Keep concise caption. |
| **Tab 3: Geospatial Map** | Scatter geo map of Indian coordinates | **KEEP** | High academic value showing GPS voucher locations across Indian states. |
| **Tab 4: Recorded Reports Table** | Full GBIF raw records table + CSV download button | **KEEP** | Essential data transparency and download capability. |
| **Tab 5: Summary Findings** | Data quality summary metrics table | **MERGE** | Merge with Tab 4 or simplify as an expandable data quality audit card. |
| **Bottom Navigation Links** | Next page shortcuts | **KEEP** | Useful workflow continuity. |

---

### Page 2: Time Series Analysis (`pages/2_Time_Series_Analysis.py`)

| Component / Section | Current State | Classification | Planned Action & Justification |
| :--- | :--- | :---: | :--- |
| **Header & Telemetry** | Page title + Diagnostic Suite badge | **KEEP** | Professional header. |
| **Parameter Toolbar** | Disease select + Smoothing window slider | **KEEP** | Core interactive controls. |
| **4 Metric KPI Cards** | Duration, Active Months, Average, Std Dev | **SIMPLIFY** | Render as compact, elegant cards without excessive margins. |
| **8 Cluttered Tabs** | Prep, Missing, Stationarity, ADF/KPSS, Diff, Decomp, ACF, Weather | **MERGE & CONSOLIDATE** | **Consolidate 8 tabs into 5 logical analytical workflows**: <br>1. *Historical Trajectory & Trend*<br>2. *Seasonal Decomposition*<br>3. *Stationarity & Differencing* (combines ADF, KPSS, and Differencing)<br>4. *Autocorrelation & Memory (ACF / PACF)*<br>5. *Weather Cross-Correlation & Data Quality* |
| **Metric Tables Before Graphs** | Metrics displayed prior to plots | **MOVE BELOW CHARTS** | Strictly prioritize **Chart First**, followed immediately by its data-driven interpretation. |
| **Heavy Mathematical Formulations** | Dickey-Fuller equations, KPSS formulas, Loess equations | **MOVE TO EXPANDABLE SECTION** | Place LaTeX equations inside elegant `st.expander("Mathematical Formulation & Test Details")`. |
| **Repeated Explanation Cards** | 8 instances of `render_graph_explanation_card` | **REMOVE** | Replace with concise, tailored, 1-2 sentence captions directly below each chart. |

---

### Page 3: Model Comparison (`pages/3_Model_Comparison.py` & `src/multi_origin_ui.py`)

| Component / Section | Current State | Classification | Planned Action & Justification |
| :--- | :--- | :---: | :--- |
| **Header & Protocol Badge** | Editorial title + Zero Leakage badge | **KEEP** | High academic presentation. |
| **Protocol Selector** | Fixed Holdout, Operational Walk-Forward, Multi-Origin Validation | **KEEP & COMPACT** | Keep all 3 evaluation modes, but render controls in a unified compact row. |
| **Section A: Classical Models** | Table of classical econometrics (MA, AR, Naive, HW, ARIMA, SARIMA) | **KEEP & REFINE** | Standardized columns: Model, RMSE, MAE, sMAPE, MASE, vs Baseline, Runtime. |
| **Section B: Advanced Models** | Table of modern forecasters (Hurdle, Prophet, Global RF, Global XGB, GRU, LSTM) | **KEEP & REFINE** | Identical metric columns as Section A for strict scientific comparability. |
| **Optimization Results Table (Prompt 19)** | Baseline vs. Optimized RMSE/MAE & % Gain | **KEEP** | Dynamically loaded from `ADVANCED_MODEL_OPTIMIZATION_RESULTS.json`. |
| **Optimization Grouped Bar Chart** | Interactive Plotly visual of optimization gains | **KEEP** | High visual clarity. |
| **Section C: Direct Head-to-Head** | 4 KPI cards + 9-row comparative matrix | **KEEP & SIMPLIFY** | Clear summary of Best Classical vs. Best Advanced across all dimensions. |
| **Section D: Empirical Visualizations** | 6 interactive Plotly tabs | **KEEP** | Outbreak Detection, Outbreak vs Dormancy MAE, Horizon Stability, etc. |
| **Section E: Production Model Policy** | Champion roles, 10-pathogen routing table, architectural pruning | **KEEP** | Complete academic transparency. |
| **WAPE Metric Remnants** | Zero in calculation; verify all tooltips and headers | **REMOVE (VERIFIED ZERO)** | Strictly maintain zero visible WAPE. |

---

### Page 4: Future Forecasts (`pages/4_Future_Forecasts.py`)

| Component / Section | Current State | Classification | Planned Action & Justification |
| :--- | :--- | :---: | :--- |
| **Header & Editorial Telemetry** | Page title + Last run timestamp | **KEEP** | Clean header. |
| **Pathogen & Model Selection Toolbar** | Pathogen select, Model architecture select, Horizon slider | **KEEP** | Core forecast generators. |
| **Date Anchoring Callout** | Shows Historical Records vs. Prediction Period | **KEEP & ENHANCE (CRITICAL)** | **Resolve Date Confusion**: Prominently display: <br>• *Latest Verified Historical Observation:* **December 2025**<br>• *Forward Forecast Window:* **January 2026 – June 2026** (Out-of-sample projection originating from Dec 2025 cutoff). |
| **4 Forecast KPI Metric Cards** | End of series report, predicted next month, peak month, average | **SIMPLIFY** | Make cards compact and cleanly aligned. |
| **Primary Forecast Plotly Chart** | Historical + Point Forecast + 80%/95% Prediction Intervals | **KEEP** | Core visual. Ensure actual-vs-forecast split line is distinctly styled. |
| **Peak Forecast Callout Card** | Highlights highest activity month | **KEEP & SIMPLIFY** | Useful early-warning indicator. |
| **Monthly Disease Outlook Cards Grid** | 3-column cards for Jan–Jun 2026 | **KEEP & STREAMLINE** | Remove redundant text inside cards. Focus on: Month, Predicted Count, Trend Arrow, Activity Badge, and 80% CI. |
| **Multi-Disease Watch Matrix** | Table comparing peak activity across all 10 pathogens | **KEEP** | Excellent high-level surveillance summary. |
| **Scenario Simulation (Monsoon/Drought)** | Weather scenario multipliers | **MOVE TO EXPANDABLE SECTION** | Place inside clean expander to prevent visual clutter for standard users. |

---

### Page 5: Data & Methodology (`pages/5_Data_and_Methodology.py`)

| Component / Section | Current State | Classification | Planned Action & Justification |
| :--- | :--- | :---: | :--- |
| **Header & License Badge** | Editorial title + CC-BY 4.0 badge | **KEEP** | Academic standard. |
| **Current 5 Tabs** | Sources, Prep, Methods, Eval, Downloads | **RESTRUCTURE INTO 8 PHASES** | Reorganize cleanly into the 8 requested academic sections:<br>1. *Data Sources*<br>2. *Dataset Coverage & Temporal Grounding*<br>3. *Preprocessing & Missing Value Imputation*<br>4. *Surveillance Methodology*<br>5. *Model Architectures (Classical & Modern)*<br>6. *Evaluation Protocols & Validation Schemes*<br>7. *Epidemiological Limitations & Boundary Conditions*<br>8. *Downloads, Data Dictionaries & Academic Citations* |
| **Repeated Methodology Paragraphs** | Redundant descriptive text | **SIMPLIFY** | Use bullet points, summary matrices, and expandable technical details. |
| **Dataset Download Links** | CSV and JSON download buttons | **KEEP** | Complete data transparency. |
| **Academic Citations** | GBIF, NASA, EPPO, Box-Jenkins citations | **KEEP** | Formal academic rigor. |

---

## 3. Cross-Cutting Design & Navigation Decisions

1. **Retirement of Simple View / Advanced View Toggle**:
   - The query parameter / sidebar toggle created confusing conditional HTML across pages, resulting in duplicate code blocks (`if not adv_mode: ... else: ...`).
   - **Resolution**: Provide a single, universally clean, professional interface that serves both students and researchers, using Streamlit's native `st.expander` for advanced mathematical formulations, deep diagnostic outputs, and technical hyperparameters.
2. **Unified Project Telemetry**:
   - Create `src/project_stats.py` to calculate dataset bounds, record counts, date ranges, and zero percentages dynamically from the underlying CSVs.
   - Eliminate all conflicting hardcoded dates (e.g., "2015–2026" vs "2015–2025").
3. **Consistent Minimalist Styling**:
   - Use standard typography (`Fraunces` for headings, `JetBrains Mono` for numbers/code, `Inter`/system sans for prose).
   - Use consistent green palette (`#102C26`, `#183D32`, `#527A50`, `#C4A66A` gold accents).
   - Remove heavy borders, oversized margins, and unnecessary background decorations.

