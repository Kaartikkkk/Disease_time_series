# Original Dataset & Architecture Audit Report

**Author**: Antigravity Technical Lead & Senior Agricultural Time-Series Architect  
**Project**: Agricultural Disease Forecasting Using Time Series  
**Date**: November 13, 2026  
**Status**: Comprehensive Baseline Pre-Improvement Audit  

---

## 1. Executive Summary & Purpose

In accordance with **PROMPT 26**, this audit establishes the empirical ground truth of the original historical dataset and project architecture before removing synthetic components and applying scientifically justified dataset improvements. 

All claims regarding dates, record counts, missingness, and column structures in this document are derived directly from the underlying disk files in [`data/`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data).

---

## 2. Original Dataset Identity & File Locations

The project contains a two-tier data architecture: raw heterogeneous observational archives and unified processed monthly panels.

### 2.1 Primary Master Panel (Original Processed Dataset)
- **File Location**: [`data/processed/india_plant_diseases_monthly.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/india_plant_diseases_monthly.csv)
- **Companion Pathogen Series**: [`data/processed/monthly_{pathogen_slug}.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed) (10 individual files)
- **Clean Event Records**: [`data/processed/clean_{pathogen_slug}_records.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed) (10 individual files)

### 2.2 Raw Upstream Source Files
- **GBIF Primary Occurrence Records**: [`data/raw/gbif/`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/raw/gbif) (10 pairs of `.csv` and `.json` files covering *Alternaria solani*, *Bipolaris oryzae*, *Fusarium oxysporum*, *Macrophomina phaseolina*, *Phytophthora infestans*, *Puccinia striiformis*, *Pyricularia oryzae*, *Ralstonia solanacearum*, *Rhizoctonia solani*, and *Xanthomonas oryzae*).
- **EPPO Quarantine & Distribution Records**: [`data/raw/eppo/eppo_india_reports.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/raw/eppo/eppo_india_reports.csv) and [`eppo_india_reports_raw.json`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/raw/eppo/eppo_india_reports_raw.json).
- **NASA POWER Agro-Climatology Reanalysis**: [`data/raw/weather/`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/raw/weather) (Monthly meteorological composites across national and regional agro-ecological zones).

---

## 3. Empirical Date Range & Frequency Verification

> [!IMPORTANT]
> **Source Data Verification regarding 1775**:
> Prompt 26 instructed: *"The original historical dataset starts from 1775. Verify the actual date range from the source data instead of assuming it."*
> 
> We performed an exhaustive audit across all raw GBIF occurrence files, EPPO records, NASA POWER files, and processed series. The findings are conclusive:
> 1. **GBIF Raw Occurrence Archives**: The oldest verified biological occurrence observation in India across the 10 target pathogens is from **2015** (e.g., *Fusarium oxysporum* eDNA and specimen accessions from 2015–2026; *Puccinia striiformis* 2015–2024; *Bipolaris oryzae* 2015–2025). There are zero records dating back to 1775 in the GBIF or EPPO source data.
> 2. **NASA POWER Meteorological Data**: Modern satellite agro-climatology reanalysis covers **2015–2025**.
> 3. **The 1775 Origin**: The year `1775-01-01` originated solely as the arbitrary starting timestamp used in the previous Prompt 25 synthetic dataset generator (to generate 250 years $\times$ 12 months = 3,000 synthetic rows).
> 4. **True Original Dataset Date Range**: The real historical surveillance dataset strictly spans **2015-01-01 to 2025-12-01**.

### Temporal Characteristics:
- **Starting Timestamp**: `2015-01-01`
- **Ending Timestamp**: `2025-12-01`
- **Temporal Frequency**: Monthly Start (`MS`), exactly continuous.
- **Duration**: 11 continuous years = 132 consecutive months per pathogen.
- **Total Record Count**: $10 \text{ pathogens} \times 132 \text{ months} = 1,320$ pathogen-month observations.

---

## 4. Column Schema & Data Semantics

The original master dataset [`data/processed/india_plant_diseases_monthly.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/india_plant_diseases_monthly.csv) contains 25 columns:

| Column Name | Data Type | Null Count | Semantic Meaning & Role |
|---|---|---|---|
| `date` | `str` (ISO YYYY-MM-DD) | 0 | First day of observation month (`MS`) |
| `documented_occurrence_count` | `int64` | 0 | **Primary Target Variable**: Documented verified disease occurrences in India |
| `is_missing` | `bool` | 0 | Data quality indicator ($1 = \text{Missing surveillance gap}$, $0 = \text{Active monitoring}$) |
| `year` | `int64` | 0 | Calendar year (2015–2025) |
| `month` | `int64` | 0 | Calendar month index (1–12) |
| `month_name` | `str` | 0 | Three-letter month abbreviation (Jan–Dec) |
| `quarter` | `int64` | 0 | Calendar quarter (1–4) |
| `crop_season` | `str` | 0 | Indian agricultural season (`Kharif`, `Rabi`, `Zaid`) |
| `pathogen_slug` | `str` | 0 | Machine identifier for pathogen (e.g., `pyricularia_oryzae`) |
| `scientific_name` | `str` | 0 | Formal binomial taxonomic name (e.g., *Pyricularia oryzae*) |
| `common_name` | `str` | 0 | Standard agronomic disease name (e.g., *Rice Blast*) |
| `host_crop` | `str` | 0 | Primary economic host plant (e.g., *Rice*, *Wheat*, *Potato*) |
| `region_id` | `str` | 0 | Regional agro-ecological zone code (`national_composite`) |
| `temp_mean_c` | `float64` | 0 | Monthly mean surface temperature (°C) from NASA POWER |
| `temp_max_c` | `float64` | 0 | Monthly maximum 2m air temperature (°C) |
| `temp_min_c` | `float64` | 0 | Monthly minimum 2m air temperature (°C) |
| `precipitation_monthly_mm` | `float64` | 0 | Total monthly rainfall precipitation (mm) |
| `relative_humidity_pct` | `float64` | 0 | Mean relative humidity at 2m (%) |
| `dew_point_c` | `float64` | 0 | Surface dew point temperature (°C) |
| `temp_mean_c_lag1` | `float64` | 10 | Lag-1 mean temperature (NaN only for Jan 2015) |
| `temp_mean_c_lag2` | `float64` | 20 | Lag-2 mean temperature (NaN only for Jan–Feb 2015) |
| `precipitation_monthly_mm_lag1` | `float64` | 10 | Lag-1 precipitation |
| `precipitation_monthly_mm_lag2` | `float64` | 20 | Lag-2 precipitation |
| `relative_humidity_pct_lag1` | `float64` | 10 | Lag-1 relative humidity |
| `relative_humidity_pct_lag2` | `float64` | 20 | Lag-2 relative humidity |

---

## 5. Data Quality, Missingness, and Zero Distributions

### 5.1 Duplication Analysis
- Duplicate records across `(date, pathogen_slug)`: **0 (Strictly unique)**.

### 5.2 Zero-Occurrence Distribution
- Overall Zero Months: **948 out of 1,320 records (71.8%)**.
- Positive Occurrence Months: **372 records (28.2%)**.
- Extreme Pathogen Heterogeneity:
  - Highest Zero Rate: *Puccinia striiformis* (Stripe Rust) at **97.0%** zeroes (only 4 positive outbreak months).
  - Lowest Zero Rate: *Fusarium oxysporum* (Wilt) at **19.7%** zeroes (ubiquitous soil-borne pathogen).
  - Bacterial/Foliar Fungi: 68% to 87% zero rates.

### 5.3 Missing Surveillance Analysis (`is_missing == True`)
- Flagged Missing Surveillance Months: **136 out of 1,320 records (10.3%)**.
- In the original pipeline, `documented_occurrence_count` was initialized to 0 for these months while `is_missing = True` was retained.
- Pathogens with highest unmonitored surveillance gaps:
  - *Phytophthora infestans*: 36 months (27.3%)
  - *Puccinia striiformis*: 20 months (15.2%)
  - *Alternaria solani*, *Macrophomina phaseolina*: 13 months each (9.8%)
  - *Fusarium oxysporum*: 1 month (0.8%)

---

## 6. Preprocessing & Feature-Engineering Pipeline Audit

The current preprocessing flow is housed in:
1. [`src/preprocessing.py`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/preprocessing.py):
   - Deduplication of GBIF specimen occurrences.
   - Date validation (`parse_and_validate_dates`).
   - Coordinate bounding box check for India (`[6.0, 37.0]` N, `[68.0, 98.0]` E).
   - Assignment of agro-climatic zones.
2. [`src/monthly_aggregation.py`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/monthly_aggregation.py):
   - Monthly resampling on continuous Month-Start (`MS`) calendar.
   - Alignment with NASA POWER weather features and generation of lag-1 and lag-2 features.
   - Setting `is_missing` for months outside the active surveillance span.

### Opportunities for Improvement Identified in the Audit:
1. **Missing Value Imputation**: Currently, missing months in `documented_occurrence_count` are hardcoded to zero. While biologically defensible during non-crop seasons, during peak crop seasons or active surveillance periods, an explicit seasonal/linear interpolation or Hurdle expectation provides a superior representation.
2. **Feature Expansion**:
   - Lagged disease features: Lags 1, 2, 3, and 12 of the target variable itself.
   - Rolling disease features: 3-month and 6-month rolling means and standard deviations (lagged by 1 month to strictly prevent data leakage).
   - Weather interaction features: Hydrothermal coefficient ($\text{Rain} / \text{Temp}$), Dew point depression ($\text{Temp} - \text{DewPoint}$).
   - Seasonality harmonics: Sine/Cosine Fourier terms for calendar month ($\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$).
   - Rate-of-change: Monthly difference in weather conditions.
3. **Data Quality Tracking**:
   - Create an explicit `imputed_flag` so any modified values can be verified and traced back to their raw source.

---

## 7. Model Inventory & Implementation Audit

There are **17 registered forecasting models** in [`src/model_registry.py`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/src/model_registry.py):

| Category | Model ID | Architecture Display Name | Status |
|---|---|---|---|
| **Classical** | `seasonal_naive` | Seasonal Naive (Baseline) | Active |
| **Statistical** | `ar` | AR(1) Univariate Autoregressive | Active |
| **Statistical** | `ma` | MA(1) Moving Average | Active |
| **Statistical** | `arima` | ARIMA(1,1,1) Integrated Box-Jenkins | Active |
| **Statistical** | `sarima` | SARIMA(1,0,1)(1,0,1)[12] Seasonal Box-Jenkins | Active |
| **Classical** | `holt_winters` | Holt-Winters Exponential Smoothing | Active |
| **Statistical** | `arimax` | ARIMAX (Weather Regressors) | Active |
| **Statistical** | `sarimax` | SARIMAX (Seasonal + Weather Regressors) | Active |
| **Machine Learning** | `gradient_boosting` | Hist Gradient Boosting (Autoregressive) | Active |
| **Machine Learning** | `random_forest` | Local Random Forest (Autoregressive) | Active |
| **Machine Learning** | `xgboost` | Local XGBoost (Autoregressive) | Active |
| **Statistical** | `prophet` | Prophet (Trend + Seasonality) | Active |
| **Deep Learning** | `lstm` | PyTorch LSTM (Sequence Neural Net) | Active |
| **Deep Learning** | `gru` | PyTorch GRU (Gated Recurrent Unit) | Active |
| **Machine Learning** | `global_xgboost` | Global XGBoost (Multi-Pathogen Panel) | Active |
| **Machine Learning** | `global_random_forest` | Global Random Forest (Multi-Pathogen Panel) | Active |
| **Machine Learning** | `hurdle_xgboost` | Hurdle XGBoost (Two-Stage Classifier + Regressor) | Active |

All 17 models are configured for the original dataset structure and evaluate seamlessly on the 10 pathogens.

---

## 8. Synthetic Dataset References to be Removed

The audit identified the exact files created exclusively for the synthetic dataset experiment:
1. `data/synthetic/universal_disease_timeseries.csv`
2. `data/synthetic/synthetic_weather.csv`
3. `data/synthetic/synthetic_latent_volatility.csv`
4. `data/synthetic/synthetic_dataset_metadata.json`
5. `data/synthetic/SYNTHETIC_DATASET_DOCUMENTATION.md`
6. `data/synthetic/` directory itself
7. `reports/figures/synthetic/` (10 PNG diagnostic plots)
8. `scripts/generate_universal_synthetic_dataset.py`
9. `tests/test_synthetic_dataset.py`
10. `SYNTHETIC_MODEL_COMPATIBILITY_REPORT.md`

All of these files can be completely removed without affecting the original dataset, pipeline, or any pre-existing unit tests.

