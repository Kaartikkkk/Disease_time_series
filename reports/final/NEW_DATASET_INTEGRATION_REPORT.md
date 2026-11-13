# New Dataset Integration Report

**Author**: Senior Agricultural Time-Series Systems Architect  
**Project**: Plant Disease Outbreak Forecasting Using Time Series (India)  
**Date**: November 13, 2026  
**Status**: APPROVED & OPERATIONAL  
**Active Collection**: [`data/processed/improved_monthly_{slug}.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed) (10 files) & [`data/processed/improved_india_plant_diseases_monthly.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/improved_india_plant_diseases_monthly.csv)  

---

## 1. Authoritative Dataset Collection Overview

The single authoritative dataset collection for this project comprises **10 pathogen-specific monthly time-series CSVs** and **1 unified cross-pathogen monthly master panel**, spanning 11 consecutive calendar years from **January 1, 2015 to December 1, 2025** ($T=132$ monthly steps per pathogen; 1,320 total panel observations).

### Pathogen Inventory & Empirical Summary

| Pathogen Slug | Binomial Scientific Name | Agronomic Common Name | Economic Host Crop | Total Obs | Zero Months (%) | Total Occurrence Cases | Max Monthly Count |
|---|---|---|---|---|---|---|---|
| `alternaria_solani` | *Alternaria solani* | Early Blight of Potato/Tomato | Potato, Tomato | 132 | 104 (78.8%) | 71 | 6 |
| `bipolaris_oryzae` | *Bipolaris oryzae* | Brown Spot of Rice | Rice (*Oryza sativa*) | 132 | 104 (78.8%) | 94 | 12 |
| `fusarium_oxysporum` | *Fusarium oxysporum* | Fusarium Wilt | Chickpea, Cotton, Tomato, Banana | 132 | 26 (19.7%) | 688 | 44 |
| `macrophomina_phaseolina` | *Macrophomina phaseolina* | Dry Root Rot / Charcoal Rot | Chickpea, Soybean, Cotton, Groundnut | 132 | 77 (58.3%) | 208 | 21 |
| `phytophthora_infestans` | *Phytophthora infestans* | Late Blight of Potato | Potato (*Solanum tuberosum*) | 132 | 123 (93.2%) | 22 | 3 |
| `puccinia_striiformis` | *Puccinia striiformis* | Yellow / Stripe Rust of Wheat | Wheat (*Triticum aestivum*) | 132 | 128 (97.0%) | 7 | 3 |
| `pyricularia_oryzae` | *Pyricularia oryzae* | Rice Blast | Rice (*Oryza sativa*) | 132 | 103 (78.0%) | 112 | 11 |
| `ralstonia_solanacearum` | *Ralstonia solanacearum* | Bacterial Wilt | Potato, Tomato, Brinjal, Ginger | 132 | 90 (68.2%) | 228 | 15 |
| `rhizoctonia_solani` | *Rhizoctonia solani* | Sheath Blight of Rice | Rice (*Oryza sativa*) | 132 | 78 (59.1%) | 272 | 18 |
| `xanthomonas_oryzae` | *Xanthomonas oryzae* | Bacterial Blight of Rice | Rice (*Oryza sativa*) | 132 | 115 (87.1%) | 69 | 7 |
| **Total Master Panel** | **10 Pathogens** | **Indian Agricultural Complex** | **Staple & Cash Crops** | **1,320** | **948 (71.8%)** | **1,771** | **44** |

---

## 2. Complete 43-Column Schema Specification

Each of the 10 improved pathogen datasets adheres to a strict 43-column schema with **zero missing values** across all rows:

### Base Identifiers and Temporal Metadata (13 Columns)
- `date`: First day of observation month (`YYYY-MM-DD`, frequency `MS`).
- `documented_occurrence_count`: **Primary ground-truth target variable** (verified historical biological occurrence count in India).
- `is_missing`: Boolean data-quality indicator ($1 = \text{Surveillance gap}$, $0 = \text{Active monitoring}$).
- `year`, `month`, `month_name`, `quarter`, `crop_season`: Categorical and integer calendar indexing.
- `pathogen_slug`, `scientific_name`, `common_name`, `host_crop`, `region_id`: BINOMIAL taxonomic and geographic descriptors.

### Surface Climatological Regressors & Imputation Indicators (12 Columns)
- `temp_mean_c`, `temp_max_c`, `temp_min_c`: Monthly mean, maximum, and minimum 2m surface temperature (°C).
- `precipitation_monthly_mm`: Total monthly rainfall precipitation (mm).
- `relative_humidity_pct`: Mean relative humidity at 2m (%).
- `dew_point_c`: Surface dew point temperature (°C).
- `temp_mean_c_lag1`, `temp_mean_c_lag2`: Lag-1 and Lag-2 temperature (climatological normal backfilled for Jan–Feb 2015).
- `precipitation_monthly_mm_lag1`, `precipitation_monthly_mm_lag2`: Lag-1 and Lag-2 precipitation.
- `relative_humidity_pct_lag1`, `relative_humidity_pct_lag2`: Lag-1 and Lag-2 relative humidity.
- `imputed_flag`: Boolean tracking column indicating whether boundary weather features were imputed ($1.5\%$ of records).

### Engineered Causal Autoregressive & Momentum Features (5 Columns)
- `occurrence_lag1`, `occurrence_lag2`, `occurrence_lag3`: Trailing monthly target lags ($y_{t-1}, y_{t-2}, y_{t-3}$).
- `occurrence_lag12`: Annual seasonal lag ($y_{t-12}$) capturing year-over-year cyclical transmission.
- `occurrence_diff_1m_lag1`: Trailing rate-of-change momentum ($y_{t-1} - y_{t-2}$).

### Shifted Rolling Window Statistics (4 Columns)
- `occurrence_rolling_mean_3m_lag1`: 3-month trailing mean over $[t-3, t-1]$.
- `occurrence_rolling_std_3m_lag1`: 3-month trailing standard deviation over $[t-3, t-1]$.
- `occurrence_rolling_mean_6m_lag1`: 6-month trailing mean over $[t-6, t-1]$.
- `occurrence_rolling_std_6m_lag1`: 6-month trailing standard deviation over $[t-6, t-1]$.
*(All rolling metrics are strictly computed on $y_{t-1}$ to prevent lookahead data leakage).*

### Agrometeorological Indices & Harmonics (6 Columns)
- `hydrothermal_index_lag1`: De Martonne hydrothermal index: $\text{Precipitation}_{t-1} / (\text{Temperature}_{t-1} + 10)$.
- `dew_point_depression_lag1`: Canopy wetness indicator derived from temperature and relative humidity: $(100 - \text{RH}_{t-1}) / 5$.
- `temp_diurnal_range_lag1`: Trailing diurnal temperature range ($\text{Temp}_{\max, t-1} - \text{Temp}_{\min, t-1}$).
- `sin_month`, `cos_month`: Continuous trigonometric Fourier harmonics representing annual cyclicality.
- `trend_index`: Normalized secular timeline index ranging from $0.000$ (Jan 2015) to $1.000$ (Dec 2025).

### Variance-Stabilizing Target Transformations (3 Columns)
- `log1p_occurrence_count`: $\log(y_t + 1)$ for exponential count suppression.
- `sqrt_occurrence_count`: $\sqrt{y_t}$ Freeman-Tukey transformation for Poisson dispersion.

---

## 3. Dynamic Concatenation vs. Separate Access Verification

Empirical verification established that:
$$\text{Concatenate}\left(\left[\text{improved\_monthly\_}\{slug\}\text{.csv} \mid slug \in \mathcal{P}\right]\right) \equiv \text{improved\_india\_plant\_diseases\_monthly.csv}$$

- **Row Count**: $10 \times 132 = 1,320$ records.
- **Column Count**: Exactly 43 identical columns.
- **Null Count**: 0 missing values across all columns.
- **Target Invariant**: Sum of target across 10 individual files is 1,771; sum of target in master panel is 1,771.

---

## 4. Centralized Data Architecture (`src/data/dataset_service.py`)

A centralized dataset management module was created to provide a unified data access layer:
- **`load_authoritative_series(slug: str)`**: Fetches, validates, and date-parses the individual pathogen series.
- **`load_authoritative_master_dataset()`**: Loads the master panel directly or dynamically synthesizes it if missing.
- **`load_clean_event_records(slug: str)`**: Loads granular geospatial occurrence records for Page 1.
- **`verify_dataset_integrity()`**: Automated test harness validating schemas, row counts, and nulls across all 10 pathogen files.
