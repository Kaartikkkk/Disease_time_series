# 🔬 Methodology, Statistical Framework & Limitations Report

**Publication Timestamp (UTC):** 2026-10-21T19:27:22.879324+00:00  
**Geographic Scope:** Republic of India (IN)  

## 1. Target Variable Definition & Epistemology

**Target Variable:** `documented_occurrence_count`  
**Measurement Unit:** Integer count of qualifying occurrence records per calendar month  

> Measures the frequency of verified scientific, diagnostic, or preserved pathogen samples deposited in official public biodiversity and agricultural repositories. It is an index of documented field surveillance activity and pathogen presence, not a complete census of farm-level disease incidence.

**Missingness Policy:** Months without observations are flagged as `is_missing = True` outside verified active surveillance windows. We never arbitrarily interpolate zeros to artificially complete a time series.

## 2. Public Data Sources & Attribution

### GBIF Occurrence Search API (Techdocs OpenAPI v1)
- **URL:** [https://api.gbif.org/v1/occurrence/search](https://api.gbif.org/v1/occurrence/search)
- **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0) / CC0

### NASA POWER Monthly Agrometeorology API
- **URL:** [https://power.larc.nasa.gov/api/temporal/monthly/point](https://power.larc.nasa.gov/api/temporal/monthly/point)
- **License:** NASA Open Data Policy (Public Domain)
- **Key Parameters:** T2M (Mean Temp), T2M_MAX, T2M_MIN, PRECTOTCORR (Rainfall), RH2M (Humidity), T2MDEW (Dew Point)

### EPPO Global Database & Reporting Service
- **URL:** [https://gd.eppo.int/](https://gd.eppo.int/)
- **License:** Open Access / Academic Fair Use

## 3. Modeling & Evaluation Hierarchy

### Classical Statistical Suite
- Seasonal Naive (Benchmark baseline)
- AR(p) Autoregressive
- MA(q) Moving Average
- ARMA(p, q)
- ARIMA(p, d, q) Integrated Autoregressive Moving Average
- SARIMA(p, d, q)(P, D, Q, 12) Seasonal ARIMA
- Holt-Winters Exponential Smoothing
- ARIMAX / SARIMAX with strictly lagged NASA POWER weather regressors

### Modern Machine Learning & Library Implementations
- Meta Prophet (Additive/multiplicative yearly seasonality + exogenous regressors)
- HistGradientBoosting and RandomForest Autoregressors (with cyclical month transforms)

**Validation Split:** Strict chronological train (70%) / validation (15%) / test (15%) splits without shuffling to prevent temporal lookahead bias.
**Evaluation Metrics:** MAE, RMSE, sMAPE (bounded 0-200%), WAPE (for count sparsity), Forecast Bias (Mean Error), Prediction Interval Coverage (80% and 95%)

## 4. Scientific Limitations & Responsible AI Disclaimers

- ⚠️ Passive surveillance bias: occurrences reflect sampling intensity of research institutions and diagnostic labs.
- ⚠️ Weather correlation does not prove biological causation: physical disease onset requires susceptible host phenology and microclimatic leaf wetness.
- ⚠️ Forecasts beyond observed weather rely on transparent climatology scenarios, not realized future weather.
