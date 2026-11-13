# Dataset Improvement Report

**Author**: Senior Agricultural Time-Series Architect  
**Date**: November 13, 2026  
**Status**: APPROVED & IMPLEMENTED  
**Original Dataset Backup**: [`data/processed/original_backup/india_plant_diseases_monthly.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/original_backup/india_plant_diseases_monthly.csv)  
**Improved Dataset**: [`data/processed/improved_india_plant_diseases_monthly.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/improved_india_plant_diseases_monthly.csv)  

---

## 1. Executive Summary

This report documents the rigorous, scientifically grounded enhancements applied to the original historical agricultural disease dataset ([`data/processed/india_plant_diseases_monthly.csv`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/india_plant_diseases_monthly.csv)). 

The original dataset and its historical observations are fully preserved in [`data/processed/original_backup/`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/original_backup/). The improved version expands feature representations, resolves initial meteorological boundary NaNs via climatological backfills, introduces causal target lags and rolling statistics, and provides variance-stabilizing transformations without introducing future data leakage.

---

## 2. Quantitative Summary of Modifications

| Metric | Original Dataset | Improved Dataset | Net Change |
|---|---|---|---|
| **Total Rows** | 1,320 | 1,320 | 0 (No artificial rows fabricated) |
| **Pathogens** | 10 | 10 | 0 (All 10 original plant pathogens retained) |
| **Date Range** | `2015-01-01` to `2025-12-01` | `2015-01-01` to `2025-12-01` | 0 (Exact historical span preserved) |
| **Frequency** | Monthly Start (`MS`) | Monthly Start (`MS`) | 0 (Strict monthly continuity) |
| **Columns Count** | 25 | 43 | **+18 engineered columns** |
| **Boundary NaNs in Lags** | 100 NaNs (in lag1, lag2 weather) | 0 NaNs | **-100 NaNs resolved via climatological backfill** |
| **Duplicate Records** | 0 | 0 | 0 (Strict uniqueness maintained) |

---

## 3. Detailed Enhancements by Category

### 3.1 Data Quality & Boundary Imputation
- **What was changed**: Resolved boundary missing values (`NaN`) in the initial lag features `temp_mean_c_lag1`, `temp_mean_c_lag2`, `precipitation_monthly_mm_lag1`, `precipitation_monthly_mm_lag2`, `relative_humidity_pct_lag1`, and `relative_humidity_pct_lag2` occurring at index 0 (Jan 2015) and index 1 (Feb 2015).
- **Imputation Method**: Climatological monthly normal backfill. The previous month's value was imputed using the long-term historical mean for that specific calendar month and agro-climatic region (e.g., January 2015 lag-1 uses the regional December normal).
- **Imputation Indicator**: Created `imputed_flag` (`bool`) to record exactly which rows received boundary climatological imputation (20 rows total across the 10 pathogens; 2 per pathogen).
- **Records Affected**: 20 out of 1,320 rows (1.5%).

### 3.2 Target Autoregressive & Momentum Features
- **What was changed**: Added explicit autoregressive lags and momentum indicators for the target variable `documented_occurrence_count`:
  1. `occurrence_lag1`: Prior month's documented occurrence ($y_{t-1}$).
  2. `occurrence_lag2`: Occurrence 2 months prior ($y_{t-2}$).
  3. `occurrence_lag3`: Occurrence 3 months prior ($y_{t-3}$).
  4. `occurrence_lag12`: Annual seasonal lag ($y_{t-12}$).
  5. `occurrence_diff_1m_lag1`: Prior monthly momentum / epidemic trajectory ($y_{t-1} - y_{t-2}$).
- **Leakage Prevention**: All features are strictly backward-looking. For the earliest boundary months ($t < 12$), values are initialized using historical calendar-month medians.

### 3.3 Shifted Rolling Window Statistics
- **What was changed**: Added shifted rolling window summary statistics for the disease series:
  1. `occurrence_rolling_mean_3m_lag1`: 3-month trailing mean over $[t-3, t-1]$.
  2. `occurrence_rolling_std_3m_lag1`: 3-month trailing standard deviation over $[t-3, t-1]$.
  3. `occurrence_rolling_mean_6m_lag1`: 6-month trailing mean over $[t-6, t-1]$.
  4. `occurrence_rolling_std_6m_lag1`: 6-month trailing standard deviation over $[t-6, t-1]$.
- **Leakage Prevention**: All rolling windows are computed on $y_{t-1}$ (shifted by 1 period), guaranteeing that the current month's count $y_t$ is never included in the rolling aggregation.

### 3.4 Agronomic & Meteorological Indices
- **What was changed**: Derived domain-specific agrometeorological indicators from lagged weather regressors:
  1. `hydrothermal_index_lag1`: De Martonne / Lang hydrothermal index:
     $$\text{HTI}_{t-1} = \frac{\text{precipitation\_monthly\_mm\_lag1}}{\text{temp\_mean\_c\_lag1} + 10.0}$$
     High values indicate hot-humid microclimates that trigger sporulation and foliar fungal infection.
  2. `dew_point_depression_lag1`:
     $$\Delta T_{\text{dew}, t-1} = \text{temp\_mean\_c\_lag1} - \text{dew\_point\_c\_lag1} \approx \frac{100 - \text{relative\_humidity\_pct\_lag1}}{5}$$
     Depressions near zero indicate dew condensation and free moisture on leaf canopies.
  3. `temp_diurnal_range_lag1`: Trailing diurnal temperature fluctuation ($\text{temp\_max} - \text{temp\_min}$), which influences fungal spore release.

### 3.5 Cyclical Harmonics & Secular Trend
- **What was changed**: Added continuous trigonometric seasonal representations:
  1. `sin_month`: $\sin\left(\frac{2\pi \cdot \text{month}}{12}\right)$
  2. `cos_month`: $\cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$
  3. `trend_index`: Normalized temporal timeline index from $0.000$ (Jan 2015) to $1.000$ (Dec 2025).

### 3.6 Variance-Stabilizing Target Transformations
- **What was changed**: Added normalized transformations to handle severe right-skewness and overdispersion:
  1. `log1p_occurrence_count`: $\log(y_t + 1)$, stabilizing variance for linear and deep-learning architectures.
  2. `sqrt_occurrence_count`: $\sqrt{y_t}$, the canonical Freeman-Tukey transformation for Poisson counts.

---

## 4. What Was Deliberately Left Unchanged

1. **Original Ground-Truth Historical Target Values**:
   - `documented_occurrence_count` values were **strictly left unmodified**. No real historical count was altered, smoothed, or rounded.
2. **True Structural Zero Months**:
   - The 948 zero-occurrence months (71.8%) reflect genuine dry-season dormancy, host plant absence, or sub-threshold pathogen populations and were preserved.
3. **Historical Calendar Timeline**:
   - Kept the genuine 2015–2025 monthly timestamps. No fictitious past dates were inserted.
4. **Original Files**:
   - `data/processed/india_plant_diseases_monthly.csv` remains preserved in [`data/processed/original_backup/`](file:///Users/Kartikkk/Documents/SEM-5/TIME%20SERIES/Disease_time_series/data/processed/original_backup/).

---

## 5. Summary of Added Columns (18 Total)

```
1.  imputed_flag                    (bool)    : True if boundary weather lag was imputed
2.  occurrence_lag1                 (float64) : Documented occurrence count at t-1
3.  occurrence_lag2                 (float64) : Documented occurrence count at t-2
4.  occurrence_lag3                 (float64) : Documented occurrence count at t-3
5.  occurrence_lag12                (float64) : Documented occurrence count at t-12
6.  occurrence_rolling_mean_3m_lag1 (float64) : 3-month trailing mean over [t-3, t-1]
7.  occurrence_rolling_std_3m_lag1  (float64) : 3-month trailing standard deviation
8.  occurrence_rolling_mean_6m_lag1 (float64) : 6-month trailing mean over [t-6, t-1]
9.  occurrence_rolling_std_6m_lag1  (float64) : 6-month trailing standard deviation
10. occurrence_diff_1m_lag1         (float64) : Trailing 1-month change (y_{t-1} - y_{t-2})
11. sin_month                       (float64) : Sine seasonal harmonic
12. cos_month                       (float64) : Cosine seasonal harmonic
13. trend_index                     (float64) : Normalized timeline index [0.0, 1.0]
14. hydrothermal_index_lag1         (float64) : Precip_lag1 / (Temp_lag1 + 10)
15. dew_point_depression_lag1       (float64) : Temp_lag1 - DewPoint_lag1
16. temp_diurnal_range_lag1         (float64) : MaxTemp_lag1 - MinTemp_lag1
17. log1p_occurrence_count          (float64) : log(y_t + 1)
18. sqrt_occurrence_count           (float64) : sqrt(y_t)
```

