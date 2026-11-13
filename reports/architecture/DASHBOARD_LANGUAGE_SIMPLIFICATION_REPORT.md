# Dashboard Language Simplification Report

**Project:** Agricultural Disease Forecasting Using Time Series  
**Execution Phase:** Post-Phase 3 Language & UX Simplification (PROMPT 22)  
**Status:** Successfully Completed & Verified  

---

## 1. Executive Summary

As part of the final academic and presentation stabilization, all user-facing pages, charts, tables, and metric cards across the application were systematically audited and updated to replace unnecessarily dense econometric jargon with clean, self-explanatory, and professional English.

Crucially, **no mathematical calculations, model configurations, forecasting pipelines, or underlying metric values were altered**. The precision and mathematical integrity of the research framework have been fully preserved, while making the entire platform accessible to students, professors, and agricultural researchers without requiring verbal interpretation.

---

## 2. Standardized Metric Label Mapping

The following authoritative label transformations were applied across all leaderboard tables, KPI cards, and comparative figures:

| Previous Technical Label | New Simplified Professional Label | Operational Definition & Scope |
|---|---|---|
| **Pooled RMSE** | **RMSE** | Calculated across all evaluated prediction records pooled together. |
| **Macro RMSE** | **Average RMSE** | Arithmetic average of the individual RMSE values across the 10 monitored crop pathogens. |
| **Pooled MAE** | **MAE** | Mean absolute error across all evaluated predictions pooled together. |
| **Macro MAE** | **Average MAE** | Arithmetic average of individual pathogen MAE values. |
| **False Alarm (%)** | **False Alarms (%)** | Percentage of inactive/dormant zero-months incorrectly alerted as outbreaks. |
| **vs Seasonal Naive** | **Improvement vs Baseline** | Percentage reduction in prediction error relative to the 12-month seasonal persistence baseline. |
| **Runtime (s)** | **Time (seconds)** | Average execution and prediction latency per forecast origin. |
| **Operational Latency** | **Prediction Time** | Computational latency for generating forward projections. |
| **Classical Rank** | **Rank** | Integer standing within the evaluated methodological cohort. |
| **Outbreak Recall** | **Outbreak Detection Rate** | Percentage of true epidemic surge occurrences ($y_t > 0$) successfully detected by the model. |
| **Outbreak F1** | **Outbreak F1 Score** | Harmonic mean of outbreak precision and outbreak detection sensitivity. |
| **Baseline Improvement** | **Improvement vs Baseline** | Consistent relative gain over the Seasonal Naive baseline. |

### Preservation of Distinction & Tooltips
Where both overall and average metrics are displayed side-by-side (such as in the Model Comparison tables and KPI rows), their distinction is clearly documented through concise UI tooltips:
* **RMSE:** `help="Calculated across all evaluated predictions together"`
* **Average RMSE:** `help="Average of individual pathogen RMSE values"`
* **MAE:** `help="Calculated across all evaluated predictions together"`
* **Average MAE:** `help="Average of individual pathogen MAE values"`

---

## 3. Page-by-Page Language Simplification

### Page 1: Overview (`app.py`)
* **Zero-Inflation Breakdown:** Replaced technical zero classification text with self-explanatory labeling:
  * `{stats['structural_zeros']} Structural · {stats['gap_imputed_zeros']} Imputed Zeros` $\rightarrow$ `{stats['structural_zeros']} Confirmed No-Occurrence · {stats['gap_imputed_zeros']} Missing-Data Estimated`.
* **Advisory Action Cards:** Simplified `"Out-of-sample forward projections support targeted agronomist advisories"` $\rightarrow$ `"Early disease forecasts support targeted agronomist advisories"`.

### Page 2: Disease Records (`pages/1_Disease_Explorer.py`)
* **Seasonal Aggregation Chart:** Replaced `"Aggregated Historical Occurrences by Calendar Month (Jan–Dec)"` $\rightarrow$ `"Historical Disease Reports by Calendar Month (Jan–Dec)"`.
* **Tab Labels:** Standardized `"Historical Timeline"`, `"Seasonal Pattern"`, `"Geospatial Map"`, and `"Recorded Reports Table"`.

### Page 3: Time Series Analysis (`pages/2_Time_Series_Analysis.py`)
* **Stationarity & Differencing Takeaway:** Clarified the mathematical verdict:
  * `"Taking first differences in zero-inflated data increases artificial variance, explaining why ARIMA differencing degrades out-of-sample forecast accuracy"` $\rightarrow$ `"Taking first differences in data with many zero months increases artificial swings, explaining why ARIMA differencing degrades forecast accuracy on unseen data"`.
* **Surveillance Missingness Quality:** Simplified `"Gap-Imputed Interval"` $\rightarrow$ `"Missing-Data Estimated Zero"`.

### Page 4: Model Comparison (`pages/3_Model_Comparison.py` & `src/multi_origin_ui.py`)
* **Top Champion Banner:**
  * Replaced `"Pooled RMSE"` and `"Outbreak F1"` with `"RMSE"`, `"MAE"`, and `"Outbreak F1 Score"`.
  * Clarified champion description: `"Ranked #1 across 14 candidate architectures in rigorous 4-origin walk-forward testing... delivering lowest RMSE (5.982) and highest Outbreak Detection Rate (86.2%)"`.
* **Classical & Advanced Tables:**
  * Replaced all occurrences of `"Classical Rank"`, `"Pooled RMSE"`, `"Macro RMSE"`, `"Pooled MAE"`, `"False Alarm (%)"`, `"vs Seasonal Naive"`, and `"Runtime (s)"` with `"Rank"`, `"RMSE"`, `"Average RMSE"`, `"MAE"`, `"False Alarms (%)"`, `"Improvement vs Baseline"`, and `"Time (seconds)"`.
* **Direct Head-to-Head Section:**
  * Summary Cards: Display `"Top Classical Method"`, `"Top Advanced Architecture"`, `"Modern Accuracy Gain"`, and `"Overall Winner"` (with subtitle `"Lowest overall prediction error"`).
  * Direct Comparison Table: Standardized column headings to `"Evaluation Metric"`, `"Best Classical"`, `"Best Advanced"`, `"Improvement"`, `"Winner"`, and `"Plain-English Explanation"`.
* **Overall Winner Box & Academic Takeaways:**
  * Replaced complex academic headers with simple, numbered scientific insights:
    1. *Seasonal Inactivity & Zero Disease Months* (explaining why classical differencing fails on 71.8% zero months).
    2. *Weather and Spore Germination Triggers* (explaining humidity $>80\%$ and temperature tipping points).
    3. *Early Action in Agricultural Surveillance* (explaining the 1–6 month operational lead time for preventive fungicide pre-positioning).

### Page 5: Future Forecasts (`pages/4_Future_Forecasts.py`)
* **Forecast Range Labels:**
  * `"80% Prediction Interval Half-Width"` $\rightarrow$ `"80% Expected Range Half-Width"`.
  * `"95% Confidence Interval"` $\rightarrow$ `"95% Expected Range"`.
  * `"80% Confidence Range"` $\rightarrow$ `"80% Expected Range"`.
* **Controls & Horizons:**
  * `"Forward Horizon (Months)"` $\rightarrow$ `"Forecast Period (Months)"`.
  * `"Forward out-of-sample projections..."` $\rightarrow$ `"Forward projections originating from the latest verified historical observations with 80% and 95% expected ranges."`

### Page 6: Data & Methodology (`pages/5_Data_and_Methodology.py`)
* **Zero Definitions:**
  * Defined `"Confirmed No-Occurrence Months"` for seasonal dormancy.
  * Defined `"Missing-Data Estimated Zeros"` for unobserved regularized intervals.
* **Validation Schemes:**
  * Reframed `"Multi-Origin Walk-Forward Scheme"` $\rightarrow$ `"Multi-Origin Step-by-Step Historical Testing"`.
  * Highlighted `"Testing on Unseen Data"` alongside the locked May 2024 – Dec 2025 final test partition.
  * Replaced `"Outbreak Recall & F1"` $\rightarrow$ `"Outbreak Detection Rate & Outbreak F1 Score"`.

---

## 4. Verification & Invariance Testing

1. **Underlying Numerical Invariance:**
   * Benchmark Champion: Hurdle XGBoost (RMSE: `5.982`, Outbreak F1: `0.774`, Outbreak Detection Rate: `86.2%`).
   * Average Error Champion: Global Random Forest (Average RMSE: `4.081`).
   * Typical Error Champion: Prophet (MAE: `2.275`).
   * Classical Champion: MA(1) (RMSE: `6.243`, Average RMSE: `4.280`, MAE: `2.446`).
   * All metrics match underlying tensors in `data/processed/multi_origin_validation_results.json` to 3 decimal places.
2. **Automated Streamlit AppTest:**
   * `app.py`: **PASSED (0 exceptions)**
   * `pages/1_Disease_Explorer.py`: **PASSED (0 exceptions)**
   * `pages/2_Time_Series_Analysis.py`: **PASSED (0 exceptions)**
   * `pages/3_Model_Comparison.py`: **PASSED (0 exceptions)**
   * `pages/4_Future_Forecasts.py`: **PASSED (0 exceptions)**
   * `pages/5_Data_and_Methodology.py`: **PASSED (0 exceptions)**
3. **Full Pytest Suite:** All 95 tests passing with zero failures.

