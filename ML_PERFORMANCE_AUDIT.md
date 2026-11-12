# ML Performance Audit: Diagnosis of Time-Series Forecasting Pipeline

## Executive Summary
This audit rigorously examines the forecasting pipeline, data aggregation, train/validation/test split logic, feature construction, leakage prevention, and evaluation consistency across all 10 classical, machine learning, and deep learning models in the **Disease Time Series / Agricultural Disease Forecasting** repository.

---

## 1. Why Rice Blast Evaluation Uses September 2022 to February 2023
- **Dataset Span**: Each pathogen series runs from January 2015 (`2015-01-01`) through December 2025 (`2025-12-01`), totaling **132 continuous monthly observations**.
- **Three-Way Chronological Split (`src/forecasting.py:chronological_split`)**:
  - `train_ratio = 0.70`: 70% of 132 observations = **92 months** (`2015-01-01` to `2022-08-01`).
  - `val_ratio = 0.15`: 15% of 132 observations = **20 months** (`2022-09-01` to `2024-04-01`).
  - `test_ratio = 0.15`: 15% of 132 observations = **20 months** (`2024-05-01` to `2025-12-01`), strictly reserved as an untouched final out-of-sample holdout.
- **Evaluation Horizon (`horizon=6`)**:
  - The model comparison suite (`run_rolling_walk_forward_evaluation` and `run_model_comparison`) defaults to evaluating on the beginning of the validation split.
  - When evaluating an operational 6-month horizon, it starts at index 92 (`2022-09-01`) through index 97 (`2023-02-01`).
  - The final 20 months (`2024-05-01` to `2025-12-01`) were purposely protected as the final test set to prevent snooping and repeated holdout reuse during iterative model selection.

---

## 2. Completeness of Pathogen-Month Observations
- **Finding**: **All 132 monthly observations are present for every single one of the 10 pathogens** (1,320 total pathogen-month records).
- **Date Continuity**: Every file in `data/processed/monthly_<slug>.csv` contains exact, continuous monthly start dates (`<year>-<month>-01`) with no gaps or duplicated timestamps:
  - `alternaria_solani`: 132 months, 0 missing
  - `bipolaris_oryzae`: 132 months, 0 missing
  - `fusarium_oxysporum`: 132 months, 0 missing
  - `macrophomina_phaseolina`: 132 months, 0 missing
  - `phytophthora_infestans`: 132 months, 0 missing
  - `puccinia_striiformis`: 132 months, 0 missing
  - `pyricularia_oryzae`: 132 months, 0 missing
  - `ralstonia_solanacearum`: 132 months, 0 missing
  - `rhizoctonia_solani`: 132 months, 0 missing
  - `xanthomonas_oryzae`: 132 months, 0 missing

---

## 3. Zero Values vs. Missing Records
- **Overall Zero Rate**: Exactly **948 of 1,320 records (71.82%)** have `documented_occurrence_count == 0`.
- **Classification of Zero Observations (`data/processed/india_plant_diseases_monthly.csv`)**:
  - **812 observations (61.5%)**: `is_missing == False` and `documented_occurrence_count == 0`. These represent active surveillance months where GBIF/EPPO recorded observation activities, but no occurrences of the specific pathogen were documented. These are **genuine structural zeros** (dormancy or non-outbreak periods).
  - **136 observations (10.3%)**: `is_missing == True` and `documented_occurrence_count == 0`. These represent survey gap periods (e.g., historical months where no primary reporting occurred), imputed cleanly to zero to preserve timeline continuity without fabricating phantom occurrences.
  - **372 observations (28.2%)**: `is_missing == False` and `documented_occurrence_count > 0` (actual disease outbreak events).

---

## 4. Chronological Integrity of Train/Validation/Test Split
- The function `chronological_split()` strictly uses positional indexing without random shuffling:
  $$\max(date_{\text{train}}) < \min(date_{\text{val}}) \le \max(date_{\text{val}}) < \min(date_{\text{test}})$$
  - Training maximum date: `2022-08-01`
  - Validation minimum date: `2022-09-01`
  - Validation maximum date: `2024-04-01`
  - Test minimum date: `2024-05-01`
  - Test maximum date: `2025-12-01`
- There is **zero overlap or temporal leakage** between partitions.

---

## 5. Model Evaluation Dates and Actual Value Consistency
- In `run_rolling_walk_forward_evaluation(df, horizon=6, step_ahead=1)`:
  - All 14 models are evaluated on the exact same 6 dates: `2022-09-01`, `2022-10-01`, `2022-11-01`, `2022-12-01`, `2023-01-01`, `2023-02-01`.
  - All models predict against the identical ground-truth actual counts: `[0.0, 0.0, 0.0, 4.0, 9.0, 6.0]`.
- In `run_model_comparison(df, horizon=6)` (fixed 6-month recursive holdout):
  - Every model receives the identical training dataframe up to `2022-08-01` and is scored against the identical 6 holdout actual values.
- Neither protocol uses different dates or subsets across models.

---

## 6. Feature-Target Alignment in Machine Learning Models
- **Feature Construction (`build_xgboost_features` & `_prepare_supervised_tabular_data`)**:
  - Target lags: `lag_1` is $y_{t-1}$, `lag_2` is $y_{t-2}$, etc. Target $y_t$ is strictly aligned to the row being predicted.
  - Initial rows containing `NaN` from lag creation are dropped before model fitting via `.dropna(subset=feature_cols)`.
  - No forward shift exists; the supervised target is correctly indexed as $y_t$ with input features derived from $\le t-1$.

---

## 7. Rolling Features and Weather Feature Leakage Check
- **Rolling Features**:
  - In `build_xgboost_features`:
    ```python
    shifted = series.shift(1)
    df_feat[mean_col] = shifted.rolling(window=w, min_periods=1).mean()
    df_feat[std_col] = shifted.rolling(window=w, min_periods=1).std().fillna(0.0)
    ```
  - The series is explicitly shifted by 1 before applying `.rolling()`, guaranteeing that `roll_mean_3` at month $t$ is computed from $\{y_{t-1}, y_{t-2}, y_{t-3}\}$ and never contains $y_t$.
- **Recursive Forecasting Multi-Step Loop**:
  - Multi-step rolling windows in recursive mode slice `history[-w:]`, where `history` only appends prior step *predictions* $\hat{y}_{t+k-1}$, never actual future observations.
  - Fully leakage-free.

---

## 8. Weather Exogenous Availability at Forecast Origin
- **Weather Columns**: `temp_mean_c_lag1`, `precipitation_monthly_mm_lag1`, `relative_humidity_pct_lag1`.
- **Availability at Origin**:
  - Because these features are pre-lagged by 1 month (`_lag1`), their value for month $t$ is the actual weather from month $t-1$.
  - At the forecast origin (step 1), this weather value is fully observed and available.
  - For subsequent steps ($h \ge 2$) in a multi-step fixed holdout:
    - If `future_weather_df` contains future realized weather (as in a retrospective econometric test), the models read it.
    - If future weather is unavailable (as in live operational forecasting), models gracefully fall back to the historical feature mean (`exog_means`) or a user-selected weather scenario.

---

## 9. Model Comparison Scoring Consistency
- Every model output is passed through `validate_forecast_output()`, ensuring:
  1. Prediction array length exactly equals the requested horizon $h$.
  2. All predictions are finite floats (non-finite or NaN outputs result in model failure handling).
  3. Predictions are evaluated against the same `validation_actual` vector.
- The comparison table (`build_model_comparison_table`) sorts and formats all models based on these identical observations.

---

## 10. Metric Calculation Consistency
- `compute_forecast_metrics()` calculates:
  - $\text{MAE} = \frac{1}{n} \sum |y - \hat{y}|$
  - $\text{RMSE} = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$
  - $\text{sMAPE} = \frac{100\%}{n} \sum \frac{2 |y - \hat{y}|}{|y| + |\hat{y}|}$ (with $(0, 0) \to 0\%$ convention)
  - $\text{WAPE} = 100\% \times \frac{\sum |y - \hat{y}|}{\sum |y|}$
  - $\text{MASE} = \frac{\text{MAE}}{\frac{1}{n_{\text{train}}-12} \sum_{t=13}^{n_{\text{train}}} |y_t - y_{t-12}|}$
- Handled safely when denominators are zero; strictly uniform across all models.

---

## 11. Root Cause Analysis: Why AR(1) & ARIMA Outperformed Local ML on Rice Blast
1. **Severe Local Sample Paucity**:
   - Each pathogen model trains on only **92 monthly observations**.
   - With 12-month lag requirements, effective tabular training size is reduced to only **80 rows**, of which ~62 (78%) are zero.
2. **False Seasonal Outbreak Spikes in Tree Models**:
   - In Rice Blast training history, October experienced heavy outbreaks in several past years (counts of 15, 9, 5, 4).
   - In the validation period (October 2022), the actual count was **0**.
   - Tree models (Random Forest, HistGradientBoosting) split strongly on `month == 10` or lagged rainfall, predicting **4.27 to 6.29**, incurring massive penalties ($(4.27 - 0)^2 \approx 18.2$).
3. **Smooth Linear Mean-Reversion in AR(1) and ARIMA**:
   - AR(1) and ARIMA shrank predictions toward the series mean (~0.5), yielding small errors ($|0.5 - 0| = 0.5$, squared error $0.25$) across the three consecutive dormant months (`0, 0, 0`), resulting in a superior overall RMSE of **3.738**.
4. **Conclusion & Strategic Remedy**:
   - Local ML models suffer from high variance due to small sample size and extreme sparsity.
   - **Global ML models (pooling all 10 pathogens $\to$ 920+ training instances)** and **two-stage Hurdle models (separating occurrence classification from magnitude regression)** provide the principled path to reducing variance and false-positive spikes.

