# Prophet Implementation & Evaluation Report

**Branch:** `milestone-3-cleanup`  
**Prophet Version:** `1.5.0`  
**Target Variable:** `documented_occurrence_count` (Monthly `MS`, Jan 2015 – Dec 2025, 132 observations/pathogen)

---

## 1. Files Modified & Created

### Created
- `tests/test_prophet_model.py`: 6 unit, negative-clipping, leakage-prevention, and integration tests
- `PROPHET_IMPLEMENTATION_REPORT.md`: This report

### Modified
- `src/models/prophet_model.py`: Added `prepare_prophet_dataframe()`, conservative `DEFAULT_PROPHET_PARAMS` (`yearly_seasonality=True`, `weekly_seasonality=False`, `daily_seasonality=False`, `seasonality_mode="additive"`, `changepoint_prior_scale=0.05`, `seasonality_prior_scale=0.1`), negative-clipping with coherent `lower_95 <= point_pred <= upper_95`, and sparse-data diagnostics
- `src/models/adapters.py`: Updated `adapt_prophet()` to run strictly univariate by default (`use_exog=False`) and attach `diagnostics` in `metadata`
- `src/model_registry.py`: Activated `prophet` (`available=HAS_PROPHET`, `category="Statistical"`, `supports_exog=False`)
- `src/forecasting.py`: Re-exported `forecast_prophet`
- `pages/4_Future_Forecasts.py`: Added `"Prophet (Trend + Seasonality)"` to model selector and forecast dispatch
- `tests/test_forecasting_pipeline.py`, `tests/test_unified_architecture.py`: Updated active model count assertions from 11 to 12

---

## 2. Rice Blast (`pyricularia_oryzae`) 6-Month Expanding Walk-Forward Comparison

**Actual 6-Month Evaluation Window:** `mean = 3.17`, `std = 3.484`, `zero_pct = 50.0%`  
**Total 12-Model Walk-Forward Runtime:** `7.36s` (`Prophet single-fit runtime = 0.464s`)

| Rank | Model | Category | RMSE | MAE | Pred Std | Near-Zero (<0.05) |
|---|---|---|---|---|---|---|
| #1 | AR(1) | Statistical | 3.738 | 2.675 | 0.819 | 0.0% |
| #2 | ARIMA(1,1,1) | Statistical | 3.739 | 2.681 | 0.805 | 0.0% |
| #3 | MA(1) | Statistical | 3.780 | 2.731 | 0.712 | 0.0% |
| #4 | SARIMA(1,0,1)(1,0,1)[12] | Statistical | 3.881 | 2.860 | 1.135 | 0.0% |
| #5 | SARIMAX (Seasonal + Weather) | Statistical | 3.884 | 2.932 | 1.284 | 0.0% |
| #6 | ARIMAX (Weather Regressors) | Statistical | 4.108 | 3.296 | 0.467 | 0.0% |
| #7 | XGBoost (Autoregressive) | Machine Learning | 4.213 | 2.976 | 1.596 | 33.3% |
| #8 | Random Forest (Autoregressive) | Machine Learning | 4.418 | 3.311 | 1.784 | 0.0% |
| #9 | Hist Gradient Boosting (Autoregressive) | Machine Learning | 4.491 | 3.493 | 2.264 | 50.0% |
| #10 | Seasonal Naive (Baseline) | Classical | 4.708 | 3.167 | 0.000 | 100.0% |
| **#11** | **Prophet (Trend + Seasonality)** | **Statistical** | **5.066** | **4.003** | **1.680** | **50.0%** |
| #12 | Holt-Winters Exponential Smoothing | Classical | 5.089 | 4.024 | 1.732 | 66.7% |

### Diagnostics & Honest Assessment
- **6M Fixed-Holdout Diagnostics:** `pred_std = 1.6791`, `near_zero_pct = 50.0%`, `pred_mean = 0.8651`, `pred_min = 0.0`, `pred_max = 4.5964`, `negative_raw_count = 2`, `train_samples = 92`, `forecast_horizon = 6`, `status = "ok"`.
- **Why Prophet Ranks #11 on Rice Blast:** On sparse, zero-inflated monthly counts ($N=92$ training points, 71.8% dataset-wide zeros), Prophet's smooth additive Fourier seasonal curve (`yearly_seasonality=True`) tends to overshoot/dip negative during dormant months (`negative_raw_count=2` clipped to `0.0`) and smooths over sharp episodic outbreak spikes, yielding higher RMSE (`5.066`) than parsimonious AR/ARIMA/MA (`3.738–3.780`) and tree models (`4.213–4.491`).

