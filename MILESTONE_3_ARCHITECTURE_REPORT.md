# Milestone 3 Unified Forecasting Architecture Report

**Branch:** `milestone-3-cleanup`  
**Scope:** Common Model Contract, Central Model Registry, Output Validation, and Evaluator Refactoring (Milestone 2 Preserved)

---

## 1. Updated Architecture Diagram

```text
Streamlit UI (app.py, pages/3_Model_Comparison.py, pages/4_Future_Forecasts.py)
        │
        ▼
src/forecasting.py  (chronological_split, run_model_comparison, run_rolling_walk_forward_evaluation)
        │   Reads hyperparameters from config/settings.yaml
        ▼
src/model_registry.py  (MODEL_SPECS, get_available_models, run_registered_model, ModelRegistry)
        │   Validates every output via src/models/base.py::validate_forecast_output()
        ▼
src/models/adapters.py  (adapt_seasonal_naive, adapt_ar, ..., adapt_gradient_boosting, adapt_prophet)
        │
        ▼
src/models/*.py  (Unchanged M2 statistical/ML algorithms + preserved prophet_model.py)
        │
        ▼
src/evaluation.py  (compute_forecast_metrics, build_model_comparison_table with Category)
```

---

## 2. List of Modified & Added Files

### Added
- `src/models/base.py`: `ForecastContext`, `validate_forecast_output()`, `build_failure_output()`
- `src/models/adapters.py`: Thin adapter functions for all 10 active Milestone 2 models + `adapt_prophet()` blueprint
- `tests/test_unified_architecture.py`: 9 architecture & leakage-prevention tests
- `MILESTONE_3_ARCHITECTURE_REPORT.md`: This report

### Modified
- `src/model_registry.py`: Added `ModelSpec`, `MODEL_SPECS` (10 active + 4 planned Milestone 3), `get_model_spec()`, `list_registered_models()`, `get_available_models()`, `get_model_category()`, `run_registered_model()`
- `src/forecasting.py`: Refactored `run_model_comparison()` and `run_rolling_walk_forward_evaluation()` to execute via `get_available_models()` + `run_registered_model()`, reading default hyperparameters from `config/settings.yaml`
- `src/evaluation.py`: Added `model_category` support to `compute_forecast_metrics()` and `"Category"` column to `build_model_comparison_table()`
- `config/settings.yaml`: Added centralized `forecasting.hyperparameters` (`seasonal_period`, `autoregressive_lags`, `weather_exog_cols`)

---

## 3. Common Model Interface (`src/models/base.py`)

Every model adapter accepts a single `ForecastContext`:
- `train_df`: Validation-safe chronological training slice (`date`, target, lagged weather columns)
- `horizon`: Requested integer forecast steps
- `target_col`: Target column name (default `"documented_occurrence_count"`)
- `exog_cols`: Optional lagged exogenous feature names
- `future_df`: Optional future dates / future lagged exogenous regressors
- `config`: Model-specific parameters (e.g., `order`, `seasonal_order`, `auto_optimize`, `lags`)

Every adapter returns a dictionary validated by `validate_forecast_output()`:
```python
{
    "predictions": np.ndarray,           # Finite 1D float array of length == horizon
    "lower_bound": Optional[np.ndarray], # None when CI is unavailable (never fabricated)
    "upper_bound": Optional[np.ndarray], # Verified lower_bound <= upper_bound
    "model_name": str,
    "metadata": dict,                    # parameters, model_family, fitted_model, status
}
```
Failed runs use `build_failure_output()`, producing `np.nan` predictions and `metadata["status"] = "failed"` (never `0.0`).

---

## 4. Central Model Registration (`src/model_registry.py`)

`MODEL_SPECS` registers 14 models with `model_id`, `display_name`, `category`, `model_family`, `available`, `supports_exog`, `requires_exog`, `has_ci`, `default_config`, and `adapter`:
- **10 Active Milestone 2 Models (`available=True`):**
  - Classical: `seasonal_naive`, `holt_winters`
  - Statistical: `ar`, `ma`, `arima`, `sarima`, `arimax`, `sarimax`
  - Machine Learning: `random_forest`, `gradient_boosting`
- **4 Planned Milestone 3 Models (`available=False`):**
  - `xgboost` (`Machine Learning`), `prophet` (`Statistical` — `adapt_prophet()` blueprint ready in `src/models/adapters.py`), `lstm` (`Deep Learning`), `gru` (`Deep Learning`).
  - Calling `run_registered_model()` on any `available=False` model raises `RuntimeError`.

---

## 5. Evaluator & Pipeline Changes (`src/forecasting.py`, `src/evaluation.py`)

- Removed 10 duplicated `try/except` model blocks in both `run_model_comparison()` and `run_rolling_walk_forward_evaluation()`, replacing them with registry-driven loops over `get_available_models()`.
- Preserved the exact **6-month chronological holdout** and **rolling/expanding walk-forward** procedures:
  - `cached_arima_order` and `cached_sarima_order` are still estimated strictly on `train_df` (`initial_y_fit`) prior to the walk-forward loop.
  - Inside each walk-forward step `k`, `train_df` is strictly `df.iloc[:n_train + k]` (expanding) or fixed-window trailing slice (rolling).
  - `use_log1p`, `apply_threshold`, `tau`, and `model_failures` tracking remain intact.
- `build_model_comparison_table()` now includes `"Category"` (`Classical`, `Statistical`, `Machine Learning`, `Deep Learning`) alongside existing Milestone 2 metrics.

---

## 6. Milestone 2 Preservation & Verification

- **Rice Blast (`pyricularia_oryzae`) 6-month fixed holdout:** Verified bit-for-bit identical RMSE/MAE across all 10 Milestone 2 models (`MA(1)=4.307`, `ARIMA(1,1,1)=4.307`, `AR(1)=4.309`, `ARIMAX=4.696`, `Seasonal Naive=4.708`, `SARIMA=4.817`, `SARIMAX=4.86`, `Random Forest=4.984`, `Holt-Winters Exponential Smoothing=5.109`, `Hist Gradient Boosting=5.366`).
- **Full Test Suite (`pytest tests/`):** **`46 passed, 0 failed`** (37 Milestone 2 tests + 9 new tests in `tests/test_unified_architecture.py`).
- **Streamlit `AppTest`:** Passed with 0 exceptions on `app.py`, `pages/3_Model_Comparison.py`, and `pages/4_Future_Forecasts.py`.

---

## 7. Remaining Limitations

1. `pages/4_Future_Forecasts.py` still dispatches its 10 models via direct `if/elif` calls rather than `run_registered_model()`; it can be switched to the registry once Milestone 3 models are integrated.
2. Milestone 3 models (`xgboost`, `prophet`, `lstm`, `gru`) remain `available=False` until their respective implementation and validation stages.
3. Deep learning (`lstm`, `gru`) will require verifying `torch` wheel compatibility on Python 3.14.6 before implementation.

