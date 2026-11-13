# Project Cleanup & Restructuring Report

**Branch:** `milestone-3-cleanup`  
**Baseline Tag:** `milestone-2-baseline`  
**Backup Location:** `../Disease_time_series_m2_backup/` (outside active repository)  
**Python Environment:** Python 3.14.6 (`.venv`)

---

## 1. Updated Project Directory Tree

```text
Disease_time_series/
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml
├── CLEANUP_REPORT.md
├── README.md
├── app.py
├── pytest.ini
├── requirements.txt
├── assets/
│   ├── cereal_harvest_sunset.jpg
│   ├── golden_crop_field.jpg
│   ├── monsoon_weather_canopy.jpg
│   ├── verdant_panoramic_hero.jpg
│   └── wheat_canopy_ears.jpg
├── config/
│   └── settings.yaml
├── data/
│   ├── processed/
│   │   ├── clean_*_records.csv (10 pathogen files)
│   │   ├── india_plant_diseases_monthly.csv (1,320 rows: 10 pathogens x 132 months)
│   │   └── monthly_*.csv (10 pathogen files, preserved for inspection)
│   ├── raw/
│   │   ├── eppo/
│   │   ├── gbif/
│   │   └── weather/
│   └── reports/
│       ├── data_quality_report.{json,md}
│       └── methodology_report.{json,md}
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_time_series_analysis.ipynb
│   └── 03_model_experiments.ipynb
├── pages/
│   ├── 1_Disease_Explorer.py
│   ├── 2_Time_Series_Analysis.py
│   ├── 3_Model_Comparison.py
│   ├── 4_Future_Forecasts.py
│   └── 5_Data_and_Methodology.py
├── reports/
│   ├── Agricultural_Disease_Forecasting_Phase_2_Evaluation_Report_and_Viva_Guide.pdf
│   └── PHASE_2_EVALUATION_REPORT.md
├── scripts/
│   └── generate_phase2_pdf.py
├── src/
│   ├── data_collection.py
│   ├── data_quality.py
│   ├── eda.py
│   ├── eppo_client.py
│   ├── evaluation.py
│   ├── forecast_insights.py
│   ├── forecasting.py
│   ├── gbif_client.py
│   ├── icons.py
│   ├── model_registry.py
│   ├── monthly_aggregation.py
│   ├── preprocessing.py
│   ├── reporting.py
│   ├── stationarity.py
│   ├── styles.css
│   ├── ui_helpers.py
│   ├── weather_client.py
│   └── models/
│       ├── __init__.py
│       ├── ar_model.py
│       ├── arima_model.py
│       ├── arimax_model.py
│       ├── gradient_boosting.py
│       ├── holt_winters.py
│       ├── ma_model.py
│       ├── ml_autoregressor.py
│       ├── prophet_model.py
│       ├── random_forest.py
│       ├── sarima_model.py
│       ├── sarimax_model.py
│       └── seasonal_naive.py
└── tests/
    ├── test_aggregation.py
    ├── test_evaluation.py
    ├── test_forecast_insights.py
    ├── test_forecasting_pipeline.py
    ├── test_individual_models.py
    └── test_preprocessing.py
```

---

## 2. List of Deleted Files

### Legacy Code & Caches
- `archive_legacy_v1/` (36 legacy v1 files)
- `gbif_audit_summary.json`
- `logs/app.log`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, `.DS_Store`

### Legacy Datasets & Checkpoints
- `data/processed/agroscope_grapevine_clean.csv`
- `data/processed/citrus_ctv_clean.csv`
- `data/processed/synthetic_demo_benchmark_clean.csv`
- `data/processed/tomato_tswv_clean.csv`
- `data/processed/wheat_blast_brazil_clean.csv`
- `data/raw/agroscope_grapevine_raw.csv`
- `data/raw/citrus_ctv_raw.csv`
- `data/raw/tomato_tswv_raw.csv`
- `data/raw/wheat_blast_brazil_raw.csv`
- `data/raw/weather_cache/weather_-19.44_-47.66_2012-04-15_2020-06-11.csv`
- `data/sample/synthetic_demo_potato_late_blight.csv`
- `models/xgboost_checkpoint.joblib`
- `models/random_forest_checkpoint.joblib`
- `models/hist_gradient_boosting_checkpoint.joblib`

### Legacy / Duplicate Reports & Unused Assets
- `reports/evaluation_summary.json` (legacy daily synthetic benchmark)
- `reports/model_comparison.csv` (legacy daily synthetic benchmark)
- `reports/Verdant_Intelligence_Phase_2_Evaluation_Report_and_Viva_Guide.pdf` (superseded duplicate)
- `assets/verdant_hero.jpg`
- `assets/seedling_sprout.jpg`

### Removed Model Modules
- `src/models/autots_model.py`
- `src/models/hurdle_model.py`

---

## 3. List of Retained Essential Files

- **Core App & Pages:** `app.py`, `pages/1_Disease_Explorer.py` through `pages/5_Data_and_Methodology.py`
- **Data & Provenance:** `data/processed/india_plant_diseases_monthly.csv`, all 10 `data/processed/clean_*_records.csv` (dynamically loaded by Disease Explorer), `data/raw/{gbif,weather,eppo}/`, `data/reports/*`
- **Preserved for Milestone 3:** `src/models/prophet_model.py`, `src/model_registry.py`
- **Milestone 2 Documentation:** `reports/Agricultural_Disease_Forecasting_Phase_2_Evaluation_Report_and_Viva_Guide.pdf`, `reports/PHASE_2_EVALUATION_REPORT.md`

---

## 4. List of Modified & Added Files

### Modified
- `src/forecasting.py`
- `src/models/__init__.py`
- `src/models/ml_autoregressor.py`, `src/models/random_forest.py`, `src/models/gradient_boosting.py`
- `src/ui_helpers.py`
- `app.py`
- `pages/2_Time_Series_Analysis.py`
- `pages/3_Model_Comparison.py`
- `pages/4_Future_Forecasts.py`
- `pages/5_Data_and_Methodology.py`
- `tests/test_individual_models.py`
- `config/settings.yaml`
- `requirements.txt`
- `.gitignore`

### Added
- `src/styles.css` (extracted 1,127 lines of active CSS from `src/ui_helpers.py`)
- `tests/test_forecasting_pipeline.py` (3 regression tests for split, 10-model comparison, and `log1p` ML behavior)
- `models/.gitkeep`
- `CLEANUP_REPORT.md`

---

## 5. Explanation of Important Code Fixes

1. **`log1p` Transformation Bug (`src/forecasting.py`):** In `run_rolling_walk_forward_evaluation()`, when `use_log1p=True`, `sub_df` is now copied into `sub_df_fit` with `sub_df_fit[target_col] = np.log1p(...)` before passing to `forecast_random_forest()` and `forecast_gradient_boosting()`. Feature lags and recursive rollouts now happen in `log1p` space and inverse-transform via `np.expm1()` exactly once.
2. **Walk-Forward Error Handling (`src/forecasting.py`):** Replaced silent `except Exception: append(0.0)` with `_record_failure()`, which logs warnings, appends `np.nan` (excluded by `compute_forecast_metrics`), and returns `model_failures` for diagnostic display on `pages/3_Model_Comparison.py`.
3. **Model Naming Consistency (`src/forecasting.py`, `pages/4_Future_Forecasts.py`):** Standardized `"Holt-Winters Exponential Smoothing"` and `"SARIMAX (Seasonal + Weather)"` across both Fixed Holdout and Walk-Forward pipelines.
4. **Removed Debug Reloads (`pages/3_Model_Comparison.py`):** Removed `importlib.reload(src.forecasting)` and `importlib.reload(src.ui_helpers)`.
5. **Dynamic Dashboard Data (`app.py`, `pages/2`, `pages/4`, `pages/5`):**
   - `app.py` now computes Rice Blast 6-month holdout metrics via cached `run_model_comparison()` and renders a real SVG sparkline from the last 6 observed months instead of hardcoded RMSE numbers (`2.311`, `2.437`, etc.).
   - `pages/4_Future_Forecasts.py` now exposes all 10 Milestone 2 models, computes `[Top Validated]` dynamically via `run_model_comparison()`, and clearly labels the dataset scope instead of presenting an inactive region filter.
   - `pages/5_Data_and_Methodology.py` now builds the pathogen table dynamically from `config/settings.yaml` and `data_quality_report.json` (eliminating non-existent `Colletotrichum falcatum` and `Puccinia graminis` rows).
   - Date spans standardized dynamically to `Jan 2015 – Dec 2025` (132 months).
6. **CSS & UI Extraction (`src/styles.css`, `src/ui_helpers.py`):** Moved active CSS out of `src/ui_helpers.py` (reducing it from 1,664 lines to 354 lines), removed obsolete `.sidebar-*` CSS and unused helpers (`get_disease_scientific_name`, `load_weather_series`).
7. **Mutable Default Fix (`ml_autoregressor.py`, `random_forest.py`, `gradient_boosting.py`):** Replaced `lags=[1, 2, 3, 12]` with `lags: Optional[List[int]] = None`.

---

## 6. Updated `requirements.txt` & `.gitignore`

- **`requirements.txt`**: Removed 11 unused packages (`autots`, `formulaic`, `interface_meta`, `rdata`, `wrapt`, `xarray`, `matplotlib`, `seaborn`, `contourpy`, `cycler`, `kiwisolver`). Preserved `xgboost==3.4.1`, `prophet==1.5.0`, `cmdstanpy`, `reportlab`, and all active runtime/test dependencies.
- **`.gitignore`**: Removed obsolete `archive_legacy_v1/` entry and added `__MACOSX/`, `.pytest_cache/`.

---

## 7. Test Results (Before & After Cleanup)

| Check | Before Cleanup | After Cleanup |
|---|---|---|
| `pytest tests/` | `37 passed` (included 3 removed `autots`/`hurdle` tests) | **`37 passed`** (34 remaining + 3 new regression tests in `test_forecasting_pipeline.py`) |
| 10 Pathogen Data Load (`load_monthly_series` + `load_clean_records`) | Not separately run | **10/10 passed** (132 months each) |
| `AppTest` across `app.py` + all 5 pages | Not separately run | **6/6 passed** (0 exceptions) |

---

## 8. Milestone 2 Preservation & Known Notes

- **All 10 Milestone 2 models** (`Seasonal Naive (Baseline)`, `AR(1)`, `MA(1)`, `ARIMA(1,1,1)`, `SARIMA(1,0,1)(1,0,1)[12]`, `Holt-Winters Exponential Smoothing`, `ARIMAX (Weather Regressors)`, `SARIMAX (Seasonal + Weather)`, `Random Forest (Autoregressive)`, `Hist Gradient Boosting (Autoregressive)`) remain intact and verified.
- **Remaining Known Notes for Milestone 3**:
  - The `.venv` runs **Python 3.14.6**. Verify PyTorch (`torch`) wheel availability on Python 3.14 before implementing **LSTM** and **GRU** (TensorFlow does not support Python 3.14).
  - `data/processed/monthly_*.csv` (10 files) were preserved per instructions; they can be safely removed later if only `india_plant_diseases_monthly.csv` is needed.
  - Streamlit's `use_container_width=True` deprecation warnings remain across pages and can be migrated to `width="stretch"` during Milestone 3 UI polish.

