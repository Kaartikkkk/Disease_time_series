# PROJECT ARCHITECTURE AUDIT & RESTRUCTURING BLUEPRINT

**Project:** Time Series Forecasting of Agricultural Plant Diseases Across India  
**Audit Date:** November 13, 2026  
**Auditor:** Antigravity AI Engineering Team  
**Scope:** Complete Codebase Architecture, File Locations, Imports, Dependencies, and Migration Strategy  

---

## 1. Executive Summary & Audit Context

The Agricultural Plant Disease Time Series Forecasting project has reached complete functional maturity, encompassing 10 distinct crop pathogens, 112 months of historical observations, multi-source data ingestion (GBIF, EPPO, NASA POWER), 18 model configurations across 6 families, systematic hyperparameter optimization, and untouched holdout evaluation (2024–2025).

However, during this iterative growth, the repository accumulated significant architectural sprawl:
- **Root Directory Proliferation:** 44 markdown reports and intermediate JSON result files reside in the project root, obscuring the primary application entry points.
- **Flat Model Structure:** 18 distinct forecasting modules, adapters, and registries are stored in a single flat directory (`src/models/`), mixing classical statistical methods, deep learning architectures, gradient boosted trees, and framework-level adapters.
- **Flat Test Suite:** 16 test files covering diverse concerns (data pipelines, deep neural nets, UI components, econometric models) reside in a flat `tests/` folder.
- **Flat Scripts Directory:** 11 scripts spanning dataset building, hyperparameter tuning, benchmarking, and figure generation are grouped together without functional division.

This audit establishes the baseline dependencies and outlines a non-destructive migration plan that organizes the codebase into a clean, modular, and maintainable architecture without breaking existing functionality, tests, or Streamlit page discovery.

---

## 2. Current Directory Inventory

```
Disease_time_series/
├── Root Markdown Reports & JSONs: 44 files (~1.4 MB)
├── app.py (Streamlit main entry point)
├── assets/ (5 environmental crop imagery assets)
├── config/ (model_selection_policy.json, settings.yaml)
├── data/
│   ├── processed/ (improved master dataset, 10 pathogen CSVs, result JSONs)
│   ├── raw/ (unprocessed GBIF/NASA downloads)
│   └── reports/ (legacy data summaries)
├── models/ (16 serialized .joblib checkpoints, registry_metadata.json)
├── notebooks/ (3 exploratory Jupyter notebooks)
├── pages/ (5 Streamlit multi-page application files)
├── pytest.ini (Pytest configuration)
├── requirements.txt (Dependencies)
├── scripts/ (11 operational scripts)
├── src/
│   ├── 19 root utility/service modules
│   └── models/ (20 model, adapter, and base classes)
└── tests/ (16 test files, 99 test cases)
```

---

## 3. Dependency & Import Audit Graph

### 3.1 Model Layer (`src/models/`)
The current model layer has clear functional groupings that can be decoupled:

1. **Classical Statistical Models:**
   - `ar_model.py`: AR(1) via statsmodels `AutoReg`
   - `ma_model.py`: MA(1) via statsmodels `SARIMAX`
   - `arima_model.py`: ARIMA(1,1,1) with auto-order search
   - `sarima_model.py`: SARIMA(1,0,1)(1,0,1)[12] seasonal Box-Jenkins
   - `arimax_model.py`: Non-seasonal ARIMAX with weather lags
   - `sarimax_model.py`: SARIMAX with seasonal cycle and lagged weather covariates
   - `holt_winters.py`: Triple Exponential Smoothing via statsmodels `ExponentialSmoothing`
   - `seasonal_naive.py`: 12-month annual persistence baseline
   *Dependencies:* `statsmodels`, `scipy`, `numpy`, `pandas`. Zero dependencies on PyTorch or XGBoost.

2. **Modern Machine Learning Models:**
   - `random_forest.py`: Scikit-learn `RandomForestRegressor`
   - `gradient_boosting.py`: Scikit-learn `HistGradientBoostingRegressor`
   - `xgboost_model.py`: Local `XGBRegressor` with multi-lag and weather features
   - `global_models.py`: Multi-pathogen pooled panel modeling (`Global RandomForest`, `Global XGBoost`, `Hurdle XGBoost`)
   - `ml_autoregressor.py`: Shared lag-matrix feature generator for tree models
   *Dependencies:* `scikit-learn`, `xgboost`, `pandas`, `numpy`.

3. **Deep Learning Models:**
   - `lstm_model.py`: PyTorch `nn.LSTM` with Adam optimizer, sequence windowing, and dropout
   - `gru_model.py`: PyTorch `nn.GRU` sharing sequence windowing with `lstm_model.py`
   *Dependencies:* `torch`, `torch.nn`, `numpy`, `pandas`.

4. **Probabilistic Forecasting Models:**
   - `prophet_model.py`: Meta Prophet with additive trend changepoints and yearly seasonality
   *Dependencies:* `prophet`, `pandas`, `numpy`.

5. **Common Framework Infrastructure:**
   - `base.py`: Defines `ForecastContext`, `validate_forecast_output()`, `build_failure_output()`, and type definitions
   - `adapters.py`: Normalizes all 18 heterogeneous model signatures into a uniform `(ctx: ForecastContext) -> Dict[str, Any]` interface
   - `model_registry.py` (in `src/`): Central catalog of `ModelSpec`, runtime dispatcher `run_registered_model()`, and `ModelPersistenceRegistry`

### 3.2 Application & Dashboard Layer (`pages/` and `app.py`)
- **`app.py`**: Imports `src.icons`, `src.project_stats`, `src.ui_helpers`.
- **`pages/1_Disease_Explorer.py`**: Imports `src.icons`, `src.project_stats`, `src.ui_helpers`.
- **`pages/2_Time_Series_Analysis.py`**: Imports `src.eda`, `src.stationarity`, `src.icons`, `src.project_stats`, `src.ui_helpers`.
- **`pages/3_Model_Comparison.py`**: Imports `src.evaluation`, `src.forecasting`, `src.icons`, `src.ui_helpers`, `src.multi_origin_ui`.
- **`pages/4_Future_Forecasts.py`**: Imports individual model forecast functions (`from src.models.ar_model import forecast_ar`, etc.), `src.forecast_insights`, `src.forecasting`, `src.ui_helpers`.
- **`pages/5_Data_and_Methodology.py`**: Imports `src.icons`, `src.project_stats`, `src.ui_helpers`.

### 3.3 Test Suite Dependencies (`tests/`)
A critical finding from the test audit is that several tests use `pytest.monkeypatch` targeting specific import paths:
- `tests/test_global_and_hurdle_models.py` monkeypatches:
  - `src.models.global_models.XGBClassifier`
  - `src.models.global_models.XGBRegressor`
  - `src.models.adapters.forecast_global_xgboost`
- `tests/test_gru_model.py` monkeypatches:
  - `src.models.adapters.forecast_gru`
- `tests/test_lstm_model.py` monkeypatches:
  - `src.models.adapters.forecast_lstm`
- `tests/test_prophet_model.py` monkeypatches:
  - `src.models.prophet_model.Prophet`
  - `src.models.prophet_model.HAS_PROPHET`
  - `src.models.adapters.HAS_PROPHET`
  - `src.models.adapters.forecast_prophet`
- `tests/test_xgboost_model.py` monkeypatches:
  - `src.models.adapters.forecast_xgboost`
- `tests/test_multi_origin_validation.py` monkeypatches:
  - `src.forecasting.run_registered_model`

**Architectural Rule:** To maintain 100% test compatibility, `src/models/adapters.py` and `src/models/__init__.py` must maintain seamless re-exports or facades so that monkeypatching `src.models.adapters.<function>` continues to work without modifying test semantics.

---

## 4. Path Configuration & Artifact Dependency Audit

### 4.1 Datasets
- Primary dataset: `data/processed/improved_india_plant_diseases_monthly.csv` (1,320 rows, 43 features)
- Per-pathogen slices: `data/processed/improved_monthly_<slug>.csv` (10 files)
- Referenced in: `src/forecasting.py`, `src/ui_helpers.py`, `src/models/global_models.py`, `scripts/*.py`

### 4.2 Model Checkpoints
- Directory: `models/` (root-level) containing 16 `.joblib` files and `models/registry_metadata.json`
- Referenced in: `src/model_registry.py` (`models_dir: Path = Path("models")`), `scripts/train_and_persist_model_artifacts.py`
- In the new structure, `artifacts/trained_models/` can be created and synchronized, while `models/` is retained or aliased to avoid broken references.

### 4.3 Policy & Configuration
- Policy file: `config/model_selection_policy.json`
- Settings file: `config/settings.yaml`
- Frozen holdout configurations: `data/processed/frozen_model_configurations.json`
- Pathogen routing table: `data/processed/pathogen_routing_matrix.json`

---

## 5. Critical Files That Must Remain in Place

1. **`app.py`**: Streamlit's primary application launch point (`streamlit run app.py`).
2. **`pages/1_Disease_Explorer.py` through `pages/5_Data_and_Methodology.py`**: Streamlit's native multi-page routing discovers pages based strictly on files matching `pages/<order>_<name>.py` at the project root. Moving or renaming these files breaks user navigation.
3. **`data/processed/improved_india_plant_diseases_monthly.csv`**: The authoritative processed data source.
4. **`pytest.ini`**: Root configuration for pytest execution (`testpaths = tests`).
5. **`requirements.txt`**: Root dependency manifest.

---

## 6. Audit of Obsolete, Pruned, and Clutter Files

1. **Root Markdown Reports (44 files):**
   - Must be moved into structured subdirectories under `reports/`:
     * `reports/architecture/`
     * `reports/dataset/`
     * `reports/experiments/`
     * `reports/evaluation/`
     * `reports/final/`
2. **Root JSON Results (3 files):**
   - `ADVANCED_MODEL_OPTIMIZATION_RESULTS.json` $\rightarrow$ `reports/experiments/`
   - `FINAL_MODEL_COMPARISON_RESULTS.json` $\rightarrow$ `reports/evaluation/`
   - `VERIFIED_ADVANCED_MODEL_CONFIGURATIONS.json` $\rightarrow$ `reports/experiments/`
3. **Pruned Models:**
   - As documented in Prompt 21 and the model audit, `arimax_model.py` and `gradient_boosting.py` were retired from the active benchmark in favor of `sarimax_model.py` and `xgboost_model.py`, but their source files remain useful for academic transparency. They will be housed in their respective modular folders (`classical/` and `machine_learning/`).
4. **Volatility (ARCH/GARCH) and Multivariate (VAR) Status:**
   - These models were empirically audited and documented as mathematically inappropriate for agricultural count time series with 71.8% zero-inflation. Per instructions, empty placeholder folders will not be created.

---

## 7. Target Directory Structure & Migration Plan

```
Disease_time_series/
│
├── app.py
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── assets/
│   ├── cereal_harvest_sunset.jpg
│   ├── golden_crop_field.jpg
│   ├── monsoon_weather_canopy.jpg
│   ├── verdant_panoramic_hero.jpg
│   └── wheat_canopy_ears.jpg
│
├── configs/
│   ├── models/
│   │   ├── model_selection_policy.json
│   │   ├── frozen_model_configurations.json
│   │   └── pathogen_routing_matrix.json
│   └── application/
│       └── settings.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── artifacts/
│   ├── trained_models/
│   ├── predictions/
│   ├── metrics/
│   └── tuning/
│
├── models/ (retained backward-compatibility alias for joblib checkpoints)
│
├── pages/ (untouched for Streamlit auto-discovery)
│   ├── 1_Disease_Explorer.py
│   ├── 2_Time_Series_Analysis.py
│   ├── 3_Model_Comparison.py
│   ├── 4_Future_Forecasts.py
│   └── 5_Data_and_Methodology.py
│
├── src/
│   ├── models/
│   │   ├── classical/
│   │   │   ├── __init__.py
│   │   │   ├── ar_model.py
│   │   │   ├── ma_model.py
│   │   │   ├── arima_model.py
│   │   │   ├── arimax_model.py
│   │   │   ├── sarima_model.py
│   │   │   ├── sarimax_model.py
│   │   │   ├── holt_winters.py
│   │   │   └── seasonal_naive.py
│   │   │
│   │   ├── machine_learning/
│   │   │   ├── __init__.py
│   │   │   ├── random_forest.py
│   │   │   ├── xgboost_model.py
│   │   │   ├── gradient_boosting.py
│   │   │   ├── global_models.py
│   │   │   └── ml_autoregressor.py
│   │   │
│   │   ├── deep_learning/
│   │   │   ├── __init__.py
│   │   │   ├── lstm_model.py
│   │   │   └── gru_model.py
│   │   │
│   │   ├── probabilistic/
│   │   │   ├── __init__.py
│   │   │   └── prophet_model.py
│   │   │
│   │   ├── common/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── adapters.py
│   │   │
│   │   ├── __init__.py (backward-compatible facade re-exports)
│   │   ├── base.py (backward-compatible facade re-exports)
│   │   └── adapters.py (backward-compatible facade re-exports)
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── data_quality.py
│   │   ├── data_collection.py
│   │   ├── monthly_aggregation.py
│   │   ├── gbif_client.py
│   │   ├── eppo_client.py
│   │   └── weather_client.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluation.py (metrics & tables)
│   │   ├── model_policy.py (pathogen routing policy)
│   │   └── forecast_insights.py (advisory insights)
│   │
│   ├── forecasting/
│   │   ├── __init__.py
│   │   └── forecasting.py (split, walk-forward, multi-step engine)
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── ui_helpers.py
│   │   ├── icons.py
│   │   ├── project_stats.py
│   │   ├── multi_origin_ui.py
│   │   └── styles.css
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── reporting.py
│   │   ├── eda.py
│   │   └── stationarity.py
│   │
│   └── model_registry.py (central model catalog & persistence engine)
│
├── tests/
│   ├── classical/
│   │   └── test_individual_models.py
│   ├── machine_learning/
│   │   ├── test_xgboost_model.py
│   │   └── test_global_and_hurdle_models.py
│   ├── deep_learning/
│   │   ├── test_lstm_model.py
│   │   └── test_gru_model.py
│   ├── probabilistic/
│   │   └── test_prophet_model.py
│   ├── data_processing/
│   │   ├── test_preprocessing.py
│   │   ├── test_aggregation.py
│   │   └── test_dataset_integrity.py
│   ├── evaluation/
│   │   ├── test_evaluation.py
│   │   ├── test_multi_origin_validation.py
│   │   └── test_forecast_insights.py
│   └── integration/
│       ├── test_forecasting_pipeline.py
│       ├── test_unified_architecture.py
│       ├── test_model_comparison_ui.py
│       └── test_model_policy.py
│
├── scripts/
│   ├── training/
│   │   ├── train_and_document_all_models.py
│   │   ├── train_and_persist_model_artifacts.py
│   │   └── generate_model_training_registry.py
│   ├── tuning/
│   │   ├── run_master_model_optimization.py
│   │   ├── optimize_advanced_models.py
│   │   └── run_comprehensive_fine_tuning.py
│   ├── benchmarking/
│   │   ├── run_multi_origin_benchmark.py
│   │   └── run_final_test_evaluation.py
│   └── utilities/
│       ├── build_improved_dataset.py
│       ├── generate_final_test_figures.py
│       └── generate_phase2_pdf.py
│
├── reports/
│   ├── architecture/ (milestone & compliance audits)
│   ├── dataset/ (dataset improvements & GBIF verification)
│   ├── experiments/ (model tuning & ablation reports)
│   ├── evaluation/ (benchmark comparisons)
│   └── final/ (holdout, pathogen matrix, robustness reports)
│
└── logs/ (runtime & evaluation logs)
```

---

## 8. Migration Safety Strategy

To guarantee that zero functionality breaks during and after the reorganization:
1. **Facade Re-Export Pattern:** `src/models/__init__.py` and `src/models/adapters.py` will re-export all model functions from their new subpackages (`src.models.classical.*`, `src.models.machine_learning.*`, `src.models.deep_learning.*`, `src.models.probabilistic.*`, `src.models.common.*`). Any module or test calling `from src.models import forecast_arima` or `from src.models.adapters import forecast_gru` will continue functioning without interruption.
2. **Backward-Compatible Root Shims:** In `src/`, thin forwarders will be maintained for `forecasting.py`, `evaluation.py`, `preprocessing.py`, etc., pointing to their subpackage implementations (`src/forecasting/`, `src/evaluation/`, `src/data_processing/`).
3. **Streamlit Multi-Page Preservation:** Files in `pages/` remain at root level with their numeric prefixes, ensuring Streamlit's built-in navigation remains 100% operational.
4. **Pytest Discovery:** Organizing tests into subfolders under `tests/` allows `pytest tests/` to discover all 99 tests automatically without changing `pytest.ini`.
5. **Git Safety:** No git delete, commit, or push operations will be performed.

---

## 9. Conclusion & Readiness

The audit confirms that the project is in a clean state, with all 99 tests passing and all 6 Streamlit pages functioning properly. We can now proceed with the step-by-step file movements and directory restructuring according to this plan.

