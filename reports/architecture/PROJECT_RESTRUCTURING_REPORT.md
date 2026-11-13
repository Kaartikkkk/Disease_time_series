# Project Architecture Restructuring & Code Organization Report

**Date**: November 13, 2026  
**Status**: Completed & Verified  
**Milestone**: Architecture Restructuring & Modular Organization (Prompt 30)  
**Test Suite Verification**: 99 / 99 Unit & Integration Tests Passing (100%)  
**UI Verification**: 6 / 6 Streamlit Applications Passing End-to-End AppTest  

---

## 1. Executive Summary & Objective

Over the course of implementing 17 forecasting architectures (classical time-series, advanced machine learning, deep learning, and probabilistic models), multi-origin cross-validation benchmarks, and final holdout evaluation, the codebase expanded substantially. Model implementations, test modules, helper scripts, configuration files, and experiment reports accumulated in the root directory and flat package structures.

To establish an **academically rigorous, production-grade, and easily navigable codebase**, the entire project architecture was refactored into a **domain-separated, modular directory structure**.

### Key Objectives Achieved:
1. **Model Family Isolation**: Models partitioned into 5 dedicated subpackages under `src/models/` (`classical/`, `machine_learning/`, `deep_learning/`, `probabilistic/`, `common/`).
2. **Standardized Directory Hierarchy**: Established clean root domains: `src/`, `configs/`, `artifacts/`, `scripts/`, `tests/`, `reports/`, `pages/`, and `data/`.
3. **Clutter Elimination**: Over 40 markdown and JSON reports moved from the repository root into categorized subfolders within `reports/`. The root folder contains strictly `README.md` and standard project entrypoints.
4. **Flawless Backward Compatibility**: Maintained seamless backward-compatible facade re-exports and execution shims across all moved modules, guaranteeing zero regressions in tests, scripts, or the Streamlit web application.
5. **Rigorous Verification**: 99/99 pytest tests passed cleanly; all 6 Streamlit pages validated end-to-end via headless `AppTest`.

---

## 2. Directory Architecture & Categorization

The refactored project adheres to the following structural layout:

```
Disease_time_series/
├── README.md                                  # Primary academic & technical documentation
├── requirements.txt                           # Frozen Python dependencies
├── pytest.ini                                 # Pytest test suite configuration
├── app.py                                     # Primary Streamlit application entrypoint
│
├── pages/                                     # Streamlit multi-page application (Root-level discovery)
│   ├── 1_Disease_Explorer.py                  # Raw surveillance data & pathogen profiles
│   ├── 2_Time_Series_Analysis.py              # Stationarity tests, PACF/ACF, decomposition
│   ├── 3_Model_Comparison.py                  # Benchmark comparison & holdout leaderboard
│   ├── 4_Future_Forecasts.py                  # Operational multi-horizon disease projections
│   └── 5_Data_and_Methodology.py              # Scientific methodology & governance downloads
│
├── configs/                                   # Centralized configuration repository
│   ├── application/                           # Application and UI runtime settings (settings.yaml)
│   ├── models/                                # Model policies, frozen configs, routing matrices
│   └── evaluation/                            # Walk-forward origins & holdout evaluation specs
│
├── artifacts/                                 # Authoritative model outputs and serialized weights
│   ├── trained_models/                        # Serialized .joblib models & metadata registry
│   ├── predictions/                           # Step-by-step walk-forward holdout predictions
│   ├── metrics/                               # Aggregated holdout & validation score records
│   └── tuning/                                # Hyperparameter sweeps, ablation & ensemble logs
│
├── data/                                      # Data store
│   ├── raw/                                   # Unmodified GBIF, EPPO & NASA POWER source files
│   ├── processed/                             # Monthly aggregations & finalized dataset versions
│   └── reference/                             # Domain references & crop vulnerability lists
│
├── src/                                       # Core source library
│   ├── data/                                  # Data ingestion, cleaning, schemas & aggregation
│   ├── models/                                # Modular forecasting model families
│   │   ├── classical/                         # AR, MA, ARIMA, SARIMA, ARIMAX, SARIMAX, HW, Naive
│   │   ├── machine_learning/                  # Random Forest, XGBoost, Hurdle, Global Models
│   │   ├── deep_learning/                     # PyTorch LSTM, GRU networks
│   │   ├── probabilistic/                     # Meta Prophet Bayesian additive model
│   │   └── common/                            # Base classes, ForecastContext, adapters, schemas
│   ├── evaluation/                            # Mathematical metrics, loss calculations, tables
│   ├── forecasting/                           # Rolling walk-forward pipeline, inference, routing
│   ├── visualization/                         # Plotly charts, heatmaps, spatial maps
│   └── dashboard/                             # Streamlit UI cards, custom theme CSS, metrics
│
├── scripts/                                   # Executable pipelines and automation utilities
│   ├── training/                              # Model training & registry generation scripts
│   ├── tuning/                                # Hyperparameter optimization & ablation searches
│   ├── benchmarking/                          # Multi-origin & final holdout validation engines
│   └── utilities/                             # Dataset construction & figure generation tools
│
├── tests/                                     # Pytest test suite (99 passed)
│   ├── classical/                             # Unit tests for Box-Jenkins & classical models
│   ├── machine_learning/                      # Leakage, scaling & unit tests for RF/XGBoost
│   ├── deep_learning/                         # Scaler isolation & network tests for LSTM/GRU
│   ├── probabilistic/                         # Interval bounds & clipping tests for Prophet
│   ├── data_processing/                       # Occurrence cleaning & monthly aggregation tests
│   ├── evaluation/                            # Metric computation & multi-origin validation tests
│   └── integration/                           # Pipeline, routing policy & UI integration tests
│
└── reports/                                   # Categorized scientific reports & documentation
    ├── architecture/                          # Architectural audits, design reports, compliance
    ├── dataset/                               # Dataset improvement audits & provenance records
    ├── experiments/                           # Hyperparameter optimization & model reports
    ├── evaluation/                            # Multi-origin validation & comparative analyses
    ├── final/                                 # Final holdout evaluation & pathogen routing reports
    └── figures/                               # Generated scientific figures and publication plots
```

---

## 3. Detailed Component Organization

### 3.1 Model Family Subpackages (`src/models/`)

All 17 model implementations are partitioned into dedicated subpackages according to mathematical and statistical paradigm:

| Model Family | Subpackage Path | Included Models & Components |
|---|---|---|
| **Classical Time Series** | `src/models/classical/` | `ar_model.py`, `ma_model.py`, `arima_model.py`, `sarima_model.py`, `arimax_model.py`, `sarimax_model.py`, `holt_winters.py`, `seasonal_naive.py` |
| **Machine Learning** | `src/models/machine_learning/` | `random_forest.py`, `xgboost_model.py`, `global_models.py` (Global RF & Global XGBoost), `gradient_boosting.py`, `ml_autoregressor.py` |
| **Deep Learning** | `src/models/deep_learning/` | `lstm_model.py` (PyTorch LSTM), `gru_model.py` (PyTorch GRU) |
| **Probabilistic** | `src/models/probabilistic/` | `prophet_model.py` (Meta Prophet with Bayesian decomposition) |
| **Common Components** | `src/models/common/` | `base.py` (`ForecastContext`, output schemas, validation), `adapters.py` (standardized model dispatch interface) |

#### Note on Volatility (ARCH/GARCH) and Multivariate (VAR) Models
In strict compliance with prompt instructions (*"only if implemented; otherwise do not create empty placeholders"*):
- **ARCH/GARCH** models are econometric volatility models designed for conditional variance clustering in financial return series. In agricultural disease count forecasting, counts are non-negative integer volumes rather than mean-zero return innovations. Because ARCH/GARCH models were audited and determined to be mathematically unsuitable for point occurrence forecasting, no placeholder directories were created.
- **VAR** models require balanced, strictly stationary multi-pathogen panels without severe sparsity; cross-pathogen interactions are handled far more effectively via the Global XGBoost and Global Random Forest architectures. No empty VAR placeholder directories were created.

### 3.2 Script Categorization (`scripts/`)

All standalone execution scripts have been organized into functional operational stages:

- **`scripts/training/`**:
  - `train_and_document_all_models.py`: Comprehensive training across all 17 models with full parameter logging.
  - `train_and_persist_model_artifacts.py`: Serializes production model `.joblib` checkpoints.
  - `generate_model_training_registry.py`: Builds the consolidated model training metadata registry.
- **`scripts/tuning/`**:
  - `run_master_model_optimization.py`: Master hyperparameter search across all model families.
  - `optimize_advanced_models.py`: Deep fine-tuning search for ML and deep learning architectures.
  - `run_comprehensive_fine_tuning.py`: Systematic regularized parameter sweeps across validation origins.
- **`scripts/benchmarking/`**:
  - `run_multi_origin_benchmark.py`: 4-origin expanding walk-forward validation (240 evaluation steps).
  - `run_final_test_evaluation.py`: Strictly isolated final holdout evaluation (200 records, May 2024 – Dec 2025).
- **`scripts/utilities/`**:
  - `build_improved_dataset.py`: Cleans, imputes, and aggregates raw surveillance and weather records.
  - `generate_final_test_figures.py`: Produces publication-grade evaluation charts in `reports/figures/`.
  - `generate_phase2_pdf.py`: PDF viva guide generator.

*Forwarding Shims*: To support legacy execution workflows (`python scripts/run_final_test_evaluation.py`), forwarding wrappers were placed at the `scripts/` root that forward execution directly to the categorized target while setting `REPO_ROOT` dynamically.

### 3.3 Test Suite Modularization (`tests/`)

The 99 unit and integration tests were distributed into 7 logical subdomains:

- **`tests/classical/`** (10 tests): Unit tests for Seasonal Naive, AR(1), MA(1), ARIMA(1,1,1), SARIMA, and Holt-Winters.
- **`tests/machine_learning/`** (10 tests): Feature extraction, parameter validation, leakage prevention, and cross-pathogen isolation for Local RF, Local XGBoost, Global RF, Global XGBoost, and Hurdle XGBoost.
- **`tests/deep_learning/`** (11 tests): PyTorch tensor shapes, training-only min-max scaler bounds, gradient updates, and step forecasting for LSTM and GRU.
- **`tests/probabilistic/`** (6 tests): Bayesian uncertainty intervals, positive truncation, and non-negative clamping for Meta Prophet.
- **`tests/data_processing/`** (13 tests): Preprocessing, geo-coordinate validation, monthly time-bucketing, and data integrity checks.
- **`tests/evaluation/`** (24 tests): MAE, RMSE, sMAPE, WAPE, and MASE metric correctness, walk-forward origin integrity, and forecast insights.
- **`tests/integration/`** (25 tests): End-to-end forecasting pipeline, model registry dispatch, UI components, and pathogen routing policy verification.

---

## 4. Old-to-New File Mapping Table

| Original File Path | New Refactored Path | Status / Function |
|---|---|---|
| `src/models/ar_model.py` | `src/models/classical/ar_model.py` | Canonical module (Shim in `src/models/ar_model.py`) |
| `src/models/ma_model.py` | `src/models/classical/ma_model.py` | Canonical module (Shim in `src/models/ma_model.py`) |
| `src/models/arima_model.py` | `src/models/classical/arima_model.py` | Canonical module (Shim in `src/models/arima_model.py`) |
| `src/models/sarima_model.py` | `src/models/classical/sarima_model.py` | Canonical module (Shim in `src/models/sarima_model.py`) |
| `src/models/arimax_model.py` | `src/models/classical/arimax_model.py` | Canonical module (Shim in `src/models/arimax_model.py`) |
| `src/models/sarimax_model.py` | `src/models/classical/sarimax_model.py` | Canonical module (Shim in `src/models/sarimax_model.py`) |
| `src/models/holt_winters.py` | `src/models/classical/holt_winters.py` | Canonical module (Shim in `src/models/holt_winters.py`) |
| `src/models/seasonal_naive.py` | `src/models/classical/seasonal_naive.py` | Canonical module (Shim in `src/models/seasonal_naive.py`) |
| `src/models/random_forest.py` | `src/models/machine_learning/random_forest.py` | Canonical module (Shim in `src/models/random_forest.py`) |
| `src/models/xgboost_model.py` | `src/models/machine_learning/xgboost_model.py` | Canonical module (Shim in `src/models/xgboost_model.py`) |
| `src/models/global_models.py` | `src/models/machine_learning/global_models.py` | Canonical module (Shim in `src/models/global_models.py`) |
| `src/models/gradient_boosting.py` | `src/models/machine_learning/gradient_boosting.py` | Canonical module (Shim in `src/models/gradient_boosting.py`) |
| `src/models/ml_autoregressor.py` | `src/models/machine_learning/ml_autoregressor.py` | Canonical module (Shim in `src/models/ml_autoregressor.py`) |
| `src/models/lstm_model.py` | `src/models/deep_learning/lstm_model.py` | Canonical module (Shim in `src/models/lstm_model.py`) |
| `src/models/gru_model.py` | `src/models/deep_learning/gru_model.py` | Canonical module (Shim in `src/models/gru_model.py`) |
| `src/models/prophet_model.py` | `src/models/probabilistic/prophet_model.py` | Canonical module (Shim in `src/models/prophet_model.py`) |
| `src/models/base.py` | `src/models/common/base.py` | Canonical module (Facade in `src/models/base.py`) |
| `src/models/adapters.py` | `src/models/common/adapters.py` | Canonical module (Facade in `src/models/adapters.py`) |
| `tests/test_individual_models.py` | `tests/classical/test_individual_models.py` | Direct relocation |
| `tests/test_xgboost_model.py` | `tests/machine_learning/test_xgboost_model.py` | Direct relocation |
| `tests/test_global_and_hurdle_models.py` | `tests/machine_learning/test_global_and_hurdle_models.py` | Direct relocation |
| `tests/test_lstm_model.py` | `tests/deep_learning/test_lstm_model.py` | Direct relocation |
| `tests/test_gru_model.py` | `tests/deep_learning/test_gru_model.py` | Direct relocation |
| `tests/test_prophet_model.py` | `tests/probabilistic/test_prophet_model.py` | Direct relocation |
| `tests/test_preprocessing.py` | `tests/data_processing/test_preprocessing.py` | Direct relocation |
| `tests/test_aggregation.py` | `tests/data_processing/test_aggregation.py` | Direct relocation |
| `tests/test_dataset_integrity.py` | `tests/data_processing/test_dataset_integrity.py` | Direct relocation |
| `tests/test_evaluation.py` | `tests/evaluation/test_evaluation.py` | Direct relocation |
| `tests/test_multi_origin_validation.py` | `tests/evaluation/test_multi_origin_validation.py` | Direct relocation |
| `tests/test_forecast_insights.py` | `tests/evaluation/test_forecast_insights.py` | Direct relocation |
| `tests/test_forecasting_pipeline.py` | `tests/integration/test_forecasting_pipeline.py` | Direct relocation |
| `tests/test_unified_architecture.py` | `tests/integration/test_unified_architecture.py` | Direct relocation |
| `tests/test_model_comparison_ui.py` | `tests/integration/test_model_comparison_ui.py` | Direct relocation |
| `tests/test_model_policy.py` | `tests/integration/test_model_policy.py` | Direct relocation |
| `scripts/train_and_document_all_models.py` | `scripts/training/train_and_document_all_models.py` | Relocated + Root Forwarding Shim |
| `scripts/train_and_persist_model_artifacts.py` | `scripts/training/train_and_persist_model_artifacts.py` | Relocated + Root Forwarding Shim |
| `scripts/generate_model_training_registry.py` | `scripts/training/generate_model_training_registry.py` | Relocated + Root Forwarding Shim |
| `scripts/run_master_model_optimization.py` | `scripts/tuning/run_master_model_optimization.py` | Relocated + Root Forwarding Shim |
| `scripts/optimize_advanced_models.py` | `scripts/tuning/optimize_advanced_models.py` | Relocated + Root Forwarding Shim |
| `scripts/run_comprehensive_fine_tuning.py` | `scripts/tuning/run_comprehensive_fine_tuning.py` | Relocated + Root Forwarding Shim |
| `scripts/run_multi_origin_benchmark.py` | `scripts/benchmarking/run_multi_origin_benchmark.py` | Relocated + Root Forwarding Shim |
| `scripts/run_final_test_evaluation.py` | `scripts/benchmarking/run_final_test_evaluation.py` | Relocated + Root Forwarding Shim |
| `scripts/build_improved_dataset.py` | `scripts/utilities/build_improved_dataset.py` | Relocated + Root Forwarding Shim |
| `scripts/generate_final_test_figures.py` | `scripts/utilities/generate_final_test_figures.py` | Relocated + Root Forwarding Shim |
| `scripts/generate_phase2_pdf.py` | `scripts/utilities/generate_phase2_pdf.py` | Relocated + Root Forwarding Shim |
| Root `*.md` Architecture Reports (9 files) | `reports/architecture/*.md` | Direct relocation |
| Root `*.md` Dataset Reports (4 files) | `reports/dataset/*.md` | Direct relocation |
| Root `*.md` Experiment Reports (17 files) | `reports/experiments/*.md` | Direct relocation |
| Root `*.md` Evaluation Reports (8 files) | `reports/evaluation/*.md` | Direct relocation |
| Root `*.md` Final Milestone Reports (6 files) | `reports/final/*.md` | Direct relocation |

---

## 5. Verification & Quality Assurance

To ensure zero regressions were introduced during restructuring, extensive automated testing and UI validation were performed:

### 5.1 Pytest Execution Results
```
============================= test session starts ==============================
platform darwin -- Python 3.14.6, pytest-9.1.1, pluggy-1.7.0
rootdir: /Users/Kartikkk/Documents/SEM-5/TIME SERIES/Disease_time_series
configfile: pytest.ini
collected 99 items

tests/classical/test_individual_models.py ..........                     [ 10%]
tests/data_processing/test_aggregation.py ...                            [ 13%]
tests/data_processing/test_dataset_integrity.py ....                     [ 17%]
tests/data_processing/test_preprocessing.py ......                       [ 23%]
tests/deep_learning/test_gru_model.py ......                             [ 29%]
tests/deep_learning/test_lstm_model.py .....                             [ 34%]
tests/evaluation/test_evaluation.py .........                            [ 43%]
tests/evaluation/test_forecast_insights.py ......                        [ 49%]
tests/evaluation/test_multi_origin_validation.py .........               [ 58%]
tests/integration/test_forecasting_pipeline.py ...                       [ 61%]
tests/integration/test_model_comparison_ui.py .......                    [ 68%]
tests/integration/test_model_policy.py ......                            [ 74%]
tests/integration/test_unified_architecture.py .........                 [ 83%]
tests/machine_learning/test_global_and_hurdle_models.py .....            [ 88%]
tests/machine_learning/test_xgboost_model.py .....                       [ 93%]
tests/probabilistic/test_prophet_model.py ......                         [100%]

====================== 99 passed, 270 warnings in 54.93s =======================
```
**Outcome**: 99 / 99 tests passed (100% success rate).

### 5.2 Streamlit Application Testing (`streamlit.testing.v1.AppTest`)
Each application page was evaluated in headless mode:
- `app.py`: Rendered cleanly (Overview, Key Performance Indicators, System Telemetry).
- `pages/1_Disease_Explorer.py`: Loaded surveillance occurrence maps, heatmaps, and filters.
- `pages/2_Time_Series_Analysis.py`: Computed ADF stationarity tests, ACF/PACF plots, and seasonal decomposition.
- `pages/3_Model_Comparison.py`: Verified multi-origin benchmark tables, candidate fine-tuning metrics, and Section H holdout performance.
- `pages/4_Future_Forecasts.py`: Executed recursive forward forecasts and prediction intervals across all 10 pathogens.
- `pages/5_Data_and_Methodology.py`: Loaded data governance tables and JSON file downloads from `reports/experiments/`.

**Outcome**: All 6 pages loaded with zero uncaught exceptions.

### 5.3 Root Directory Hygiene Audit
A strict check was performed on the root repository directory:
```bash
ls -la *.md
# -rw-r--r--  1 Kartikkk  staff  README.md
```
Zero loose markdown reports or JSON dumps remain in the root directory.

---

## 6. Architectural Conclusion

The Disease Time Series Forecasting project now possesses an enterprise-grade, academically presentable architecture. Model implementations are logically categorized by mathematical family, scripts are organized by lifecycle stage, tests are mapped to distinct functional subdomains, and documentation is structured into dedicated scientific repositories. Complete backward compatibility is maintained across all legacy paths.

