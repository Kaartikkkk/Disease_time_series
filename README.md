# 🌾 Agricultural Disease Forecasting: Plant Disease Outbreak Forecasting Platform

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Statistical Suite](https://img.shields.io/badge/Statsmodels-0.14+-brightgreen.svg)](https://www.statsmodels.org/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn%20%7C%20XGBoost-1.5+-orange.svg)](https://xgboost.readthedocs.io/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-LSTM%20%7C%20GRU-red.svg)](https://pytorch.org/)
[![Probabilistic](https://img.shields.io/badge/Meta%20Prophet-Bayesian-informational.svg)](https://facebook.github.io/prophet/)
[![Testing](https://img.shields.io/badge/Pytest-99%20Passed%20(100%25)-success.svg)](https://pytest.org/)
[![Design System](https://img.shields.io/badge/Design%20System-Editorial%20Climate--Tech-102C26.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Academic and Portfolio-Grade Time-Series Surveillance & Outbreak Forecasting Platform for Indian Agriculture**  
> An end-to-end epidemiological forecasting and agro-meteorological intelligence platform utilizing **100% real, publicly available scientific surveillance data** from the **Global Biodiversity Information Facility (GBIF)**, climate analytics from **NASA POWER**, and phytosanitary records from the **European and Mediterranean Plant Protection Organization (EPPO)**.
>
> Built with an editorial climate-tech design system (*Fraunces* + *Plus Jakarta Sans* + *JetBrains Mono*), earthy botanical aesthetic (`#102C26`, `#527A50`, `#F6F4EC`, `#C4A66A`), unified SVG iconography, 17 multi-horizon forecasting models across 4 paradigm families, rigorous multi-origin walk-forward validation, and an untouched 20-month final holdout benchmark.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [Target Variable & Scientific Epistemology](#-target-variable--scientific-epistemology)
3. [System Architecture & Modular Design](#-system-architecture--modular-design)
4. [Directory Hierarchy & Code Organization](#-directory-hierarchy--code-organization)
5. [Forecasting Model Suite (17 Models)](#-forecasting-model-suite-17-models)
6. [Evaluation Methodology & Benchmark Results](#-evaluation-methodology--benchmark-results)
7. [Streamlit Multi-Page Web Platform](#-streamlit-multi-page-web-platform)
8. [Automation & Execution Scripts](#-automation--execution-scripts)
9. [Scientific Documentation & Reports Catalog](#-scientific-documentation--reports-catalog)
10. [Local Installation & Setup](#-local-installation--setup)
11. [Testing & Quality Verification (99/99 Passed)](#-testing--quality-verification-9999-passed)
12. [Scientific Governance & Attributions](#-scientific-governance--attributions)

---

## 🔬 Executive Summary & Problem Statement

Plant pathogens cause severe harvest losses and biosecurity threats across India's staple food and commercial crop belts. Timely, multi-step epidemiological forecasting enables state departments of agriculture, research institutions (ICAR), agricultural extension officers, and farming cooperatives to implement preventative interventions prior to catastrophic outbreak peaks.

Unlike conventional academic demonstrations relying on synthetic data or naive train/test splits, **Agricultural Disease Forecasting Using Time Series** enforces rigorous scientific integrity:
- **Zero Synthetic Observation Injection:** Grounded exclusively in verified, real-world historical field surveillance records across 10 critical Indian plant pathogens (132 calendar months, January 2015 – December 2025).
- **Leakage-Free Weather Conditioning:** Climate covariates from NASA POWER (precipitation, relative humidity, surface temperature) are strictly lagged ($t-1, t-2$) so future meteorology is never leaked into training models.
- **Multi-Family Benchmark:** Evaluates 17 forecasting architectures spanning classical Box-Jenkins econometrics, advanced machine learning (random forest, gradient boosting, two-stage hurdle models), deep learning recurrent networks (PyTorch LSTM, GRU), and Bayesian probabilistic modeling (Meta Prophet).
- **Strict Chronological Holdout Isolation:** The final 20 months (May 2024 – December 2025; 200 pathogen-months) were permanently isolated and untouched throughout hyperparameter tuning and model selection.

---

## 🎯 Target Variable & Scientific Epistemology

### Primary Target: `documented_occurrence_count`
$$\text{Target} = y_t = \sum_{i \in \mathcal{D}_t} 1$$
where $\mathcal{D}_t$ is the set of verified, geo-referenced disease occurrence records deposited in international and national repositories (GBIF, INSDC, ICAR, Herbarium accessions) within calendar month $t$.

### Monitored Pathogen Taxa (10 Indian Agricultural Diseases):
1. **Puccinia graminis** (Black / Stem Rust of Wheat)
2. **Puccinia striiformis** (Stripe / Yellow Rust of Wheat)
3. **Puccinia triticina** (Leaf / Brown Rust of Wheat)
4. **Magnaporthe oryzae** (Rice Blast Disease)
5. **Pyricularia oryzae** (Rice Blast Anamorph)
6. **Xanthomonas oryzae** (Bacterial Leaf Blight of Rice)
7. **Bipolaris oryzae** (Brown Spot of Rice)
8. **Ralstonia solanacearum** (Bacterial Wilt of Solanaceous Crops)
9. **Phytophthora infestans** (Late Blight of Potato & Tomato)
10. **Fusarium oxysporum** (Vascular Wilt Complex)

---

## 🏛️ System Architecture & Modular Design

The platform is designed around strict domain separation, modular components, and dependency inversion:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PUBLIC SURVEILLANCE & CLIMATE INGESTION              │
│       GBIF Occurrence API   │   NASA POWER Meteorology   │   EPPO DB   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  SPATIO-TEMPORAL CLEANING & AGGREGATION                │
│   src/data/ (deduplication, bounding-box geofencing, MS time-grid)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     MODULAR FORECASTING MODEL SUITE                    │
│                        (src/models/common/base.py)                     │
│                                                                        │
│   Classical Models         Machine Learning       Deep Learning        │
│   (src/models/classical/)  (src/models/machine_   (src/models/deep_    │
│   • Seasonal Naive         learning/)              learning/)          │
│   • AR(1), MA(1)           • Local RF & XGBoost   • PyTorch LSTM       │
│   • ARIMA(1,1,1)           • Global RF & XGBoost  • PyTorch GRU        │
│   • SARIMA, SARIMAX        • Hurdle XGBoost                            │
│   • Holt-Winters ESM                              Probabilistic        │
│                                                   (src/models/         │
│                                                   probabilistic/)      │
│                                                   • Meta Prophet       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  RIGOROUS WALK-FORWARD VALIDATION PIPELINE             │
│   src/forecasting/ & src/evaluation/ (4 Origins, 240 Evaluation Steps) │
│   Production Selection Policy: Pathogen-specific automated routing    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  STREAMLIT SCIENTIFIC WEB PLATFORM                     │
│   app.py + pages/ (Overview, Explorer, Analysis, Benchmark, Forecast)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Hierarchy & Code Organization

The repository is organized into cleanly partitioned subdirectories:

```
Disease_time_series/
├── README.md                                  # Complete academic & architectural documentation
├── requirements.txt                           # Frozen project dependencies
├── pytest.ini                                 # Pytest execution configuration
├── app.py                                     # Primary Streamlit web application
│
├── pages/                                     # Streamlit multi-page interface
│   ├── 1_Disease_Explorer.py                  # Raw surveillance data & pathogen profiles
│   ├── 2_Time_Series_Analysis.py              # Stationarity tests, PACF/ACF, decomposition
│   ├── 3_Model_Comparison.py                  # Benchmark comparison & holdout leaderboard
│   ├── 4_Future_Forecasts.py                  # Multi-step projections & prediction intervals
│   └── 5_Data_and_Methodology.py              # Governance downloads & mathematical formulas
│
├── configs/                                   # Central configuration repository
│   ├── application/                           # Application and UI runtime settings (settings.yaml)
│   ├── models/                                # Model policies, frozen configs, routing matrices
│   └── evaluation/                            # Walk-forward origins & holdout evaluation specs
│
├── artifacts/                                 # Serialized model artifacts & evaluation records
│   ├── trained_models/                        # Serialized .joblib models & registry metadata
│   ├── predictions/                           # Step-by-step walk-forward holdout predictions
│   ├── metrics/                               # Final holdout and multi-origin score records
│   └── tuning/                                # Hyperparameter sweeps, ablation & ensemble logs
│
├── data/                                      # Data store
│   ├── raw/                                   # Unmodified GBIF, EPPO & NASA source archives
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
│   ├── evaluation/                            # Metrics (RMSE, MAE, MASE, WAPE), loss calculations
│   ├── forecasting/                           # Rolling walk-forward pipeline, recursive inference
│   ├── visualization/                         # Plotly charts, heatmaps, spatial maps
│   └── dashboard/                             # Streamlit UI cards, custom theme CSS, metrics
│
├── scripts/                                   # Automation scripts and execution pipelines
│   ├── training/                              # Model training & registry generation scripts
│   ├── tuning/                                # Hyperparameter optimization & ablation searches
│   ├── benchmarking/                          # Multi-origin & final holdout validation engines
│   └── utilities/                             # Dataset construction & figure generation tools
│
├── tests/                                     # Pytest test suite (99 passed across 7 domains)
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

## 🤖 Forecasting Model Suite (17 Models)

The system features 17 rigorously implemented and evaluated forecasting models:

| Category | Model Name | Module Location | Description & Specification |
|---|---|---|---|
| **Baseline** | **Seasonal Naive** | `src.models.classical` | Annual persistence benchmark ($y_t = y_{t-12}$); reference for MASE and improvement ratios. |
| **Classical** | **AR(1)** | `src.models.classical` | Autoregressive model of lag order 1 with estimated drift constant. |
| **Classical** | **MA(1)** | `src.models.classical` | Moving average model with single invertibility-constrained innovation term. |
| **Classical** | **ARIMA(1,1,1)** | `src.models.classical` | Differenced Box-Jenkins autoregressive integrated moving average model. |
| **Classical** | **SARIMA** | `src.models.classical` | Seasonal $(1,0,1) \times (1,0,1)_{12}$ specification capturing annual monsoon cycles. |
| **Classical** | **Holt-Winters** | `src.models.classical` | Additive exponential smoothing with damped trend and 12-month seasonality. |
| **Classical** | **ARIMAX** | `src.models.classical` | ARIMAX model conditioned on 1-month lagged NASA POWER temperature and humidity. |
| **Classical** | **SARIMAX** | `src.models.classical` | Operational seasonal model with 1-month lagged weather regressors. |
| **Machine Learning** | **Local Random Forest** | `src.models.machine_learning` | Pathogen-specific ensemble of 180 trees (max depth 5, feature subsampling 0.7). |
| **Machine Learning** | **Local XGBoost** | `src.models.machine_learning` | Regularized gradient boosted tree ($\alpha=0.6, \lambda=2.5$, learning rate 0.025). |
| **Machine Learning** | **Global Random Forest** | `src.models.machine_learning` | Cross-pathogen pooled panel model (120 trees, depth 4, 10-pathogen one-hot encoding). |
| **Machine Learning** | **Global XGBoost** | `src.models.machine_learning` | Multi-pathogen gradient boosted tree with joint cross-series learning. |
| **Machine Learning** | **Hurdle XGBoost** | `src.models.machine_learning` | Two-stage architecture: binary outbreak classifier multiplied by non-zero count regressor. |
| **Deep Learning** | **PyTorch LSTM** | `src.models.deep_learning` | Recurrent Long Short-Term Memory network (lookback 6, 16 hidden units, dropout 0.1). |
| **Deep Learning** | **PyTorch GRU** | `src.models.deep_learning` | Gated Recurrent Unit network (lookback 12, 8 hidden units, learning rate 0.003). |
| **Probabilistic** | **Meta Prophet** | `src.models.probabilistic` | Bayesian additive model with piecewise linear growth, additive annual Fourier components. |
| **Ensemble** | **Production Policy** | `src.forecasting` | Automated selection policy routing each pathogen to its historically validated optimal model. |

---

## 📊 Evaluation Methodology & Benchmark Results

### 1. Multi-Origin Expanding Walk-Forward Validation (Prompt 18 & 19)
- **Period**: September 2022 to April 2024 (20 months)
- **Origins**: 4 chronological cutoff dates ($t_0 \in \{\text{Aug 22, Feb 23, Aug 23, Oct 23}\}$)
- **Horizons**: 1 to 6 months ahead ($H \in \{1, 2, 3, 4, 5, 6\}$)
- **Scale**: 240 evaluation steps (10 pathogens $\times$ 4 origins $\times$ 6 horizons)

#### Validation Performance Leaderboard:
| Rank | Model Architecture | Family | RMSE | MAE | Outbreak F1 | Improvement vs Baseline |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 🥇 | **GRU (Optimized)** | Deep Learning | **5.967** | **2.246** | 0.564 | **+7.3%** |
| 🥈 | **Hurdle XGBoost** | Machine Learning | **5.996** | 2.338 | **0.651** | **+6.8%** |
| 🥉 | **Prophet (Meta)** | Probabilistic | 6.021 | 2.294 | 0.739 | +6.4% |
| 4 | **Global XGBoost** | Machine Learning | 6.027 | 2.391 | 0.618 | +6.3% |
| 5 | **Global Random Forest** | Machine Learning | 6.044 | 2.372 | 0.603 | +6.1% |
| 6 | **Local Random Forest** | Machine Learning | 6.052 | 2.423 | 0.636 | +5.9% |
| 7 | **LSTM (Optimized)** | Deep Learning | 6.075 | 2.379 | 0.428 | +5.6% |
| 8 | **MA(1)** | Classical | 6.243 | 2.385 | 0.578 | +3.0% |
| 9 | **ARIMA(1,1,1)** | Classical | 6.248 | 2.388 | 0.578 | +2.9% |
| 10 | **AR(1)** | Classical | 6.251 | 2.392 | 0.578 | +2.8% |
| 11 | **Seasonal Naive** | Baseline | 6.435 | 2.490 | 0.490 | 0.0% (Ref) |

### 2. Strictly Preserved Final Holdout Evaluation (Prompt 29)
- **Period**: May 2024 to December 2025 (20 months; records 112 to 131)
- **Isolation**: Permanently locked and unaccessed during model tuning
- **Scale**: 200 pathogen-month observations (10 pathogens $\times$ 20 test steps)

#### Holdout Results (Generalization on Unseen Future):
| Model | RMSE | MAE | MASE | WAPE (%) | Outbreak Detection Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| **Hurdle XGBoost** | **5.589** | **2.285** | **0.868** | **94.8%** | **78.2%** |
| **Global Random Forest** | 5.621 | 2.304 | 0.875 | 95.6% | 74.5% |
| **Global XGBoost** | 5.645 | 2.312 | 0.878 | 95.9% | 76.4% |
| **GRU Network** | 5.698 | 2.341 | 0.889 | 97.1% | 69.1% |
| **Meta Prophet** | 5.742 | 2.368 | 0.899 | 98.3% | 70.9% |
| **MA(1) Classical** | 5.814 | 2.395 | 0.910 | 99.4% | 61.8% |
| **Seasonal Naive (Baseline)** | 6.210 | 2.512 | 1.000 | 104.2% | 52.7% |

---

## 💻 Streamlit Multi-Page Web Platform

Launch the dashboard to interact with all data and models:
```bash
streamlit run app.py
```

### Application Page Guide:
1. **`app.py` (Executive Overview)**: High-level system KPIs, monitoring status across 10 pathogens, data governance summary, and quick navigation cards.
2. **`pages/1_Disease_Explorer.py`**: Real GBIF surveillance records, interactive map with Indian agro-climatic boundaries, temporal trend graphs, and data filtering.
3. **`pages/2_Time_Series_Analysis.py`**: Rigorous statistical econometrics: Augmented Dickey-Fuller (ADF) tests, ACF/PACF with 95% Bartlett confidence limits, and seasonal decomposition.
4. **`pages/3_Model_Comparison.py`**: Self-explanatory comparative analysis:
   - Walk-forward benchmark leaderboard (4 origins, 240 steps)
   - Advanced model fine-tuning results (before vs. after regularized optimization)
   - Dynamic pathogen routing table
   - Untouched holdout evaluation & robustness analysis (May 2024 – Dec 2025)
5. **`pages/4_Future_Forecasts.py`**: Forward disease projections (1 to 12 months) with 80% and 95% prediction intervals, historical overlay, and outbreak risk level.
6. **`pages/5_Data_and_Methodology.py`**: Complete scientific epistemology, data provenance audit, equations, and downloadable evaluation registry JSON files.

---

## 🛠️ Automation & Execution Scripts

All execution pipelines are located in `scripts/`:

```bash
# 1. Training & Registry Generation
python scripts/training/train_and_document_all_models.py
python scripts/training/train_and_persist_model_artifacts.py
python scripts/training/generate_model_training_registry.py

# 2. Hyperparameter Tuning & Sweeps
python scripts/tuning/run_master_model_optimization.py
python scripts/tuning/optimize_advanced_models.py
python scripts/tuning/run_comprehensive_fine_tuning.py

# 3. Benchmarking Engines
python scripts/benchmarking/run_multi_origin_benchmark.py
python scripts/benchmarking/run_final_test_evaluation.py

# 4. Utilities & Visualizations
python scripts/utilities/build_improved_dataset.py
python scripts/utilities/generate_final_test_figures.py
```
*(Note: Forwarding shims allow invoking `python scripts/run_final_test_evaluation.py` directly).*

---

## 📚 Scientific Documentation & Reports Catalog

Over 40 comprehensive technical reports and audit logs are cataloged in `reports/`:

- **Architecture (`reports/architecture/`)**:
  - `PROJECT_ARCHITECTURE_AUDIT.md`: System-wide dependency and import audit.
  - `PROJECT_RESTRUCTURING_REPORT.md`: Full refactoring report and old-to-new file mappings.
  - `FINAL_PHASE_3_AUDIT_REPORT.md` & `FINAL_UI_CLEANUP_REPORT.md`.
- **Dataset (`reports/dataset/`)**:
  - `ORIGINAL_DATASET_AUDIT.md`: Audit of original historical surveillance data.
  - `DATASET_IMPROVEMENT_REPORT.md`: Scientific documentation of covariate imputation and name standardization.
  - `SYNTHETIC_REMOVAL_REPORT.md`: Formal verification of complete synthetic data removal.
- **Experiments & Tuning (`reports/experiments/`)**:
  - `ADVANCED_MODELS_FINE_TUNING_REPORT.md`: Systematic fine-tuning of ML and deep learning models.
  - `HYPERPARAMETER_OPTIMIZATION_REPORT.md`: Search space and validation score logs.
  - `FEATURE_ABLATION_REPORT.md`: Impact analysis of climate lags vs. autoregressive features.
- **Evaluation & Benchmarks (`reports/evaluation/`)**:
  - `MULTI_ORIGIN_VALIDATION_REPORT.md`: Complete 240-step walk-forward benchmark.
  - `FINAL_TEST_EVALUATION_REPORT.md`: Initial holdout evaluation analysis.
  - `CLASSICAL_VS_ADVANCED_FINAL_COMPARISON.md`: Cross-family comparative analysis.
- **Final Milestone Reports (`reports/final/`)**:
  - `FINAL_HOLDOUT_EVALUATION_REPORT.md`: Final untouched holdout evaluation results.
  - `PATHOGEN_WISE_MODEL_COMPARISON.md`: Individual pathogen breakdown and performance rankings.
  - `MODEL_ROBUSTNESS_REPORT.md`: Generalization, horizon degradation, and volatility analysis.
  - `FINAL_MODEL_SELECTION_REPORT.md`: Definitive production routing matrix and recommendations.

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/plant-disease-forecasting.git
cd plant-disease-forecasting

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing & Quality Verification (99/99 Passed)

Run the full pytest suite:
```bash
pytest tests/ -v
```

All 99 unit, leakage-prevention, and integration tests pass cleanly:
```
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

====================== 99 passed in 54.93s =======================
```

---

## 📜 Scientific Governance & Attributions

- **GBIF.org**: Global Biodiversity Information Facility ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **NASA POWER**: Atmospheric Science Data Center at NASA Langley Research Center.
- **EPPO**: European and Mediterranean Plant Protection Organization Global Database.
- **License**: Creative Commons Attribution 4.0 International ([CC-BY-4.0](LICENSE)).
