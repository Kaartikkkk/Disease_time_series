# Plant Disease Outbreak Forecasting Using Time Series Analysis

An end-to-end, scientifically rigorous machine learning and statistical time series forecasting platform for predicting plant disease epidemics, assessing outbreak risk, and assisting agricultural decision-making across diverse crop-pathogen pathosystems.

---

## 🌾 Project Overview

Plant disease outbreaks cause severe annual yield losses, threatening food security and agricultural supply chains. This repository implements a full-stack time-series engineering framework designed to forecast disease incidence, quantify epidemic risk levels, and guide targeted fungicide/containment interventions before catastrophic spread occurs.

### Key Capabilities
- **Exploratory Data Analysis (EDA)**: Missingness profiling, seasonal subseries decomposition, variance stabilization, and weather cross-correlation (CCF).
- **Statistical Stationarity Verification**: Dual ADF (unit root) and KPSS (trend-stationarity) hypothesis testing with automated order determination ($d, D$).
- **Comprehensive Model Suite**:
  - **Baselines**: Naïve Persistence, Seasonal Naïve ($s=7$), Moving Average ($w=7$), Historical Mean.
  - **Classical Autoregressive & Moving Average**: $\text{AR}(p)$, $\text{MA}(q)$, $\text{ARMA}(p,q)$, $\text{ARIMA}(p,d,q)$, $\text{SARIMA}(p,d,q)(P,D,Q)_s$, Holt-Winters Exponential Smoothing.
  - **Exogenous & Multivariate Modeling**: $\text{ARIMAX}$, $\text{SARIMAX}$ (with temperature/humidity covariates), and $\text{VAR}(p)$ (Vector Autoregression).
  - **Modern Time Series Libraries**: Meta's **Prophet** (Bayesian structural time series with additive seasonalities) and **AutoTS** (Automated Machine Learning genetic algorithm search).
  - **Machine Learning Regressors**: Random Forest, Histogram Gradient Boosting, and XGBoost with recursive multi-step inference.
- **Evaluation Framework**: Chronological expanding-window backtesting across multiple folds without future data leakage, evaluated on **RMSE**, **MAE**, and **MAPE**.
- **Probabilistic Risk Engine**: Gaussian tail integration $P(Y_t > \tau)$ over empirical historical percentiles with actionable agronomic advisories.
- **Streamlit Web Application**: 7-page interactive dashboard for data exploration, model diagnostics, multi-horizon forecasting, and risk management.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer [Data Ingestion & Provenance]
        D1[Agroscope Grapevine qPCR]
        D2[Embrapa Wheat Blast/FHB]
        D3[Gottwald Citrus Tristeza]
        D4[Bald Tomato TSWV]
        D5[Synthetic Potato Late Blight]
        W1[Open-Meteo Reanalysis API]
        D1 & D2 & D3 & D4 & D5 & W1 --> V[Schema Validation & Deduplication]
        V --> C[Processed Data Cache]
    end

    subgraph Feature_Layer [Feature & Stationarity Engineering]
        C --> S[ADF & KPSS Stationarity Checks]
        C --> DEC[STL & Classical Decomposition]
        C --> F1[Cyclical Sin/Cos Calendar Features]
        C --> F2[Leakage-Free Lags k>=1 & Rolling Stats]
        C --> F3[Microclimate Indices THI & RH]
    end

    subgraph Model_Layer [Model Registry & Training]
        S & F1 & F2 & F3 --> B[Baselines: Persistence, SNaive, MA]
        S & F1 & F2 & F3 --> CL[Classical: AR, MA, ARMA, ARIMA, SARIMA, Holt-Winters]
        S & F1 & F2 & F3 --> EX[Exogenous: ARIMAX, SARIMAX, VAR]
        S & F1 & F2 & F3 --> MOD[Modern: Prophet & AutoTS AutoML]
        S & F1 & F2 & F3 --> ML[ML: Random Forest, HistGB, XGBoost]
    end

    subgraph Eval_Layer [Chronological Backtesting & Evaluation]
        B & CL & EX & MOD & ML --> BT[Expanding-Window Backtesting]
        BT --> MET[Metrics: RMSE, MAE, MAPE, sMAPE, R2, MASE]
        BT --> RES[Residual Ljung-Box & Bias Diagnostics]
    end

    subgraph Decision_Layer [Risk Engine & Application]
        MET --> RISK[Risk Thresholds & Tail Probability P(Y > tau)]
        RISK --> APP[Interactive Streamlit Dashboard :8501]
        APP --> P1[Overview & Workflow]
        APP --> P2[Data Explorer]
        APP --> P3[Time Series Analysis]
        APP --> P4[Forecasting Studio]
        APP --> P5[Outbreak Risk Assessment]
        APP --> P6[Model Evaluation]
        APP --> P7[Data Management]
    end
```

---

## 📊 Theoretical Model Formulations

### 1. Autoregressive Model — $\text{AR}(p)$
Models disease incidence as a linear combination of its own previous $p$ realizations:
$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \epsilon_t$$
*Domain Interpretation*: Captures endogenous pathogen inoculum accumulation from recent incubation cycles.

### 2. Moving Average Model — $\text{MA}(q)$
Expresses current incidence as a function of the current white noise shock and past $q$ random shock innovations:
$$y_t = \mu + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$
*Domain Interpretation*: Represents temporary shocks, such as transient microclimatic spore-wash events or sudden precipitation pulses.

### 3. Autoregressive Moving Average — $\text{ARMA}(p,q)$
Combines persistence and shock absorption:
$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

### 4. Autoregressive Integrated Moving Average — $\text{ARIMA}(p,d,q)$
Applies degree $d$ non-seasonal differencing $\Delta^d y_t = (1 - B)^d y_t$ to achieve mean stationarity prior to $\text{ARMA}$ fitting:
$$\phi(B)(1 - B)^d y_t = c + \theta(B)\epsilon_t$$

### 5. Seasonal ARIMA — $\text{SARIMA}(p,d,q)(P,D,Q)_s$
Incorporates both non-seasonal and seasonal lag polynomials with seasonal periodicity $s$ (e.g., $s=7$ for weekly cycles or $s=365$ for annual agronomic seasons):
$$\Phi_P(B^s)\phi(B)(1 - B)^d(1 - B^s)^D y_t = \Theta_Q(B^s)\theta(B)\epsilon_t$$

### 6. Holt-Winters Triple Exponential Smoothing
Decomposes the series into level ($\ell_t$), trend ($b_t$), and seasonal component ($s_t$) via recursive smoothing equations:
$$\hat{y}_{t+h\mid t} = \ell_t + h b_t + s_{t+h-m(k+1)}$$
Adjusted with dampening factor $\phi$ to prevent unrealistic epidemic runaway.

### 7. $\text{ARIMAX}$ and $\text{SARIMAX}$
Extends $\text{ARIMA}/\text{SARIMA}$ with exogenous climatic regressors $\mathbf{X}_t$ (daily mean temperature, relative humidity):
$$y_t = \beta^T \mathbf{X}_t + \eta_t, \quad \phi(B)(1 - B)^d \eta_t = \theta(B)\epsilon_t$$
*Domain Interpretation*: Weather creates favorable infection windows that amplify the baseline autoregressive trajectory.

### 8. Vector Autoregression — $\text{VAR}(p)$
Multivariate system modeling simultaneous linear interdependencies across $K$ time series:
$$\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^{p} \mathbf{A}_i \mathbf{Y}_{t-i} + \mathbf{E}_t$$
where $\mathbf{Y}_t = [\text{Incidence}_t, \text{Temperature}_t, \text{Humidity}_t]^T$.

### 9. Prophet (Meta)
A modular Bayesian generalized additive model:
$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
where $g(t)$ is a piecewise linear/logistic growth trend, $s(t)$ represents Fourier seasonalities, and $h(t)$ incorporates irregular holiday/agronomic events.

### 10. AutoTS (Automated Time Series)
An automated machine learning framework utilizing genetic algorithms and evolutionary optimization over preprocessing transformers (Box-Cox, differencing, scaling) and diverse model families (ARIMA, Theta, ETS, GLM, Ensembles).

### 11. Machine Learning Regressors (Random Forest, HistGBM, XGBoost)
Recursive autoregressive multi-step forecasting:
$$\hat{y}_{t+h} = f\left(\hat{y}_{t+h-1}, \dots, \hat{y}_{t+h-k}, \sin\left(\frac{2\pi d}{365}\right), \cos\left(\frac{2\pi d}{365}\right), \mathbf{X}_{t+h}\right)$$
Features are strictly computed with lag $k \ge 1$ to preserve temporal integrity.

---

## 📈 Evaluation Metrics & Agronomic Interpretation

| Metric | Formulation | Agricultural & Decision Context |
| :--- | :--- | :--- |
| **MAE** | $\frac{1}{N}\sum_{t=1}^{N} \vert y_t - \hat{y}_t \vert$ | Mean error in natural disease incidence units (% or severity index). Unbiased expected error. |
| **RMSE** | $\sqrt{\frac{1}{N}\sum_{t=1}^{N} (y_t - \hat{y}_t)^2}$ | Penalizes large errors quadratically. Critical in epidemiology where underestimating a major spike leads to crop devastation. |
| **MAPE (%)** | $\frac{100\%}{N}\sum_{t=1}^{N}\frac{\vert y_t - \hat{y}_t \vert}{\vert y_t \vert}$ | Percentage scale error. Masked on near-zero values to prevent numerical explosion. |
| **sMAPE (%)** | $\frac{100\%}{N}\sum_{t=1}^{N}\frac{2\vert y_t - \hat{y}_t \vert}{\vert y_t \vert + \vert \hat{y}_t \vert}$ | Symmetric percentage metric bounded between 0% and 200%. Handles low incidence counts smoothly. |
| **$R^2$** | $1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}$ | Proportion of out-of-sample variance explained. Can be negative for models worse than the historical mean. |
| **MASE** | $\frac{\text{MAE}_{\text{model}}}{\frac{1}{T-1}\sum_{i=2}^{T}\vert y_i - y_{i-1} \vert}$ | Scaled against the in-sample naïve persistence benchmark. $\text{MASE} < 1.0$ confirms algorithmic superiority. |

---

## 🧪 Data Sources & Scientific Provenance

| Dataset Key | Host Crop & Disease | Temporal Span | Source Organization & License |
| :--- | :--- | :--- | :--- |
| `agroscope_grapevine` | Grapevine (*Plasmopara viticola* - Downy Mildew) | 2020–2022 (qPCR daily/weekly) | Agroscope Switzerland & Zenodo (`10.5281/zenodo.7341235`). Open Access. |
| `embrapa_wheat` | Wheat (*Magnaporthe oryzae* & *Fusarium*) | 2018–2021 (field trials) | Embrapa Brazil Open Data Portal. CC-BY 4.0. |
| `gottwald_citrus` | Citrus (*Citrus tristeza virus* - CTV) | Multi-year canopy surveys | Gottwald et al., packaged in R `epiphy`. Research usage. |
| `bald_tomato` | Tomato (*Tomato spotted wilt virus* - TSWV) | Multi-week nursery spread | Bald et al., packaged in R `epiphy`. Research usage. |
| `synthetic_demo_benchmark`| Potato Late Blight (*Phytophthora infestans*) | 2021–2023 (1095 daily steps) | Synthetic benchmark calibrated with biological degree-day formulas. Explicitly tagged. |

All external microclimatic covariates are enriched via Open-Meteo Historical Weather API using geographical coordinates corresponding to field site coordinates.

---

## 🚨 Outbreak Risk Estimation & Thresholds

Historical disease distribution percentiles define empirical risk boundaries:
- **Baseline ($\tau_{\text{base}}$)**: Median incidence ($50^{\text{th}}$ percentile). Normal endemic presence.
- **Moderate ($\tau_{75}$)**: $75^{\text{th}}$ percentile. Early warning threshold.
- **High ($\tau_{90}$)**: $90^{\text{th}}$ percentile. Epidemic alert.
- **Very High ($\tau_{97.5}$)**: $97.5^{\text{th}}$ percentile. Severe outbreak event.

### Probability of Outbreak
For forecast horizon $t+h$ with predicted mean $\hat{y}_{t+h}$ and forecast standard error $\sigma_h$:
$$P(Y_{t+h} > \tau_{90}) = 1 - \Phi\left(\frac{\tau_{90} - \hat{y}_{t+h}}{\sigma_h}\right)$$

### Actionable Agronomic Matrix
- **Low Risk ($P < 0.25$)**: Routine monitoring; no preventative chemical spray required.
- **Moderate Risk ($0.25 \le P < 0.50$)**: Increase scouting frequency to 48-hour intervals; verify bio-control readiness.
- **High Risk ($0.50 \le P < 0.75$)**: Preventive systemic fungicide application; check canopy aeration and drainage.
- **Severe Outbreak ($P \ge 0.75$)**: Emergency curative spray application, quarantine affected zones, restrict irrigation.

---

## 💻 Installation & Quickstart

### Prerequisites
- macOS, Linux, or Windows (WSL recommended for Windows)
- Python 3.10 to 3.14
- C++ compiler (for Stan/Prophet backend if compiling from source)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/plant-disease-forecasting.git
cd plant-disease-forecasting

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that all unit tests pass:
```bash
PYTHONPATH=. pytest -v tests/
```

### 3. Generate Benchmark Evaluation Reports
Run chronological backtesting and serialize model checkpoints:
```bash
PYTHONPATH=. python src/generate_evaluation_reports.py
```
Output files will be generated in `reports/model_comparison.csv` and `reports/evaluation_summary.json`.

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Navigation

1. **Overview & Workflow**: Interactive high-level summary, system architecture diagram, and pipeline methodology.
2. **Data Explorer**: Dataset selection (Agroscope, Embrapa, Gottwald, Bald, Synthetic), raw table inspection, missingness heatmap, and stationarity diagnostics.
3. **Time Series Analysis**: STL seasonal decomposition, rolling statistics, autocorrelation (ACF), partial autocorrelation (PACF), and weather cross-correlation (CCF).
4. **Forecasting Studio**: Interactive model selection (AR, MA, ARMA, ARIMA, SARIMA, Holt-Winters, ARIMAX, SARIMAX, VAR, Prophet, AutoTS, Random Forest, HistGBM, XGBoost), horizon slider ($H=7$ to $60$), and 95% confidence intervals.
5. **Outbreak Risk Assessment**: Exceedance probability plots, historical threshold references, and operational agronomic decision advisories.
6. **Model Evaluation & Leaderboard**: Expanding-window backtest leaderboard comparing models across MAE, RMSE, MAPE, sMAPE, R², and MASE, with horizon degradation curves and residual diagnostics.
7. **Data Management & Ingestion**: Data source provenance auditing, schema verification, and Open-Meteo weather reanalysis synchronization.

---

## 🔬 Directory Structure

```
Disease_time_series/
├── .venv/                          # Python virtual environment
├── config/
│   └── settings.yaml               # Pipeline hyperparameters and risk percentiles
├── data/
│   ├── raw/                        # Ingested raw datasets
│   └── processed/                  # Normalized, validated, and cached series
├── models/                         # Serialized production model checkpoints
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Comprehensive EDA & distribution diagnostics
│   ├── 02_time_series_analysis.ipynb# Stationarity, ADF/KPSS, ACF/PACF, STL
│   └── 03_model_experiments.ipynb  # Backtesting, model selection & AutoTS
├── reports/
│   ├── model_comparison.csv        # Benchmarking leaderboard across all models
│   └── evaluation_summary.json     # Machine-readable performance metrics
├── src/
│   ├── analysis/                   # Stationarity (ADF/KPSS), decomposition, EDA
│   ├── data/                       # Ingestion adapters, Open-Meteo client, cleaning
│   ├── evaluation/                 # Metrics (RMSE, MAE, MAPE), splits, backtesting
│   ├── features/                   # Cyclical calendar, lag generation, weather indices
│   ├── models/                     # Classical, ARIMAX, VAR, Prophet, AutoTS, ML
│   ├── risk/                       # Percentile thresholds, tail probabilities
│   ├── ui/                         # Streamlit custom components and plotting
│   └── utils/                      # Paths, logging, schema validation
├── tests/                          # 17 pytest verification test cases
├── app.py                          # Streamlit multi-page web application
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Complete documentation
```

---

## 📜 Academic Integrity & Responsible AI
- **No Future Leakage**: All feature extraction, differencing parameters, and rolling statistics are calculated exclusively on prior observations ($k \ge 1$).
- **Transparent Provenance**: Public research repositories are attributed. The synthetic potato late blight benchmark is explicitly tagged as a demonstration series.
- **Physical Bounds**: Disease incidence predictions are strictly bounded ($y \ge 0$).
