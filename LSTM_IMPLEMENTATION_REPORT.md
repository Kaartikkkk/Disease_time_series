# LSTM Implementation and Evaluation Report

## 1. Environment & Framework Selection
- **Python**: `3.14.6` (`arm64` macOS, 8 CPUs)
- **Deep Learning Framework**: `torch==2.15.0` (TensorFlow/Keras is not installed in Python 3.14.6).
- **Execution Device**: Explicit `cpu` device (`torch.device("cpu")`) inside `forecast_lstm()` to avoid MPS kernel-launch overhead on small `(batch=8, seq=12, feat=1)` tensors.
- **Thread Safety**: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)` configured in `src/models/lstm_model.py` to prevent OpenMP thread-pool contention / segmentation faults on macOS `arm64` when running alongside XGBoost, LightGBM, and CmdStan/Prophet in the same process.

## 2. Architecture & Leakage-Free Design (`src/models/lstm_model.py`)
- **Model Topology (`_LSTMNet`)**:
  - `LSTM(input_size=1, hidden_size=16, batch_first=True)`
  - `Dropout(p=0.1)`
  - `Linear(in_features=16, out_features=1)`
- **Sequence Construction (`create_lstm_sequences`)**:
  - Univariate chronological sliding window with `lookback=12` (months `t-12 .. t-1` -> month `t`).
  - No shuffling across time; `val_fraction=0.15` reserves the chronological tail of the training split for early stopping.
- **Scaling (`MinMaxScaler(feature_range=(0.0, 1.0))`)**:
  - Fitted strictly on `raw_series[:n_fit_points]` **before** transforming the chronological internal validation tail (`raw_series[n_fit_points:]`), preventing any validation or test scaling leakage.
- **Training & Early Stopping**:
  - `Adam` optimizer (`learning_rate=0.005`), `MSELoss`, `batch_size=8`, `max_epochs=80`, `patience=10`, deterministic `seed=42`.
  - Predictions are generated recursively over `horizon`, inverse-transformed back to original occurrence counts, and clipped at `>= 0.0`.
- **Forecast Contract (`src/models/adapters.py`, `src/model_registry.py`)**:
  - Registered as `"lstm"` (`display_name="LSTM (Sequence Neural Net)"`, `category="Deep Learning"`, `available=HAS_TORCH`, `has_ci=False`, `supports_exog=False`).
  - Returns `lower_bound=None` and `upper_bound=None` rather than fabricating prediction intervals.

## 3. Single-Fit LSTM Diagnostics (`pyricularia_oryzae`, 126 Train Months, 6-Month Horizon)
- **Status**: `ok` (`framework: PyTorch 2.15.0`)
- **Samples**: `train_samples=95`, `val_samples=19`
- **Training**: `epochs_trained=12`, `best_epoch=2`, `training_runtime_sec=0.9843`
- **6-Month Forecast**: `[1.1501, 1.1233, 1.1151, 1.1117, 1.1104, 1.1099]`
- **Prediction Summary**: `pred_mean=1.1201`, `pred_std=0.0142`, `pred_min=1.1099`, `pred_max=1.1501`, `near_zero_pct=0.0%`

## 4. Rice Blast (`pyricularia_oryzae`) 6-Month Walk-Forward Benchmark (All 13 Active Models)
- **Dataset**: `data/processed/monthly_pyricularia_oryzae.csv` (`132` monthly rows, `2015-01-01` to `2025-12-01`)
- **Validation Dates**: `2022-09-01`, `2022-10-01`, `2022-11-01`, `2022-12-01`, `2023-01-01`, `2023-02-01`
- **Validation Actuals**: `[0.0, 0.0, 0.0, 4.0, 9.0, 6.0]` (`std=3.4841`, `zero_pct=50.00%`)
- **Total 13-Model Walk-Forward Runtime**: `7.52s` (`model_failures={}`)

| Model | Category | RMSE | MAE | MASE | sMAPE (%) | WAPE (%) | Pred Std | Near-Zero % (<0.05) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AR(1) | Statistical | 3.738 | 2.675 | 2.931 | 163.71% | 84.5% | 0.8187 | 0.00% |
| ARIMA(1,1,1) | Statistical | 3.739 | 2.681 | 2.938 | 163.72% | 84.7% | 0.8052 | 0.00% |
| MA(1) | Statistical | 3.780 | 2.731 | 2.992 | 164.91% | 86.2% | 0.7118 | 0.00% |
| SARIMA(1,0,1)(1,0,1)[12] | Statistical | 3.881 | 2.860 | 3.134 | 167.01% | 90.3% | 1.1350 | 0.00% |
| SARIMAX (Seasonal + Weather) | Statistical | 3.884 | 2.932 | 3.213 | 166.68% | 92.6% | 1.2841 | 0.00% |
| ARIMAX (Weather Regressors) | Statistical | 4.108 | 3.296 | 3.612 | 170.84% | 104.1% | 0.4667 | 0.00% |
| **LSTM (Sequence Neural Net)** | **Deep Learning** | **4.174** | **2.917** | **3.197** | **176.44%** | **92.1%** | **0.6707** | **0.00%** |
| XGBoost (Autoregressive) | Machine Learning | 4.213 | 2.976 | 3.262 | 137.52% | 94.0% | 1.5961 | 33.33% |
| Random Forest (Autoregressive) | Machine Learning | 4.418 | 3.311 | 3.628 | 172.57% | 104.5% | 1.7841 | 0.00% |
| Hist Gradient Boosting (Autoregressive) | Machine Learning | 4.491 | 3.493 | 3.828 | 102.27% | 110.3% | 2.2638 | 50.00% |
| Seasonal Naive (Baseline) | Classical | 4.708 | 3.167 | 3.470 | 100.00% | 100.0% | 0.0000 | 100.00% |
| Prophet (Trend + Seasonality) | Statistical | 5.066 | 4.003 | 4.387 | 199.79% | 126.4% | 1.6802 | 50.00% |
| Holt-Winters Exponential Smoothing | Classical | 5.089 | 4.024 | 4.410 | 200.00% | 127.1% | 1.7319 | 66.67% |

### Key Observations
- LSTM (`RMSE=4.174`, `MAE=2.917`) ranked 7th of 13 overall, outperforming all tree-based ML baselines (`XGBoost`, `Random Forest`, `Hist Gradient Boosting`), `Seasonal Naive`, `Prophet`, and `Holt-Winters` on RMSE, while trailing the linear/seasonal statistical models (`AR(1)` at `RMSE=3.738`).
- In single-fit multi-step recursive mode (`horizon=6`), LSTM rapidly mean-reverts toward ~`1.11` (`pred_std=0.0142`), whereas in 1-step-ahead expanding-window walk-forward evaluation (`[0.648, 0.304, 0.259, 0.229, 0.358, 2.121]`, `pred_std=0.6707`), it responds to recent observed spikes (reaching `2.121` in `2023-02-01` after `2023-01-01` reached `9.0`).
- **Limitations**: With ~`95` training sequences and `0` exogenous weather features, recursive multi-step forecasts flatten quickly and underestimate high-count outbreak peaks (`9.0`). No uncertainty intervals are produced (`has_ci=False`).

## 5. Verification Results
- **Test Suite (`pytest tests/ -v`)**: `62 passed, 0 failed` (`44.09s`), including `tests/test_lstm_model.py` (5 tests).
- **Streamlit `AppTest`**:
  - `app.py`: `exceptions=0`
  - `pages/3_Model_Comparison.py`: `exceptions=0`
  - `pages/4_Future_Forecasts.py`: `exceptions=0`

