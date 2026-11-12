# GRU Implementation and Evaluation Report

## 1. Framework & Environment
- **Framework**: PyTorch `2.15.0` (reused from the existing LSTM environment; TensorFlow was not installed).
- **Platform**: Python `3.14.6`, macOS `arm64`.
- **Execution**: Explicit `torch.device("cpu")` with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)` to prevent OpenMP thread-pool contention on macOS `arm64`.

## 2. Architecture (`src/models/gru_model.py`)
- **Topology (`_GRUNet`)**:
  - `nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)`
  - `nn.Dropout(p=0.1)`
  - `nn.Linear(in_features=16, out_features=1)`
- Kept compact (16 hidden units, single recurrent layer) to suit the 132-month series length per pathogen.

## 3. Reused LSTM Utilities
- `src/models/gru_model.py` directly imports and reuses from `src/models/lstm_model.py` without modifying LSTM behavior:
  - `HAS_TORCH`
  - `DEFAULT_LSTM_CONFIG` (`DEFAULT_GRU_CONFIG = dict(DEFAULT_LSTM_CONFIG)`)
  - `create_lstm_sequences()`

## 4. Sequence Preparation
- Chronological sliding windows with `lookback=12` (months `t-12 .. t-1` -> month `t`), preserving temporal order without shuffling.
- Univariate historical `documented_occurrence_count` input only (no weather regressors in this initial GRU implementation).
- Insufficient history (`len(raw_series) <= lookback + 4`) raises `ValueError` and is handled by the unified failure-to-NaN mechanism.

## 5. Scaling
- Uses `MinMaxScaler(feature_range=(0.0, 1.0))` fitted strictly on `raw_series[:n_fit_points]` **before** transforming the chronological internal validation tail (`raw_series[n_fit_points:]`).
- Predictions are inverse-transformed back to original counts and clipped at `>= 0.0`.
- Recorded in diagnostics as `scaling_method="MinMaxScaler(0, 1)"`.

## 6. Training Configuration
- `lookback`: `12`
- `units`: `16`
- `dropout`: `0.1`
- `learning_rate`: `0.005` (`Adam`)
- `loss`: `MSELoss`
- `batch_size`: `8`
- `max_epochs`: `80`
- `patience`: `10`
- `val_fraction`: `0.15`
- `seed`: `42` (`np.random.seed(42)` and `torch.manual_seed(42)`)

## 7. Early Stopping
- Uses the final 15% chronological tail of the training split (`n_val_points = max(4, int(round(n_total * 0.15)))`).
- Saves and restores `best_state` achieving the lowest validation MSE (`val_loss < best_val_loss - 1e-6`), stopping after `patience=10` epochs without improvement.
- Never touches the 6-month evaluation period.

## 8. Leakage Prevention
- Scaler is fitted strictly on `raw_series[:n_fit_points]` (prior to internal validation).
- In walk-forward evaluation (`window_type="expanding"`), each step receives only observations strictly prior to `eval_date` and fits a fresh scaler and model.
- In fixed holdout evaluation, recursive multi-step forecasts feed prior model predictions back into the 12-month lookback window without reading holdout actuals.

## 9. Walk-Forward Evaluation (`pyricularia_oryzae`, 6 Months Expanding-Window)
- **Validation Dates**: `2022-09-01`, `2022-10-01`, `2022-11-01`, `2022-12-01`, `2023-01-01`, `2023-02-01`
- **Actual Values**: `[0.0, 0.0, 0.0, 4.0, 9.0, 6.0]` (`std=3.4841`, `zero_pct=50.00%`)
- **Total 14-Model Runtime**: `8.7s` (`model_failures={}`)

| Model | Category | RMSE | MAE | MASE | sMAPE (%) | WAPE (%) | Pred Std | Near-Zero % (<0.05) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AR(1) | Statistical | 3.738 | 2.675 | 2.931 | 163.71% | 84.5% | 0.8187 | 0.00% |
| ARIMA(1,1,1) | Statistical | 3.739 | 2.681 | 2.938 | 163.72% | 84.7% | 0.8052 | 0.00% |
| MA(1) | Statistical | 3.780 | 2.731 | 2.992 | 164.91% | 86.2% | 0.7118 | 0.00% |
| SARIMA(1,0,1)(1,0,1)[12] | Statistical | 3.881 | 2.860 | 3.134 | 167.01% | 90.3% | 1.1350 | 0.00% |
| SARIMAX (Seasonal + Weather) | Statistical | 3.884 | 2.932 | 3.213 | 166.68% | 92.6% | 1.2841 | 0.00% |
| **GRU (Gated Recurrent Unit)** | **Deep Learning** | **3.995** | **2.853** | **3.127** | **170.01%** | **90.1%** | **0.6327** | **0.00%** |
| ARIMAX (Weather Regressors) | Statistical | 4.108 | 3.296 | 3.612 | 170.84% | 104.1% | 0.4667 | 0.00% |
| LSTM (Sequence Neural Net) | Deep Learning | 4.174 | 2.917 | 3.197 | 176.44% | 92.1% | 0.6707 | 0.00% |
| XGBoost (Autoregressive) | Machine Learning | 4.213 | 2.976 | 3.262 | 137.52% | 94.0% | 1.5961 | 33.33% |
| Random Forest (Autoregressive) | Machine Learning | 4.418 | 3.311 | 3.628 | 172.57% | 104.5% | 1.7841 | 0.00% |
| Hist Gradient Boosting (Autoregressive) | Machine Learning | 4.491 | 3.493 | 3.828 | 102.27% | 110.3% | 2.2638 | 50.00% |
| Seasonal Naive (Baseline) | Classical | 4.708 | 3.167 | 3.470 | 100.00% | 100.0% | 0.0000 | 100.00% |
| Prophet (Trend + Seasonality) | Statistical | 5.066 | 4.003 | 4.387 | 199.79% | 126.4% | 1.6802 | 50.00% |
| Holt-Winters Exponential Smoothing | Classical | 5.089 | 4.024 | 4.410 | 200.00% | 127.1% | 1.7319 | 66.67% |

## 10. Fixed Holdout Evaluation (`pyricularia_oryzae`, 6-Month Recursive Holdout)
- **Total 14-Model Runtime**: `2.34s` (`model_failures=None`)
- **Validation Dates**: `2022-09-01` to `2023-02-01` (same actuals `[0.0, 0.0, 0.0, 4.0, 9.0, 6.0]`)
- **LSTM (Sequence Neural Net)**: `RMSE=4.266`, `MAE=3.152`, `MASE=3.455`, `sMAPE=178.27%`, `WAPE=99.5%`, `pred_std=0.0194`, `near_zero_pct=0.00%` (Rank 1 of 14)
- **AR(1)**: `RMSE=4.309`, `MAE=3.144`, `pred_std=0.0412`
- **GRU (Gated Recurrent Unit)**: `RMSE=4.410`, `MAE=3.162`, `MASE=3.465`, `sMAPE=185.07%`, `WAPE=99.8%`, `pred_std=0.0110`, `near_zero_pct=0.00%` (Rank 5 of 14)
- **Seasonal Naive (Baseline)**: `RMSE=4.708`, `MAE=3.167`, `pred_std=0.0000`, `near_zero_pct=100.00%`
- **XGBoost (Autoregressive)**: `RMSE=4.809`, `MAE=3.582`, `pred_std=0.8869`, `near_zero_pct=66.67%`
- **Prophet (Trend + Seasonality)**: `RMSE=5.072`, `MAE=4.029`, `pred_std=1.6791`, `near_zero_pct=50.00%`

## 11. Rice Blast Benchmark Summary
- In **1-step-ahead expanding-window walk-forward evaluation**, GRU ranked **6th of 14** (`RMSE=3.995`, `MAE=2.853`), outperforming LSTM (`4.174`), ARIMAX (`4.108`), XGBoost (`4.213`), Random Forest (`4.418`), Hist Gradient Boosting (`4.491`), Seasonal Naive (`4.708`), Prophet (`5.066`), and Holt-Winters (`5.089`), while trailing the linear/seasonal statistical models (`AR(1)` at `3.738`).
- In **6-step recursive fixed holdout**, LSTM (`RMSE=4.266`) slightly outperformed GRU (`RMSE=4.410`), though both ranked in the top 5 of 14.

## 12. LSTM vs. GRU Comparison (Single-Fit Diagnostics on 126 Train Months)
| Metric | LSTM | GRU |
|---|---:|---:|
| `train_samples` / `val_samples` | 95 / 19 | 95 / 19 |
| `epochs_trained` / `best_epoch` | 12 / 2 | 11 / 1 |
| `training_runtime_sec` | 0.2229s | 0.1432s |
| `pred_mean` | 1.1201 | 1.1406 |
| `pred_std` | 0.0142 | 0.0044 |
| `pred_min` / `pred_max` | 1.1099 / 1.1501 | 1.1355 / 1.1479 |
| `near_zero_pct` | 0.0% | 0.0% |
| Walk-Forward RMSE / MAE | 4.174 / 2.917 | 3.995 / 2.853 |
| Fixed Holdout RMSE / MAE | 4.266 / 3.152 | 4.410 / 3.162 |

- **Takeaway**: Neither recurrent model dominates across both protocols. GRU trains ~36% faster (`0.1432s` vs `0.2229s` single fit) and achieves lower walk-forward error (`RMSE 3.995` vs `4.174`) because its 1-step-ahead update responds slightly more strongly after the `2023-01-01` spike (`2.226` vs `2.121`), whereas LSTM has slightly lower fixed recursive holdout error (`4.266` vs `4.410`).

## 13. Sparse-Data Diagnostics
- Neither GRU nor LSTM collapses to all-zero predictions (`near_zero_pct=0.00%` in both walk-forward and fixed holdout), unlike `Seasonal Naive` (`100%`) or `Hist Gradient Boosting` (`50%`).
- However, in 6-step recursive holdout, both models rapidly damp toward a nearly constant mean level (`pred_std=0.0110` for GRU; `0.0194` for LSTM), under-predicting the `9.0` outbreak peak and over-predicting dormant `0.0` months (~`0.43`–`0.46`).

## 14. Limitations
- Small sample size (~95 training sequences) limits deep recurrent capacity.
- Univariate target-only input omits weather covariates (NASA POWER) in this initial implementation.
- Recursive multi-step damping flattens predictions over horizons beyond 1–2 months.
- No prediction intervals (`has_ci=False`, `lower_bound=None`, `upper_bound=None`).

## 15. Files Created and Modified
- **Created**:
  - `src/models/gru_model.py`
  - `tests/test_gru_model.py`
  - `GRU_IMPLEMENTATION_REPORT.md`
- **Modified**:
  - `src/models/adapters.py` (added `adapt_gru`)
  - `src/models/__init__.py` (exported `forecast_gru`)
  - `src/model_registry.py` (activated `"gru"` with `available=HAS_TORCH`, `supports_exog=False`, `has_ci=False`, `adapter=adapt_gru`)
  - `src/forecasting.py` (re-exported `forecast_gru`)
  - `pages/4_Future_Forecasts.py` (added `"GRU (Gated Recurrent Unit)"` selector & dispatch)
  - `tests/test_forecasting_pipeline.py` (updated model count assertions `13 -> 14`)
  - `tests/test_unified_architecture.py` (updated available model assertions `13 -> 14` and `"gru"` active)

## 16. Verification
- `pytest tests/ -v`: `68 passed, 0 failed` (`61.44s`).
- Streamlit `AppTest`: `app.py`, `pages/3_Model_Comparison.py`, and `pages/4_Future_Forecasts.py` all ran with `exceptions=0`; Page 4 `Prediction Method` selectbox includes `'Deep Learning (GRU)'`.

