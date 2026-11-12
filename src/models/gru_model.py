"""PyTorch GRU Univariate Time-Series Forecasting Model.

Designed for compact, sparse monthly plant disease surveillance series (~80-120
training observations), mirroring the LSTM pipeline for a fair comparison:
- Architecture: 1-layer GRU (16 hidden units) + Dropout (0.1) + Linear(16, 1)
- Sequence Window: Reuses chronological 12-month lookback (`create_lstm_sequences`)
- Scaling: MinMaxScaler(0, 1) fitted strictly on the training split (zero leakage)
- Early Stopping: Chronological internal validation tail (last ~15% of training sequences)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from src.models.lstm_model import (
    DEFAULT_LSTM_CONFIG,
    HAS_TORCH,
    create_lstm_sequences,
)

if HAS_TORCH:
    import torch
    import torch.nn as nn
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    class _GRUNet(nn.Module):
        """Compact single-layer GRU + Dropout + Linear head."""

        def __init__(self, hidden_size: int = 16, dropout: float = 0.1) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.gru(x)
            last_step = out[:, -1, :]
            return self.fc(self.dropout(last_step))
else:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _GRUNet = None  # type: ignore[assignment]


DEFAULT_GRU_CONFIG: Dict[str, Any] = dict(DEFAULT_LSTM_CONFIG)


def forecast_gru(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    gru_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, None, None, Any, Dict[str, Any]]:
    """Fit a compact univariate PyTorch GRU on chronological sequences and forecast.

    Returns:
        Tuple of (point_predictions, None, None, fitted_model, diagnostics).
    """
    if not HAS_TORCH or torch is None or _GRUNet is None:
        raise ImportError("PyTorch (torch) is not installed or unavailable in this environment.")

    t_start = time.perf_counter()
    cfg = {**DEFAULT_GRU_CONFIG, **(gru_config or {})}

    lookback = int(cfg["lookback"])
    units = int(cfg["units"])
    dropout = float(cfg["dropout"])
    lr = float(cfg["learning_rate"])
    batch_size = int(cfg["batch_size"])
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["patience"])
    val_fraction = float(cfg["val_fraction"])
    seed = int(cfg["seed"])

    df_sorted = train_df.copy()
    if "date" in df_sorted.columns:
        df_sorted["date"] = pd.to_datetime(df_sorted["date"])
        df_sorted = df_sorted.sort_values("date").reset_index(drop=True)

    raw_series = pd.to_numeric(df_sorted[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if len(raw_series) <= lookback + 4:
        raise ValueError(
            f"Insufficient training observations ({len(raw_series)}) for GRU with lookback={lookback}."
        )

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Fit MinMaxScaler strictly on pre-validation training slice to avoid scaling leakage
    n_total = len(raw_series)
    n_val_points = max(4, int(round(n_total * val_fraction)))
    n_fit_points = n_total - n_val_points

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(raw_series[:n_fit_points].reshape(-1, 1))
    scaled_all = scaler.transform(raw_series.reshape(-1, 1)).reshape(-1).astype(np.float32)

    X_all, y_all = create_lstm_sequences(scaled_all, lookback=lookback)
    n_train_seq = max(1, n_fit_points - lookback)
    X_train, y_train = X_all[:n_train_seq], y_all[:n_train_seq]
    X_val, y_val = X_all[n_train_seq:], y_all[n_train_seq:]

    device = torch.device("cpu")
    model = _GRUNet(hidden_size=units, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device) if len(X_val) > 0 else None
    y_val_t = torch.from_numpy(y_val).to(device) if len(y_val) > 0 else None

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0
    epochs_trained = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for start_idx in range(0, len(X_train_t), batch_size):
            xb = X_train_t[start_idx : start_idx + batch_size]
            yb = y_train_t[start_idx : start_idx + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        epochs_trained = epoch

        if X_val_t is not None and y_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(X_val_t), y_val_t).item())
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        else:
            best_epoch = epoch

    if X_val_t is not None:
        model.load_state_dict(best_state)

    model.eval()
    window = list(scaled_all[-lookback:])
    scaled_preds: List[float] = []
    with torch.no_grad():
        for _ in range(horizon):
            x_step = torch.tensor(window[-lookback:], dtype=torch.float32).view(1, lookback, 1).to(device)
            y_hat_scaled = float(model(x_step).item())
            scaled_preds.append(y_hat_scaled)
            window.append(y_hat_scaled)

    inv_preds = scaler.inverse_transform(np.asarray(scaled_preds, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    preds_arr = np.maximum(0.0, inv_preds.astype(float))
    runtime_sec = round(time.perf_counter() - t_start, 4)

    diagnostics: Dict[str, Any] = {
        "framework": f"PyTorch {torch.__version__}",
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "lookback": lookback,
        "units": units,
        "dropout": dropout,
        "learning_rate": lr,
        "batch_size": batch_size,
        "scaling_method": "MinMaxScaler(0, 1)",
        "training_runtime_sec": runtime_sec,
        "training_runtime": runtime_sec,
        "pred_std": round(float(np.std(preds_arr)), 4),
        "near_zero_pct": round(float(np.mean(preds_arr < 0.05) * 100.0), 2),
        "pred_mean": round(float(np.mean(preds_arr)), 4),
        "pred_min": round(float(np.min(preds_arr)), 4),
        "pred_max": round(float(np.max(preds_arr)), 4),
        "status": "ok",
    }

    return preds_arr, None, None, model, diagnostics

