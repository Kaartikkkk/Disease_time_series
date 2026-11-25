"""PyTorch Time Series Transformer (PatchTST / Temporal Multi-Head Attention) Model.

Implements a compact, state-of-the-art Transformer architecture specifically tuned
for monthly agricultural disease surveillance series (~80-120 training points):
- Sequence Window: Chronological 12-month lookback (months t-12..t-1 -> month t)
- Input Projection: Linear tokenization with 1D positional encoding
- Multi-Head Attention: 2 attention heads, d_model=16, 1 TransformerEncoderLayer
- Regularization: Dropout (0.1) and LayerNorm to prevent catastrophic overfitting on small samples
- Scaling: MinMaxScaler(0, 1) fitted strictly on the chronological training split (zero leakage)
- Early Stopping: Chronological internal validation tail (last ~15% of training sequences)
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


DEFAULT_TRANSFORMER_CONFIG: Dict[str, Any] = {
    "lookback": 12,
    "d_model": 16,
    "nhead": 2,
    "dim_feedforward": 32,
    "dropout": 0.1,
    "learning_rate": 0.005,
    "batch_size": 8,
    "max_epochs": 80,
    "patience": 10,
    "val_fraction": 0.15,
    "seed": 42,
}


def create_transformer_sequences(
    values: np.ndarray,
    lookback: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build chronological sliding-window (X, y) sequences without temporal leakage."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(arr) <= lookback:
        return np.empty((0, lookback, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for i in range(len(arr) - lookback):
        X_list.append(arr[i : i + lookback].reshape(lookback, 1))
        y_list.append(np.array([arr[i + lookback]], dtype=np.float32))
    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)


if HAS_TORCH:
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for sequence order awareness."""

        def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x shape: (batch_size, seq_len, d_model)
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)

    class _TimeSeriesTransformerNet(nn.Module):
        """Compact Time Series Transformer with Multi-Head Self-Attention."""

        def __init__(
            self,
            d_model: int = 16,
            nhead: int = 2,
            dim_feedforward: int = 32,
            dropout: float = 0.1,
            lookback: int = 12,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max(lookback + 10, 50), dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.layer_norm = nn.LayerNorm(d_model)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x shape: (batch_size, seq_len, 1)
            h = self.input_proj(x)
            h = self.pos_encoder(h)
            h = self.transformer_encoder(h)
            h = self.layer_norm(h)
            # Global pooled / last-token representation
            last_token = h[:, -1, :]
            out = self.fc(last_token)
            return out
else:
    PositionalEncoding = None  # type: ignore[assignment]
    _TimeSeriesTransformerNet = None  # type: ignore[assignment]


def forecast_transformer(
    train_df: pd.DataFrame,
    horizon: int,
    target_col: str = "documented_occurrence_count",
    transformer_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Any, Dict[str, Any]]:
    """Fit a compact univariate Time Series Transformer on chronological sequences and forecast.

    Args:
        train_df: Chronological training DataFrame.
        horizon: Months ahead to forecast.
        target_col: Target variable column name.
        transformer_config: Optional hyperparameter overrides.

    Returns:
        Tuple of (point_predictions, lower_bound, upper_bound, fitted_model, diagnostics).
    """
    if not HAS_TORCH or torch is None or _TimeSeriesTransformerNet is None:
        raise ImportError("PyTorch (torch) is not installed or unavailable in this environment.")

    t_start = time.perf_counter()
    cfg = {**DEFAULT_TRANSFORMER_CONFIG, **(transformer_config or {})}

    lookback = int(cfg["lookback"])
    d_model = int(cfg["d_model"])
    nhead = int(cfg["nhead"])
    dim_ff = int(cfg["dim_feedforward"])
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
            f"Insufficient training observations ({len(raw_series)}) for Transformer with lookback={lookback}."
        )

    # Deterministic seeding
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Split raw series chronologically before fitting scaler (zero data leakage)
    n_total = len(raw_series)
    n_val_points = max(4, int(round(n_total * val_fraction)))
    n_fit_points = n_total - n_val_points

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(raw_series[:n_fit_points].reshape(-1, 1))
    scaled_all = scaler.transform(raw_series.reshape(-1, 1)).reshape(-1).astype(np.float32)

    X_all, y_all = create_transformer_sequences(scaled_all, lookback=lookback)
    n_train_seq = max(1, n_fit_points - lookback)
    X_train, y_train = X_all[:n_train_seq], y_all[:n_train_seq]
    X_val, y_val = X_all[n_train_seq:], y_all[n_train_seq:]

    device = torch.device("cpu")
    model = _TimeSeriesTransformerNet(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_ff,
        dropout=dropout,
        lookback=lookback,
    ).to(device)

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
        permutation = torch.randperm(len(X_train_t))
        for start_idx in range(0, len(X_train_t), batch_size):
            batch_indices = permutation[start_idx : start_idx + batch_size]
            X_batch = X_train_t[batch_indices]
            y_batch = y_train_t[batch_indices]

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epochs_trained = epoch
        if X_val_t is not None and y_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = float(criterion(val_pred, y_val_t).item())
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        else:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # Out-of-sample iterative multi-step autoregressive roll-forward
    current_window = scaled_all[-lookback:].copy()
    predictions_scaled: List[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            inp = torch.from_numpy(current_window.reshape(1, lookback, 1)).to(device)
            nxt_pred = float(model(inp).item())
            nxt_clamped = max(0.0, nxt_pred)
            predictions_scaled.append(nxt_clamped)
            current_window = np.append(current_window[1:], nxt_clamped).astype(np.float32)

    pred_scaled_arr = np.array(predictions_scaled, dtype=np.float32).reshape(-1, 1)
    point_predictions = scaler.inverse_transform(pred_scaled_arr).reshape(-1)
    point_predictions = np.maximum(0.0, point_predictions)

    # Uncertainty bounds estimated via residual variance on train sequences
    with torch.no_grad():
        in_sample_pred = model(X_train_t).numpy().reshape(-1, 1)
        in_sample_orig = scaler.inverse_transform(in_sample_pred).reshape(-1)
        actual_orig = scaler.inverse_transform(y_train).reshape(-1)
        residual_std = float(np.std(actual_orig - in_sample_orig)) if len(actual_orig) > 1 else 1.0

    lower_bound = np.maximum(0.0, point_predictions - 1.96 * residual_std)
    upper_bound = np.maximum(0.0, point_predictions + 1.96 * residual_std)

    fit_time = time.perf_counter() - t_start
    diagnostics: Dict[str, Any] = {
        "model_type": "TemporalTransformer",
        "architecture": f"TransformerEncoder(d_model={d_model}, nhead={nhead}, layers=1)",
        "lookback": lookback,
        "d_model": d_model,
        "nhead": nhead,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if np.isfinite(best_val_loss) else None,
        "fit_time_seconds": round(fit_time, 3),
        "parameters": f"d_model={d_model}, nhead={nhead}, lookback={lookback}, epochs={best_epoch}",
    }

    return point_predictions, lower_bound, upper_bound, model, diagnostics
