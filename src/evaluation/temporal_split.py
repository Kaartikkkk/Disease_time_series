"""Leakage-free chronological dataset splitting utilities."""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from src.utils.logging_config import logger


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.15,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strictly chronological train/validation/test split without future data leakage.

    Guarantees:
    - No random shuffling.
    - All training dates precede all validation dates.
    - All validation dates precede all test dates.
    - Preprocessing must be fitted strictly on the train partition.

    Returns:
    (train_df, val_df, test_df)
    """
    if df.empty:
        raise ValueError("Cannot split empty DataFrame.")

    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    n = len(df_sorted)

    if n < 6:
        raise ValueError(f"Series has only {n} rows; cannot perform 3-way chronological split.")

    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))
    n_train = n - n_test - n_val

    if n_train < 3:
        raise ValueError(
            f"Insufficient training observations ({n_train}) after allocating test ({n_test}) and val ({n_val})."
        )

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train : n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val :].copy()

    # Leakage assertion check
    max_train_date = pd.to_datetime(train_df[date_col].max())
    min_val_date = pd.to_datetime(val_df[date_col].min())
    max_val_date = pd.to_datetime(val_df[date_col].max())
    min_test_date = pd.to_datetime(test_df[date_col].min())

    assert max_train_date <= min_val_date, (
        f"Temporal leakage detected: max train date {max_train_date} > min val date {min_val_date}"
    )
    assert max_val_date <= min_test_date, (
        f"Temporal leakage detected: max val date {max_val_date} > min test date {min_test_date}"
    )

    logger.info(
        f"Chronological split: Train={len(train_df)} rows ({train_df[date_col].min().date()} to {max_train_date.date()}), "
        f"Val={len(val_df)} rows ({min_val_date.date()} to {max_val_date.date()}), "
        f"Test={len(test_df)} rows ({min_test_date.date()} to {test_df[date_col].max().date()})"
    )
    return train_df, val_df, test_df
