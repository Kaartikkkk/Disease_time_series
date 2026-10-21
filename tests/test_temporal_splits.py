"""Unit tests for chronological splitting and leakage prevention assertions."""

import pandas as pd
import pytest
from src.evaluation.temporal_split import chronological_train_test_split


def test_chronological_split_ordering_and_no_leakage():
    """Verify that train, validation, and test sets are strictly chronological."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "crop": "Grapevine",
        "disease": "Downy Mildew",
        "incidence_rate": range(100),
        "source": "test",
    })

    train_df, val_df, test_df = chronological_train_test_split(df, test_size=0.20, val_size=0.15)

    assert len(train_df) == 65
    assert len(val_df) == 15
    assert len(test_df) == 20

    # Strict date progression check
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()

    # Target values order check
    assert train_df["incidence_rate"].max() < val_df["incidence_rate"].min()
    assert val_df["incidence_rate"].max() < test_df["incidence_rate"].min()


def test_chronological_split_insufficient_data():
    """Verify ValueError is raised if series has insufficient rows for 3-way split."""
    df_small = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4, freq="D"),
        "incidence_rate": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="cannot perform 3-way chronological split"):
        chronological_train_test_split(df_small, test_size=0.2, val_size=0.2)
