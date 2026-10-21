"""Unit tests for leakage-free feature engineering."""

import numpy as np
import pandas as pd
import pytest
from src.features.lag_features import generate_lag_features, create_supervised_matrix
from src.features.calendar_features import add_calendar_features


def test_lag_features_no_future_leakage():
    """Verify target lag k strictly references observation from t - k and never t or t + 1."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "crop": "Grapevine",
        "disease": "Downy Mildew",
        "incidence_rate": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "source": "test",
    })
    feat_df = generate_lag_features(df, lags=[1, 2, 3], rolling_windows=[3], frequency="D")

    # Lag 1 at index 1 must be value from index 0 (10.0)
    assert np.isnan(feat_df["target_lag_1"].iloc[0])
    assert feat_df["target_lag_1"].iloc[1] == 10.0
    assert feat_df["target_lag_1"].iloc[4] == 40.0

    # Lag 2 at index 2 must be value from index 0 (10.0)
    assert np.isnan(feat_df["target_lag_2"].iloc[0])
    assert np.isnan(feat_df["target_lag_2"].iloc[1])
    assert feat_df["target_lag_2"].iloc[2] == 10.0

    # Rolling mean 3 at index 3 must be mean of [10.0, 20.0, 30.0] = 20.0 (strictly BEFORE index 3)
    # Notice: index 3 itself is 40.0, so if 40.0 were included, mean would be (20+30+40)/3 = 30.0
    roll_val_at_idx_3 = feat_df["target_roll_mean_3"].iloc[3]
    assert np.isclose(roll_val_at_idx_3, 20.0), f"Leakage detected: expected 20.0, got {roll_val_at_idx_3}"


def test_calendar_cyclical_bounds():
    """Verify cyclical trigonometric features are bounded within [-1.0, 1.0]."""
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    df = pd.DataFrame({"date": dates})
    df_cal = add_calendar_features(df, frequency="D")

    assert (df_cal["sin_dayofyear"] >= -1.0).all() and (df_cal["sin_dayofyear"] <= 1.0).all()
    assert (df_cal["cos_dayofyear"] >= -1.0).all() and (df_cal["cos_dayofyear"] <= 1.0).all()
    assert (df_cal["sin_dayofweek"] >= -1.0).all() and (df_cal["sin_dayofweek"] <= 1.0).all()
