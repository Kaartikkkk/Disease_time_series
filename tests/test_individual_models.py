"""Unit tests verifying standalone execution of every individual model file in src/models/."""

import numpy as np
import pandas as pd
import pytest

from src.models.seasonal_naive import forecast_seasonal_naive
from src.models.holt_winters import forecast_holt_winters
from src.models.ar_model import forecast_ar
from src.models.ma_model import forecast_ma
from src.models.arima_model import forecast_arima
from src.models.sarima_model import forecast_sarima
from src.models.gradient_boosting import forecast_gradient_boosting
from src.models.random_forest import forecast_random_forest


@pytest.fixture
def sample_monthly_series():
    """Create synthetic 36-month non-negative series with seasonality."""
    np.random.seed(42)
    t = np.arange(36)
    seasonal_pattern = 5.0 + 3.0 * np.sin(2 * np.pi * t / 12.0)
    noise = np.random.normal(0, 0.5, size=36)
    values = np.maximum(0.0, seasonal_pattern + noise)
    return pd.Series(values)


@pytest.fixture
def sample_monthly_df(sample_monthly_series):
    """Create dataframe with dates and weather regressors."""
    dates = pd.date_range("2021-01-01", periods=len(sample_monthly_series), freq="MS")
    return pd.DataFrame({
        "date": dates,
        "documented_occurrence_count": sample_monthly_series.values,
        "temp_mean_c_lag1": np.random.uniform(15, 35, len(dates)),
        "precipitation_monthly_mm_lag1": np.random.uniform(0, 150, len(dates)),
        "relative_humidity_pct_lag1": np.random.uniform(40, 95, len(dates)),
    })


def test_seasonal_naive_standalone(sample_monthly_series):
    preds, lb, ub = forecast_seasonal_naive(sample_monthly_series, horizon=6)
    assert len(preds) == 6
    assert len(lb) == 6
    assert len(ub) == 6
    assert np.all(preds >= 0.0)
    assert np.all(lb <= ub)


def test_holt_winters_standalone(sample_monthly_series):
    preds, lb, ub, res = forecast_holt_winters(sample_monthly_series, horizon=6)
    assert len(preds) == 6
    assert np.all(preds >= 0.0)
    assert res is not None


def test_ar_standalone(sample_monthly_series):
    preds, lb, ub, res = forecast_ar(sample_monthly_series, horizon=6, p=1)
    assert len(preds) == 6
    assert np.all(preds >= 0.0)


def test_ma_standalone(sample_monthly_series):
    preds, lb, ub, res = forecast_ma(sample_monthly_series, horizon=6, q=1)
    assert len(preds) == 6
    assert np.all(preds >= 0.0)


def test_arima_standalone(sample_monthly_series):
    preds, lb, ub, res = forecast_arima(sample_monthly_series, horizon=6, order=(1, 1, 1))
    assert len(preds) == 6
    assert np.all(preds >= 0.0)


def test_sarima_standalone(sample_monthly_series):
    preds, lb, ub, res = forecast_sarima(sample_monthly_series, horizon=6, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
    assert len(preds) == 6
    assert np.all(preds >= 0.0)


def test_gradient_boosting_standalone(sample_monthly_df):
    preds, lb, ub, model = forecast_gradient_boosting(sample_monthly_df, horizon=6)
    assert len(preds) == 6
    assert np.all(preds >= 0.0)


def test_random_forest_standalone(sample_monthly_df):
    preds, lb, ub, model = forecast_random_forest(sample_monthly_df, horizon=6)
    assert len(preds) == 6
    assert np.all(preds >= 0.0)
