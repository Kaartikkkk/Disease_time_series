"""Unit tests for forecasting pipelines, model instantiation, horizons, and bounds."""

import numpy as np
import pandas as pd
import pytest
from src.models.forecasting import generate_forecast
from src.models.model_registry import list_available_models, get_model
from src.models.baselines import PersistenceForecaster, SeasonalNaiveForecaster, MovingAverageForecaster


@pytest.fixture
def sample_series_df():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "date": dates,
        "crop": "Potato",
        "disease": "Late Blight",
        "incidence_rate": 10.0 + 5.0 * np.sin(np.arange(60) * 2 * np.pi / 7.0) + np.random.normal(0, 0.5, 60),
        "temperature": 18.0 + np.random.normal(0, 1.0, 60),
        "humidity": 70.0 + np.random.normal(0, 5.0, 60),
        "rainfall": np.random.exponential(2.0, 60),
        "source": "test",
        "is_synthetic": True,
    })


def test_forecast_dates_and_horizon_length(sample_series_df):
    """Verify forecast dates strictly follow the last training date and match horizon length."""
    horizon = 14
    last_train_date = sample_series_df["date"].iloc[-1]

    fc_res = generate_forecast(
        df=sample_series_df,
        model_name="persistence",
        target_col="incidence_rate",
        horizon=horizon,
        frequency="D",
    )

    assert len(fc_res.point_forecast) == horizon
    assert len(fc_res.forecast_dates) == horizon
    assert fc_res.forecast_dates[0] > last_train_date
    assert fc_res.forecast_dates[0] == last_train_date + pd.Timedelta(days=1)
    assert fc_res.forecast_dates[-1] == last_train_date + pd.Timedelta(days=horizon)


def test_forecasts_are_finite_and_non_negative(sample_series_df):
    """Verify all forecasts are finite and bounded >= 0.0."""
    models_to_test = ["persistence", "moving_average", "arima", "random_forest"]
    for m in models_to_test:
        fc = generate_forecast(
            df=sample_series_df,
            model_name=m,
            target_col="incidence_rate",
            horizon=7,
            frequency="D",
        )
        assert np.isfinite(fc.point_forecast).all()
        assert (fc.point_forecast >= 0.0).all()
        assert np.isfinite(fc.lower_bound).all()
        assert (fc.lower_bound >= 0.0).all()


def test_baseline_forecasters_exact_logic():
    """Verify mathematical exactness of baseline forecasters."""
    s = pd.Series([10.0, 15.0, 20.0, 25.0])

    # Persistence should repeat last value (25.0)
    p_model = PersistenceForecaster().fit(s)
    preds_p = p_model.predict(3)
    np.testing.assert_array_equal(preds_p, [25.0, 25.0, 25.0])

    # Moving Average with window 2 should predict mean of [20.0, 25.0] = 22.5
    ma_model = MovingAverageForecaster(window=2).fit(s)
    preds_ma = ma_model.predict(2)
    np.testing.assert_array_equal(preds_ma, [22.5, 22.5])
