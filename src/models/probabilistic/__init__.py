"""Probabilistic Forecasting Models.

Contains Meta Prophet generalized additive models with Bayesian piecewise linear trend
changepoint detection and Fourier seasonal harmonics.
"""

from src.models.probabilistic.prophet_model import (
    HAS_PROPHET,
    forecast_prophet,
    prepare_prophet_dataframe,
)

__all__ = [
    "HAS_PROPHET",
    "forecast_prophet",
    "prepare_prophet_dataframe",
]

