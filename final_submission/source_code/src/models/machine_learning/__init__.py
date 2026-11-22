"""Advanced Machine Learning Models.

Contains non-linear tree partitioners, gradient boosted ensembles, multi-pathogen pooled
panel models, and specialized zero-inflated two-stage hurdle architectures.
"""

from src.models.machine_learning.global_models import (
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_hurdle_xgboost,
)
from src.models.machine_learning.gradient_boosting import forecast_gradient_boosting
from src.models.machine_learning.ml_autoregressor import forecast_ml_autoregressor
from src.models.machine_learning.random_forest import forecast_random_forest
from src.models.machine_learning.xgboost_model import HAS_XGBOOST, forecast_xgboost

__all__ = [
    "HAS_XGBOOST",
    "forecast_global_random_forest",
    "forecast_global_xgboost",
    "forecast_gradient_boosting",
    "forecast_hurdle_xgboost",
    "forecast_ml_autoregressor",
    "forecast_random_forest",
    "forecast_xgboost",
]

