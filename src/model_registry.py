"""Model Registry and Serialization Engine for Plant Disease Forecasting Models.

Manages:
1. Central declarative registration of classical, ML, and Milestone 3 models
2. Unified adapter execution with standardized output validation
3. Serialization and persistence of fitted model checkpoints via `joblib`
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from src.models.adapters import (
    adapt_ar,
    adapt_arima,
    adapt_arimax,
    adapt_global_random_forest,
    adapt_global_xgboost,
    adapt_gradient_boosting,
    adapt_gru,
    adapt_holt_winters,
    adapt_hurdle_xgboost,
    adapt_lstm,
    adapt_ma,
    adapt_prophet,
    adapt_random_forest,
    adapt_sarima,
    adapt_sarimax,
    adapt_seasonal_naive,
    adapt_xgboost,
)
from src.models.base import ForecastContext, ModelAdapterFn, validate_forecast_output
from src.models.lstm_model import HAS_TORCH
from src.models.prophet_model import HAS_PROPHET
from src.models.xgboost_model import HAS_XGBOOST

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Declarative specification for a forecasting model in the central registry."""

    model_id: str
    display_name: str
    category: str  # "Classical", "Statistical", "Machine Learning", "Deep Learning"
    model_family: str
    available: bool
    supports_exog: bool = False
    requires_exog: bool = False
    has_ci: bool = True
    default_config: Dict[str, Any] = field(default_factory=dict)
    adapter: Optional[ModelAdapterFn] = None
    notes: str = ""


MODEL_SPECS: Dict[str, ModelSpec] = {
    # --------------------------------------------------------------------------
    # Active Milestone 2 Models (10)
    # --------------------------------------------------------------------------
    "seasonal_naive": ModelSpec(
        model_id="seasonal_naive",
        display_name="Seasonal Naive (Baseline)",
        category="Classical",
        model_family="Persistence Baseline",
        available=True,
        supports_exog=False,
        has_ci=True,
        default_config={"season_length": 12},
        adapter=adapt_seasonal_naive,
    ),
    "ar": ModelSpec(
        model_id="ar",
        display_name="AR(1)",
        category="Statistical",
        model_family="Box-Jenkins Univariate",
        available=True,
        supports_exog=False,
        has_ci=True,
        default_config={"p": 1},
        adapter=adapt_ar,
    ),
    "ma": ModelSpec(
        model_id="ma",
        display_name="MA(1)",
        category="Statistical",
        model_family="Box-Jenkins Univariate",
        available=True,
        supports_exog=False,
        has_ci=True,
        default_config={"q": 1},
        adapter=adapt_ma,
    ),
    "arima": ModelSpec(
        model_id="arima",
        display_name="ARIMA(1,1,1)",
        category="Statistical",
        model_family="Box-Jenkins Integrated",
        available=True,
        supports_exog=False,
        has_ci=True,
        default_config={"order": (1, 1, 1), "auto_optimize": True},
        adapter=adapt_arima,
    ),
    "sarima": ModelSpec(
        model_id="sarima",
        display_name="SARIMA(1,0,1)(1,0,1)[12]",
        category="Statistical",
        model_family="Seasonal Box-Jenkins",
        available=True,
        supports_exog=False,
        has_ci=True,
        default_config={"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 12), "auto_optimize": True},
        adapter=adapt_sarima,
    ),
    "holt_winters": ModelSpec(
        model_id="holt_winters",
        display_name="Holt-Winters Exponential Smoothing",
        category="Classical",
        model_family="Exponential Smoothing",
        available=True,
        supports_exog=False,
        has_ci=True,
        default_config={"season_length": 12, "auto_optimize": True},
        adapter=adapt_holt_winters,
    ),
    "arimax": ModelSpec(
        model_id="arimax",
        display_name="ARIMAX (Weather Regressors)",
        category="Statistical",
        model_family="Exogenous Econometric",
        available=True,
        supports_exog=True,
        requires_exog=True,
        has_ci=True,
        default_config={"order": (1, 0, 1), "auto_optimize": True},
        adapter=adapt_arimax,
    ),
    "sarimax": ModelSpec(
        model_id="sarimax",
        display_name="SARIMAX (Seasonal + Weather)",
        category="Statistical",
        model_family="Exogenous Econometric",
        available=True,
        supports_exog=True,
        requires_exog=True,
        has_ci=True,
        default_config={"order": (1, 0, 1), "seasonal_order": (1, 0, 1, 12), "auto_optimize": True},
        adapter=adapt_sarimax,
    ),
    "gradient_boosting": ModelSpec(
        model_id="gradient_boosting",
        display_name="Hist Gradient Boosting (Autoregressive)",
        category="Machine Learning",
        model_family="Tree Ensemble ML",
        available=True,
        supports_exog=True,
        has_ci=True,
        default_config={"lags": [1, 2, 3, 12]},
        adapter=adapt_gradient_boosting,
    ),
    "random_forest": ModelSpec(
        model_id="random_forest",
        display_name="Random Forest (Autoregressive)",
        category="Machine Learning",
        model_family="Tree Ensemble ML",
        available=True,
        supports_exog=True,
        has_ci=True,
        default_config={"lags": [1, 2, 3, 12]},
        adapter=adapt_random_forest,
    ),
    # --------------------------------------------------------------------------
    # Planned Milestone 3 Models (Unavailable until implemented & tested)
    # --------------------------------------------------------------------------
    "xgboost": ModelSpec(
        model_id="xgboost",
        display_name="XGBoost (Autoregressive)",
        category="Machine Learning",
        model_family="Gradient Boosted Trees (XGBoost)",
        available=HAS_XGBOOST,
        supports_exog=True,
        has_ci=True,
        default_config={"xgb_lags": [1, 2, 3, 6, 12]},
        adapter=adapt_xgboost,
        notes="Milestone 3 regularized XGBoost with lag, shifted rolling, calendar, and weather features",
    ),
    "prophet": ModelSpec(
        model_id="prophet",
        display_name="Prophet (Trend + Seasonality)",
        category="Statistical",
        model_family="Generalized Additive (Prophet)",
        available=HAS_PROPHET,
        supports_exog=False,
        has_ci=True,
        default_config={"use_exog": False},
        adapter=adapt_prophet,
        notes="Milestone 3 univariate Prophet (additive yearly seasonality, conservative priors)",
    ),
    "lstm": ModelSpec(
        model_id="lstm",
        display_name="LSTM (Sequence Neural Net)",
        category="Deep Learning",
        model_family="Recurrent Neural Network (PyTorch)",
        available=HAS_TORCH,
        supports_exog=False,
        has_ci=False,
        default_config={"lookback": 12},
        adapter=adapt_lstm,
        notes="Milestone 3 univariate PyTorch LSTM (16 units, 12-month lookback, MinMaxScaler)",
    ),
    "gru": ModelSpec(
        model_id="gru",
        display_name="GRU (Gated Recurrent Unit)",
        category="Deep Learning",
        model_family="Recurrent Neural Network (PyTorch)",
        available=HAS_TORCH,
        supports_exog=False,
        has_ci=False,
        default_config={"lookback": 12},
        adapter=adapt_gru,
        notes="Milestone 3 univariate PyTorch GRU (16 units, 12-month lookback, MinMaxScaler)",
    ),
    "global_xgboost": ModelSpec(
        model_id="global_xgboost",
        display_name="Global XGBoost (Multi-Pathogen)",
        category="Machine Learning",
        model_family="Pooled Gradient Boosted Trees",
        available=HAS_XGBOOST,
        supports_exog=True,
        has_ci=True,
        default_config={"lags": [1, 2, 3, 6, 12]},
        adapter=adapt_global_xgboost,
        notes="Global XGBoost trained on cross-pathogen pooled panel history",
    ),
    "global_random_forest": ModelSpec(
        model_id="global_random_forest",
        display_name="Global Random Forest (Multi-Pathogen)",
        category="Machine Learning",
        model_family="Pooled Tree Ensemble",
        available=True,
        supports_exog=True,
        has_ci=True,
        default_config={"lags": [1, 2, 3, 6, 12]},
        adapter=adapt_global_random_forest,
        notes="Global Random Forest trained on cross-pathogen pooled panel history",
    ),
    "hurdle_xgboost": ModelSpec(
        model_id="hurdle_xgboost",
        display_name="Hurdle XGBoost (Classifier + Regressor)",
        category="Machine Learning",
        model_family="Two-Stage Zero-Inflated ML",
        available=HAS_XGBOOST,
        supports_exog=True,
        has_ci=True,
        default_config={"lags": [1, 2, 3, 6, 12]},
        adapter=adapt_hurdle_xgboost,
        notes="Two-stage Hurdle XGBoost separating occurrence probability and positive magnitude",
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    """Return the ModelSpec for a registered model_id or raise KeyError."""
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model_id '{model_id}'. Registered: {list(MODEL_SPECS.keys())}")
    return MODEL_SPECS[model_id]


def list_registered_models(
    include_unavailable: bool = True,
    category: Optional[str] = None,
) -> List[ModelSpec]:
    """List registered ModelSpec entries filtered by availability and category."""
    specs = list(MODEL_SPECS.values())
    if not include_unavailable:
        specs = [s for s in specs if s.available]
    if category is not None:
        specs = [s for s in specs if s.category == category]
    return specs


def get_available_models() -> List[ModelSpec]:
    """Return all currently integrated and runnable models."""
    return list_registered_models(include_unavailable=False)


def get_model_category(display_name: str) -> str:
    """Resolve model category from its display name."""
    for spec in MODEL_SPECS.values():
        if spec.display_name == display_name:
            return spec.category
    return "Time Series"


def run_registered_model(
    model_id: str,
    ctx: ForecastContext,
    clip_non_negative: bool = True,
) -> Dict[str, Any]:
    """Execute a registered, available model via its adapter and validate output."""
    spec = get_model_spec(model_id)
    if not spec.available or spec.adapter is None:
        raise RuntimeError(
            f"Model '{spec.display_name}' ({model_id}) is registered as unavailable ({spec.notes or 'not yet implemented'})."
        )

    merged_cfg = {**spec.default_config, **(ctx.config or {})}
    exec_ctx = ForecastContext(
        train_df=ctx.train_df,
        horizon=ctx.horizon,
        target_col=ctx.target_col,
        exog_cols=ctx.exog_cols,
        future_df=ctx.future_df,
        config=merged_cfg,
    )
    raw_output = spec.adapter(exec_ctx)
    return validate_forecast_output(
        raw_output,
        expected_horizon=ctx.horizon,
        clip_non_negative=clip_non_negative,
    )


class ModelRegistry:
    """Registry for managing time series model checkpoints and evaluation metadata."""

    def __init__(self, models_dir: Path = Path("models")):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.models_dir / "registry_metadata.json"
        self._registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load persistent registry index."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to load model registry index: %s", e)
        return {}

    def _save_registry(self) -> None:
        """Save persistent registry index."""
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            logger.error("Failed to save model registry index: %s", e)

    def save_model(
        self,
        model_artifact: Any,
        model_name: str,
        pathogen_slug: str,
        validation_metrics: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None,
        train_start: str = "",
        train_end: str = "",
    ) -> str:
        """Persist model artifact and register metadata."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug_clean = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "_")
        filename = f"{pathogen_slug}_{slug_clean}_{timestamp}.joblib"
        file_path = self.models_dir / filename

        joblib.dump(model_artifact, file_path)

        reg_entry = {
            "model_name": model_name,
            "pathogen_slug": pathogen_slug,
            "filename": filename,
            "file_path": str(file_path),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_start": train_start,
            "train_end": train_end,
            "validation_metrics": validation_metrics,
            "hyperparameters": hyperparameters or {},
        }

        key = f"{pathogen_slug}::{model_name}"
        self._registry[key] = reg_entry
        self._save_registry()
        logger.info("Saved and registered model checkpoint to %s", file_path)
        return str(file_path)

    def load_model(self, pathogen_slug: str, model_name: str) -> Optional[Any]:
        """Load model checkpoint for a pathogen."""
        key = f"{pathogen_slug}::{model_name}"
        entry = self._registry.get(key)
        if not entry:
            logger.warning("No registered model found for %s", key)
            return None

        file_path = Path(entry["file_path"])
        if file_path.exists():
            return joblib.load(file_path)
        logger.error("Model file %s does not exist on disk.", file_path)
        return None

    def list_models(self, pathogen_slug: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered models, optionally filtered by pathogen."""
        entries = list(self._registry.values())
        if pathogen_slug:
            entries = [e for e in entries if e.get("pathogen_slug") == pathogen_slug]
        return entries
