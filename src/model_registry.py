"""Model Registry and Serialization Engine for Plant Disease Forecasting Models.

Manages:
- Serialization and persistence of fitted models via `joblib`
- Versioning and metadata tracking (fitted timestamp, training date bounds, hyperparameters, validation metrics)
- Safe loading and retrieval of persisted model checkpoints
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

logger = logging.getLogger(__name__)


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
