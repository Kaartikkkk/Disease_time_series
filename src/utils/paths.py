"""Path management and configuration loader for Plant Disease Outbreak Forecasting."""

from pathlib import Path
from typing import Any, Dict
import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Config path
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load settings.yaml configuration dictionary."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_path(path_key: str) -> Path:
    """Retrieve absolute path for a configured directory key."""
    config = load_config()
    relative_path = config.get("paths", {}).get(path_key)
    if not relative_path:
        raise KeyError(f"Path key '{path_key}' not found in configuration paths.")
    abs_path = PROJECT_ROOT / relative_path
    abs_path.mkdir(parents=True, exist_ok=True)
    return abs_path


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
