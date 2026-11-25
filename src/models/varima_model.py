"""Re-export forecast_varima from classical submodule."""
from src.models.classical.varima_model import forecast_varima

__all__ = ["forecast_varima"]
