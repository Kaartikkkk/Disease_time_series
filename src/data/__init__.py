"""Data processing, ingestion, quality, and aggregation package."""
from src.data.preprocessing import (
    assign_agro_climatic_region,
    deduplicate_records,
    parse_and_validate_dates,
    validate_and_clean_dataset,
)
from src.data.monthly_aggregation import (
    build_continuous_monthly_series,
    process_all_pathogens_to_monthly,
)
from src.data.data_quality import generate_data_quality_report
from src.data.data_collection import run_data_collection
from src.data.gbif_client import GBIFClient
from src.data.eppo_client import EPPOClient
from src.data.weather_client import WeatherClient

__all__ = [
    "assign_agro_climatic_region",
    "deduplicate_records",
    "parse_and_validate_dates",
    "validate_and_clean_dataset",
    "build_continuous_monthly_series",
    "process_all_pathogens_to_monthly",
    "generate_data_quality_report",
    "run_data_collection",
    "GBIFClient",
    "EPPOClient",
    "WeatherClient",
]
