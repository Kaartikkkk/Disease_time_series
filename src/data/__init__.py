"""Data processing, ingestion, quality, aggregation, authoritative dataset, and service package."""
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
from src.data.dataset_service import (
    SPECIFICATION,
    DatasetSpecification,
    get_authoritative_master_path,
    get_clean_records_path,
    get_pathogen_dataset_path,
    list_authoritative_pathogen_files,
    load_authoritative_master_dataset,
    load_authoritative_series,
    load_clean_event_records,
    verify_dataset_integrity,
)
from src.data.imputation import (
    TrainingAwareImputer,
    fit_and_impute_chronological_split,
)

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
    "SPECIFICATION",
    "DatasetSpecification",
    "get_authoritative_master_path",
    "get_clean_records_path",
    "get_pathogen_dataset_path",
    "list_authoritative_pathogen_files",
    "load_authoritative_master_dataset",
    "load_authoritative_series",
    "load_clean_event_records",
    "verify_dataset_integrity",
    "TrainingAwareImputer",
    "fit_and_impute_chronological_split",
]
