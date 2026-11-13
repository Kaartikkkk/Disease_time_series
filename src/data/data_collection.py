"""Master Data Ingestion Orchestrator for Real Public Data.

Executes data collection across:
1. GBIF Occurrence API (India 2015-2026 pathogen records)
2. NASA POWER Monthly Agrometeorology API (Regional and National Indian Climate)
3. EPPO Global Database / Reporting Service (Verified Phytosanitary Alerts)
"""

import logging
from pathlib import Path
from typing import Any, Dict
import yaml

from src.eppo_client import EPPOClient
from src.gbif_client import GBIFClient
from src.weather_client import WeatherClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Load project YAML settings."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_data_collection(force_download: bool = False, config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Execute complete real-world data collection pipeline for India."""
    settings = load_settings(config_path)
    logger.info("=== STARTING AGROCAST DATA INGESTION PIPELINE (INDIA) ===")

    gbif_client = GBIFClient()
    weather_client = WeatherClient()
    eppo_client = EPPOClient()

    results_summary: Dict[str, Any] = {
        "gbif": {},
        "weather": {},
        "eppo": {},
    }

    # 1. Ingest GBIF occurrence data for all configured pathogens
    pathogens_dict = settings.get("pathogens", {})
    start_year = settings.get("data_collection", {}).get("start_year", 2015)
    end_year = settings.get("data_collection", {}).get("end_year", 2026)
    target_country = settings.get("data_collection", {}).get("target_country", "IN")

    for key, info in pathogens_dict.items():
        sci_name = info.get("scientific_name")
        logger.info("Processing pathogen '%s' (%s)...", sci_name, info.get("common_name"))

        # Check if cached raw file exists
        cached = None if force_download else gbif_client.load_cached_raw(sci_name)
        if cached:
            logger.info("Found existing raw GBIF archive for '%s' (%d records).", sci_name, len(cached.get("records", [])))
            results_summary["gbif"][key] = {
                "scientific_name": sci_name,
                "common_name": info.get("common_name"),
                "records_retrieved": len(cached.get("records", [])),
                "source": "Cached Raw Archive",
            }
        else:
            archive = gbif_client.fetch_occurrences(
                scientific_name=sci_name,
                country=target_country,
                start_year=start_year,
                end_year=end_year,
                max_records=2500,
            )
            results_summary["gbif"][key] = {
                "scientific_name": sci_name,
                "common_name": info.get("common_name"),
                "records_retrieved": len(archive.get("records", [])),
                "source": "Live GBIF API",
            }

    # 2. Ingest NASA POWER monthly weather data for Indian regions
    weather_regions = settings.get("weather_regions", {})
    weather_end_year = min(end_year, 2025)  # NASA POWER verified through 2025

    for reg_id, reg_info in weather_regions.items():
        reg_name = reg_info.get("name")
        lat = reg_info.get("latitude")
        lon = reg_info.get("longitude")

        cached_df = None if force_download else weather_client.load_cached_weather(reg_id)
        if cached_df is not None:
            logger.info("Found existing NASA POWER weather for '%s' (%d months).", reg_name, len(cached_df))
            results_summary["weather"][reg_id] = {
                "region_name": reg_name,
                "months_covered": len(cached_df),
                "source": "Cached Weather CSV",
            }
        else:
            w_res = weather_client.fetch_monthly_weather(
                latitude=lat,
                longitude=lon,
                start_year=start_year,
                end_year=weather_end_year,
                region_id=reg_id,
                region_name=reg_name,
            )
            results_summary["weather"][reg_id] = {
                "region_name": reg_name,
                "months_covered": len(w_res["dataframe"]),
                "source": "Live NASA POWER API",
            }

    # 3. Ingest EPPO reports
    eppo_archive = eppo_client.extract_reports()
    results_summary["eppo"] = {
        "reports_count": len(eppo_archive.get("reports", [])),
        "source": "EPPO Global Database Archive",
    }

    logger.info("=== DATA COLLECTION COMPLETED SUCCESSFULLY ===")
    return results_summary


if __name__ == "__main__":
    run_data_collection(force_download=False)
