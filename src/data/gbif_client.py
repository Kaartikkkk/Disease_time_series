"""GBIF Occurrence and Taxonomic API Client for India Plant Disease Surveillance.

Connects to the official Global Biodiversity Information Facility (GBIF) REST APIs:
- Occurrence Search API: https://api.gbif.org/v1/occurrence/search
- Species Match API: https://api.gbif.org/v1/species/match
- Dataset Metadata API: https://api.gbif.org/v1/dataset/{key}

Features robust pagination, rate-limiting, error handling, and raw untouched archiving.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class GBIFClient:
    """Client for retrieving plant pathogen occurrences in India from GBIF."""

    OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
    SPECIES_MATCH_URL = "https://api.gbif.org/v1/species/match"
    DATASET_URL = "https://api.gbif.org/v1/dataset"

    def __init__(
        self,
        raw_data_dir: Path = Path("data/raw/gbif"),
        timeout_seconds: int = 25,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "AgroCast-India-PlantDisease-Forecasting/2.0 (Academic Research; contact: agrocast-india@research.org)",
                "Accept": "application/json",
            }
        )

    def match_species(self, name: str) -> Dict[str, Any]:
        """Resolve scientific name against GBIF taxonomic backbone."""
        params = {"name": name, "verbose": True}
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.SPECIES_MATCH_URL, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.warning("GBIF species match attempt %d failed for '%s': %s", attempt + 1, name, e)
                if attempt == self.max_retries - 1:
                    return {"matchType": "NONE", "error": str(e)}
                time.sleep(self.backoff_factor ** attempt)
        return {"matchType": "NONE"}

    def fetch_occurrences(
        self,
        scientific_name: str,
        country: str = "IN",
        start_year: int = 2015,
        end_year: int = 2026,
        max_records: int = 2000,
        page_size: int = 300,
    ) -> Dict[str, Any]:
        """Fetch occurrences for a scientific name in a target country with pagination.
        
        Saves untouched raw API payloads and returns structured archive metadata.
        """
        all_records: List[Dict[str, Any]] = []
        offset = 0
        total_count = 0
        retrieval_time = datetime.now(timezone.utc).isoformat()
        query_params = {
            "country": country,
            "scientificName": scientific_name,
            "year": f"{start_year},{end_year}",
            "limit": page_size,
        }

        logger.info(
            "Querying GBIF for '%s' in %s (Years: %d-%d)",
            scientific_name, country, start_year, end_year
        )

        while offset < max_records:
            current_params = {**query_params, "offset": offset}
            success = False
            for attempt in range(self.max_retries):
                try:
                    resp = self.session.get(
                        self.OCCURRENCE_URL, params=current_params, timeout=self.timeout
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    success = True
                    break
                except requests.RequestException as exc:
                    logger.warning(
                        "GBIF search request failed (offset=%d, attempt=%d): %s",
                        offset, attempt + 1, exc
                    )
                    time.sleep(self.backoff_factor ** attempt)

            if not success:
                logger.error("Failed to retrieve batch at offset %d after %d retries.", offset, self.max_retries)
                break

            total_count = data.get("count", 0)
            results = data.get("results", [])
            if not results:
                break

            all_records.extend(results)
            offset += len(results)
            logger.info("Retrieved %d / %d records for '%s'", len(all_records), total_count, scientific_name)

            if offset >= total_count or len(results) < page_size:
                break
            time.sleep(0.3)  # Polite rate limiting

        # Build clean raw record archive
        slug = scientific_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        archive = {
            "metadata": {
                "source": "GBIF Occurrence API",
                "api_endpoint": self.OCCURRENCE_URL,
                "scientific_name": scientific_name,
                "country": country,
                "start_year": start_year,
                "end_year": end_year,
                "total_available_in_source": total_count,
                "total_retrieved": len(all_records),
                "retrieval_timestamp_utc": retrieval_time,
                "license": "CC-BY-4.0 / CC0 (per individual record license fields)",
                "citation": "GBIF.org Occurrence Download. Accessed via GBIF Occurrence Search API.",
            },
            "records": all_records,
        }

        # Save untouched raw JSON
        json_path = self.raw_data_dir / f"gbif_{slug}_raw.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)

        # Also save flat raw CSV preserving original fields
        if all_records:
            df_raw = pd.json_normalize(all_records)
            csv_path = self.raw_data_dir / f"gbif_{slug}_raw.csv"
            df_raw.to_csv(csv_path, index=False)
            logger.info("Saved raw CSV to %s (%d rows)", csv_path, len(df_raw))

        logger.info("Saved raw JSON archive to %s", json_path)
        return archive

    def load_cached_raw(self, scientific_name: str) -> Optional[Dict[str, Any]]:
        """Load previously saved raw JSON archive if present."""
        slug = scientific_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        json_path = self.raw_data_dir / f"gbif_{slug}_raw.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Error reading cached raw file %s: %e", json_path, e)
        return None
