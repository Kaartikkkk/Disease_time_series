"""NASA POWER Meteorological API Client for Indian Agricultural Regions.

Connects to NASA POWER (Prediction Of Worldwide Energy Resources) Monthly Agrometeorology API:
https://power.larc.nasa.gov/api/temporal/monthly/point

Retrieves validated monthly climate parameters:
- T2M: Temperature at 2 Meters (°C)
- T2M_MAX: Maximum Temperature at 2 Meters (°C)
- T2M_MIN: Minimum Temperature at 2 Meters (°C)
- PRECTOTCORR: Precipitation Corrected (mm/day) -> converted to mm/month
- RH2M: Relative Humidity at 2 Meters (%)
- T2MDEW: Dew/Frost Point at 2 Meters (°C)
"""

from calendar import monthrange
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class WeatherClient:
    """Client for retrieving NASA POWER monthly meteorological series across India."""

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    DEFAULT_PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "T2MDEW"]

    def __init__(
        self,
        raw_data_dir: Path = Path("data/raw/weather"),
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "AgroCast-India-PlantDisease-Forecasting/2.0",
                "Accept": "application/json",
            }
        )

    def fetch_monthly_weather(
        self,
        latitude: float,
        longitude: float,
        start_year: int = 2015,
        end_year: int = 2025,
        region_id: str = "national_composite",
        region_name: str = "India National Composite",
        parameters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fetch monthly weather time-series from NASA POWER for coordinates."""
        if parameters is None:
            parameters = self.DEFAULT_PARAMETERS

        params = {
            "parameters": ",".join(parameters),
            "community": "AG",
            "longitude": f"{longitude:.4f}",
            "latitude": f"{latitude:.4f}",
            "start": str(start_year),
            "end": str(end_year),
            "format": "JSON",
        }

        retrieval_time = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Fetching NASA POWER weather for %s (lat=%.2f, lon=%.2f, %d-%d)",
            region_name, latitude, longitude, start_year, end_year
        )

        raw_data: Optional[Dict[str, Any]] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                raw_data = resp.json()
                break
            except requests.RequestException as exc:
                logger.warning(
                    "NASA POWER attempt %d/%d failed for %s: %s",
                    attempt + 1, self.max_retries, region_id, exc
                )
                time.sleep(2.0 * (attempt + 1))

        if not raw_data:
            raise RuntimeError(f"Failed to fetch NASA POWER data for {region_id} after {self.max_retries} attempts.")

        # Save untouched raw JSON payload
        raw_archive = {
            "metadata": {
                "source": "NASA POWER Monthly Agrometeorology API",
                "api_endpoint": self.BASE_URL,
                "region_id": region_id,
                "region_name": region_name,
                "latitude": latitude,
                "longitude": longitude,
                "start_year": start_year,
                "end_year": end_year,
                "parameters_requested": parameters,
                "retrieval_timestamp_utc": retrieval_time,
                "license": "NASA Open Data Policy (Public Domain)",
                "citation": "NASA Langley Research Center POWER Project. Monthly Meteorological Data.",
            },
            "raw_payload": raw_data,
        }

        json_path = self.raw_data_dir / f"nasa_power_{region_id}_raw.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_archive, f, indent=2)

        # Parse into tidy tabular format
        df_weather = self._parse_power_payload(raw_data, region_id, region_name, latitude, longitude)
        csv_path = self.raw_data_dir / f"nasa_power_{region_id}_monthly.csv"
        df_weather.to_csv(csv_path, index=False)
        logger.info("Saved parsed weather series to %s (%d months)", csv_path, len(df_weather))

        return {
            "metadata": raw_archive["metadata"],
            "dataframe": df_weather,
            "raw_json_path": str(json_path),
            "csv_path": str(csv_path),
        }

    def _parse_power_payload(
        self,
        payload: Dict[str, Any],
        region_id: str,
        region_name: str,
        latitude: float,
        longitude: float,
    ) -> pd.DataFrame:
        """Parse NASA POWER JSON into a continuous monthly DataFrame.
        
        Note: NASA POWER returns 'YYYY13' for the annual average.
        This method filters out '13' and retains strictly standard months 01-12.
        """
        prop_param = payload.get("properties", {}).get("parameter", {})
        if not prop_param:
            return pd.DataFrame()

        rows = []
        # Keys are formatted as 'YYYYMM'
        sample_param = list(prop_param.keys())[0]
        ym_keys = sorted(list(prop_param[sample_param].keys()))

        for ym in ym_keys:
            if len(ym) != 6:
                continue
            year_str = ym[:4]
            month_str = ym[4:6]
            if month_str == "13":
                # Skip 13 which is the annual aggregate
                continue

            year = int(year_str)
            month = int(month_str)
            date_str = f"{year_str}-{month_str}-01"

            # Number of days in this month to calculate monthly total precipitation
            days_in_month = monthrange(year, month)[1]

            row = {
                "date": date_str,
                "year": year,
                "month": month,
                "region_id": region_id,
                "region_name": region_name,
                "latitude": latitude,
                "longitude": longitude,
            }

            for p_name, p_dict in prop_param.items():
                val = p_dict.get(ym)
                # NASA POWER sentinel for missing data is -999.0
                if val is not None and val != -999.0:
                    row[p_name] = float(val)
                else:
                    row[p_name] = None

            # Calculate total monthly precipitation: PRECTOTCORR is mm/day in NASA POWER
            if row.get("PRECTOTCORR") is not None:
                row["precipitation_monthly_mm"] = round(row["PRECTOTCORR"] * days_in_month, 2)
            else:
                row["precipitation_monthly_mm"] = None

            # Standardized column names
            row["temp_mean_c"] = row.get("T2M")
            row["temp_max_c"] = row.get("T2M_MAX")
            row["temp_min_c"] = row.get("T2M_MIN")
            row["relative_humidity_pct"] = row.get("RH2M")
            row["dew_point_c"] = row.get("T2MDEW")

            rows.append(row)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def load_cached_weather(self, region_id: str) -> Optional[pd.DataFrame]:
        """Load cached monthly weather CSV if available."""
        csv_path = self.raw_data_dir / f"nasa_power_{region_id}_monthly.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return None
