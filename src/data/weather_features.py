"""Weather data integration, Open-Meteo historical retrieval, and derived agro-meteorological features."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from src.utils.logging_config import logger
from src.utils.paths import RAW_DATA_DIR

WEATHER_CACHE_DIR = RAW_DATA_DIR / "weather_cache"
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_open_meteo_historical(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily historical weather from the public Open-Meteo Historical Archive API.

    Variables retrieved:
    - temperature_2m_max (°C)
    - temperature_2m_min (°C)
    - temperature_2m_mean (°C)
    - relative_humidity_2m_mean (%)
    - precipitation_sum (mm)
    - wind_speed_10m_max (km/h)

    Zero authentication required; full global coverage.
    """
    cache_key = f"weather_{latitude:.2f}_{longitude:.2f}_{start_date}_{end_date}.csv"
    cache_file = WEATHER_CACHE_DIR / cache_key

    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"])
            logger.info(f"Loaded cached weather records from {cache_file} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cached weather file {cache_file}: {e}. Re-fetching.")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "UTC",
    }

    logger.info(f"Querying Open-Meteo API for ({latitude}, {longitude}) from {start_date} to {end_date}...")
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        daily = payload.get("daily", {})

        if not daily or "time" not in daily:
            logger.error("Open-Meteo returned empty daily weather structure.")
            return pd.DataFrame()

        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily.get("temperature_2m_max"),
            "temp_min": daily.get("temperature_2m_min"),
            "temperature": daily.get("temperature_2m_mean"),
            "humidity": daily.get("relative_humidity_2m_mean"),
            "rainfall": daily.get("precipitation_sum"),
            "wind_speed": daily.get("wind_speed_10m_max"),
        })

        if use_cache:
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved weather cache to {cache_file}")

        return df
    except Exception as e:
        logger.error(f"Error fetching Open-Meteo historical weather: {e}")
        return pd.DataFrame()


def compute_derived_weather_features(df_weather: pd.DataFrame) -> pd.DataFrame:
    """Compute biologically meaningful lagged and rolling weather covariates.

    - temp_roll_7: 7-day rolling mean temperature (°C)
    - humidity_roll_7: 7-day rolling mean relative humidity (%)
    - rain_sum_7: 7-day cumulative rainfall (mm)
    - rain_sum_14: 14-day cumulative rainfall (mm)
    - infection_favorability_index: Heuristic agro-meteorological favorability index (0-1)
      based on favorable fungal temperature (15-25°C) and elevated humidity (>75%).
    """
    if df_weather.empty or "date" not in df_weather.columns:
        return df_weather

    df = df_weather.sort_values(by="date").copy()

    if "temperature" in df.columns:
        df["temp_roll_3"] = df["temperature"].rolling(window=3, min_periods=1).mean()
        df["temp_roll_7"] = df["temperature"].rolling(window=7, min_periods=1).mean()
        df["temp_lag_1"] = df["temperature"].shift(1)
        df["temp_lag_3"] = df["temperature"].shift(3)
        df["temp_lag_7"] = df["temperature"].shift(7)

    if "humidity" in df.columns:
        df["humidity_roll_7"] = df["humidity"].rolling(window=7, min_periods=1).mean()
        df["humidity_lag_1"] = df["humidity"].shift(1)
        df["humidity_lag_3"] = df["humidity"].shift(3)

    if "rainfall" in df.columns:
        df["rain_sum_3"] = df["rainfall"].rolling(window=3, min_periods=1).sum()
        df["rain_sum_7"] = df["rainfall"].rolling(window=7, min_periods=1).sum()
        df["rain_sum_14"] = df["rainfall"].rolling(window=14, min_periods=1).sum()
        df["rain_lag_1"] = df["rainfall"].shift(1)
        df["rain_lag_3"] = df["rainfall"].shift(3)

    # Biological favorability index: fungal sporulation and germination surge
    # when temp is in optimal range (15 - 25°C) and relative humidity > 70% or rainfall occurs
    if "temperature" in df.columns and "humidity" in df.columns:
        temp_score = np.exp(-0.5 * ((df["temperature"] - 20.0) / 4.5) ** 2)
        humidity_score = np.clip((df["humidity"] - 50.0) / 45.0, 0.0, 1.0)
        df["infection_favorability_index"] = np.clip(temp_score * humidity_score, 0.0, 1.0)

    return df
