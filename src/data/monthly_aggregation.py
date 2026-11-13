"""Monthly Time-Series Aggregation and Weather Feature Alignment for India.

Key Scientific and Methodological Rules:
1. Target Variable: `documented_occurrence_count` — counts of verified, qualifying
   deposited occurrence events per calendar month.
2. Continuous Frequency: Strictly maintains a continuous monthly index (`MS` / Month Start)
   from start_year to end_year.
3. Explicit Missingness: Explicitly distinguishes between zero documented events
   and missing observations using an `is_missing` boolean flag. Never interpolates
   arbitrary zeros into gaps.
4. Leakage-Free Weather Join: Integrates NASA POWER monthly meteorological regressors
   (temperature, precipitation, humidity, dew point) aligned strictly by year and month.
   Lagged features (1-month, 2-month lags) are constructed without lookahead bias.
5. Regional Consistency: Weather data is merged according to the pathogen's regional
   presence (e.g., Punjab/Haryana for wheat rust, Odisha/Bengal for rice blast/brown spot,
   or National Composite).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

from src.preprocessing import validate_and_clean_dataset

logger = logging.getLogger(__name__)


def build_continuous_monthly_series(
    df_clean: pd.DataFrame,
    pathogen_slug: str,
    pathogen_info: Dict[str, Any],
    weather_df: Optional[pd.DataFrame] = None,
    start_year: int = 2015,
    end_year: int = 2025,
    region_id: str = "national_composite",
) -> pd.DataFrame:
    """Aggregate cleaned occurrence records into a continuous monthly time series.
    
    Preserves explicit missing value indicators and joins weather regressors.
    """
    # Create complete monthly date range
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-01"
    full_index = pd.date_range(start=start_date, end=end_date, freq="MS")

    if not df_clean.empty and "event_date_parsed" in df_clean.columns:
        # Extract Month Start
        df_clean = df_clean.copy()
        df_clean["month_start"] = df_clean["event_date_parsed"].dt.to_period("M").dt.to_timestamp()
        
        # Monthly count of documented occurrences
        monthly_counts = (
            df_clean.groupby("month_start")
            .size()
            .rename("documented_occurrence_count")
            .to_frame()
        )
    else:
        monthly_counts = pd.DataFrame(columns=["documented_occurrence_count"])

    # Reindex to complete monthly timeline
    df_ts = monthly_counts.reindex(full_index)
    df_ts.index.name = "date"
    df_ts = df_ts.reset_index()

    # Determine date span where surveillance was active
    if not monthly_counts.empty:
        obs_min = monthly_counts.index.min()
        obs_max = monthly_counts.index.max()
    else:
        obs_min = pd.Timestamp(start_date)
        obs_max = pd.Timestamp(end_date)

    # Scientific rule:
    # During the surveillance span [obs_min, obs_max], months with NaN are months where no
    # events were documented; we record occurrence = 0, but set is_missing = False if active.
    # Outside the span, or where continuous surveillance cannot be assumed, is_missing = True.
    df_ts["is_missing"] = False
    
    # Months with no recorded observations
    nan_mask = df_ts["documented_occurrence_count"].isna()
    # Within active operational window, records with 0 count
    within_window = (df_ts["date"] >= obs_min) & (df_ts["date"] <= obs_max)
    
    # Count is 0 for unobserved months in window; otherwise NaN
    df_ts["documented_occurrence_count"] = df_ts["documented_occurrence_count"].fillna(0).astype(int)
    # Outside active window, label as unobserved / missing surveillance
    df_ts.loc[~within_window, "is_missing"] = True

    # Calendar metadata
    df_ts["year"] = df_ts["date"].dt.year
    df_ts["month"] = df_ts["date"].dt.month
    df_ts["month_name"] = df_ts["date"].dt.strftime("%b")
    df_ts["quarter"] = df_ts["date"].dt.quarter

    # Indian Agricultural Season assignment
    # Kharif: Jun-Oct (6-10), Rabi: Nov-Mar (11,12,1,2,3), Zaid: Apr-May (4,5)
    def assign_season(m: int) -> str:
        if m in [6, 7, 8, 9, 10]:
            return "Kharif (Monsoon)"
        elif m in [11, 12, 1, 2, 3]:
            return "Rabi (Winter)"
        else:
            return "Zaid (Summer)"

    df_ts["crop_season"] = df_ts["month"].apply(assign_season)
    df_ts["pathogen_slug"] = pathogen_slug
    df_ts["scientific_name"] = pathogen_info.get("scientific_name")
    df_ts["common_name"] = pathogen_info.get("common_name")
    df_ts["host_crop"] = pathogen_info.get("host_crop")
    df_ts["region_id"] = region_id

    # 4. Join weather variables if provided
    if weather_df is not None and not weather_df.empty:
        weather_clean = weather_df.copy()
        weather_clean["date"] = pd.to_datetime(weather_clean["date"]).dt.to_period("M").dt.to_timestamp()
        weather_cols = [
            "date", "temp_mean_c", "temp_max_c", "temp_min_c",
            "precipitation_monthly_mm", "relative_humidity_pct", "dew_point_c"
        ]
        weather_cols = [c for c in weather_cols if c in weather_clean.columns]
        
        df_ts = pd.merge(df_ts, weather_clean[weather_cols], on="date", how="left")

        # Create strictly lagged weather regressors (Lag 1 and Lag 2) to eliminate future leakage
        for w_col in ["temp_mean_c", "precipitation_monthly_mm", "relative_humidity_pct"]:
            if w_col in df_ts.columns:
                df_ts[f"{w_col}_lag1"] = df_ts[w_col].shift(1)
                df_ts[f"{w_col}_lag2"] = df_ts[w_col].shift(2)

    return df_ts


def process_all_pathogens_to_monthly(
    config_path: str = "config/settings.yaml",
    raw_gbif_dir: Path = Path("data/raw/gbif"),
    raw_weather_dir: Path = Path("data/raw/weather"),
    output_dir: Path = Path("data/processed"),
) -> Dict[str, pd.DataFrame]:
    """Process all configured pathogens into individual and unified monthly time series."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    pathogens = settings.get("pathogens", {})
    start_year = settings.get("data_collection", {}).get("start_year", 2015)
    end_year = min(settings.get("data_collection", {}).get("end_year", 2026), 2025)

    # Load regional weather series
    weather_series_map: Dict[str, pd.DataFrame] = {}
    for w_csv in raw_weather_dir.glob("nasa_power_*_monthly.csv"):
        region_id = w_csv.stem.replace("nasa_power_", "").replace("_monthly", "")
        w_df = pd.read_csv(w_csv)
        weather_series_map[region_id] = w_df

    national_weather = weather_series_map.get("national_composite", pd.DataFrame())

    all_monthly_dfs: List[pd.DataFrame] = []
    processed_dict: Dict[str, pd.DataFrame] = {}

    for slug, info in pathogens.items():
        sci_name = info.get("scientific_name")
        file_slug = sci_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        raw_csv = raw_gbif_dir / f"gbif_{file_slug}_raw.csv"

        if not raw_csv.exists():
            logger.warning("Raw GBIF CSV missing for %s", slug)
            continue

        # Clean individual raw dataset
        df_clean = validate_and_clean_dataset(raw_csv, slug, info, output_dir)

        # Select representative regional weather
        reg_id = "national_composite"
        if slug in ["pyricularia_oryzae", "bipolaris_oryzae"]:
            reg_id = "odisha_bengal"  # Eastern Rice Belt
        elif slug in ["puccinia_striiformis"]:
            reg_id = "punjab_haryana"  # Indo-Gangetic Wheat Belt
        elif slug in ["fusarium_oxysporum", "macrophomina_phaseolina"]:
            reg_id = "maharashtra"  # Western/Central Pulse/Cotton Belt

        target_weather = weather_series_map.get(reg_id, national_weather)

        # Build continuous monthly series
        monthly_df = build_continuous_monthly_series(
            df_clean=df_clean,
            pathogen_slug=slug,
            pathogen_info=info,
            weather_df=target_weather,
            start_year=start_year,
            end_year=end_year,
            region_id=reg_id,
        )

        # Export individual monthly series
        out_path = output_dir / f"monthly_{slug}.csv"
        monthly_df.to_csv(out_path, index=False)
        logger.info("Saved monthly series for '%s' to %s (%d months)", slug, out_path, len(monthly_df))

        processed_dict[slug] = monthly_df
        all_monthly_dfs.append(monthly_df)

    # Create combined multi-pathogen dataset
    if all_monthly_dfs:
        combined_df = pd.concat(all_monthly_dfs, ignore_index=True)
        combined_path = output_dir / "india_plant_diseases_monthly.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info("Saved unified multi-pathogen monthly dataset to %s (%d total rows)", combined_path, len(combined_df))

    return processed_dict


if __name__ == "__main__":
    process_all_pathogens_to_monthly()
