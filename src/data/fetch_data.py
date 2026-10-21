"""Data acquisition adapters for public agricultural surveillance and demonstration datasets."""

import io
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import rdata
import requests
from src.data.source_registry import DATASET_REGISTRY, DatasetMetadata
from src.data.validate_schema import ensure_canonical_dataframe
from src.data.weather_features import fetch_open_meteo_historical
from src.utils.logging_config import logger
from src.utils.paths import RAW_DATA_DIR, SAMPLE_DATA_DIR


def fetch_agroscope_grapevine(use_cache: bool = True) -> pd.DataFrame:
    """Download and compile multi-year Grapevine Downy Mildew surveillance from Agroscope.

    Datasets:
    - 2023, 2024, 2025 airborne spore concentrations (qPCR) for Chasselas and Pinot Noir
    - Synchronized meteorological station at Changins (Vaud, Switzerland)
    """
    raw_save_path = RAW_DATA_DIR / "agroscope_grapevine_raw.csv"
    if use_cache and raw_save_path.exists():
        try:
            df = pd.read_csv(raw_save_path, parse_dates=["date"])
            logger.info(f"Loaded cached Agroscope Grapevine data ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cached Agroscope file: {e}")

    logger.info("Fetching Agroscope Grapevine surveillance data from GitHub...")
    base_url = "https://raw.githubusercontent.com/agroscope-ch/plasmopy/main/data/input"
    years = ["2023", "2024", "2025"]
    cultivars = ["chasselas", "pinot.noir"]

    all_disease_records = []
    all_meteo_records = []

    # 1. Fetch weather files
    for yr in years:
        meteo_url = f"{base_url}/{yr}_meteo_changins.csv"
        try:
            r = requests.get(meteo_url, timeout=15)
            if r.status_code == 200:
                m_df = pd.read_csv(io.StringIO(r.text), sep=";")
                # Columns: Date, CHANGINS - Température moy. +2 m (°C), Humidité moy. (%), Précipitations intensité (mm/h)
                m_df["parsed_date"] = pd.to_datetime(m_df["Date"], format="%d.%m.%Y %H:%M", errors="coerce")
                m_df = m_df.dropna(subset=["parsed_date"])
                m_df["day"] = m_df["parsed_date"].dt.normalize()

                # Extract available columns safely
                temp_col = [c for c in m_df.columns if "Temp" in c or "temp" in c]
                hum_col = [c for c in m_df.columns if "Humid" in c or "humid" in c]
                prec_col = [c for c in m_df.columns if "Précip" in c or "precip" in c or "rain" in c]

                agg_dict = {}
                if temp_col:
                    agg_dict[temp_col[0]] = "mean"
                if hum_col:
                    agg_dict[hum_col[0]] = "mean"
                if prec_col:
                    agg_dict[prec_col[0]] = "sum"

                if agg_dict:
                    daily_meteo = m_df.groupby("day").agg(agg_dict).reset_index()
                    daily_meteo = daily_meteo.rename(columns={
                        "day": "date",
                        temp_col[0] if temp_col else "": "temperature",
                        hum_col[0] if hum_col else "": "humidity",
                        prec_col[0] if prec_col else "": "rainfall",
                    })
                    all_meteo_records.append(daily_meteo)
        except Exception as e:
            logger.warning(f"Could not load meteo for year {yr}: {e}")

    df_meteo = pd.concat(all_meteo_records, ignore_index=True).drop_duplicates(subset=["date"]) if all_meteo_records else pd.DataFrame()

    # 2. Fetch disease observations (qPCR airborne spore counts)
    for yr in years:
        for cv in cultivars:
            qpcr_url = f"{base_url}/{yr}_qPCR_changins.{cv}.csv"
            try:
                r = requests.get(qpcr_url, timeout=15)
                if r.status_code == 200:
                    q_df = pd.read_csv(io.StringIO(r.text), sep=";")
                    q_df["parsed_date"] = pd.to_datetime(q_df["Date"], format="%d.%m.%Y %H:%M", errors="coerce")
                    q_df = q_df.dropna(subset=["parsed_date"])
                    q_df["date"] = q_df["parsed_date"].dt.normalize()

                    # Target: Counts (copies/m³ or log10 copies)
                    # Notice: Drop completely missing unobserved sampling rows
                    # But retain measured values
                    q_df["spore_count"] = pd.to_numeric(q_df["Counts"], errors="coerce")
                    valid_q = q_df.dropna(subset=["spore_count"]).copy()

                    valid_q["crop"] = f"Grapevine ({cv.replace('.', ' ').title()})"
                    valid_q["disease"] = "Downy Mildew"
                    valid_q["location_id"] = "Changins_Switzerland"
                    valid_q["latitude"] = 46.40
                    valid_q["longitude"] = 6.23
                    valid_q["source"] = "Agroscope_Switzerland"
                    valid_q["incidence_rate"] = valid_q["spore_count"]
                    valid_q["is_synthetic"] = False

                    all_disease_records.append(valid_q)
            except Exception as e:
                logger.warning(f"Could not load qPCR for {yr} {cv}: {e}")

    if not all_disease_records:
        raise RuntimeError("Failed to fetch Agroscope disease records from public repository.")

    df_disease = pd.concat(all_disease_records, ignore_index=True)

    # 3. Merge with weather records on date
    if not df_meteo.empty:
        df_merged = pd.merge(df_disease, df_meteo, on="date", how="left")
    else:
        df_merged = df_disease
        df_merged["temperature"] = np.nan
        df_merged["humidity"] = np.nan
        df_merged["rainfall"] = np.nan

    canonical_df = ensure_canonical_dataframe(df_merged)
    canonical_df.to_csv(raw_save_path, index=False)
    logger.info(f"Successfully processed Agroscope dataset: {len(canonical_df)} observations")
    return canonical_df


def fetch_wheat_blast_brazil(use_cache: bool = True) -> pd.DataFrame:
    """Download and parse Del Ponte et al. Wheat Blast field surveillance data."""
    raw_save_path = RAW_DATA_DIR / "wheat_blast_brazil_raw.csv"
    if use_cache and raw_save_path.exists():
        try:
            return pd.read_csv(raw_save_path, parse_dates=["date"])
        except Exception:
            pass

    url = "https://raw.githubusercontent.com/emdelponte/r4pde/main/data/BlastWheat.rda"
    logger.info(f"Fetching Wheat Blast dataset from {url}...")
    r = requests.get(url, timeout=20)
    parsed = rdata.parser.parse_data(r.content)
    converted = rdata.conversion.convert(parsed)
    df_raw = converted["BlastWheat"].copy()

    # Format dates from heading column (format: 'DD-MM-YYYY')
    df_raw["date"] = pd.to_datetime(df_raw["heading"], format="%d-%m-%Y", errors="coerce")
    df_raw = df_raw.dropna(subset=["date"]).sort_values("date")

    df_out = pd.DataFrame({
        "date": df_raw["date"],
        "crop": "Wheat",
        "disease": "Wheat Blast",
        "location_id": df_raw["location"].astype(str) + "_" + df_raw["state"].astype(str),
        "latitude": pd.to_numeric(df_raw["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(df_raw["longitude"], errors="coerce"),
        "incidence_rate": pd.to_numeric(df_raw["inc_mean"], errors="coerce"),
        "severity": pd.to_numeric(df_raw["index_mean"], errors="coerce"),
        "source": "DelPonte_Embrapa_WheatBlast",
        "is_synthetic": False,
    })

    # Fetch weather covariates for site centroid if coordinates exist
    valid_coords = df_out.dropna(subset=["latitude", "longitude"])
    if not valid_coords.empty:
        mean_lat = float(valid_coords["latitude"].mean())
        mean_lon = float(valid_coords["longitude"].mean())
        start_d = str(df_out["date"].min().date())
        end_d = str(df_out["date"].max().date())
        weather_df = fetch_open_meteo_historical(mean_lat, mean_lon, start_d, end_d)
        if not weather_df.empty:
            df_out = pd.merge(df_out, weather_df[["date", "temperature", "humidity", "rainfall"]], on="date", how="left")

    canonical_df = ensure_canonical_dataframe(df_out)
    canonical_df.to_csv(raw_save_path, index=False)
    return canonical_df


def fetch_citrus_ctv(use_cache: bool = True) -> pd.DataFrame:
    """Download and aggregate Citrus Tristeza Virus orchard surveillance data from epiphy."""
    raw_save_path = RAW_DATA_DIR / "citrus_ctv_raw.csv"
    if use_cache and raw_save_path.exists():
        try:
            return pd.read_csv(raw_save_path, parse_dates=["date"])
        except Exception:
            pass

    url = "https://raw.githubusercontent.com/chgigot/epiphy/master/data/citrus_ctv.rda"
    logger.info(f"Fetching Citrus CTV dataset from {url}...")
    r = requests.get(url, timeout=20)
    parsed = rdata.parser.parse_data(r.content)
    converted = rdata.conversion.convert(parsed)
    ctv_dict = converted["citrus_ctv"]

    records = []
    # Contains orchard sites: 'IVI3and4', 'IVI6and7', 'El_Realengo'
    for site_name, df_site in ctv_dict.items():
        # Columns: x, y, t (year), i (infected trees), n (total trees)
        # Aggregate by year and site
        annual = df_site.groupby("t").agg({"i": "sum", "n": "sum"}).reset_index()
        for _, row in annual.iterrows():
            year_int = int(row["t"])
            cases = float(row["i"])
            total = float(row["n"])
            rate = (cases / total) * 100.0 if total > 0 else 0.0
            records.append({
                "date": pd.Timestamp(year=year_int, month=6, day=1),
                "crop": "Citrus",
                "disease": "Citrus Tristeza Virus (CTV)",
                "location_id": f"Spain_{site_name}",
                "latitude": 39.4699,
                "longitude": -0.3763,
                "cases_or_affected": cases,
                "inspected_or_total": total,
                "incidence_rate": rate,
                "source": "Gottwald_epiphy_CitrusCTV",
                "is_synthetic": False,
            })

    df_out = pd.DataFrame(records).sort_values("date")
    canonical_df = ensure_canonical_dataframe(df_out)
    canonical_df.to_csv(raw_save_path, index=False)
    return canonical_df


def fetch_tomato_tswv(use_cache: bool = True) -> pd.DataFrame:
    """Download and aggregate Tomato Spotted Wilt Virus field surveillance data from epiphy."""
    raw_save_path = RAW_DATA_DIR / "tomato_tswv_raw.csv"
    if use_cache and raw_save_path.exists():
        try:
            return pd.read_csv(raw_save_path, parse_dates=["date"])
        except Exception:
            pass

    url = "https://raw.githubusercontent.com/chgigot/epiphy/master/data/tomato_tswv.rda"
    logger.info(f"Fetching Tomato TSWV dataset from {url}...")
    r = requests.get(url, timeout=20)
    parsed = rdata.parser.parse_data(r.content)
    converted = rdata.conversion.convert(parsed)
    tswv_dict = converted["tomato_tswv"]

    records = []
    base_dates = {
        "field_1928": pd.Timestamp("1928-10-15"),
        "field_1929": pd.Timestamp("1929-10-15"),
    }

    for field_name, df_field in tswv_dict.items():
        base_d = base_dates.get(field_name, pd.Timestamp("1928-10-15"))
        # Columns: x, y, t (survey week 1..N), i (infected plants), n (total plants)
        weekly = df_field.groupby("t").agg({"i": "sum", "n": "sum"}).reset_index()
        for _, row in weekly.iterrows():
            week_num = int(row["t"])
            obs_date = base_d + pd.Timedelta(weeks=week_num)
            cases = float(row["i"])
            total = float(row["n"])
            rate = (cases / total) * 100.0 if total > 0 else 0.0
            records.append({
                "date": obs_date,
                "crop": "Tomato",
                "disease": "Spotted Wilt Virus",
                "location_id": f"Waite_Institute_{field_name}",
                "latitude": -34.97,
                "longitude": 138.63,
                "cases_or_affected": cases,
                "inspected_or_total": total,
                "incidence_rate": rate,
                "source": "Bald_epiphy_TomatoTSWV",
                "is_synthetic": False,
            })

    df_out = pd.DataFrame(records).sort_values("date")
    canonical_df = ensure_canonical_dataframe(df_out)
    canonical_df.to_csv(raw_save_path, index=False)
    return canonical_df


def generate_synthetic_benchmark(
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-31",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a biologically grounded demonstration time series for high-frequency daily evaluation.

    MANDATORY LABELING: Clearly marked as synthetic demonstration benchmark (is_synthetic=True).
    Coupled with realistic annual seasonal temperature/humidity curves, Ullrich-Schrödter late blight
    favorability equations, auto-regressive contagion persistence, and stochastic disease outbreaks.
    """
    sample_save_path = SAMPLE_DATA_DIR / "synthetic_demo_potato_late_blight.csv"
    np.random.seed(random_seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    day_of_year = dates.dayofyear.values

    # 1. Realistic seasonal weather simulation for temperate agricultural zone
    # Mean annual temperature cycle: cold winter (2°C), warm summer (24°C)
    temperature = 13.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25) + np.random.normal(0, 2.2, n)
    # Humidity: higher in late spring/summer rain events and autumn
    humidity = 70.0 + 15.0 * np.cos(2 * np.pi * (day_of_year - 30) / 365.25) + np.random.normal(0, 6.0, n)
    humidity = np.clip(humidity, 35.0, 99.0)

    # Rainfall occurrences: Poisson rain events modulated by season
    rain_prob = 0.25 + 0.12 * np.sin(2 * np.pi * day_of_year / 365.25)
    has_rain = np.random.binomial(1, np.clip(rain_prob, 0.1, 0.45), n)
    rainfall = has_rain * np.random.exponential(scale=7.5, size=n)

    # 2. Agro-meteorological late blight favorability index (Ullrich-Schrödter model proxy)
    # Phytophthora infestans thrives between 15°C and 22°C when humidity >= 75%
    temp_favorability = np.exp(-0.5 * ((temperature - 18.5) / 4.0) ** 2)
    humidity_favorability = np.clip((humidity - 65.0) / 30.0, 0.0, 1.0)
    environmental_pressure = temp_favorability * humidity_favorability * (1.0 + 0.3 * (rainfall > 2.0))

    # 3. Dynamic disease progression: Autoregressive epidemic process
    # Low winter baseline (dormant inoculum), exponential expansion in favorable summer weeks
    incidence = np.zeros(n)
    incidence[0] = 1.5

    for t in range(1, n):
        # Base seasonal survival
        seasonal_potential = 30.0 * environmental_pressure[t]
        # Inoculum carryover from yesterday (persistence) + new infection pressure + noise
        autocorr = 0.88 * incidence[t - 1]
        new_infection = 0.15 * seasonal_potential
        shock = np.random.normal(0, 0.8)

        # Inject periodic discrete outbreak events (e.g. prolonged warm rainstorms)
        # Every year around mid-summer (days 180 to 220)
        if 180 <= day_of_year[t] <= 215 and environmental_pressure[t] > 0.65 and np.random.rand() < 0.08:
            shock += np.random.uniform(8.0, 22.0)

        daily_val = autocorr + new_infection + shock
        incidence[t] = max(0.0, min(100.0, daily_val))

    df_synth = pd.DataFrame({
        "date": dates,
        "crop": "Potato",
        "disease": "Late Blight (Demo)",
        "location_id": "Simulated_Temperate_Valley",
        "latitude": 43.50,
        "longitude": 1.50,
        "cases_or_affected": np.round(incidence * 10),
        "inspected_or_total": 1000,
        "incidence_rate": np.round(incidence, 2),
        "severity": np.round(incidence * 0.75, 2),
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "rainfall": np.round(rainfall, 1),
        "source": "SYNTHETIC_BENCHMARK_DEMONSTRATION",
        "is_synthetic": True,
    })

    canonical_df = ensure_canonical_dataframe(df_synth)
    canonical_df.to_csv(sample_save_path, index=False)
    logger.info(f"Generated synthetic demonstration benchmark ({len(canonical_df)} daily observations).")
    return canonical_df
