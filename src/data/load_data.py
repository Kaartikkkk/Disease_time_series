"""Unified data loader and caching interface for Plant Disease Outbreak Forecasting."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from src.data.clean_data import clean_plant_disease_data, DataQualityReport
from src.data.fetch_data import (
    fetch_agroscope_grapevine,
    fetch_wheat_blast_brazil,
    fetch_citrus_ctv,
    fetch_tomato_tswv,
    generate_synthetic_benchmark,
)
from src.data.source_registry import DATASET_REGISTRY, DatasetMetadata, get_dataset_metadata
from src.utils.logging_config import logger
from src.utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_dataset(dataset_id: str, use_cache: bool = True) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Load, clean, and validate a dataset by its registered identifier.

    Caches the cleaned canonical output in data/processed/.
    """
    processed_file = PROCESSED_DATA_DIR / f"{dataset_id}_clean.csv"

    if use_cache and processed_file.exists():
        try:
            df = pd.read_csv(processed_file, parse_dates=["date"])
            _, report = clean_plant_disease_data(df)
            logger.info(f"Loaded cached processed dataset: {dataset_id} ({len(df)} rows)")
            return df, report
        except Exception as e:
            logger.warning(f"Failed to load cached processed dataset {dataset_id}: {e}. Reprocessing.")

    # Dispatch to specific data fetcher
    if dataset_id == "agroscope_grapevine":
        raw_df = fetch_agroscope_grapevine(use_cache=use_cache)
    elif dataset_id == "wheat_blast_brazil":
        raw_df = fetch_wheat_blast_brazil(use_cache=use_cache)
    elif dataset_id == "citrus_ctv":
        raw_df = fetch_citrus_ctv(use_cache=use_cache)
    elif dataset_id == "tomato_tswv":
        raw_df = fetch_tomato_tswv(use_cache=use_cache)
    elif dataset_id == "synthetic_demo_benchmark":
        raw_df = generate_synthetic_benchmark()
    else:
        raise ValueError(
            f"Unknown dataset_id '{dataset_id}'. Choose from: {list(DATASET_REGISTRY.keys())}"
        )

    clean_df, report = clean_plant_disease_data(raw_df)
    clean_df.to_csv(processed_file, index=False)
    logger.info(f"Saved processed dataset {dataset_id} to {processed_file}")
    return clean_df, report


def load_all_available_datasets(use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """Pre-fetch and clean all supported real and demonstration datasets."""
    datasets = {}
    for did in DATASET_REGISTRY.keys():
        try:
            df, _ = load_dataset(did, use_cache=use_cache)
            datasets[did] = df
        except Exception as e:
            logger.error(f"Failed to load dataset {did}: {e}")
    return datasets
