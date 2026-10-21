"""Public Plant Disease Dataset Source Registry and Provenance Metadata."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class DatasetMetadata:
    dataset_id: str
    name: str
    publisher: str
    source_url: str
    license: str
    crop: str
    disease: str
    pathogen: str
    geographic_coverage: str
    observation_frequency: str
    date_range: str
    target_variable: str
    target_units: str
    observation_type: str  # e.g., 'Real field surveillance', 'Experimental trial', 'Inoculum qPCR measurement', 'Demonstration'
    has_weather_data: bool
    is_synthetic: bool
    limitations: List[str]
    description: str


DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    "agroscope_grapevine": DatasetMetadata(
        dataset_id="agroscope_grapevine",
        name="Agroscope Grapevine Downy Mildew Airborne Inoculum & Microclimate Surveillance",
        publisher="Agroscope - Swiss Federal Centre of Excellence for Agricultural Research",
        source_url="https://github.com/agroscope-ch/plasmopy",
        license="CC-BY-4.0 / Open Access",
        crop="Grapevine",
        disease="Downy Mildew",
        pathogen="Plasmopara viticola",
        geographic_coverage="Changins, Nyon, Vaud, Switzerland (46.40° N, 6.23° E)",
        observation_frequency="Semi-weekly / sequential 2-3 days during infection season",
        date_range="2023-05-01 to 2025-08-31 (multi-year seasons)",
        target_variable="spore_concentration_qpcr",
        target_units="log10(copies/m³)",
        observation_type="Real field surveillance qPCR quantification of airborne sporangia",
        has_weather_data=True,
        is_synthetic=False,
        limitations=[
            "Target is airborne sporangial concentration, representing primary and secondary infection pressure rather than visible foliar lesions.",
            "Winter months (dormancy period: October - April) have naturally halted spore release and are excluded or near-zero.",
            "Weather is sampled from an in-vineyard meteorological station at 10-minute intervals and aggregated to daily means.",
        ],
        description="Comprehensive real-world agricultural surveillance from Switzerland tracking Grapevine Downy Mildew primary and secondary infection dynamics, airborne sporangial load, and microclimate (temperature, relative humidity, precipitation, leaf wetness).",
    ),
    "wheat_blast_brazil": DatasetMetadata(
        dataset_id="wheat_blast_brazil",
        name="Wheat Blast Multi-Region Field Surveillance & Epidemiology Trials",
        publisher="Del Ponte et al., Embrapa Trigo & University of Passo Fundo",
        source_url="https://github.com/emdelponte/r4pde",
        license="GPL-3.0 / Open Access Research",
        crop="Wheat",
        disease="Wheat Blast",
        pathogen="Magnaporthe oryzae (Pyricularia oryzae)",
        geographic_coverage="Central-South Brazil (Mato Grosso do Sul, Paraná, São Paulo)",
        observation_frequency="Multi-location seasonal cohort evaluations",
        date_range="2012 to 2018",
        target_variable="incidence_rate",
        target_units="percentage (%)",
        observation_type="Real agricultural field trial observations",
        has_weather_data=True,
        is_synthetic=False,
        limitations=[
            "Data points correspond to specific field study harvests and heading evaluation periods across multiple geographical regions.",
            "Requires aggregation by date or geographic cohort for continuous regular time series modeling.",
            "Matched with real ERA5 / Open-Meteo historical weather archives using exact site coordinates.",
        ],
        description="Empirical field observations of Wheat Blast spike incidence and severity across major Brazilian wheat growing regions with precise geospatial coordinates.",
    ),
    "citrus_ctv": DatasetMetadata(
        dataset_id="citrus_ctv",
        name="Citrus Tristeza Virus (CTV) Longitudinal Orchard Surveillance",
        publisher="Gottwald et al., APS Press / epiphy repository",
        source_url="https://github.com/chgigot/epiphy",
        license="Academic Open Data / GPL-2.0",
        crop="Citrus",
        disease="Citrus Tristeza Virus (CTV)",
        pathogen="Citrus tristeza virus (Closterovirus)",
        geographic_coverage="Valencia, Spain and Corsica, France",
        observation_frequency="Annual / multi-season census of tree health",
        date_range="1981 to 1993",
        target_variable="incidence_rate",
        target_units="proportion (0.0 to 1.0)",
        observation_type="Real field tree-by-tree visual and serological (ELISA) diagnosis",
        has_weather_data=False,
        is_synthetic=False,
        limitations=[
            "Annual inspection cadence suited for long-term multi-year epidemic progression rather than short-term daily forecasts.",
            "Zero values indicate verified uninfected status at inspection time.",
        ],
        description="Classic multi-year longitudinal epidemiological surveys tracking tree-level Citrus Tristeza Virus spread across commercial orchards.",
    ),
    "tomato_tswv": DatasetMetadata(
        dataset_id="tomato_tswv",
        name="Tomato Spotted Wilt Virus (TSWV) Temporal Epidemic Spread",
        publisher="Bald et al. / epiphy",
        source_url="https://github.com/chgigot/epiphy",
        license="Academic Open Data / GPL-2.0",
        crop="Tomato",
        disease="Spotted Wilt Virus",
        pathogen="Tomato spotted wilt orthotospovirus (TSWV)",
        geographic_coverage="Waite Agricultural Research Institute, Australia",
        observation_frequency="Weekly survey across the growing season",
        date_range="Multi-week seasonal cohorts (1928-1929)",
        target_variable="incidence_rate",
        target_units="proportion (0.0 to 1.0)",
        observation_type="Real sequential field plant inspections",
        has_weather_data=False,
        is_synthetic=False,
        limitations=[
            "Tracks progression over a closed vegetable growing season.",
            "Captures classical S-shaped logistic epidemic curve.",
        ],
        description="Seminal quantitative plant disease epidemiology dataset documenting the rapid temporal spread of thrips-vectored Tomato Spotted Wilt Virus across experimental fields.",
    ),
    "synthetic_demo_benchmark": DatasetMetadata(
        dataset_id="synthetic_demo_benchmark",
        name="Synthetic Demonstration Benchmark: Potato Late Blight Daily Dynamics",
        publisher="Project Synthetic Epidemiological Generator (Explicitly Labeled Demonstration)",
        source_url="internal://synthetic_generator",
        license="MIT",
        crop="Potato",
        disease="Late Blight",
        pathogen="Phytophthora infestans",
        geographic_coverage="Simulated Temperate Region (43.5° N, 1.5° E)",
        observation_frequency="Continuous Daily",
        date_range="2021-01-01 to 2024-12-31 (4 full annual cycles, 1461 daily records)",
        target_variable="incidence_rate",
        target_units="percentage (%)",
        observation_type="Synthetic demonstration dataset generated from modified Ullrich-Schrödter late blight disease progress equations and weather couplings",
        has_weather_data=True,
        is_synthetic=True,
        limitations=[
            "THIS IS A SYNTHETIC DEMONSTRATION BENCHMARK CREATED STRICTLY FOR TESTING, INTERFACE VALIDATION, AND HIGH-FREQUENCY DAILY BACKTESTING.",
            "DO NOT CITE AS REAL-WORLD AGRICULTURAL SURVEILLANCE DATA.",
            "Simulates multi-year seasonality, temperature and humidity dependency, and stochastic epidemic outbreaks.",
        ],
        description="Explicitly labeled synthetic demonstration time series modeling multi-year daily potato late blight incidence with realistic seasonal peaks, microclimate drivers, and outbreak anomalies.",
    ),
}


def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    """Retrieve metadata for a dataset."""
    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{dataset_id}' not found. Available datasets: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_id]


def list_all_datasets() -> pd.DataFrame:
    """Return a summary table of all registered public and demonstration datasets."""
    records = []
    for d in DATASET_REGISTRY.values():
        records.append({
            "Dataset ID": d.dataset_id,
            "Name": d.name,
            "Crop": d.crop,
            "Disease": d.disease,
            "Publisher": d.publisher,
            "Type": d.observation_type,
            "Is Synthetic": "YES (DEMO)" if d.is_synthetic else "NO (REAL PUBLIC)",
            "Date Coverage": d.date_range,
            "Target": f"{d.target_variable} ({d.target_units})",
            "Has Weather": "Yes" if d.has_weather_data else "No",
        })
    return pd.DataFrame(records)
