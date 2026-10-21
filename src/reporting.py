"""Reporting and Artifact Generation Module for Plant Disease Forecasting System.

Generates:
1. Methodology and System Architecture Report (Markdown and JSON)
2. Model Comparison and Evaluation Summary Tables
3. CSV Export Bundles for processed observations and forward forecasts
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def generate_methodology_report(
    output_dir: Path = Path("data/reports"),
) -> Dict[str, Any]:
    """Generate comprehensive methodology, data provenance, and limitations report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_time = datetime.now(timezone.utc).isoformat()

    report_content = {
        "title": "Methodology, Statistical Assumptions, and System Constraints",
        "generated_at_utc": report_time,
        "country_scope": "Republic of India (IN)",
        "target_definition": {
            "name": "documented_occurrence_count",
            "unit": "Integer count of qualifying occurrence records per calendar month",
            "meaning": (
                "Measures the frequency of verified scientific, diagnostic, or preserved "
                "pathogen samples deposited in official public biodiversity and agricultural "
                "repositories. It is an index of documented field surveillance activity and "
                "pathogen presence, not a complete census of farm-level disease incidence."
            ),
            "missingness_policy": (
                "Months without observations are flagged as `is_missing = True` outside "
                "verified active surveillance windows. We never arbitrarily interpolate zeros "
                "to artificially complete a time series."
            ),
        },
        "data_sources": [
            {
                "name": "GBIF Occurrence Search API (Techdocs OpenAPI v1)",
                "url": "https://api.gbif.org/v1/occurrence/search",
                "license": "Creative Commons Attribution 4.0 International (CC-BY 4.0) / CC0",
                "temporal_range": "2015-01-01 to 2026-04-30",
                "attribution": "GBIF.org (2026). Occurrence Download accessed via GBIF REST API.",
            },
            {
                "name": "NASA POWER Monthly Agrometeorology API",
                "url": "https://power.larc.nasa.gov/api/temporal/monthly/point",
                "license": "NASA Open Data Policy (Public Domain)",
                "variables": ["T2M (Mean Temp)", "T2M_MAX", "T2M_MIN", "PRECTOTCORR (Rainfall)", "RH2M (Humidity)", "T2MDEW (Dew Point)"],
                "attribution": "NASA Langley Research Center POWER Project.",
            },
            {
                "name": "EPPO Global Database & Reporting Service",
                "url": "https://gd.eppo.int/",
                "license": "Open Access / Academic Fair Use",
                "role": "Verified regulatory pest alerts, separating publication dates from event dates.",
            },
        ],
        "statistical_framework": {
            "classical_models": [
                "Seasonal Naive (Benchmark baseline)",
                "AR(p) Autoregressive",
                "MA(q) Moving Average",
                "ARMA(p, q)",
                "ARIMA(p, d, q) Integrated Autoregressive Moving Average",
                "SARIMA(p, d, q)(P, D, Q, 12) Seasonal ARIMA",
                "Holt-Winters Exponential Smoothing",
                "ARIMAX / SARIMAX with strictly lagged NASA POWER weather regressors",
            ],
            "modern_models": [
                "Meta Prophet (Additive/multiplicative yearly seasonality + exogenous regressors)",
                "HistGradientBoosting and RandomForest Autoregressors (with cyclical month transforms)",
            ],
            "validation_strategy": "Strict chronological train (70%) / validation (15%) / test (15%) splits without shuffling to prevent temporal lookahead bias.",
            "metrics": ["MAE", "RMSE", "sMAPE (bounded 0-200%)", "WAPE (for count sparsity)", "Forecast Bias (Mean Error)", "Prediction Interval Coverage (80% and 95%)"],
        },
        "scientific_limitations": [
            "Passive surveillance bias: occurrences reflect sampling intensity of research institutions and diagnostic labs.",
            "Weather correlation does not prove biological causation: physical disease onset requires susceptible host phenology and microclimatic leaf wetness.",
            "Forecasts beyond observed weather rely on transparent climatology scenarios, not realized future weather.",
        ],
    }

    # Save JSON report
    json_path = output_dir / "methodology_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_content, f, indent=2)

    # Save Markdown report
    md_path = output_dir / "methodology_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🔬 Methodology, Statistical Framework & Limitations Report\n\n")
        f.write(f"**Publication Timestamp (UTC):** {report_time}  \n")
        f.write(f"**Geographic Scope:** {report_content['country_scope']}  \n\n")

        f.write("## 1. Target Variable Definition & Epistemology\n\n")
        f.write(f"**Target Variable:** `{report_content['target_definition']['name']}`  \n")
        f.write(f"**Measurement Unit:** {report_content['target_definition']['unit']}  \n\n")
        f.write(f"> {report_content['target_definition']['meaning']}\n\n")
        f.write(f"**Missingness Policy:** {report_content['target_definition']['missingness_policy']}\n\n")

        f.write("## 2. Public Data Sources & Attribution\n\n")
        for src in report_content["data_sources"]:
            f.write(f"### {src['name']}\n")
            f.write(f"- **URL:** [{src['url']}]({src['url']})\n")
            f.write(f"- **License:** {src['license']}\n")
            if "variables" in src:
                f.write(f"- **Key Parameters:** {', '.join(src['variables'])}\n")
            f.write("\n")

        f.write("## 3. Modeling & Evaluation Hierarchy\n\n")
        f.write("### Classical Statistical Suite\n")
        for m in report_content["statistical_framework"]["classical_models"]:
            f.write(f"- {m}\n")
        f.write("\n### Modern Machine Learning & Library Implementations\n")
        for m in report_content["statistical_framework"]["modern_models"]:
            f.write(f"- {m}\n")

        f.write(f"\n**Validation Split:** {report_content['statistical_framework']['validation_strategy']}\n")
        f.write(f"**Evaluation Metrics:** {', '.join(report_content['statistical_framework']['metrics'])}\n\n")

        f.write("## 4. Scientific Limitations & Responsible AI Disclaimers\n\n")
        for lim in report_content["scientific_limitations"]:
            f.write(f"- ⚠️ {lim}\n")

    logger.info("Saved methodology reports to %s and %s", json_path, md_path)
    return report_content


def export_forecast_csv(
    forecast_df: pd.DataFrame,
    pathogen_slug: str,
    model_name: str,
    output_dir: Path = Path("data/reports"),
) -> Path:
    """Export forward forecast table to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug_clean = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    csv_path = output_dir / f"forecast_{pathogen_slug}_{slug_clean}.csv"
    forecast_df.to_csv(csv_path, index=False)
    return csv_path
