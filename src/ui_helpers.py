"""Shared UI Components, Styling, and Data Loaders for Verdant Intelligence."""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.icons import (
    icon_arrow_right,
    icon_download,
    icon_forecast,
    icon_leaf,
    icon_logo,
    icon_model_compare,
    icon_pathology,
    icon_pin,
    icon_pulse,
    icon_shield,
    icon_thermometer,
    icon_timeseries,
    icon_weather,
)

PROCESSED_DIR = Path("data/processed")
RAW_GBIF_DIR = Path("data/raw/gbif")
RAW_WEATHER_DIR = Path("data/raw/weather")
REPORTS_DIR = Path("data/reports")
ASSETS_DIR = Path("assets")


def get_image_base64(image_path: Path) -> str:
    """Read an image file and return base64 string for seamless HTML embedding."""
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = image_path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def inject_custom_css():
    """Inject Verdant Intelligence editorial design system into Streamlit."""
    st.markdown(
        """
        <style>
        /* =====================================================================
           GOOGLE FONTS & BASE DESIGN TOKENS
           ===================================================================== */
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --forest-deep: #102C26;
            --evergreen: #183D32;
            --botanical: #527A50;
            --sage-muted: #A9BEA0;
            --sage-light: #E3EBE0;
            --warm-ivory: #F6F4EC;
            --paper-white: #FFFEFA;
            --charcoal: #202822;
            --charcoal-muted: #5A665D;
            --muted-gold: #C4A66A;
            --gold-light: #FDF9F0;
            --border-subtle: rgba(82, 122, 80, 0.16);
            --shadow-subtle: 0 4px 20px rgba(16, 44, 38, 0.05);
            --shadow-float: 0 12px 32px rgba(16, 44, 38, 0.09);
        }

        /* Base Typography & Background */
        .stApp {
            background-color: var(--warm-ivory);
            color: var(--charcoal);
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-image: radial-gradient(rgba(82, 122, 80, 0.04) 1px, transparent 1px);
            background-size: 28px 28px;
        }

        /* Top App Padding & Header Hide */
        header[data-testid="stHeader"] {
            background-color: rgba(246, 244, 236, 0.85);
            backdrop-filter: blur(10px);
            z-index: 99;
        }

        /* Typography Hierarchy */
        h1, h2, h3, .editorial-heading {
            font-family: 'Fraunces', 'DM Serif Display', Georgia, serif !important;
            color: var(--forest-deep) !important;
            letter-spacing: -0.025em;
            font-feature-settings: "liga" 1, "kern" 1;
        }

        h1 {
            font-size: 2.5rem !important;
            font-weight: 600 !important;
            line-height: 1.18 !important;
        }

        h2 {
            font-size: 1.85rem !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
        }

        h3 {
            font-size: 1.35rem !important;
            font-weight: 500 !important;
        }

        p, div, label, span {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--charcoal);
        }

        /* Navigation Bar */
        .verdant-navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 254, 250, 0.85);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 14px 24px;
            margin-bottom: 28px;
            box-shadow: var(--shadow-subtle);
        }
        .verdant-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: var(--forest-deep);
        }
        .verdant-brand-title {
            font-family: 'Fraunces', serif;
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--forest-deep);
            line-height: 1;
        }
        .verdant-brand-sub {
            font-size: 0.72rem;
            color: var(--botanical);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 600;
        }
        .verdant-nav-links {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .verdant-nav-link {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--charcoal-muted);
            text-decoration: none;
            transition: color 0.2s ease;
        }
        .verdant-nav-link:hover {
            color: var(--forest-deep);
        }

        /* Hero Split Section */
        .hero-container {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 36px;
            align-items: center;
            background: linear-gradient(135deg, rgba(255, 254, 250, 0.95) 0%, rgba(246, 244, 236, 0.8) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 44px 48px;
            margin-bottom: 36px;
            box-shadow: var(--shadow-float);
            position: relative;
            overflow: hidden;
        }
        .hero-left {
            position: relative;
            z-index: 2;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--gold-light);
            border: 1px solid rgba(196, 166, 106, 0.35);
            color: #8C6F2D;
            padding: 5px 13px;
            border-radius: 30px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 18px;
        }
        .hero-headline {
            font-family: 'Fraunces', serif !important;
            font-size: 2.75rem !important;
            font-weight: 600 !important;
            line-height: 1.12 !important;
            color: var(--forest-deep) !important;
            margin-bottom: 18px;
            letter-spacing: -0.03em;
        }
        .hero-lead {
            font-size: 1.05rem;
            line-height: 1.6;
            color: var(--charcoal-muted);
            margin-bottom: 28px;
            max-width: 540px;
        }
        .hero-actions {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }
        .btn-primary-cta {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--forest-deep);
            color: var(--paper-white) !important;
            font-weight: 600;
            font-size: 0.92rem;
            padding: 12px 24px;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(16, 44, 38, 0.2);
        }
        .btn-primary-cta:hover {
            background: var(--evergreen);
            transform: translateY(-2px);
            box-shadow: 0 8px 18px rgba(16, 44, 38, 0.25);
            color: #FFFFFF !important;
        }
        .btn-secondary-cta {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: transparent;
            color: var(--forest-deep) !important;
            font-weight: 600;
            font-size: 0.92rem;
            padding: 11px 20px;
            border-radius: 12px;
            border: 1px solid rgba(16, 44, 38, 0.25);
            text-decoration: none;
            transition: all 0.2s ease;
        }
        .btn-secondary-cta:hover {
            background: rgba(16, 44, 38, 0.05);
            border-color: var(--forest-deep);
        }

        /* Telemetry status badge */
        .telemetry-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.78rem;
            color: var(--charcoal-muted);
            font-weight: 500;
        }
        .telemetry-status strong {
            color: var(--botanical);
        }

        /* Hero Right Visual Stack */
        .hero-right {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .hero-visual-frame {
            width: 100%;
            height: 380px;
            border-radius: 20px;
            overflow: hidden;
            border: 2px solid var(--paper-white);
            box-shadow: 0 16px 36px rgba(16, 44, 38, 0.14);
            position: relative;
        }
        .hero-visual-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 6s ease;
        }
        .hero-visual-frame:hover .hero-visual-img {
            transform: scale(1.03);
        }
        .hero-overlay-card {
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(255, 254, 250, 0.92);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(169, 190, 160, 0.35);
            border-radius: 14px;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        }

        /* Research Metric Cards */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 36px;
        }
        .research-metric-card {
            background: var(--paper-white);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: var(--shadow-subtle);
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }
        .research-metric-card::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--botanical) 0%, var(--muted-gold) 100%);
            opacity: 0;
            transition: opacity 0.25s ease;
        }
        .research-metric-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-float);
        }
        .research-metric-card:hover::after {
            opacity: 1;
        }
        .metric-icon-wrap {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: var(--warm-ivory);
            color: var(--botanical);
            margin-bottom: 12px;
        }
        .metric-label-text {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--charcoal-muted);
            margin-bottom: 4px;
        }
        .metric-value-text {
            font-family: 'Fraunces', serif;
            font-size: 1.85rem;
            font-weight: 600;
            color: var(--forest-deep);
            line-height: 1.1;
        }
        .metric-note-text {
            font-size: 0.78rem;
            color: var(--botanical);
            margin-top: 6px;
            font-weight: 500;
        }

        /* Process Steps Grid */
        .process-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 28px 0 36px 0;
            position: relative;
        }
        .process-card {
            background: var(--paper-white);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 24px 20px;
            box-shadow: var(--shadow-subtle);
            position: relative;
            transition: all 0.25s ease;
        }
        .process-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-float);
            border-color: var(--sage-muted);
        }
        .process-num {
            font-family: 'Fraunces', serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--muted-gold);
            margin-bottom: 12px;
            line-height: 1;
        }
        .process-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.98rem;
            font-weight: 600;
            color: var(--forest-deep);
            margin-bottom: 8px;
        }
        .process-desc {
            font-size: 0.82rem;
            line-height: 1.5;
            color: var(--charcoal-muted);
        }

        /* Model Capabilities Grid */
        .model-cards-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 24px 0 36px 0;
        }
        .model-card {
            background: var(--paper-white);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 24px 22px;
            box-shadow: var(--shadow-subtle);
            transition: all 0.25s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .model-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-float);
            border-color: var(--botanical);
        }
        .model-badge {
            display: inline-block;
            background: rgba(82, 122, 80, 0.12);
            color: var(--botanical);
            font-size: 0.72rem;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 10px;
        }
        .model-card-title {
            font-family: 'Fraunces', serif;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--forest-deep);
            margin-bottom: 8px;
        }
        .model-card-desc {
            font-size: 0.82rem;
            color: var(--charcoal-muted);
            line-height: 1.5;
            margin-bottom: 16px;
        }
        .model-stat-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-top: 14px;
            border-top: 1px dashed var(--sage-light);
            font-size: 0.78rem;
            color: var(--charcoal-muted);
        }
        .model-stat-val {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--forest-deep);
        }

        /* Final CTA Banner */
        .final-cta-section {
            background: linear-gradient(135deg, var(--forest-deep) 0%, var(--evergreen) 100%);
            border-radius: 24px;
            padding: 48px;
            color: var(--paper-white);
            text-align: center;
            margin: 44px 0 28px 0;
            box-shadow: 0 16px 40px rgba(16, 44, 38, 0.2);
            position: relative;
            overflow: hidden;
        }
        .final-cta-title {
            font-family: 'Fraunces', serif !important;
            font-size: 2.3rem !important;
            color: #FFFFFF !important;
            margin-bottom: 14px;
        }
        .final-cta-lead {
            font-size: 1rem;
            color: #D1FAE5;
            max-width: 620px;
            margin: 0 auto 28px auto;
            line-height: 1.6;
        }
        .btn-gold-cta {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: var(--muted-gold);
            color: var(--forest-deep) !important;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 13px 28px;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(196, 166, 106, 0.35);
        }
        .btn-gold-cta:hover {
            background: #D4B67A;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(196, 166, 106, 0.45);
        }

        /* Editorial Footer */
        .verdant-footer {
            border-top: 1px solid var(--border-subtle);
            padding: 32px 0;
            margin-top: 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--charcoal-muted);
        }

        /* Keyframes */
        @keyframes pulseGlow {
            0% { transform: scale(0.95); opacity: 0.6; }
            50% { transform: scale(1.6); opacity: 0; }
            100% { transform: scale(0.95); opacity: 0.6; }
        }

        /* Accessibility: Reduced Motion */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation: none !important;
                transition: none !important;
            }
        }

        /* Responsive Breakpoints */
        @media (max-width: 960px) {
            .hero-container {
                grid-template-columns: 1fr;
                padding: 32px 24px;
            }
            .metric-grid, .process-grid, .model-cards-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        @media (max-width: 600px) {
            .metric-grid, .process-grid, .model-cards-grid {
                grid-template-columns: 1fr;
            }
            .hero-headline {
                font-size: 2.1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_config_settings() -> Dict[str, Any]:
    """Load settings from config/settings.yaml."""
    config_file = Path("config/settings.yaml")
    if not config_file.exists():
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_all_pathogen_options() -> Dict[str, str]:
    """Return dictionary of pathogen slug -> formatted display label."""
    settings = get_config_settings()
    pathogens = settings.get("pathogens", {})
    return {
        slug: f"{info.get('common_name')} ({info.get('scientific_name')})"
        for slug, info in pathogens.items()
    }


@st.cache_data
def load_monthly_series(slug: str) -> Optional[pd.DataFrame]:
    """Load processed continuous monthly series for a pathogen."""
    path = PROCESSED_DIR / f"monthly_{slug}.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


@st.cache_data
def load_clean_records(slug: str) -> Optional[pd.DataFrame]:
    """Load cleaned occurrence records for a pathogen."""
    path = PROCESSED_DIR / f"clean_{slug}_records.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        if "event_date_parsed" in df.columns:
            df["event_date_parsed"] = pd.to_datetime(df["event_date_parsed"])
        return df
    return None


@st.cache_data
def load_data_quality_report_data() -> Optional[Dict[str, Any]]:
    """Load data quality report JSON."""
    path = REPORTS_DIR / "data_quality_report.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_weather_series(region_id: str) -> Optional[pd.DataFrame]:
    """Load NASA POWER monthly weather series for a region."""
    path = RAW_WEATHER_DIR / f"nasa_power_{region_id}_monthly.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def format_plotly_figure(
    fig: go.Figure,
    title: str = "",
    height: int = 420,
    show_legend: bool = True,
) -> go.Figure:
    """Apply high-contrast editorial styling to Plotly charts."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {
                "size": 15,
                "family": "Fraunces, DM Serif Display, Georgia, serif",
                "color": "#102C26",
            },
        },
        template="plotly_white",
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        height=height,
        margin=dict(l=40, r=40, t=55, b=40),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, family="Plus Jakarta Sans"),
        ),
        hoverlabel=dict(
            bgcolor="#102C26",
            font_size=12,
            font_family="Plus Jakarta Sans",
            font_color="#FFFFFF",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E3EBE0",
        linecolor="#A9BEA0",
        tickfont=dict(size=11, family="Plus Jakarta Sans", color="#5A665D"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E3EBE0",
        linecolor="#A9BEA0",
        tickfont=dict(size=11, family="Plus Jakarta Sans", color="#5A665D"),
    )
    return fig
