"""Shared UI Components, Styling, and Data Loaders for Agricultural Disease Forecasting Using Time Series."""

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.icons import (
    icon_dashboard,
    icon_database,
    icon_info,
    icon_leaf,
    icon_microscope,
    icon_model_compare,
    icon_pulse,
    icon_target,
    icon_timeseries,
    icon_trending_up,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_GBIF_DIR = BASE_DIR / "data" / "raw" / "gbif"
RAW_WEATHER_DIR = BASE_DIR / "data" / "raw" / "weather"
REPORTS_DIR = BASE_DIR / "data" / "reports"
ASSETS_DIR = BASE_DIR / "assets"
STYLES_PATH = Path(__file__).resolve().parent / "styles.css"


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
    """Inject editorial design system stylesheet into Streamlit."""
    if STYLES_PATH.exists():
        css_content = STYLES_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)



def is_advanced_mode() -> bool:
    """Return True if user toggled to Advanced View (for researchers/evaluators), False for Simple View."""
    # Check query parameter first
    qp = st.query_params
    if "mode" in qp:
        val = str(qp.get("mode", "")).lower()
        if val in ["advanced", "adv"]:
            st.session_state["view_mode"] = "Advanced View"
            return True
        elif val in ["simple", "sim"]:
            st.session_state["view_mode"] = "Simple View"
            return False
    return st.session_state.get("view_mode", "Simple View") == "Advanced View"


def is_simple_mode() -> bool:
    """Return True if in Simple View (default for farmers/students)."""
    return not is_advanced_mode()


def render_what_does_this_mean(simple_text: str, technical_details: Optional[str] = None):
    """Render a dedicated, elegant 'What Does This Mean?' explanation card."""
    tech_html = ""
    if is_advanced_mode() and technical_details:
        tech_html = f"""
        <div class="what-mean-tech">
            <strong>Technical Note / Formulation:</strong> {technical_details}
        </div>
        """
    card_html = f"""
    <div class="what-does-this-mean-card">
        <div class="what-mean-header">
            {icon_info(16, "#527A50")}
            <span>What Does This Mean?</span>
        </div>
        <p class="what-mean-body">{simple_text}</p>
        {tech_html}
    </div>
    """
    st.html(card_html)


def render_graph_explanation_card(
    what_it_shows: str = "",
    parameters: Optional[str] = None,
    what_it_compares: Optional[str] = None,
    key_observation: Optional[str] = None,
    why_it_matters: Optional[str] = None,
    technical_details: Optional[str] = None,
    **kwargs,
):
    """Render a structured summary card directly below graphs detailing what it shows, parameters & variables, and what it compares."""
    param_text = parameters or kwargs.get("parameters") or key_observation or kwargs.get("key_observation") or ""
    compare_text = what_it_compares or kwargs.get("what_it_compares") or why_it_matters or kwargs.get("why_it_matters") or ""

    lines_html = []
    if what_it_shows:
        lines_html.append(f"""
        <div class="expl-line">
            <span class="expl-label">{icon_target(14, "#183D32")} <strong>What It Shows:</strong></span>
            <span class="expl-text">{what_it_shows}</span>
        </div>
        """)
    if param_text:
        label = "Parameters & Variables:" if parameters else "Observation:"
        icon = icon_pulse(14, "#2E5C4D") if parameters else icon_target(14, "#183D32")
        lines_html.append(f"""
        <div class="expl-line">
            <span class="expl-label">{icon} <strong>{label}</strong></span>
            <span class="expl-text">{param_text}</span>
        </div>
        """)
    if compare_text:
        label = "What It Compares:" if what_it_compares else "Takeaway:"
        icon = icon_model_compare(14, "#527A50") if what_it_compares else icon_leaf(14, "#527A50")
        lines_html.append(f"""
        <div class="expl-line">
            <span class="expl-label">{icon} <strong>{label}</strong></span>
            <span class="expl-text">{compare_text}</span>
        </div>
        """)
    if technical_details:
        lines_html.append(f"""
        <div class="expl-line" style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(82, 122, 80, 0.2);">
            <span class="expl-label">{icon_info(14, "#527A50")} <strong>Technical Note:</strong></span>
            <span class="expl-text">{technical_details}</span>
        </div>
        """)

    card_html = f"""
    <div class="graph-explanation-card">
        {''.join(lines_html)}
    </div>
    """
    st.html(card_html)


def render_top_navigation(active_page: str = "overview"):
    """Render modern horizontal top navigation bar replacing the sidebar."""
    is_adv = is_advanced_mode()
    
    nav_items = [
        ("overview", "/", "Overview", icon_dashboard),
        ("explorer", "/Disease_Explorer", "Disease Explorer", icon_microscope),
        ("timeseries", "/Time_Series_Analysis", "Time Series Analysis", icon_timeseries),
        ("comparison", "/Model_Comparison", "Model Comparison", icon_model_compare),
        ("forecasts", "/Future_Forecasts", "Future Forecasts", icon_trending_up),
        ("methodology", "/Data_and_Methodology", "Data & Methodology", icon_database),
    ]

    nav_links_html = []
    for slug, url, label, icon_fn in nav_items:
        is_active = (active_page == slug)
        cls = "verdant-nav-tab active" if is_active else "verdant-nav-tab"
        icon_col = "#166534" if is_active else "#5A665D"
        icon_markup = icon_fn(16, icon_col)
        nav_links_html.append(
            f'<a href="{url}" target="_self" class="{cls}">{icon_markup}<span>{label}</span></a>'
        )

    all_links = "".join(nav_links_html)
    simple_active = "active" if not is_adv else ""
    adv_active = "active" if is_adv else ""

    navbar_html = f"""
    <div class="verdant-top-navbar">
        <a href="/" target="_self" class="verdant-top-brand">
            {icon_leaf(22, "#166534")}
            <div>
                <div class="verdant-brand-name">Agricultural Disease Forecasting Using Time Series</div>
            </div>
        </a>
        <nav class="verdant-nav-links-wrap">
            {all_links}
        </nav>
        <div class="verdant-top-utils">
            <div class="mode-toggle-group" title="Toggle between Simple View and Advanced Research View">
                <a href="?mode=simple" target="_self" class="mode-toggle-btn {simple_active}">Simple View</a>
                <a href="?mode=advanced" target="_self" class="mode-toggle-btn {adv_active}">Advanced View</a>
            </div>
        </div>
    </div>
    """
    st.html(navbar_html)


def render_app_sidebar(active_page: str = "overview"):
    """Backward-compatible wrapper: Renders the horizontal top navigation."""
    render_top_navigation(active_page)


def render_top_header():
    """No-op: The horizontal top navigation bar serves as the header across all pages."""
    pass






@st.cache_data
def get_config_settings() -> Dict[str, Any]:
    """Load settings from config/settings.yaml."""
    config_file = Path("config/settings.yaml")
    if not config_file.exists():
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def get_pathogen_display_name(slug: str, force_scientific: bool = False) -> str:
    """Return friendly display label: Common name first, with scientific name in parentheses in Advanced Mode."""
    settings = get_config_settings()
    pathogens = settings.get("pathogens", {})
    info = pathogens.get(slug, {})
    common = info.get("common_name", slug.replace("_", " ").title())
    scientific = info.get("scientific_name", "")
    if is_advanced_mode() or force_scientific:
        return f"{common} ({scientific})" if scientific else common
    return common


@st.cache_data
def get_disease_common_name(slug: str) -> str:
    """Return common everyday disease name (e.g. 'Rice Blast')."""
    settings = get_config_settings()
    info = settings.get("pathogens", {}).get(slug, {})
    return info.get("common_name", slug.replace("_", " ").title())


@st.cache_data
def load_all_pathogen_options() -> Dict[str, str]:
    """Return dictionary of pathogen slug -> formatted display label with familiar common name first."""
    settings = get_config_settings()
    pathogens = settings.get("pathogens", {})
    return {
        slug: f"{info.get('common_name')} ({info.get('scientific_name')})"
        for slug, info in pathogens.items()
    }


@st.cache_data
def load_friendly_pathogen_options() -> Dict[str, str]:
    """Return dictionary of pathogen slug -> common name in Simple View, or common (scientific) in Advanced View."""
    settings = get_config_settings()
    pathogens = settings.get("pathogens", {})
    return {
        slug: get_pathogen_display_name(slug)
        for slug in pathogens.keys()
    }


@st.cache_data
def load_monthly_series(slug: str) -> Optional[pd.DataFrame]:
    """Load continuous monthly series for a pathogen from the single master dataset."""
    from src.data.dataset_service import load_authoritative_series
    return load_authoritative_series(slug)


@st.cache_data
def load_clean_records(slug: str) -> Optional[pd.DataFrame]:
    """Load occurrence records for a pathogen from the master dataset."""
    return load_authoritative_series(slug)


@st.cache_data
def load_data_quality_report_data() -> Optional[Dict[str, Any]]:
    """Load data quality report JSON."""
    path = REPORTS_DIR / "data_quality_report.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
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
