"""Agricultural Disease Forecasting Using Time Series.

Plant Disease Outbreak Forecasting & Climate Intelligence Platform.
Overview Page — High-impact, minimal, professional executive summary.
"""

from datetime import datetime, timezone
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure main app page
st.set_page_config(
    page_title="AgriForecast — Agricultural Disease Forecasting",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.icons import (
    icon_arrow_right,
    icon_database,
    icon_info,
    icon_layers,
    icon_leaf,
    icon_logo,
    icon_network,
    icon_pathology,
    icon_trending_up,
)
from src.project_stats import get_project_telemetry
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_image_base64,
    inject_custom_css,
    load_monthly_series,
    render_top_navigation,
)

inject_custom_css()
render_top_navigation(active_page="overview")

# Load authoritative dynamic project telemetry
stats = get_project_telemetry()

# Load tailored image assets for the AgriForecast redesign
hero_ear_b64 = get_image_base64(ASSETS_DIR / "hero_ear_blended.png")
if not hero_ear_b64:
    hero_ear_b64 = get_image_base64(ASSETS_DIR / "wheat_canopy_ears.jpg")

obj_hills_b64 = get_image_base64(ASSETS_DIR / "objective_hills_blended.png")

img_monsoon_b64 = get_image_base64(ASSETS_DIR / "monsoon_weather_canopy.jpg")
img_sunset_b64 = get_image_base64(ASSETS_DIR / "cereal_harvest_sunset.jpg")
img_wheat_b64 = get_image_base64(ASSETS_DIR / "wheat_canopy_ears.jpg")
img_golden_b64 = get_image_base64(ASSETS_DIR / "golden_crop_field.jpg")

# Load real monthly series for primary preview
df_rice_blast = load_monthly_series("pyricularia_oryzae")
df_brown_spot = load_monthly_series("bipolaris_oryzae")

# ==============================================================================
# 1. HERO SECTION (MATCHING AGRIFORECAST SPECIFICATION)
# ==============================================================================
hero_markup = f"""
<div class="dashboard-hero-card">
    <div class="hero-text-content">
        <div class="hero-tag">TIME SERIES • DISEASE FORECASTING</div>
        <h1 class="hero-headline">
            Forecast Crop Diseases.<br>Protect Harvests.
        </h1>
        <p class="hero-subtitle">
            A time-series forecasting platform for Indian plant pathogens using classical, ML and deep learning models.
        </p>
        <div class="hero-actions">
            <a href="/Future_Forecasts" target="_self" class="hero-btn-primary">
                {icon_trending_up(16, "#FFFFFF")} Start Forecasting →
            </a>
            <a href="/Disease_Explorer" target="_self" class="hero-btn-secondary">
                {icon_layers(16, "#111827")} Explore Data
            </a>
        </div>
    </div>
    <div class="hero-image-wrapper">
        <img src="{hero_ear_b64}" class="hero-ear-img" alt="Agricultural crop forecasting" />
    </div>
</div>
"""
st.html(hero_markup)

# ==============================================================================
# 2. FOUR KPI CARDS (STUDY PERIOD, TOTAL RECORDS, PATHOGENS, FORECAST HORIZON)
# ==============================================================================
total_months_span = round(stats.get("total_months", 132) / 12.0)
start_label = stats.get("monthly_start_date", "Jan 2015")
end_label = stats.get("monthly_end_date", "Dec 2025")
total_obs = f"{stats.get('total_observations', 1320):,}"
n_taxa = stats.get("total_pathogens", 10)

kpi_markup = f"""
<div class="kpi-row-grid">
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_leaf(22, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label">Study Period</div>
            <div class="kpi-value">{total_months_span} Years</div>
            <div class="kpi-subtext">{start_label} – {end_label}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_database(22, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label">Total Records</div>
            <div class="kpi-value">{total_obs}</div>
            <div class="kpi-subtext">Monthly Observations</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_network(22, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label">Plant Pathogens</div>
            <div class="kpi-value">{n_taxa}</div>
            <div class="kpi-subtext">Across Indian Crops</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_trending_up(22, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label">Forecast Horizon</div>
            <div class="kpi-value">6 Months</div>
            <div class="kpi-subtext">Out-of-Sample Prediction</div>
        </div>
    </div>
</div>
"""
st.html(kpi_markup)

# ==============================================================================
# 3. PROJECT OBJECTIVE CARD (MATCHING AGRIFORECAST SPECIFICATION)
# ==============================================================================
obj_markup = f"""
<div class="project-objective-card">
    <div class="obj-content-wrap">
        <div class="obj-icon-badge">{icon_info(18, "#2D6A4F")}</div>
        <div>
            <div class="obj-title">Project Objective</div>
            <div class="obj-desc">
                Forecast the occurrence of key agricultural diseases using time-series and machine learning models to enable early surveillance and better crop protection.
            </div>
        </div>
    </div>
    <img src="{obj_hills_b64}" class="obj-hills-img" alt="Agricultural fields contours" />
</div>
"""
st.html(obj_markup)

# ==============================================================================
# 4. PRIMARY DATASET VISUAL (RICE BLAST VS BROWN SPOT 2015–2025)
# ==============================================================================
st.markdown(
    f"""
    <div style="margin-top: 12px; margin-bottom: 12px;">
        <h3 style="font-family: 'Fraunces', serif; font-size: 1.35rem; color: #111827; margin: 0 0 4px 0;">
            Longitudinal Disease Surveillance Trajectory ({stats['monthly_start_date']} – {stats['monthly_end_date']})
        </h3>
        <p style="color: #4B5563; font-size: 0.88rem; margin: 0;">
            11-year monthly incidence dynamics contrasting episodic monsoonal surges (Rice Blast) against endemic background reports (Brown Spot of Rice).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

fig_preview = go.Figure()
if df_rice_blast is not None and not df_rice_blast.empty:
    fig_preview.add_trace(
        go.Scatter(
            x=df_rice_blast["date"],
            y=df_rice_blast["documented_occurrence_count"],
            name="Rice Blast (Pyricularia oryzae)",
            line=dict(color="#047857", width=2.4),
            mode="lines+markers",
            marker=dict(size=4),
        )
    )
if df_brown_spot is not None and not df_brown_spot.empty:
    fig_preview.add_trace(
        go.Scatter(
            x=df_brown_spot["date"],
            y=df_brown_spot["documented_occurrence_count"],
            name="Brown Spot of Rice (Bipolaris oryzae)",
            line=dict(color="#C4A66A", width=2.0),
            mode="lines+markers",
            marker=dict(size=4),
        )
    )

format_plotly_figure(
    fig_preview,
    title=f"Monthly Verified Disease Occurrences in India ({stats['monthly_start_date']} – {stats['monthly_end_date']})",
    height=320,
)
st.plotly_chart(fig_preview, use_container_width=True)

# ==============================================================================
# 5. AGRO-ECOLOGICAL ENVIRONMENTS (IN EXPANDABLE DRAWER FOR CLEAN MINIMALISM)
# ==============================================================================
with st.expander("🌾 Explore Indian Agro-Ecological Environments & Micro-Climates", expanded=False):
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 12px 0;">
            <div class="verdant-card" style="padding: 0; overflow: hidden;">
                <div style="height: 140px; overflow: hidden; position: relative;">
                    <img src="{img_monsoon_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Monsoon Precipitation" />
                    <div style="position: absolute; bottom: 6px; left: 8px; background: rgba(16, 44, 38, 0.85); color: #FFF; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">
                        KHARIF MONSOON
                    </div>
                </div>
                <div style="padding: 12px;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: #102C26; margin-bottom: 4px;">Leaf Wetness & Rain Splashing</div>
                    <div style="font-size: 0.78rem; color: #5A665D; line-height: 1.4;">Extended leaf wetness (>10h) during monsoon rains accelerates fungal spore germination.</div>
                </div>
            </div>
            <div class="verdant-card" style="padding: 0; overflow: hidden;">
                <div style="height: 140px; overflow: hidden; position: relative;">
                    <img src="{img_sunset_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Rabi Winter Season" />
                    <div style="position: absolute; bottom: 6px; left: 8px; background: rgba(16, 44, 38, 0.85); color: #FFF; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">
                        RABI WINTER
                    </div>
                </div>
                <div style="padding: 12px;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: #102C26; margin-bottom: 4px;">Cool Dew Cycles</div>
                    <div style="font-size: 0.78rem; color: #5A665D; line-height: 1.4;">Cool temperatures and morning dew drive seasonal stripe rust susceptibility in cereals.</div>
                </div>
            </div>
            <div class="verdant-card" style="padding: 0; overflow: hidden;">
                <div style="height: 140px; overflow: hidden; position: relative;">
                    <img src="{img_wheat_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Canopy Tillering" />
                    <div style="position: absolute; bottom: 6px; left: 8px; background: rgba(16, 44, 38, 0.85); color: #FFF; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">
                        CANOPY MICRO-CLIMATE
                    </div>
                </div>
                <div style="padding: 12px;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: #102C26; margin-bottom: 4px;">Humidity Traps</div>
                    <div style="font-size: 0.78rem; color: #5A665D; line-height: 1.4;">Dense foliage traps relative humidity (>80%), sustaining localized lesion propagation.</div>
                </div>
            </div>
            <div class="verdant-card" style="padding: 0; overflow: hidden;">
                <div style="height: 140px; overflow: hidden; position: relative;">
                    <img src="{img_golden_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Golden Harvest" />
                    <div style="position: absolute; bottom: 6px; left: 8px; background: rgba(16, 44, 38, 0.85); color: #FFF; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 4px;">
                        PROTECTIVE SURVEILLANCE
                    </div>
                </div>
                <div style="padding: 12px;">
                    <div style="font-weight: 700; font-size: 0.92rem; color: #102C26; margin-bottom: 4px;">Advance Warning</div>
                    <div style="font-size: 0.78rem; color: #5A665D; line-height: 1.4;">Early disease forecasts support targeted agronomist advisories.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 6. MINIMAL, ACADEMIC FOOTER
# ==============================================================================
st.markdown(
    f"""
    <div class="verdant-footer" style="margin-top: 40px; padding: 18px 0; border-top: 1px solid rgba(82,122,80,0.16); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; font-size: 0.8rem; color: #5A665D;">
        <div style="display: flex; align-items: center; gap: 8px;">
            {icon_logo(18, "#527A50")}
            <span style="font-family: 'Fraunces', serif; font-weight: 600; color: #102C26;">Agricultural Disease Forecasting Using Time Series</span>
            <span>· Academic &amp; Research Portfolio Platform</span>
        </div>
        <div>
            Open Data: <a href="https://www.gbif.org" style="color: #527A50; text-decoration: underline;">GBIF</a> (CC-BY 4.0) •
            <a href="https://power.larc.nasa.gov" style="color: #527A50; text-decoration: underline;">NASA POWER</a> •
            <a href="https://gd.eppo.int" style="color: #527A50; text-decoration: underline;">EPPO</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
