"""Verdant Intelligence — Plant Disease Outbreak Forecasting & Climate Intelligence.

Executive Landing Page and Portfolio Showcase.
"""

import base64
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure main app page
st.set_page_config(
    page_title="Verdant Intelligence — Agricultural Disease Forecasting",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.icons import (
    icon_alert_triangle,
    icon_arrow_right,
    icon_bar_chart,
    icon_bell,
    icon_book,
    icon_chevron_down,
    icon_database,
    icon_download,
    icon_forecast,
    icon_home,
    icon_india_map,
    icon_layers,
    icon_leaf,
    icon_logo,
    icon_model_compare,
    icon_pathology,
    icon_pin,
    icon_pulse,
    icon_search,
    icon_settings,
    icon_shield,
    icon_sprout,
    icon_sun,
    icon_thermometer,
    icon_timeseries,
    icon_trending_up,
    icon_weather,
)
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_config_settings,
    get_image_base64,
    inject_custom_css,
    is_advanced_mode,
    is_simple_mode,
    load_all_pathogen_options,
    load_data_quality_report_data,
    load_monthly_series,
    render_app_sidebar,
    render_top_header,
    render_what_does_this_mean,
)

inject_custom_css()

# Render unified left sidebar and top bar
render_app_sidebar(active_page="overview")
render_top_header()

adv_mode = is_advanced_mode()

# Load real quality report metrics
quality_data = load_data_quality_report_data()
meta = quality_data.get("report_metadata", {}) if quality_data else {}
total_records = meta.get("total_raw_records", 3076)
total_dates = meta.get("total_records_with_date", 2071)
total_coords = meta.get("total_records_with_coords", 835)

# Load real image assets
panoramic_hero_path = ASSETS_DIR / "verdant_panoramic_hero.jpg"
if not panoramic_hero_path.exists():
    panoramic_hero_path = ASSETS_DIR / "cereal_harvest_sunset.jpg"
panoramic_hero_b64 = get_image_base64(panoramic_hero_path)

img_monsoon_b64 = get_image_base64(ASSETS_DIR / "monsoon_weather_canopy.jpg")
img_sunset_b64 = get_image_base64(ASSETS_DIR / "cereal_harvest_sunset.jpg")
img_wheat_b64 = get_image_base64(ASSETS_DIR / "wheat_canopy_ears.jpg")
img_golden_b64 = get_image_base64(ASSETS_DIR / "golden_crop_field.jpg")

# ==============================================================================
# 1. PANORAMIC HERO WITH FLOATING GLASS CARD
# ==============================================================================
sparkline_raw_svg = """<svg width="270" height="60" viewBox="0 0 270 60" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="heroSparkGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#E11D48" stop-opacity="0.32"/>
      <stop offset="100%" stop-color="#E11D48" stop-opacity="0.0"/>
    </linearGradient>
  </defs>
  <path d="M10 46 Q 50 42, 90 36 T 170 26 T 255 10 L 255 58 L 10 58 Z" fill="url(#heroSparkGrad)"/>
  <path d="M10 46 Q 50 42, 90 36 T 170 26 T 255 10" stroke="#E11D48" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="10" cy="46" r="3.5" fill="#E11D48" stroke="#FFFFFF" stroke-width="1.5"/>
  <circle cx="90" cy="36" r="3.5" fill="#E11D48" stroke="#FFFFFF" stroke-width="1.5"/>
  <circle cx="170" cy="26" r="3.5" fill="#E11D48" stroke="#FFFFFF" stroke-width="1.5"/>
  <circle cx="255" cy="10" r="4" fill="#E11D48" stroke="#FFFFFF" stroke-width="2"/>
</svg>"""
sparkline_b64 = base64.b64encode(sparkline_raw_svg.strip().encode("utf-8")).decode("utf-8")
sparkline_img_tag = f'<img src="data:image/svg+xml;base64,{sparkline_b64}" width="270" height="60" style="display:block;margin:0 auto;" alt="Disease Trajectory" />'
sparkline_svg = (
    f'{sparkline_img_tag}'
    '<div style="display: flex; justify-content: space-between; padding: 0 10px; font-size: 0.72rem; color: #7D8A82; font-weight: 600; margin-top: 2px; margin-bottom: 12px;">'
    '<span>Nov</span><span>Dec</span><span>Jan</span>'
    '</div>'
)

hero_stat_label = "TOP VALIDATED MODEL" if adv_mode else "TOP METHOD"
hero_stat_val = "ARIMA(1,1,1)" if adv_mode else "ARIMA"
hero_metric_label = "OUT-OF-SAMPLE RMSE" if adv_mode else "AVERAGE ERROR"
hero_metric_val = "2.438" if adv_mode else "2.4 reports"

hero_markup = (
    f'<div class="dashboard-hero-banner" style="background-image: url(\'{panoramic_hero_b64}\');">'
    f'<div class="dashboard-hero-overlay">'
    f'<div class="hero-left-content">'
    f'<div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(16, 44, 38, 0.7); border: 1px solid rgba(196, 166, 106, 0.5); border-radius: 30px; padding: 5px 14px; margin-bottom: 16px; backdrop-filter: blur(8px);">'
    f'<span style="font-size: 0.68rem; font-weight: 700; color: #F5E8C7; letter-spacing: 0.08em; text-transform: uppercase;">'
    f'{icon_leaf(12, "#C4A66A")} AGRICULTURAL FORECASTING · INDIA'
    f'</span></div>'
    f'<div class="hero-main-title">'
    f'Anticipate disease.<br><span>Protect the harvest.</span></div>'
    f'<p style="color: rgba(246, 244, 236, 0.95); font-size: 0.95rem; line-height: 1.6; margin: 0 0 24px 0; max-width: 490px; text-shadow: 0 1px 4px rgba(0,0,0,0.4);">'
    f'A clear, data-driven plant disease forecasting platform transforming real-world disease records from <b>GBIF</b> and weather data from <b>NASA POWER</b> into simple, reliable future predictions across India\'s crop regions.'
    f'</p>'
    f'<div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">'
    f'<a href="/Disease_Explorer" target="_self" class="btn-primary-cta" style="padding: 10px 22px; font-size: 0.88rem; border-radius: 10px; background: #102C26; border: 1px solid rgba(255,255,255,0.2);">'
    f'View Disease Records {icon_arrow_right(14, "#FFFFFF")}'
    f'</a>'
    f'<a href="/About_the_Data" target="_self" class="btn-secondary-cta" style="padding: 9px 18px; font-size: 0.88rem; border-radius: 10px; color: #FFFFFF !important; border: 1px solid rgba(255,255,255,0.4); background: rgba(255,255,255,0.15); backdrop-filter: blur(8px);">'
    f'▷ About the Data'
    f'</a>'
    f'</div></div>'
    f'<div class="hero-glass-card">'
    f'<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">'
    f'<div style="width: 24px; height: 24px; border-radius: 50%; background: #E8F5E9; display: flex; align-items: center; justify-content: center;">'
    f'{icon_trending_up(14, "#166534")}'
    f'</div>'
    f'<div style="font-size: 0.82rem; font-weight: 700; color: #102C26;">Predicted Disease Activity</div>'
    f'</div>'
    f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">'
    f'<div style="font-size: 0.95rem; font-weight: 700; color: #102C26;">Rice — Rice Blast</div>'
    f'<span style="background: #FEE2E2; color: #DC2626; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;">Active Season</span>'
    f'</div>'
    f'<div style="font-size: 0.72rem; color: #7D8A82; font-weight: 500; margin-bottom: 12px;">Next 3 Months</div>'
    f'{sparkline_svg}'
    f'<div style="border-top: 1px solid #EAE6DC; padding-top: 12px; display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 8px;">'
    f'<div>'
    f'<div style="font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #7D8A82;">{hero_stat_label}</div>'
    f'<div style="font-size: 0.88rem; font-weight: 700; color: #102C26; margin-top: 2px;">{hero_stat_val}</div>'
    f'</div>'
    f'<div style="text-align: right;">'
    f'<div style="font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #7D8A82;">{hero_metric_label}</div>'
    f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 1.05rem; font-weight: 700; color: #166534; margin-top: 1px;">{hero_metric_val}</div>'
    f'</div></div></div></div></div>'
)
st.html(hero_markup)

# ==============================================================================
# 2. FOUR FEATURE / METRIC CARDS
# ==============================================================================
features_markup = (
    f'<div class="feature-cards-grid">'
    f'<a href="/Disease_Explorer" target="_self" class="feature-card">'
    f'<div class="feature-card-top">'
    f'<div class="feature-icon-circle" style="background: #E8F5E9;">{icon_sprout(20, "#166534")}</div>'
    f'<div class="feature-arrow-btn">{icon_arrow_right(13, "currentColor")}</div>'
    f'</div>'
    f'<div><div style="display: flex; align-items: baseline; gap: 6px;"><span class="feature-card-stat">10+</span><span class="feature-card-title">Crops Monitored</span></div>'
    f'<div class="feature-card-sub">Rice, Wheat, Potato, Chickpea and more</div></div>'
    f'</a>'
    f'<a href="/Disease_Explorer" target="_self" class="feature-card">'
    f'<div class="feature-card-top">'
    f'<div class="feature-icon-circle" style="background: #FEF2F2;">{icon_pathology(20, "#DC2626")}</div>'
    f'<div class="feature-arrow-btn">{icon_arrow_right(13, "currentColor")}</div>'
    f'</div>'
    f'<div><div style="display: flex; align-items: baseline; gap: 6px;"><span class="feature-card-stat">10</span><span class="feature-card-title">Plant Diseases</span></div>'
    f'<div class="feature-card-sub">Rice Blast, Stem Rust, Blight & more</div></div>'
    f'</a>'
    f'<a href="/Weather_Intelligence" target="_self" class="feature-card">'
    f'<div class="feature-card-top">'
    f'<div class="feature-icon-circle" style="background: #EFF6FF;">{icon_weather(20, "#2563EB")}</div>'
    f'<div class="feature-arrow-btn">{icon_arrow_right(13, "currentColor")}</div>'
    f'</div>'
    f'<div><div class="feature-card-title" style="font-size: 0.95rem; font-weight: 700;">Weather Insights</div>'
    f'<div class="feature-card-sub">Rainfall, humidity & temperature trends</div></div>'
    f'</a>'
    f'<a href="/Model_Comparison" target="_self" class="feature-card">'
    f'<div class="feature-card-top">'
    f'<div class="feature-icon-circle" style="background: #FAF5FF;">{icon_model_compare(20, "#9333EA")}</div>'
    f'<div class="feature-arrow-btn">{icon_arrow_right(13, "currentColor")}</div>'
    f'</div>'
    f'<div><div class="feature-card-title" style="font-size: 0.95rem; font-weight: 700;">Compare Models</div>'
    f'<div class="feature-card-sub">11 prediction methods tested side-by-side</div></div>'
    f'</a>'
    f'</div>'
)
st.html(features_markup)

# ==============================================================================
# 3. KEY INSIGHTS SECTION
# ==============================================================================
insights_markup = (
    f'<div class="insights-header-row">'
    f'<div class="insights-title-wrap">'
    f'<div class="insights-accent-bar"></div>'
    f'<div><div class="insights-heading">Key Insights</div>'
    f'<div class="insights-subtitle">Plain-English summaries of disease trends and weather patterns across India</div></div>'
    f'</div>'
    f'</div>'
    f'<div class="insight-cards-grid">'
    f'<a href="/Forecasts" target="_self" class="insight-card">'
    f'<div class="insight-icon-circle" style="background: #FEF2F2;">{icon_alert_triangle(20, "#DC2626")}</div>'
    f'<div class="insight-card-content">'
    f'<div class="insight-card-header">'
    f'<span class="insight-card-title">Seasonal Notice</span>'
    f'<span class="insight-badge" style="background: #FEE2E2; color: #DC2626;">Higher Activity</span>'
    f'</div>'
    f'<div class="insight-card-desc">Rice Blast reports typically peak during wet Kharif harvest months in eastern belts.</div>'
    f'</div>'
    f'<div class="insight-arrow-btn">{icon_arrow_right(12, "currentColor")}</div>'
    f'</a>'
    f'<a href="/Forecasts" target="_self" class="insight-card">'
    f'<div class="insight-icon-circle" style="background: #E8F5E9;">{icon_trending_up(20, "#166534")}</div>'
    f'<div class="insight-card-content">'
    f'<div class="insight-card-header">'
    f'<span class="insight-card-title">Seasonal Patterns</span>'
    f'<span class="insight-badge" style="background: #DCFCE7; color: #166534;">Cyclic</span>'
    f'</div>'
    f'<div class="insight-card-desc">Foliar diseases follow yearly patterns that repeat each monsoon and winter season.</div>'
    f'</div>'
    f'<div class="insight-arrow-btn">{icon_arrow_right(12, "currentColor")}</div>'
    f'</a>'
    f'<a href="/Weather_Intelligence" target="_self" class="insight-card">'
    f'<div class="insight-icon-circle" style="background: #EFF6FF;">{icon_weather(20, "#2563EB")}</div>'
    f'<div class="insight-card-content">'
    f'<div class="insight-card-header">'
    f'<span class="insight-card-title">Weather Factors</span>'
    f'<span class="insight-badge" style="background: #DBEAFE; color: #1E40AF;">Humidity</span>'
    f'</div>'
    f'<div class="insight-card-desc">High humidity (above 80%) creates favorable conditions for fungal disease spread.</div>'
    f'</div>'
    f'<div class="insight-arrow-btn">{icon_arrow_right(12, "currentColor")}</div>'
    f'</a>'
    f'</div>'
)
st.html(insights_markup)





# ==============================================================================
# SECTION A: RESEARCH METRICS
# ==============================================================================
st.markdown('<div id="research-metrics"></div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="metric-grid">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(20, "#527A50")}</div>
            <div class="metric-label-text">Historical Coverage</div>
            <div class="metric-value-text">11.2 Yrs</div>
            <div class="metric-note-text">Jan 2015 – Apr 2026</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
            <div class="metric-label-text">Usable Dated Records</div>
            <div class="metric-value-text">{total_dates:,}</div>
            <div class="metric-note-text">67.3% temporal integrity</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
            <div class="metric-label-text">Analyzable Pathogens</div>
            <div class="metric-value-text">10 Taxa</div>
            <div class="metric-note-text">Rice, wheat, pulses, solanaceous</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_weather(20, "#527A50")}</div>
            <div class="metric-label-text">Climate Telemetry</div>
            <div class="metric-value-text">6 Factors</div>
            <div class="metric-note-text">Temp, rainfall, RH, dew point</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SECTION B: HOW IT WORKS
# ==============================================================================
st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
st.markdown("---")

p1_title = "01 · Collect Disease Records" if not adv_mode else "01 · Surveillance Ingestion"
p1_desc = (
    "Pulls verified disease reports for 10 plant diseases across India from GBIF and botanical research herbaria."
    if not adv_mode
    else "Pulls verified occurrence records for 10 candidate agricultural pathogens across India from GBIF REST API and EPPO."
)

p2_title = "02 · Add Weather Data" if not adv_mode else "02 · Climate Integration"
p2_desc = (
    "Retrieves 11 years of monthly temperature, rainfall, and relative humidity data from NASA POWER satellites."
    if not adv_mode
    else "Retrieves 132 continuous months of agrometeorology from NASA POWER across five Indian agro-climatic zones."
)

p3_title = "03 · Check Data Patterns" if not adv_mode else "03 · Stationarity & Diagnostics"
p3_desc = (
    "Tests whether disease reports follow steady seasonal patterns each year and adjusts for long-term changes."
    if not adv_mode
    else "Performs ADF and KPSS unit-root hypothesis testing, variance differencing, and ACF/PACF Bartlett confidence bands."
)

p4_title = "04 · Predict Future Reports" if not adv_mode else "04 · Multi-Model Forecasting"
p4_desc = (
    "Tests 11 forecasting methods on past history to find the most accurate model and predict future monthly disease activity."
    if not adv_mode
    else "Executes chronological out-of-sample benchmarking across 11 model families, computing RMSE, MAE, and prediction intervals."
)

st.markdown(
    f"""
    <div style="text-align: center; max-width: 680px; margin: 36px auto 12px auto;">
        <div class="hero-eyebrow" style="margin-bottom: 8px;">Step-by-Step Architecture</div>
        <h2 style="font-size: 2.1rem; margin-bottom: 12px;">How Predictions Are Made</h2>
        <p style="color: #5A665D; font-size: 0.95rem; line-height: 1.6;">
            A transparent four-stage pipeline connecting verified field records and satellite weather data to accurate forecasting models.
        </p>
    </div>
    <div class="process-grid">
        <div class="process-card">
            <div class="process-num">01</div>
            <div class="process-title">{p1_title}</div>
            <div class="process-desc">{p1_desc}</div>
        </div>
        <div class="process-card">
            <div class="process-num">02</div>
            <div class="process-title">{p2_title}</div>
            <div class="process-desc">{p2_desc}</div>
        </div>
        <div class="process-card">
            <div class="process-num">03</div>
            <div class="process-title">{p3_title}</div>
            <div class="process-desc">{p3_desc}</div>
        </div>
        <div class="process-card">
            <div class="process-num">04</div>
            <div class="process-title">{p4_title}</div>
            <div class="process-desc">{p4_desc}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# AGRO-ECOLOGICAL TERRAINS & CANOPY REGIMES (USER IMAGES GALLERY)
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; max-width: 720px; margin: 36px auto 20px auto;">
        <div class="hero-eyebrow" style="margin-bottom: 8px;">Field Surveillance Environments</div>
        <h2 style="font-size: 2.1rem; margin-bottom: 12px;">Agro-Ecological Terrains & Micro-Climate Regimes</h2>
        <p style="color: #5A665D; font-size: 0.95rem; line-height: 1.6;">
            Understanding crop growth, localized humidity in leaf canopies, and seasonal monsoon rains across India's agricultural heartlands.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-bottom: 40px;">
        <div class="verdant-card" style="padding: 0; overflow: hidden; display: flex; flex-direction: column;">
            <div style="height: 190px; overflow: hidden; position: relative;">
                <img src="{img_monsoon_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Monsoon Precipitation Front" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    MONSOON RAIN PATTERNS
                </div>
            </div>
            <div style="padding: 16px;">
                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem; color: #102C26;">Leaf Wetness & Rain Splashing</h4>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    Monsoon rain across the plains keeps crop leaves wet for over 10 hours, creating favorable conditions for blast and blight fungi to multiply.
                </p>
            </div>
        </div>
        <div class="verdant-card" style="padding: 0; overflow: hidden; display: flex; flex-direction: column;">
            <div style="height: 190px; overflow: hidden; position: relative;">
                <img src="{img_sunset_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Rabi Cereal Sunset" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    RABI WINTER CROPS
                </div>
            </div>
            <div style="padding: 16px;">
                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem; color: #102C26;">Grain Growth & Timing Protection</h4>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    Cool winter temperatures and evening dew determine the critical weeks when wheat and pulses are most susceptible to rusts.
                </p>
            </div>
        </div>
        <div class="verdant-card" style="padding: 0; overflow: hidden; display: flex; flex-direction: column;">
            <div style="height: 190px; overflow: hidden; position: relative;">
                <img src="{img_wheat_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Vegetative Canopy Tillering" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    DENSE CROP CANOPY
                </div>
            </div>
            <div style="padding: 16px;">
                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem; color: #102C26;">Foliar Micro-Climates</h4>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    Dense plant canopies trap moisture near the ground, providing an ideal micro-climate for leaf spots and rot pathogens to develop.
                </p>
            </div>
        </div>
        <div class="verdant-card" style="padding: 0; overflow: hidden; display: flex; flex-direction: column;">
            <div style="height: 190px; overflow: hidden; position: relative;">
                <img src="{img_golden_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Golden Crop Field" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    HARVEST PRESERVATION
                </div>
            </div>
            <div style="padding: 16px;">
                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem; color: #102C26;">Protecting Crop Yields</h4>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    Knowing disease trends months in advance helps farmers and agricultural advisors take preventive action before damage spreads.
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SECTION C: FORECASTING CAPABILITIES
# ==============================================================================
st.markdown('<div id="capabilities"></div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; max-width: 700px; margin: 36px auto 16px auto;">
        <div class="hero-eyebrow" style="margin-bottom: 8px;">Prediction Methods Suite</div>
        <h2 style="font-size: 2.1rem; margin-bottom: 12px;">Statistical &amp; Machine Learning Models</h2>
        <p style="color: #5A665D; font-size: 0.95rem; line-height: 1.6;">
            We evaluate both classical time-series methods and modern machine learning forecasters,
            strictly testing each method on past unseen months.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="model-cards-grid">
        <div class="model-card">
            <div>
                <span class="model-badge">Time-Tested Method</span>
                <div class="model-card-title">ARIMA &amp; SARIMA</div>
                <div class="model-card-desc">
                    Captures past patterns and 12-month annual seasonality, reflecting India's Kharif (monsoon) and Rabi (winter) crop cycles.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">2.438 (ARIMA 1,1,1)</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge" style="background: rgba(196, 166, 106, 0.15); color: #8C6F2D;">Weather-Informed</span>
                <div class="model-card-title">SARIMAX &amp; ARIMAX</div>
                <div class="model-card-desc">
                    Includes past temperature, rainfall, and humidity readings to see how upcoming weather may influence disease reports.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">2.506 (ARIMAX)</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge">Adaptive Smoothing</span>
                <div class="model-card-title">Holt-Winters</div>
                <div class="model-card-desc">
                    Exponential smoothing with trend and seasonality, adapting quickly to recent shifts in disease reports.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">3.074</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge" style="background: rgba(37, 99, 235, 0.12); color: #2563EB;">Modern Decomposition</span>
                <div class="model-card-title">Meta Prophet</div>
                <div class="model-card-desc">
                    Separates overall trend, yearly seasonal patterns, and weather factors into intuitive, individual components.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">2.809</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge" style="background: rgba(16, 44, 38, 0.1); color: #102C26;">Machine Learning</span>
                <div class="model-card-title">Gradient Boosting &amp; Trees</div>
                <div class="model-card-desc">
                    Modern decision-tree algorithms learning non-linear relationships between calendar months, weather, and reports.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">2.763 (Random Forest)</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge">Simple Benchmark</span>
                <div class="model-card-title">Seasonal Naive</div>
                <div class="model-card-desc">
                    A simple baseline that assumes disease reports this year will equal the same month last year.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Baseline RMSE:</span>
                <span class="model-stat-val">2.693</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SECTION D: DATA INTEGRITY & RESEARCH PREVIEW
# ==============================================================================
st.markdown('<div id="research-preview"></div>', unsafe_allow_html=True)
st.markdown("---")

col_prev_left, col_prev_right = st.columns([1.1, 0.9])

with col_prev_left:
    st.markdown(
        """
        <div class="hero-eyebrow" style="margin-bottom: 6px;">Recorded History</div>
        <h2 style="font-size: 2.1rem; margin-bottom: 12px;">Recorded Disease Reports in India</h2>
        <p style="color: #5A665D; font-size: 0.92rem; line-height: 1.6;">
            Recorded reports over the past 10 years for <b>Rice Blast</b> and <b>Brown Spot of Rice</b>.
            Notice how reports rise during the Kharif monsoon and harvest season (August–November).
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Plot miniature multi-pathogen trend chart from real processed data
    df_rice_blast = load_monthly_series("pyricularia_oryzae")
    df_brown_spot = load_monthly_series("bipolaris_oryzae")

    fig_preview = go.Figure()
    if df_rice_blast is not None and not df_rice_blast.empty:
        fig_preview.add_trace(
            go.Scatter(
                x=df_rice_blast["date"],
                y=df_rice_blast["documented_occurrence_count"],
                name="Rice Blast",
                line=dict(color="#047857", width=2.5),
                mode="lines+markers",
                marker=dict(size=4),
            )
        )
    if df_brown_spot is not None and not df_brown_spot.empty:
        fig_preview.add_trace(
            go.Scatter(
                x=df_brown_spot["date"],
                y=df_brown_spot["documented_occurrence_count"],
                name="Brown Spot of Rice",
                line=dict(color="#C4A66A", width=2),
                mode="lines+markers",
                marker=dict(size=4),
            )
        )

    format_plotly_figure(
        fig_preview,
        title="Monthly Recorded Disease Reports (2015–2025)",
        height=320,
    )
    st.plotly_chart(fig_preview, use_container_width=True)

    render_what_does_this_mean(
        "This chart shows actual recorded disease reports over the past 10 years. In India, diseases like Rice Blast tend to peak during the Kharif monsoon (August to November) when humidity and rainfall are highest.",
        "Zero-bounded continuous monthly timeline aggregated from georeferenced GBIF accessions with strict temporal deduplication."
    )

with col_prev_right:
    st.markdown(
        """
        <div style="background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.16); border-radius: 18px; padding: 26px; box-shadow: 0 4px 20px rgba(16, 44, 38, 0.05); margin-top: 10px;">
            <div style="font-family: 'Fraunces', serif; font-size: 1.25rem; font-weight: 600; color: #102C26; margin-bottom: 12px;">
                Scientific Integrity Safeguards
            </div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6; margin-bottom: 16px;">
                <b style="color: #102C26;">1. Honest Records:</b> Months with no search or recording activity are marked as missing rather than assumed to be zero.
            </div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6; margin-bottom: 16px;">
                <b style="color: #102C26;">2. No Future Cheating:</b> Weather information uses only past and current values, never future weather.
            </div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6; margin-bottom: 16px;">
                <b style="color: #102C26;">3. Weather Creates Conditions:</b> Favorable weather creates the right environment, but plants must also be vulnerable for a disease to occur.
            </div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6;">
                <b style="color: #102C26;">4. Real-World Findings:</b> The absence of Wheat Stem Rust reports post-2015 reflects successful rust-resistant wheat varieties grown in India.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# SECTION F: FINAL CALL TO ACTION & FOOTER
# ==============================================================================
st.markdown(
    f"""
    <div class="final-cta-section">
        <h2 class="final-cta-title">Turn agricultural data into earlier insights.</h2>
        <p class="final-cta-lead">
            Explore recorded disease reports, seasonal patterns, and future predictions across Indian farming regions.
        </p>
        <a href="/Forecasts" target="_self" class="btn-gold-cta">
            View Future Predictions {icon_arrow_right(16, "#102C26")}
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    f"""
    <div class="verdant-footer">
        <div style="display: flex; align-items: center; gap: 8px;">
            {icon_logo(20, "#527A50")}
            <span style="font-family: 'Fraunces', serif; font-weight: 600; color: #102C26;">Verdant Intelligence</span>
            <span>· Academic & Research Portfolio Platform</span>
        </div>
        <div>
            Public Data: <a href="https://www.gbif.org" style="color: #527A50; text-decoration: underline;">GBIF</a> (CC-BY 4.0) •
            <a href="https://power.larc.nasa.gov" style="color: #527A50; text-decoration: underline;">NASA POWER</a> •
            <a href="https://gd.eppo.int" style="color: #527A50; text-decoration: underline;">EPPO</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
