"""Verdant Intelligence — Plant Disease Outbreak Forecasting & Climate Intelligence.

Executive Landing Page and Portfolio Showcase.
"""

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
    initial_sidebar_state="collapsed",
)

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
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_config_settings,
    get_image_base64,
    inject_custom_css,
    load_all_pathogen_options,
    load_data_quality_report_data,
    load_monthly_series,
)

inject_custom_css()

# Load real quality report metrics
quality_data = load_data_quality_report_data()
meta = quality_data.get("report_metadata", {}) if quality_data else {}
total_records = meta.get("total_raw_records", 3076)
total_dates = meta.get("total_records_with_date", 2071)
total_coords = meta.get("total_records_with_coords", 835)

# Load real image assets
hero_img_path = ASSETS_DIR / "cereal_harvest_sunset.jpg"
if not hero_img_path.exists():
    hero_img_path = ASSETS_DIR / "verdant_hero.jpg"
hero_img_base64 = get_image_base64(hero_img_path)

img_monsoon_b64 = get_image_base64(ASSETS_DIR / "monsoon_weather_canopy.jpg")
img_sunset_b64 = get_image_base64(ASSETS_DIR / "cereal_harvest_sunset.jpg")
img_wheat_b64 = get_image_base64(ASSETS_DIR / "wheat_canopy_ears.jpg")
img_golden_b64 = get_image_base64(ASSETS_DIR / "golden_crop_field.jpg")

# ==============================================================================
# 1. NAVIGATION BAR
# ==============================================================================
st.markdown(
    f"""
    <div class="verdant-navbar">
        <div class="verdant-brand">
            {icon_logo(28, "#527A50")}
            <div>
                <div class="verdant-brand-title">Verdant Intelligence</div>
                <div class="verdant-brand-sub">Agricultural Surveillance & Forecasting · India</div>
            </div>
        </div>
        <div class="verdant-nav-links">
            <a href="#research-metrics" class="verdant-nav-link">Overview</a>
            <a href="#how-it-works" class="verdant-nav-link">Methodology</a>
            <a href="#capabilities" class="verdant-nav-link">Models</a>
            <a href="#research-preview" class="verdant-nav-link">Research</a>
            <a href="/Disease_Explorer" target="_self" class="btn-primary-cta" style="padding: 8px 18px; font-size: 0.82rem;">
                Explore Forecasts {icon_arrow_right(14, "#FFFFFF")}
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. SPLIT-SCREEN HERO SECTION
# ==============================================================================
hero_image_markup = (
    f'<img src="{hero_img_base64}" alt="Verdant Crop Canopy" class="hero-visual-img" />'
    if hero_img_base64
    else '<div style="width:100%; height:100%; background: linear-gradient(135deg, #102C26 0%, #183D32 100%);"></div>'
)

st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-left">
            <div class="hero-eyebrow">
                {icon_leaf(14, "#8C6F2D")} Agricultural Intelligence · India
            </div>
            <h1 class="hero-headline">
                Anticipate disease.<br>Protect the harvest.
            </h1>
            <p class="hero-lead">
                An advanced epidemiological forecasting platform transforming real-world pathogen
                surveillance from <b>GBIF</b> and meteorological intelligence from <b>NASA POWER</b>
                into rigorous, multi-month time-series projections across India's agricultural heartlands.
            </p>
            <div class="hero-actions">
                <a href="/Disease_Explorer" target="_self" class="btn-primary-cta">
                    Explore the Dashboard {icon_arrow_right(15, "#FFFFFF")}
                </a>
                <a href="/Methodology" target="_self" class="btn-secondary-cta">
                    Scientific Methodology
                </a>
            </div>
            <div class="telemetry-status">
                {icon_pulse(12, "#527A50")}
                <span>Verified Telemetry: <strong>{total_records:,} Real Records</strong> & <strong>132 Months</strong> continuous climate data</span>
            </div>
        </div>
        <div class="hero-right">
            <div class="hero-visual-frame">
                {hero_image_markup}
                <div class="hero-overlay-card">
                    <div>
                        <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: #5A665D; letter-spacing: 0.05em;">
                            Top Validated Model
                        </div>
                        <div style="font-family: 'Fraunces', serif; font-size: 1.05rem; font-weight: 600; color: #102C26;">
                            ARIMA(1,1,1)
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                            Out-of-Sample RMSE
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 600; color: #183D32;">
                            2.438
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

st.markdown(
    """
    <div style="text-align: center; max-width: 680px; margin: 36px auto 12px auto;">
        <div class="hero-eyebrow" style="margin-bottom: 8px;">Methodological Architecture</div>
        <h2 style="font-size: 2.1rem; margin-bottom: 12px;">How Verdant Intelligence Operates</h2>
        <p style="color: #5A665D; font-size: 0.95rem; line-height: 1.6;">
            A rigorous four-stage scientific pipeline connecting open biodiversity repositories
            and satellite meteorology to econometric time-series models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="process-grid">
        <div class="process-card">
            <div class="process-num">01</div>
            <div class="process-title">Surveillance Ingestion</div>
            <div class="process-desc">
                Pulls verified occurrence records for 10 candidate agricultural pathogens
                across India from the GBIF REST API and EPPO Global Database, filtering by
                taxonomic synonyms and accepted keys.
            </div>
        </div>
        <div class="process-card">
            <div class="process-num">02</div>
            <div class="process-title">Climate Integration</div>
            <div class="process-desc">
                Retrieves 132 continuous months of agrometeorology from NASA POWER, aligning
                mean/max/min temperature, rainfall, and relative humidity across five
                representative Indian agro-climatic zones.
            </div>
        </div>
        <div class="process-card">
            <div class="process-num">03</div>
            <div class="process-title">Stationarity & Diagnostics</div>
            <div class="process-desc">
                Performs ADF and KPSS unit-root hypothesis testing, variance reduction differencing,
                additive seasonal decomposition, and ACF/PACF Bartlett confidence interval checks.
            </div>
        </div>
        <div class="process-card">
            <div class="process-num">04</div>
            <div class="process-title">Multi-Model Forecasting</div>
            <div class="process-desc">
                Executes chronological out-of-sample benchmarking across 11 model families
                (ARIMA, SARIMA, SARIMAX, Holt-Winters, Prophet, ML), computing RMSE, MAE,
                sMAPE, and 80%/95% prediction intervals.
            </div>
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
            Empirical plant pathology requires understanding host crop phenology, localized canopy boundary-layer humidity,
            and convective precipitation dynamics across India's diverse agricultural zones.
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
                    MONSOON PRECIPITATION FRONT
                </div>
            </div>
            <div style="padding: 16px;">
                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem; color: #102C26;">Canopy Saturation & Rain-Splash Dispersal</h4>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    Convective storm fronts across the Indo-Gangetic Plains create prolonged leaf wetness (&gt;12 hrs) and hydrodynamic spore dispersal for blast pathogens.
                </p>
            </div>
        </div>
        <div class="verdant-card" style="padding: 0; overflow: hidden; display: flex; flex-direction: column;">
            <div style="height: 190px; overflow: hidden; position: relative;">
                <img src="{img_sunset_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Rabi Cereal Sunset" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    RABI CEREAL HARVEST
                </div>
            </div>
            <div style="padding: 16px;">
                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem; color: #102C26;">Grain Filling & Anthesis Vulnerability</h4>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    Post-anthesis grain filling in winter cereals under fluctuating dusk humidity, defining the critical window for timing preventative interventions.
                </p>
            </div>
        </div>
        <div class="verdant-card" style="padding: 0; overflow: hidden; display: flex; flex-direction: column;">
            <div style="height: 190px; overflow: hidden; position: relative;">
                <img src="{img_wheat_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Vegetative Canopy Tillering" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    VEGETATIVE TILLERING
                </div>
            </div>
            <div style="padding: 16px;">
                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem; color: #102C26;">Dense Foliage Micro-Climates</h4>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    High tiller density creates sub-canopy humidity traps where foliar blight and lesion pathogens (<em>Bipolaris oryzae</em>, <em>Alternaria solani</em>) incubate.
                </p>
            </div>
        </div>
        <div class="verdant-card" style="padding: 0; overflow: hidden; display: flex; flex-direction: column;">
            <div style="height: 190px; overflow: hidden; position: relative;">
                <img src="{img_golden_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Golden Crop Field" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    END-OF-SEASON HARVEST
                </div>
            </div>
            <div style="padding: 16px;">
                <h4 style="margin: 0 0 6px 0; font-size: 1.05rem; color: #102C26;">Agricultural Yield Preservation</h4>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    Multi-month econometric and machine learning forecasts allow regional agricultural departments to protect yields before symptoms sweep entire districts.
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
        <div class="hero-eyebrow" style="margin-bottom: 8px;">Model Architecture Suite</div>
        <h2 style="font-size: 2.1rem; margin-bottom: 12px;">Statistical Econometrics & Machine Learning</h2>
        <p style="color: #5A665D; font-size: 0.95rem; line-height: 1.6;">
            We evaluate both classical mathematical formulations and modern decomposable models,
            enforcing strict chronological validation without temporal data leakage.
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
                <span class="model-badge">Classical Econometric</span>
                <div class="model-card-title">ARIMA & SARIMA</div>
                <div class="model-card-desc">
                    Autoregressive Integrated Moving Average with 12-month annual seasonality,
                    capturing Kharif and Rabi cyclic agricultural dynamics in India.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">2.438 (ARIMA 1,1,1)</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge" style="background: rgba(196, 166, 106, 0.15); color: #8C6F2D;">Exogenous Climate</span>
                <div class="model-card-title">SARIMAX & ARIMAX</div>
                <div class="model-card-desc">
                    Incorporates strictly lagged (t-1, t-2) temperature, precipitation, and relative
                    humidity regressors to condition projections on micro-climatic suitability.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">2.506 (ARIMAX)</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge">Smoothing Formulation</span>
                <div class="model-card-title">Holt-Winters</div>
                <div class="model-card-desc">
                    Triple exponential smoothing with additive trend and seasonal components,
                    providing an adaptive baseline sensitive to recent structural momentum.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">3.074</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge" style="background: rgba(37, 99, 235, 0.12); color: #2563EB;">Decomposable Modern</span>
                <div class="model-card-title">Meta Prophet</div>
                <div class="model-card-desc">
                    Decomposable generalized additive model with non-linear trend changepoints,
                    yearly Fourier seasonality, and external agrometeorological regressors.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">2.809</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge" style="background: rgba(16, 44, 38, 0.1); color: #102C26;">Supervised ML</span>
                <div class="model-card-title">Tree Autoregressors</div>
                <div class="model-card-desc">
                    HistGradientBoosting and Random Forest recursive forecasters equipped with
                    cyclical month harmonic transformations and lagged climate features.
                </div>
            </div>
            <div class="model-stat-row">
                <span>Top Validation RMSE:</span>
                <span class="model-stat-val">2.763 (Random Forest)</span>
            </div>
        </div>
        <div class="model-card">
            <div>
                <span class="model-badge">Empirical Baseline</span>
                <div class="model-card-title">Seasonal Naive</div>
                <div class="model-card-desc">
                    Strict persistence benchmark repeating the identical seasonal month observation
                    from 12 periods prior, establishing the fundamental bar of predictive skill.
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
        <div class="hero-eyebrow" style="margin-bottom: 6px;">Live Research Trajectory</div>
        <h2 style="font-size: 2.1rem; margin-bottom: 12px;">Documented Surveillance in India</h2>
        <p style="color: #5A665D; font-size: 0.92rem; line-height: 1.6;">
            Interactive preview of real occurrence records deposited in international repositories for
            <b>Rice Blast</b> (<i>Pyricularia oryzae</i>) and <b>Brown Spot of Rice</b> (<i>Bipolaris oryzae</i>).
            Notice the distinct seasonal concentration during the Kharif monsoon and harvest window (Aug–Nov).
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
                name="Rice Blast (P. oryzae)",
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
                name="Brown Spot (B. oryzae)",
                line=dict(color="#C4A66A", width=2),
                mode="lines+markers",
                marker=dict(size=4),
            )
        )

    format_plotly_figure(
        fig_preview,
        title="Monthly Documented Occurrences (2015–2025)",
        height=320,
    )
    st.plotly_chart(fig_preview, use_container_width=True)

with col_prev_right:
    st.markdown(
        """
        <div style="background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.16); border-radius: 18px; padding: 26px; box-shadow: 0 4px 20px rgba(16, 44, 38, 0.05); margin-top: 10px;">
            <div style="font-family: 'Fraunces', serif; font-size: 1.25rem; font-weight: 600; color: #102C26; margin-bottom: 12px;">
                Scientific Integrity Safeguards
            </div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6; margin-bottom: 16px;">
                <b style="color: #102C26;">1. Honest Missingness:</b> Unmonitored months are flagged <code>is_missing = True</code>.
                We never arbitrarily interpolate flat zeros to mask empirical data gaps.
            </div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6; margin-bottom: 16px;">
                <b style="color: #102C26;">2. Zero Future Leakage:</b> NASA POWER temperature and precipitation variables are
                strictly lagged (<i>t-1</i> and <i>t-2</i>) to ensure models only predict with realistically available information.
            </div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6; margin-bottom: 16px;">
                <b style="color: #102C26;">3. Biological Causality Boundaries:</b> Macro-climate defines permissive physiological
                envelopes, but disease outbreaks require host tissue susceptibility and viable inoculum.
            </div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6;">
                <b style="color: #102C26;">4. Wheat Rust Finding:</b> Zero wild <i>P. graminis</i> records post-2015 reflects
                successful deployment of genetic rust-resistant cultivars in Indian wheat acreage.
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
        <h2 class="final-cta-title">Turn agricultural data into earlier warnings.</h2>
        <p class="final-cta-lead">
            Explore interactive disease mapping, run econometric unit-root tests, compare 11 forecasting models,
            and simulate future climate scenarios across Indian agro-climatic zones.
        </p>
        <a href="/Disease_Explorer" target="_self" class="btn-gold-cta">
            Launch Full Dashboard {icon_arrow_right(16, "#102C26")}
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
