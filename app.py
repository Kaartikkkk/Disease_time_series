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
    page_title="Agricultural Disease Forecasting Using Time Series",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.icons import (
    icon_arrow_right,
    icon_leaf,
    icon_logo,
    icon_model_compare,
    icon_pathology,
    icon_shield,
    icon_sprout,
    icon_timeseries,
    icon_trending_up,
    icon_weather,
)
from src.project_stats import get_project_telemetry
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_image_base64,
    inject_custom_css,
    load_monthly_series,
    render_app_sidebar,
    render_top_header,
)

inject_custom_css()
render_app_sidebar(active_page="overview")
render_top_header()

# Load authoritative dynamic project telemetry
stats = get_project_telemetry()

# Load real image assets
panoramic_hero_path = ASSETS_DIR / "verdant_panoramic_hero.jpg"
if not panoramic_hero_path.exists():
    panoramic_hero_path = ASSETS_DIR / "cereal_harvest_sunset.jpg"
panoramic_hero_b64 = get_image_base64(panoramic_hero_path)

img_monsoon_b64 = get_image_base64(ASSETS_DIR / "monsoon_weather_canopy.jpg")
img_sunset_b64 = get_image_base64(ASSETS_DIR / "cereal_harvest_sunset.jpg")
img_wheat_b64 = get_image_base64(ASSETS_DIR / "wheat_canopy_ears.jpg")
img_golden_b64 = get_image_base64(ASSETS_DIR / "golden_crop_field.jpg")

# Load real monthly series for primary preview
df_rice_blast = load_monthly_series("pyricularia_oryzae")
df_brown_spot = load_monthly_series("bipolaris_oryzae")


def _build_real_sparkline_svg(df_series: pd.DataFrame, n_points: int = 6) -> str:
    """Build an inline SVG sparkline from actual recent monthly occurrence counts."""
    if df_series is None or df_series.empty:
        return '<div style="font-size:0.78rem;color:#7D8A82;padding:12px 0;text-align:center;">No recent observations available.</div>'
    tail_df = df_series.tail(n_points).copy()
    vals = tail_df["documented_occurrence_count"].astype(float).tolist()
    labels = pd.to_datetime(tail_df["date"]).dt.strftime("%b '%y").tolist()
    max_v = max(max(vals), 1.0)
    w, h, pad_x, pad_y = 260, 52, 10, 6
    xs = [pad_x + i * (w - 2 * pad_x) / max(1, len(vals) - 1) for i in range(len(vals))]
    ys = [h - pad_y - (v / max_v) * (h - 2 * pad_y) for v in vals]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area_pts = f"{xs[0]:.1f},{h - 2} {pts} {xs[-1]:.1f},{h - 2}"
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#E11D48" stroke="#FFFFFF" stroke-width="1.5"/>'
        for x, y in zip(xs, ys)
    )
    svg = (
        f'<svg width="260" height="52" viewBox="0 0 260 52" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<defs><linearGradient id="heroSparkGrad" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="#E11D48" stop-opacity="0.25"/>'
        f'<stop offset="100%" stop-color="#E11D48" stop-opacity="0.0"/></linearGradient></defs>'
        f'<polygon points="{area_pts}" fill="url(#heroSparkGrad)"/>'
        f'<polyline points="{pts}" fill="none" stroke="#E11D48" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'{circles}</svg>'
    )
    import base64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    lbl_html = "".join(f"<span>{lbl}</span>" for lbl in labels)
    return (
        f'<img src="data:image/svg+xml;base64,{b64}" width="260" height="52" style="display:block;margin:0 auto;" alt="Recent Rice Blast Trajectory" />'
        f'<div style="display:flex;justify-content:space-between;padding:0 8px;font-size:0.68rem;color:#7D8A82;font-weight:600;margin-top:2px;margin-bottom:8px;">{lbl_html}</div>'
    )


sparkline_svg = _build_real_sparkline_svg(df_rice_blast, n_points=6)

# ==============================================================================
# 1. HERO SECTION (ATTRACTIVE, ELEGANT, SCIENTIFIC)
# ==============================================================================
hero_markup = f"""
<div class="dashboard-hero-banner" style="background-image: url('{panoramic_hero_b64}');">
    <div class="dashboard-hero-overlay">
        <div class="hero-left-content">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(16, 44, 38, 0.7); border: 1px solid rgba(196, 166, 106, 0.5); border-radius: 30px; padding: 5px 14px; margin-bottom: 16px; backdrop-filter: blur(8px);">
                <span style="font-size: 0.68rem; font-weight: 700; color: #F5E8C7; letter-spacing: 0.08em; text-transform: uppercase;">
                    {icon_leaf(12, "#C4A66A")} AGRICULTURAL FORECASTING · INDIA
                </span>
            </div>
            <div class="hero-main-title">
                Anticipate disease.<br><span>Protect the harvest.</span>
            </div>
            <p style="color: rgba(246, 244, 236, 0.95); font-size: 0.96rem; line-height: 1.6; margin: 0 0 24px 0; max-width: 500px; text-shadow: 0 1px 4px rgba(0,0,0,0.4);">
                A unified time-series intelligence platform benchmarking classical econometrics and modern AI across 10 major Indian plant pathogens and 11 years of NASA satellite agrometeorology.
            </p>
            <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                <a href="/Disease_Explorer" target="_self" class="btn-primary-cta" style="padding: 10px 22px; font-size: 0.88rem; border-radius: 10px; background: #102C26; border: 1px solid rgba(255,255,255,0.2);">
                    Explore Disease Records {icon_arrow_right(14, "#FFFFFF")}
                </a>
                <a href="/Future_Forecasts" target="_self" class="btn-secondary-cta" style="padding: 9px 18px; font-size: 0.88rem; border-radius: 10px; color: #FFFFFF !important; border: 1px solid rgba(255,255,255,0.4); background: rgba(255,255,255,0.18); backdrop-filter: blur(8px);">
                    Future Forecasts {icon_arrow_right(14, "#FFFFFF")}
                </a>
            </div>
        </div>
        <div class="hero-glass-card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="width: 24px; height: 24px; border-radius: 50%; background: #E8F5E9; display: flex; align-items: center; justify-content: center;">
                    {icon_trending_up(14, "#166534")}
                </div>
                <div style="font-size: 0.82rem; font-weight: 700; color: #102C26;">Surveillance Signal</div>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #102C26;">Rice Blast (P. oryzae)</div>
                <span style="background: #E8F5E9; color: #166534; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;">Active Signal</span>
            </div>
            <div style="font-size: 0.72rem; color: #7D8A82; font-weight: 500; margin-bottom: 10px;">End-of-Series Observed Trajectory (Last 6 Months)</div>
            {sparkline_svg}
            <div style="border-top: 1px solid #EAE6DC; padding-top: 10px; display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 8px;">
                <div>
                    <div style="font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #7D8A82;">Validation Champion</div>
                    <div style="font-size: 0.88rem; font-weight: 700; color: #102C26; margin-top: 2px;">Hurdle XGBoost</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #7D8A82;">Validation RMSE</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 700; color: #166534; margin-top: 1px;">5.996</div>
                </div>
            </div>
        </div>
    </div>
</div>
"""
st.html(hero_markup)

# ==============================================================================
# 2. DATASET TELEMETRY (AUTHORITATIVE DYNAMIC STRIP)
# ==============================================================================
years_span = round(stats["total_months"] / 12.0, 1)
telemetry_markup = f"""
<div class="metric-grid" style="margin-top: 16px; margin-bottom: 24px;">
    <div class="research-metric-card">
        <div class="metric-icon-wrap">{icon_timeseries(20, "#527A50")}</div>
        <div class="metric-label-text">Temporal Scope</div>
        <div class="metric-value-text">{years_span} Years</div>
        <div class="metric-note-text">{stats['monthly_start_date']} – {stats['monthly_end_date']} ({stats['total_months']} Months)</div>
    </div>
    <div class="research-metric-card">
        <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
        <div class="metric-label-text">Verified Surveillance</div>
        <div class="metric-value-text">{stats['dated_gbif_records']:,} Records</div>
        <div class="metric-note-text">{stats['curated_records']:,} Quality-Filtered Accessions</div>
    </div>
    <div class="research-metric-card">
        <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
        <div class="metric-label-text">Monitored Pathogens</div>
        <div class="metric-value-text">{stats['total_pathogens']} Taxa</div>
        <div class="metric-note-text">Rice, Wheat, Potato, Chickpea, Pigeonpea</div>
    </div>
    <div class="research-metric-card">
        <div class="metric-icon-wrap">{icon_weather(20, "#527A50")}</div>
        <div class="metric-label-text">Zero-Inflation Baseline</div>
        <div class="metric-value-text">{stats['zero_percentage']}% Zeros</div>
        <div class="metric-note-text">Natural Seasonal Dormancy · Off-Season Non-Occurrence</div>
    </div>
</div>
"""
st.html(telemetry_markup)

# ==============================================================================
# 3. FAST DIRECT NAVIGATION SHORTCUTS
# ==============================================================================
st.markdown("<div style='font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #5A665D; margin-bottom: 10px;'>System Navigation</div>", unsafe_allow_html=True)

nav_cols = st.columns(4)
with nav_cols[0]:
    st.markdown(
        f"""
        <a href="/Disease_Explorer" target="_self" style="text-decoration: none; color: inherit;">
            <div class="verdant-card" style="padding: 16px; height: 100%; transition: transform 0.15s ease, box-shadow 0.15s ease;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="background: #E8F5E9; border-radius: 8px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;">
                        {icon_pathology(18, "#166534")}
                    </div>
                    {icon_arrow_right(14, "#527A50")}
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.05rem; font-weight: 600; color: #102C26; margin-bottom: 4px;">
                    Disease Records
                </div>
                <div style="font-size: 0.8rem; color: #5A665D; line-height: 1.4;">
                    Historical occurrences, GPS collection coordinates, and host crops.
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with nav_cols[1]:
    st.markdown(
        f"""
        <a href="/Time_Series_Analysis" target="_self" style="text-decoration: none; color: inherit;">
            <div class="verdant-card" style="padding: 16px; height: 100%; transition: transform 0.15s ease, box-shadow 0.15s ease;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="background: #EFF6FF; border-radius: 8px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;">
                        {icon_timeseries(18, "#2563EB")}
                    </div>
                    {icon_arrow_right(14, "#527A50")}
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.05rem; font-weight: 600; color: #102C26; margin-bottom: 4px;">
                    Time Series Analysis
                </div>
                <div style="font-size: 0.8rem; color: #5A665D; line-height: 1.4;">
                    Seasonality decomposition, ADF/KPSS stationarity, and weather lags.
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with nav_cols[2]:
    st.markdown(
        f"""
        <a href="/Model_Comparison" target="_self" style="text-decoration: none; color: inherit;">
            <div class="verdant-card" style="padding: 16px; height: 100%; transition: transform 0.15s ease, box-shadow 0.15s ease;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="background: #FAF5FF; border-radius: 8px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;">
                        {icon_model_compare(18, "#9333EA")}
                    </div>
                    {icon_arrow_right(14, "#527A50")}
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.05rem; font-weight: 600; color: #102C26; margin-bottom: 4px;">
                    Model Comparison
                </div>
                <div style="font-size: 0.8rem; color: #5A665D; line-height: 1.4;">
                    Classical vs. modern benchmark, multi-origin walk-forward validation.
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with nav_cols[3]:
    st.markdown(
        f"""
        <a href="/Future_Forecasts" target="_self" style="text-decoration: none; color: inherit;">
            <div class="verdant-card" style="padding: 16px; height: 100%; transition: transform 0.15s ease, box-shadow 0.15s ease;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="background: #FEF3C7; border-radius: 8px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center;">
                        {icon_trending_up(18, "#D97706")}
                    </div>
                    {icon_arrow_right(14, "#527A50")}
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.05rem; font-weight: 600; color: #102C26; margin-bottom: 4px;">
                    Future Forecasts
                </div>
                <div style="font-size: 0.8rem; color: #5A665D; line-height: 1.4;">
                    Forward projections originating from {stats['forecast_origin_date']}, intervals, and risk outlook.
                </div>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 4. PRIMARY DATASET VISUAL (RICE BLAST VS BROWN SPOT 2015–2025)
# ==============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="margin-bottom: 12px;">
        <h3 style="font-family: 'Fraunces', serif; font-size: 1.4rem; color: #102C26; margin: 0 0 4px 0;">
            Longitudinal Disease Surveillance Trajectory ({stats['monthly_start_date']} – {stats['monthly_end_date']})
        </h3>
        <p style="color: #5A665D; font-size: 0.88rem; margin: 0;">
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
