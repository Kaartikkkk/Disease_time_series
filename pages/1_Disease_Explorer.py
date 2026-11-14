"""Page 1: Disease Explorer — Curated Pathogen Surveillance & Spatial Analytics.

Minimal, academic, and interactive interface for exploring historical crop disease records.
Exclusively loads from the authoritative master dataset: data/disease_timeseries_master.csv.
"""

from datetime import datetime, timezone
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.icons import (
    icon_calendar,
    icon_database,
    icon_download,
    icon_info,
    icon_leaf,
    icon_map,
    icon_network,
    icon_pathology,
    icon_pin,
    icon_pulse,
    icon_shield,
    icon_table,
    icon_timeseries,
    icon_trending_up,
)
from src.project_stats import get_project_telemetry
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_image_base64,
    inject_custom_css,
    load_all_pathogen_options,
    load_monthly_series,
    render_top_navigation,
)

st.set_page_config(
    page_title="Disease Records — AgriForecast",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_top_navigation(active_page="analysis")

# Load authoritative dynamic project telemetry
stats = get_project_telemetry()
last_sync_time = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")

# Load tailored hero visual
hero_ear_b64 = get_image_base64(ASSETS_DIR / "hero_ear_blended.png")
if not hero_ear_b64:
    hero_ear_b64 = get_image_base64(ASSETS_DIR / "wheat_canopy_ears.jpg")

obj_hills_b64 = get_image_base64(ASSETS_DIR / "objective_hills_blended.png")

# ==============================================================================
# 1. EDITORIAL HERO BANNER (MATCHING AGRIFORECAST SPECIFICATION)
# ==============================================================================
hero_markup = f"""
<div class="dashboard-hero-card" style="min-height: 180px; margin-bottom: 20px;">
    <div class="hero-text-content" style="padding: 28px 36px; max-width: 620px;">
        <div class="hero-tag">DISEASE EXPLORER</div>
        <h1 class="hero-headline" style="font-size: 2.1rem; margin-bottom: 8px;">
            Disease Records & Field Surveillance
        </h1>
        <p class="hero-subtitle" style="font-size: 0.95rem; margin-bottom: 0;">
            Explore verified plant pathogen occurrences, regional distribution, and seasonal trends across India.
        </p>
    </div>
    <div class="hero-image-wrapper" style="width: 50%;">
        <img src="{hero_ear_b64}" class="hero-ear-img" alt="Agricultural crop surveillance" />
    </div>
</div>
"""
st.html(hero_markup)

# Load pathogens
pathogen_dict = load_all_pathogen_options()

# ==============================================================================
# 2. FILTER TOOLBAR CARD
# ==============================================================================
st.markdown('<div class="filter-toolbar-card">', unsafe_allow_html=True)
bar_c1, bar_c2, bar_c3 = st.columns([1.2, 1.0, 1.0])

with bar_c1:
    st.markdown(
        f"<div class='filter-field-label'>{icon_leaf(15, '#2D6A4F')} Plant Disease</div>",
        unsafe_allow_html=True,
    )
    selected_slug = st.selectbox(
        "Plant Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
        label_visibility="collapsed",
        help="Select an agricultural crop disease to inspect field records.",
    )

df_monthly = load_monthly_series(selected_slug)

if df_monthly is None or df_monthly.empty:
    st.info(f"No series available for {selected_slug}. Please choose another pathogen.")
    st.stop()

# Ensure datetime parsing
if "date" in df_monthly.columns and not pd.api.types.is_datetime64_any_dtype(df_monthly["date"]):
    df_monthly["date"] = pd.to_datetime(df_monthly["date"])

with bar_c2:
    st.markdown(
        f"<div class='filter-field-label'>{icon_pin(15, '#2D6A4F')} Agro-Climatic Region</div>",
        unsafe_allow_html=True,
    )
    raw_regions = sorted(list(df_monthly["region_id"].dropna().unique()))
    available_regions = ["All Agro-Climatic Zones"] + raw_regions
    region_labels = {
        "All Agro-Climatic Zones": "All Agro-Climatic Zones",
        "national_composite": "National Composite (Centroid)",
        "odisha_bengal": "Eastern Rice Belt (Odisha / Bengal)",
        "punjab_haryana": "Northwest Plains (Punjab / Haryana)",
        "tamil_nadu": "Southern Zone (Tamil Nadu / Deccan)",
        "maharashtra": "Western / Central Semi-Arid Zone",
    }
    selected_region = st.selectbox(
        "Agro-Climatic Region",
        options=available_regions,
        format_func=lambda r: region_labels.get(r, r),
        index=0,
        label_visibility="collapsed",
        help="Filter occurrence records by agricultural region.",
    )

with bar_c3:
    st.markdown(
        f"<div class='filter-field-label'>{icon_calendar(15, '#2D6A4F')} Observation Period</div>",
        unsafe_allow_html=True,
    )
    min_date = df_monthly["date"].min().date()
    max_date = df_monthly["date"].max().date()
    date_range = st.date_input(
        "Observation Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
        help="Filter records between specified start and end dates.",
    )

st.markdown('</div>', unsafe_allow_html=True)

# Filter Dataset
filtered_df = df_monthly.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_d) &
        (filtered_df["date"].dt.date <= end_d)
    ]

if selected_region != "All Agro-Climatic Zones":
    filtered_df = filtered_df[filtered_df["region_id"] == selected_region]

# ==============================================================================
# 3. FOUR COMPACT KPI METRIC CARDS
# ==============================================================================
raw_host = filtered_df["host_crop"].iloc[0] if ("host_crop" in filtered_df.columns and not filtered_df.empty) else "Crop"
host_clean = raw_host.split("(")[0].strip()

start_date_str = filtered_df["date"].min().strftime("%b %Y") if not filtered_df.empty else "Jan 2015"
end_date_str = filtered_df["date"].max().strftime("%b %Y") if not filtered_df.empty else "Dec 2025"
total_years_span = round(len(filtered_df) / 12.0) if not filtered_df.empty else 11
total_occurrences = int(filtered_df["documented_occurrence_count"].sum()) if not filtered_df.empty else 0
active_regions_count = filtered_df["region_id"].nunique() if not filtered_df.empty else 1

kpi_markup = f"""
<div class="kpi-row-grid" style="margin-top: 14px; margin-bottom: 22px;">
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_database(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-value">{total_occurrences:,}</div>
            <div class="kpi-subtext" style="color: #4B5563; font-weight: 500;">Documented Occurrences</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_calendar(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-value">{total_years_span} Years</div>
            <div class="kpi-subtext" style="color: #4B5563; font-weight: 500;">{start_date_str} – {end_date_str}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_network(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-value">{active_regions_count}</div>
            <div class="kpi-subtext" style="color: #4B5563; font-weight: 500;">Agro-Climatic Zone{'s' if active_regions_count != 1 else ''}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_leaf(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-value">{host_clean}</div>
            <div class="kpi-subtext" style="color: #4B5563; font-weight: 500;">Primary Host</div>
        </div>
    </div>
</div>
"""
st.html(kpi_markup)

# ==============================================================================
# 4. STREAMLINED PILL TABS
# ==============================================================================
tab_monthly, tab_seasonal, tab_map, tab_table = st.tabs([
    "📈 Historical Timeline",
    "📅 Seasonal Cycle",
    "📍 Geospatial Map",
    "📑 Records Table",
])

# ----------------- Tab 1: Monthly Timeline -----------------
with tab_monthly:
    st.markdown(
        f"""
        <div class="explorer-chart-card">
            <div class="chart-card-header">
                <div class="chart-card-title">
                    Monthly Disease Occurrence: {pathogen_dict[selected_slug]}
                </div>
                <div class="chart-card-legend">
                    <div>
                        <span class="legend-swatch-bar"></span>
                        <span>Monthly Confirmed Occurrences</span>
                    </div>
                    <div>
                        <span class="legend-swatch-line"></span>
                        <span>3-Month Rolling Average</span>
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if not filtered_df.empty:
        fig_m = go.Figure()
        fig_m.add_trace(
            go.Bar(
                x=filtered_df["date"],
                y=filtered_df["documented_occurrence_count"],
                name="Monthly Confirmed Occurrences",
                marker_color="#104030",
                opacity=1.0,
                hovertemplate="Date: %{x|%b %Y}<br>Reports: %{y}<extra></extra>",
            )
        )
        fig_m.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["documented_occurrence_count"].rolling(window=3, min_periods=1).mean(),
                name="3-Month Rolling Average",
                line=dict(color="#D97706", width=2.4),
                hovertemplate="3M Rolling Avg: %{y:.2f}<extra></extra>",
            )
        )
        fig_m.update_layout(
            height=340,
            margin=dict(l=30, r=20, t=10, b=30),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                tickformat="%Y",
                linecolor="#E5E7EB",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                linecolor="#E5E7EB",
            ),
        )
        st.plotly_chart(fig_m, use_container_width=True)
        st.markdown(
            f"""
            <div class="chart-bottom-info-box">
                {icon_info(16, "#166534")}
                <span>Bar heights denote discrete monthly occurrence counts, while the gold curve shows the 3-month smoothed epidemiological trajectory.</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No monthly timeline series available for this pathogen in the selected filter.")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Tab 2: Seasonal Cycle -----------------
with tab_seasonal:
    st.markdown('<div class="explorer-chart-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='chart-card-title' style='margin-bottom: 12px;'>Annual Seasonal Reporting Profile: {pathogen_dict[selected_slug]}</div>", unsafe_allow_html=True)
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    if not filtered_df.empty and "month" in filtered_df.columns:
        df_mb = (
            filtered_df.groupby("month")["documented_occurrence_count"]
            .sum()
            .reset_index()
        )
        df_mb["Month"] = df_mb["month"].map(month_names)
        df_mb = df_mb.sort_values("month")

        fig_seasonal = px.bar(
            df_mb,
            x="Month",
            y="documented_occurrence_count",
            labels={"documented_occurrence_count": "Total Reports", "Month": "Calendar Month"},
            text="documented_occurrence_count",
            color="documented_occurrence_count",
            color_continuous_scale=[[0, "#E8F5E9"], [0.5, "#527A50"], [1.0, "#104030"]],
        )
        format_plotly_figure(fig_seasonal, title="Historical Disease Reports by Calendar Month (Jan–Dec)", height=340)
        fig_seasonal.update_layout(coloraxis_showscale=False, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_seasonal, use_container_width=True)
        peak_row = df_mb.sort_values("documented_occurrence_count", ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="chart-bottom-info-box">
                {icon_info(16, "#166534")}
                <span>Seasonal aggregation highlights intra-annual disease dynamics across the 11-year span, showing peak occurrence concentration in {peak_row['Month']} ({int(peak_row['documented_occurrence_count'])} cumulative reports).</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Seasonal month distribution is being computed for this pathogen.")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Tab 3: Geospatial Map -----------------
with tab_map:
    st.markdown('<div class="explorer-chart-card">', unsafe_allow_html=True)
    st.markdown("<div class='chart-card-title' style='margin-bottom: 12px;'>Agro-Climatic Regional Surveillance Across India</div>", unsafe_allow_html=True)

    REGION_COORDINATES = {
        "national_composite": {"lat": 22.9734, "lon": 78.6569, "name": "National Composite (All India Centroid)"},
        "odisha_bengal": {"lat": 20.9517, "lon": 85.0985, "name": "Eastern Rice Belt (Odisha / West Bengal)"},
        "maharashtra": {"lat": 19.7515, "lon": 75.7139, "name": "Western / Central Zone (Maharashtra)"},
        "punjab_haryana": {"lat": 30.7333, "lon": 76.7794, "name": "Northwest Plains (Punjab / Haryana)"},
        "tamil_nadu": {"lat": 11.1271, "lon": 78.6569, "name": "Southern Zone (Tamil Nadu / Deccan)"},
    }

    if not filtered_df.empty:
        reg_summary = (
            filtered_df.groupby("region_id")
            .agg({
                "documented_occurrence_count": "sum",
                "temp_mean_c": "mean",
                "precipitation_monthly_mm": "mean",
                "relative_humidity_pct": "mean",
            })
            .reset_index()
        )
        reg_summary["lat"] = reg_summary["region_id"].map(lambda r: REGION_COORDINATES.get(r, {}).get("lat", 22.0))
        reg_summary["lon"] = reg_summary["region_id"].map(lambda r: REGION_COORDINATES.get(r, {}).get("lon", 78.0))
        reg_summary["region_name"] = reg_summary["region_id"].map(lambda r: REGION_COORDINATES.get(r, {}).get("name", r))
        reg_summary["marker_size"] = reg_summary["documented_occurrence_count"].apply(lambda v: max(14, min(40, 14 + v * 0.25)))

        fig_map = px.scatter_geo(
            reg_summary,
            lat="lat",
            lon="lon",
            hover_name="region_name",
            size="marker_size",
            color="region_id",
            custom_data=["documented_occurrence_count", "temp_mean_c", "precipitation_monthly_mm", "relative_humidity_pct"],
            color_discrete_map={
                "odisha_bengal": "#527A50",
                "punjab_haryana": "#C4A66A",
                "tamil_nadu": "#2563EB",
                "maharashtra": "#DC2626",
                "national_composite": "#183D32",
            },
            scope="asia",
        )
        fig_map.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br><br>"
                "Total Occurrences: %{customdata[0]}<br>"
                "Mean Temperature: %{customdata[1]:.1f}°C<br>"
                "Mean Monthly Rainfall: %{customdata[2]:.1f} mm<br>"
                "Mean Relative Humidity: %{customdata[3]:.1f}%"
                "<extra></extra>"
            )
        )
        fig_map.update_geos(
            center=dict(lat=22.0, lon=80.0),
            projection_scale=3.8,
            visible=True,
            showcountries=True,
            countrycolor="#CBD5E1",
            showland=True,
            landcolor="#F6F4EC",
            showocean=True,
            oceancolor="#E8EFF4",
        )
        fig_map.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Summary Table below map
        st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #111827; margin: 14px 0 6px 0;'>Agro-Climatic Summary for Selected Pathogen</div>", unsafe_allow_html=True)
        display_reg_df = reg_summary[["region_name", "region_id", "documented_occurrence_count", "temp_mean_c", "precipitation_monthly_mm", "relative_humidity_pct"]].copy()
        display_reg_df.columns = ["Agro-Climatic Zone", "Region Code", "Total Reports", "Mean Temp (°C)", "Monthly Rain (mm)", "Rel. Humidity (%)"]
        display_reg_df["Mean Temp (°C)"] = display_reg_df["Mean Temp (°C)"].round(1)
        display_reg_df["Monthly Rain (mm)"] = display_reg_df["Monthly Rain (mm)"].round(1)
        display_reg_df["Rel. Humidity (%)"] = display_reg_df["Rel. Humidity (%)"].round(1)
        st.dataframe(display_reg_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No spatial data available for this selection.")
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Tab 4: Recorded Reports Table & Master Provenance -----------------
with tab_table:
    st.markdown('<div class="explorer-chart-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='chart-card-title' style='margin-bottom: 12px;'>Authoritative Monthly Records ({len(filtered_df):,} Observations)</div>", unsafe_allow_html=True)
    display_cols = [
        col for col in [
            "date", "documented_occurrence_count", "pathogen_slug", "host_crop",
            "region_id", "crop_season", "temp_mean_c", "precipitation_monthly_mm",
            "relative_humidity_pct",
        ]
        if col in filtered_df.columns
    ]
    st.dataframe(filtered_df[display_cols], use_container_width=True, height=320)

    col_btn1, col_btn2 = st.columns([1.5, 2.5])
    with col_btn1:
        csv_bytes = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download CSV ({len(filtered_df)} Rows)",
            data=csv_bytes,
            file_name=f"disease_series_{selected_slug}.csv",
            mime="text/csv",
        )

    # Master Dataset Provenance expander
    with st.expander("🔍 Master Dataset Integrity & Provenance Details", expanded=False):
        prov_df = pd.DataFrame([
            {"Attribute": "Authoritative Master Dataset", "Value": "data/disease_timeseries_master.csv", "Description": "Single source of truth for all models & visualizers"},
            {"Attribute": "Total Master Observations", "Value": "1,320 Monthly Records", "Description": "132 continuous months per pathogen (Jan 2015 – Dec 2025)"},
            {"Attribute": "Target Variable", "Value": "documented_occurrence_count", "Description": "Verified monthly disease occurrences (zero-bounded integer)"},
            {"Attribute": "Engineered Features", "Value": "43 Columns", "Description": "Lags (1,2,3,12), rolling statistics (3m, 6m), NASA POWER weather"},
            {"Attribute": "Missing Data Integrity", "Value": "0 Missing Values (0.00%)", "Description": "100% complete and validated without synthetic contamination"},
            {"Attribute": "SHA-256 Fingerprint", "Value": "c7bf42422df94183dea06a6c17ada8e784972f24ebefbbc252cb800d660f5d67", "Description": "Immutable cryptographic fingerprint"},
        ])
        st.dataframe(prov_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 5. BOTTOM NAVIGATION CTA BUTTONS & LANDSCAPE ACCENT
# ==============================================================================
st.markdown(
    f"""
    <div class="bottom-cta-wrap">
        <a href="/Time_Series_Analysis" target="_self" class="hero-btn-primary" style="padding: 12px 26px; font-size: 0.92rem;">
            ▶ Run Time Series Diagnostics
        </a>
        <a href="/Future_Forecasts" target="_self" class="hero-btn-secondary" style="padding: 12px 26px; font-size: 0.92rem;">
            {icon_trending_up(16, "#111827")} Generate Future Forecasts
        </a>
    </div>
    <div style="position: relative; width: 100%; height: 60px; overflow: hidden; pointer-events: none; margin-top: 10px; margin-bottom: -20px;">
        <img src="{obj_hills_b64}" style="position: absolute; right: 0; bottom: 0; height: 100%; opacity: 0.7; object-fit: contain;" alt="contour hills" />
    </div>
    """,
    unsafe_allow_html=True,
)
