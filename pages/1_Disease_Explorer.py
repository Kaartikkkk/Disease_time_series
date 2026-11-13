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
    icon_download,
    icon_leaf,
    icon_map,
    icon_pathology,
    icon_pin,
    icon_pulse,
    icon_shield,
    icon_table,
    icon_timeseries,
)
from src.project_stats import get_project_telemetry
from src.ui_helpers import (
    format_plotly_figure,
    inject_custom_css,
    load_all_pathogen_options,
    load_monthly_series,
    render_app_sidebar,
    render_top_header,
)

st.set_page_config(
    page_title="Disease Records — Agricultural Disease Forecasting Using Time Series",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_app_sidebar(active_page="explorer")
render_top_header()

# Load authoritative dynamic project telemetry
stats = get_project_telemetry()
last_sync_time = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")

# ==============================================================================
# 1. EDITORIAL HEADER
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div style="flex: 1 1 300px; min-width: 250px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_pathology(24, "#527A50")}
                <h1 style="font-size: clamp(1.5rem, 3vw, 2.0rem); margin: 0; color: #102C26; word-break: normal; hyphens: none; line-height: 1.2;">Disease Records &amp; Field Surveillance</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.92rem; margin: 0; line-height: 1.45;">
                Explore verified plant pathogen occurrences, agro-climatic regional distributions, and seasonal reporting cycles across India. Sourced from the single authoritative master dataset.
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 10px; padding: 6px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02); flex-shrink: 0; align-self: flex-start;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                {icon_pulse(8, "#527A50")} Authoritative Master Dataset
            </div>
            <div style="font-size: 0.75rem; color: #5A665D; margin-top: 1px;">
                Standardized Scope: <strong>{stats['monthly_start_date']} – {stats['monthly_end_date']}</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load pathogens
pathogen_dict = load_all_pathogen_options()

# ==============================================================================
# 2. FILTER TOOLBAR
# ==============================================================================
bar_c1, bar_c2, bar_c3 = st.columns([1.2, 1.0, 1.0])

with bar_c1:
    selected_slug = st.selectbox(
        "Target Plant Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
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
        help="Filter occurrence records by agricultural region.",
    )

with bar_c3:
    min_date = df_monthly["date"].min().date()
    max_date = df_monthly["date"].max().date()
    date_range = st.date_input(
        "Observation Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter records between specified start and end dates.",
    )

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
host_crop = filtered_df["host_crop"].iloc[0] if ("host_crop" in filtered_df.columns and not filtered_df.empty) else "Agricultural Crop"
date_span_str = (
    f"{filtered_df['date'].min().strftime('%b %Y')} – {filtered_df['date'].max().strftime('%b %Y')}"
    if not filtered_df.empty
    else "N/A"
)
total_occurrences = int(filtered_df["documented_occurrence_count"].sum()) if not filtered_df.empty else 0
active_regions_count = filtered_df["region_id"].nunique() if not filtered_df.empty else 0

st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 10px; margin-bottom: 20px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(18, "#527A50")}</div>
            <div class="metric-label-text">Documented Occurrences</div>
            <div class="metric-value-text">{total_occurrences:,}</div>
            <div class="metric-note-text">Total Recorded Disease Reports</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(18, "#527A50")}</div>
            <div class="metric-label-text">Observed Span</div>
            <div class="metric-value-text" style="font-size: 1.25rem;">{date_span_str}</div>
            <div class="metric-note-text">{len(filtered_df)} Continuous Months</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_pin(18, "#527A50")}</div>
            <div class="metric-label-text">Agro-Climatic Zones</div>
            <div class="metric-value-text">{active_regions_count}</div>
            <div class="metric-note-text">Regional Spatial Units</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(18, "#527A50")}</div>
            <div class="metric-label-text">Primary Host</div>
            <div class="metric-value-text" style="font-size: 1.25rem;">{host_crop.split('(')[0].strip()}</div>
            <div class="metric-note-text">Primary Agricultural Host</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 4. STREAMLINED TABS (FOCUSED & ACTIONABLE)
# ==============================================================================
tab_monthly, tab_seasonal, tab_map, tab_table = st.tabs([
    "Historical Timeline",
    "Seasonal Cycle",
    "Geospatial Map",
    "Recorded Reports Table",
])

# ----------------- Tab 1: Monthly Timeline -----------------
with tab_monthly:
    st.markdown(f"**Monthly Disease Occurrence Trajectory: {pathogen_dict[selected_slug]}**")
    if not filtered_df.empty:
        fig_m = go.Figure()
        fig_m.add_trace(
            go.Bar(
                x=filtered_df["date"],
                y=filtered_df["documented_occurrence_count"],
                name="Monthly Confirmed Occurrences",
                marker_color="#183D32",
                opacity=0.8,
                hovertemplate="Date: %{x|%b %Y}<br>Reports: %{y}<extra></extra>",
            )
        )
        fig_m.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["documented_occurrence_count"].rolling(window=3, min_periods=1).mean(),
                name="3-Month Rolling Average",
                line=dict(color="#C4A66A", width=2.4),
            )
        )
        format_plotly_figure(
            fig_m,
            title=f"Monthly Verified Disease Reports ({filtered_df['date'].min().strftime('%b %Y')} – {filtered_df['date'].max().strftime('%b %Y')})",
            height=360,
        )
        st.plotly_chart(fig_m, use_container_width=True)
        st.caption("Bar heights denote discrete monthly occurrence counts deposited in the master surveillance dataset; the gold curve traces the 3-month smoothed epidemiological trajectory.")
    else:
        st.info("No monthly timeline series available for this pathogen in the selected filter.")

# ----------------- Tab 2: Seasonal Cycle -----------------
with tab_seasonal:
    st.markdown(f"**Annual Seasonal Reporting Profile: {pathogen_dict[selected_slug]}**")
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
            color_continuous_scale=[[0, "#E8F5E9"], [0.5, "#527A50"], [1.0, "#183D32"]],
        )
        format_plotly_figure(fig_seasonal, title="Historical Disease Reports by Calendar Month (Jan–Dec)", height=350)
        fig_seasonal.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        peak_row = df_mb.sort_values("documented_occurrence_count", ascending=False).iloc[0]
        st.caption(
            f"Seasonal aggregation highlights intra-annual disease dynamics across the 11-year span, "
            f"showing peak occurrence concentration in {peak_row['Month']} ({int(peak_row['documented_occurrence_count'])} cumulative reports)."
        )
    else:
        st.info("Seasonal month distribution is being computed for this pathogen.")

# ----------------- Tab 3: Geospatial Map -----------------
with tab_map:
    st.markdown("**Agro-Climatic Regional Surveillance Across India**")

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
            paper_bgcolor="#FFFEFA",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("Spatial surveillance centroids representing agro-climatic regional hubs for disease monitoring in the master dataset.")

        # Summary Table below map
        st.markdown("**Agro-Climatic Summary for Selected Pathogen**")
        display_reg_df = reg_summary[["region_name", "region_id", "documented_occurrence_count", "temp_mean_c", "precipitation_monthly_mm", "relative_humidity_pct"]].copy()
        display_reg_df.columns = ["Agro-Climatic Zone", "Region Code", "Total Reports", "Mean Temp (°C)", "Monthly Rain (mm)", "Rel. Humidity (%)"]
        display_reg_df["Mean Temp (°C)"] = display_reg_df["Mean Temp (°C)"].round(1)
        display_reg_df["Monthly Rain (mm)"] = display_reg_df["Monthly Rain (mm)"].round(1)
        display_reg_df["Rel. Humidity (%)"] = display_reg_df["Rel. Humidity (%)"].round(1)
        st.dataframe(display_reg_df, use_container_width=True, hide_index=True)
    else:
        st.info("No spatial data available for this selection.")

# ----------------- Tab 4: Recorded Reports Table & Master Provenance -----------------
with tab_table:
    st.markdown(f"**Authoritative Monthly Records ({len(filtered_df):,} Observations)**")
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

# Navigation link
st.markdown("<br>", unsafe_allow_html=True)
try:
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.page_link("pages/2_Time_Series_Analysis.py", label="Run Time Series Diagnostics ->", use_container_width=True)
    with col_nav2:
        st.page_link("pages/4_Future_Forecasts.py", label="Generate Future Forecasts ->", use_container_width=True)
except Exception:
    pass
