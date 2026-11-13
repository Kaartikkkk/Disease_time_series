"""Page 1: Disease Explorer — Curated Pathogen Surveillance & Spatial Analytics.

Minimal, academic, and interactive interface for exploring historical crop disease records.
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
    load_clean_records,
    load_data_quality_report_data,
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
                Explore verified plant pathogen occurrences, GPS voucher locations, and historical seasonal reporting cycles across India.
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 10px; padding: 6px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02); flex-shrink: 0; align-self: flex-start;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                {icon_pulse(8, "#527A50")} Verified Research Database
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

df_clean = load_clean_records(selected_slug)
df_monthly = load_monthly_series(selected_slug)

if df_clean is None or df_clean.empty:
    st.info(f"No processed records available for {selected_slug}. Please choose another pathogen.")
    st.stop()

with bar_c2:
    available_regions = ["All Agro-Climatic Zones"] + sorted(list(df_clean["region_id"].dropna().unique()))
    region_labels = {
        "All Agro-Climatic Zones": "All Indian Regions",
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
    min_date = df_clean["event_date_parsed"].dropna().min().date()
    max_date = df_clean["event_date_parsed"].dropna().max().date()
    date_range = st.date_input(
        "Observation Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter records between specified start and end dates.",
    )

# Filter Dataset
filtered_df = df_clean.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[
        (filtered_df["event_date_parsed"].dt.date >= start_d) &
        (filtered_df["event_date_parsed"].dt.date <= end_d)
    ]

if selected_region != "All Agro-Climatic Zones":
    filtered_df = filtered_df[filtered_df["region_id"] == selected_region]

# ==============================================================================
# 3. FOUR COMPACT KPI METRIC CARDS
# ==============================================================================
host_crop = filtered_df["host_crop"].iloc[0] if ("host_crop" in filtered_df.columns and not filtered_df.empty) else "Cereal Crop"
date_span_str = f"{filtered_df['event_date_parsed'].min().strftime('%b %Y')} – {filtered_df['event_date_parsed'].max().strftime('%b %Y')}" if not filtered_df.empty else "N/A"
geocoded_count = (filtered_df["decimalLatitude"].notna() & filtered_df["decimalLongitude"].notna()).sum() if not filtered_df.empty else 0
geocoded_pct = (geocoded_count / max(1, len(filtered_df))) * 100 if not filtered_df.empty else 0.0

st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 10px; margin-bottom: 20px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(18, "#527A50")}</div>
            <div class="metric-label-text">Verified Reports</div>
            <div class="metric-value-text">{len(filtered_df):,}</div>
            <div class="metric-note-text">GBIF &amp; Institutional Herbaria</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(18, "#527A50")}</div>
            <div class="metric-label-text">Observed Span</div>
            <div class="metric-value-text" style="font-size: 1.25rem;">{date_span_str}</div>
            <div class="metric-note-text">Raw Accession Timestamps</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_pin(18, "#527A50")}</div>
            <div class="metric-label-text">Geocoded Sites</div>
            <div class="metric-value-text">{geocoded_count:,}</div>
            <div class="metric-note-text">{geocoded_pct:.1f}% with GPS coordinates</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(18, "#527A50")}</div>
            <div class="metric-label-text">Primary Host</div>
            <div class="metric-value-text" style="font-size: 1.25rem;">{host_crop.split('(')[0].strip()}</div>
            <div class="metric-note-text">{filtered_df['crop_category'].iloc[0] if ('crop_category' in filtered_df.columns and not filtered_df.empty) else 'Agricultural Host'}</div>
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
    if df_monthly is not None and not df_monthly.empty:
        fig_m = go.Figure()
        fig_m.add_trace(
            go.Bar(
                x=df_monthly["date"],
                y=df_monthly["documented_occurrence_count"],
                name="Monthly Confirmed Occurrences",
                marker_color="#183D32",
                opacity=0.8,
                hovertemplate="Date: %{x|%b %Y}<br>Reports: %{y}<extra></extra>",
            )
        )
        fig_m.add_trace(
            go.Scatter(
                x=df_monthly["date"],
                y=df_monthly["documented_occurrence_count"].rolling(window=3, min_periods=1).mean(),
                name="3-Month Rolling Average",
                line=dict(color="#C4A66A", width=2.4),
            )
        )
        format_plotly_figure(
            fig_m,
            title=f"Monthly Verified Disease Reports ({stats['monthly_start_date']} – {stats['monthly_end_date']})",
            height=360,
        )
        st.plotly_chart(fig_m, use_container_width=True)
        st.caption("Bar heights denote discrete monthly accession counts deposited in public surveillance databases; the gold curve traces the 3-month smoothed epidemiological trajectory.")
    else:
        st.info("No monthly timeline series available for this pathogen.")

# ----------------- Tab 2: Seasonal Cycle -----------------
with tab_seasonal:
    st.markdown(f"**Annual Seasonal Reporting Profile: {pathogen_dict[selected_slug]}**")
    q_data = load_data_quality_report_data()
    md = {}
    if q_data:
        audits = q_data.get("pathogen_audits", [])
        current_audit = next((a for a in audits if a.get("slug") == selected_slug), None)
        if current_audit:
            md = current_audit.get("month_distribution", {})

    month_names = {
        "1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr", "5": "May", "6": "Jun",
        "7": "Jul", "8": "Aug", "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
    }
    if md:
        df_mb = pd.DataFrame([
            {"MonthNum": int(k), "Month": month_names.get(str(k), str(k)), "Reports": v}
            for k, v in md.items()
        ]).sort_values("MonthNum")

        fig_seasonal = px.bar(
            df_mb,
            x="Month",
            y="Reports",
            text="Reports",
            color="Reports",
            color_continuous_scale=[[0, "#E8F5E9"], [0.5, "#527A50"], [1.0, "#183D32"]],
        )
        format_plotly_figure(fig_seasonal, title="Historical Disease Reports by Calendar Month (Jan–Dec)", height=350)
        fig_seasonal.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_seasonal, use_container_width=True)
        peak_month = df_mb.sort_values("Reports", ascending=False).iloc[0]["Month"]
        st.caption(f"Seasonal aggregation highlights intra-annual disease dynamics, showing peak occurrence concentration in {peak_month} during the primary cropping window.")
    else:
        st.info("Seasonal month distribution is being computed for this pathogen.")

# ----------------- Tab 3: Geospatial Map -----------------
with tab_map:
    st.markdown("**GPS Voucher Locations Across Indian Agricultural Tracts**")
    coords_df = filtered_df.dropna(subset=["decimalLatitude", "decimalLongitude"]).copy()

    if not coords_df.empty:
        coords_df["lat"] = coords_df["decimalLatitude"].astype(float)
        coords_df["lon"] = coords_df["decimalLongitude"].astype(float)

        fig_map = px.scatter_geo(
            coords_df,
            lat="lat",
            lon="lon",
            hover_name="scientific_name",
            hover_data=["event_date_parsed", "region_id", "basisOfRecord"],
            color="region_id",
            color_discrete_map={
                "odisha_bengal": "#527A50",
                "punjab_haryana": "#C4A66A",
                "tamil_nadu": "#2563EB",
                "maharashtra": "#DC2626",
                "national_composite": "#183D32",
            },
            scope="asia",
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
        st.caption(f"Spatial distribution of {len(coords_df):,} GPS-tagged observations across Indian agro-climatic zones.")
    else:
        st.info("Exact map coordinates are not available for this pathogen; historical occurrences are indexed at state/national centroid resolution.")

# ----------------- Tab 4: Recorded Reports Table & Data Quality -----------------
with tab_table:
    st.markdown(f"**Verified Occurrence Records ({len(filtered_df):,} Records)**")
    display_cols = [
        col for col in [
            "gbifID", "scientific_name", "event_date_parsed", "year", "month",
            "region_id", "stateProvince", "decimalLatitude", "decimalLongitude",
            "basisOfRecord", "license",
        ]
        if col in filtered_df.columns
    ]
    st.dataframe(filtered_df[display_cols], use_container_width=True, height=320)

    col_btn1, col_btn2 = st.columns([1.5, 2.5])
    with col_btn1:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download CSV ({len(filtered_df)} Rows)",
            data=csv_bytes,
            file_name=f"disease_records_{selected_slug}.csv",
            mime="text/csv",
        )

    # Data Quality Summary expander inside the table tab
    with st.expander("🔍 Pathogen Data Quality Audit & Metadata Details", expanded=False):
        q_data = load_data_quality_report_data()
        if q_data:
            audits = q_data.get("pathogen_audits", [])
            current_audit = next((a for a in audits if a.get("slug") == selected_slug), None)
            if current_audit:
                summary_metrics_df = pd.DataFrame([
                    {"Metric": "Total Raw Records", "Value": f"{current_audit['total_records']:,}", "Description": "Raw occurrences retrieved from GBIF REST API"},
                    {"Metric": "Dated Records", "Value": f"{current_audit['records_with_date']:,} ({current_audit['pct_with_date']}%)", "Description": "Verified year/month timestamps"},
                    {"Metric": "Geocoded in India", "Value": f"{current_audit['records_with_india_coords']:,} ({current_audit['pct_with_coords']}%)", "Description": "Verified Indian GPS coordinates"},
                    {"Metric": "Temporal Coverage", "Value": f"{current_audit['temporal_span_years']} Years", "Description": f"{current_audit['min_date']} to {current_audit['max_date']}"},
                    {"Metric": "Forecasting Suitability", "Value": "Benchmark Ready" if current_audit["is_suitable_for_monthly_forecasting"] else "Contextual Historical Series", "Description": current_audit["suitability_reason"]},
                ])
                st.dataframe(summary_metrics_df, use_container_width=True, hide_index=True)

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
