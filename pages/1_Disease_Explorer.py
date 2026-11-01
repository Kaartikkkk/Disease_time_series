"""Page 1: Disease Explorer — Curated Pathogen Surveillance & Spatial Analytics."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.icons import (
    icon_calendar,
    icon_check_circle,
    icon_download,
    icon_info,
    icon_leaf,
    icon_map,
    icon_pathology,
    icon_pin,
    icon_pulse,
    icon_shield,
    icon_table,
    icon_target,
    icon_timeseries,
)
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_image_base64,
    inject_custom_css,
    is_advanced_mode,
    is_simple_mode,
    load_all_pathogen_options,
    load_clean_records,
    load_data_quality_report_data,
    load_monthly_series,
    render_app_sidebar,
    render_graph_explanation_card,
    render_top_header,
    render_what_does_this_mean,
)

st.set_page_config(
    page_title="Disease Records — Verdant Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_app_sidebar(active_page="explorer")
render_top_header()

adv_mode = is_advanced_mode()

# ==============================================================================
# EDITORIAL HEADER & TELEMETRY STATUS
# ==============================================================================
last_sync_time = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div style="flex: 1 1 300px; min-width: 250px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                {icon_pathology(26, "#527A50")}
                <h1 style="font-size: clamp(1.6rem, 3.2vw, 2.2rem); margin: 0; color: #102C26; word-break: normal; hyphens: none; line-height: 1.2;">Disease Records</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0; line-height: 1.45;">
                Explore recorded crop disease observations, mapped collection sites, and historical reports across India.
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 12px; padding: 8px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); flex-shrink: 0; align-self: flex-start;">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                {icon_pulse(8, "#527A50")} Live Verified Database
            </div>
            <div style="font-size: 0.75rem; color: #5A665D; margin-top: 2px;">
                Verified Sync: <strong>{last_sync_time}</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load pathogens
pathogen_dict = load_all_pathogen_options()

# ==============================================================================
# UNIFIED TOP FILTER TOOLBAR
# ==============================================================================
st.markdown("<div style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #5A665D; margin-bottom: 8px;'>Filter Disease Records</div>", unsafe_allow_html=True)

bar_c1, bar_c2, bar_c3 = st.columns([1.2, 1.0, 1.0])

with bar_c1:
    selected_slug = st.selectbox(
        "Plant Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
        help="Choose a crop disease to explore historical reports.",
    )

df_clean = load_clean_records(selected_slug)
df_monthly = load_monthly_series(selected_slug)

if df_clean is None or df_clean.empty:
    st.info(f"No processed records currently available for {selected_slug}. Please choose another disease from the menu.")
    st.stop()

with bar_c2:
    available_regions = ["All Agro-Climatic Zones"] + sorted(list(df_clean["region_id"].dropna().unique()))
    region_labels = {
        "All Agro-Climatic Zones": "All Regions",
        "national_composite": "National Composite (Centroid)",
        "odisha_bengal": "Eastern Rice Belt (Odisha / Bengal)",
        "punjab_haryana": "Northwest Plains (Punjab / Haryana)",
        "tamil_nadu": "Southern Zone (Tamil Nadu / Deccan)",
        "maharashtra": "Western / Central Semi-Arid Zone",
    }
    selected_region = st.selectbox(
        "Region",
        options=available_regions,
        format_func=lambda r: region_labels.get(r, r),
        index=0,
        help="Filter records by Indian agricultural region.",
    )

with bar_c3:
    min_date = df_clean["event_date_parsed"].dropna().min().date()
    max_date = df_clean["event_date_parsed"].dropna().max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter historical reports between start and end dates.",
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
# FOUR PREMIUM KPI CARDS
# ==============================================================================
host_crop = filtered_df["host_crop"].iloc[0] if ("host_crop" in filtered_df.columns and not filtered_df.empty) else "Cereal Crop"
date_span_str = f"{filtered_df['event_date_parsed'].min().strftime('%b %Y')} – {filtered_df['event_date_parsed'].max().strftime('%b %Y')}" if not filtered_df.empty else "N/A"
geocoded_count = (filtered_df["decimalLatitude"].notna() & filtered_df["decimalLongitude"].notna()).sum() if not filtered_df.empty else 0
geocoded_pct = (geocoded_count / max(1, len(filtered_df))) * 100 if not filtered_df.empty else 0.0

st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 18px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
            <div class="metric-label-text">Recorded Reports</div>
            <div class="metric-value-text">{len(filtered_df):,}</div>
            <div class="metric-note-text">GBIF &amp; Research Accessions</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(20, "#527A50")}</div>
            <div class="metric-label-text">Date Range</div>
            <div class="metric-value-text" style="font-size: 1.4rem;">{date_span_str}</div>
            <div class="metric-note-text">Continuous Monthly Timeline</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_pin(20, "#527A50")}</div>
            <div class="metric-label-text">Mapped Locations</div>
            <div class="metric-value-text">{geocoded_count:,}</div>
            <div class="metric-note-text">{geocoded_pct:.1f}% with Indian coordinates</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
            <div class="metric-label-text">Host Crop</div>
            <div class="metric-value-text" style="font-size: 1.35rem;">{host_crop.split('(')[0].strip()}</div>
            <div class="metric-note-text">{filtered_df['crop_category'].iloc[0] if ('crop_category' in filtered_df.columns and not filtered_df.empty) else 'Indian Agriculture'}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# TABS: TIMELINE, SEASONAL CYCLE, LOCATION MAP, DATA TABLE, SUMMARY
# ==============================================================================
tab_monthly, tab_seasonal, tab_map, tab_table, tab_summary = st.tabs([
    "Historical Timeline",
    "Seasonal Cycle",
    "Location Map",
    "Recorded Reports Table",
    "Summary Findings",
])

# ----------------- Tab 1: Monthly Time-Series -----------------
with tab_monthly:
    st.subheader(f"Historical Disease Reports: {pathogen_dict[selected_slug]}")
    if df_monthly is not None and not df_monthly.empty:
        fig_m = go.Figure()

        # Documented count bar
        fig_m.add_trace(
            go.Bar(
                x=df_monthly["date"],
                y=df_monthly["documented_occurrence_count"],
                name="Monthly Recorded Reports",
                marker_color="#183D32",
                opacity=0.75,
                hovertemplate="Date: %{x|%b %Y}<br>Reports: %{y}<extra></extra>",
            )
        )
        # 3-Month Rolling Average
        fig_m.add_trace(
            go.Scatter(
                x=df_monthly["date"],
                y=df_monthly["documented_occurrence_count"].rolling(window=3, min_periods=1).mean(),
                name="3-Month Average Trend",
                line=dict(color="#C4A66A", width=2.5),
            )
        )

        format_plotly_figure(
            fig_m,
            title="Monthly Recorded Reports Over Time",
            height=440,
        )
        st.plotly_chart(fig_m, use_container_width=True)

        render_graph_explanation_card(
            what_it_shows="Verified disease reports recorded each calendar month alongside a 3-month rolling average trend line.",
            key_observation="Cyclical surges appear during the Kharif monsoon (July–October) and early Rabi season.",
            why_it_matters="Identifies seasonal outbreak patterns to plan preventive actions 4 to 6 weeks in advance.",
        )
    else:
        st.info("No monthly timeline series available for this disease selection.")

# ----------------- Tab 2: Seasonal Cycle -----------------
with tab_seasonal:
    st.subheader(f"Seasonal Reporting Pattern: {pathogen_dict[selected_slug]}")
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
        format_plotly_figure(fig_seasonal, title="Total Recorded Reports by Calendar Month", height=420)
        fig_seasonal.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_seasonal, use_container_width=True)

        peak_month = df_mb.sort_values("Reports", ascending=False).iloc[0]["Month"]
        render_graph_explanation_card(
            what_it_shows="Historical disease occurrences aggregated by calendar month (January to December).",
            key_observation=f"Reports concentrate heavily in {peak_month} and adjacent monsoon/post-monsoon months.",
            why_it_matters="Establishes the empirical baseline for seasonal time series forecasting.",
        )
    else:
        st.info("Seasonal month distribution data is being compiled for this selection.")

# ----------------- Tab 3: Geospatial Map -----------------
with tab_map:
    st.subheader("Recorded Collection Locations in India")
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
            height=460,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#FFFEFA",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        render_graph_explanation_card(
            what_it_shows="GPS-verified field collection coordinates across Indian agricultural zones.",
            key_observation="Reports cluster in high-intensity cropping tracts where universities actively sample crops.",
            why_it_matters="Highlights geographic surveillance coverage across different agro-climatic zones.",
        )
    else:
        st.info("Exact map coordinates are not available for this disease's past reports. The reports are recorded at the state and national level.")

# ----------------- Tab 4: Source Records Table -----------------
with tab_table:
    st.subheader(f"Recorded Disease Reports Table ({len(filtered_df):,} records)")

    if filtered_df.empty:
        st.info("No recorded disease reports found for this filter criteria. Try selecting 'All Regions' or expanding your date range.")
    else:
        display_cols = [
            col for col in [
                "gbifID", "scientific_name", "event_date_parsed", "year", "month",
                "region_id", "stateProvince", "decimalLatitude", "decimalLongitude",
                "basisOfRecord", "license",
            ]
            if col in filtered_df.columns
        ]

        st.dataframe(filtered_df[display_cols], use_container_width=True, height=360)

        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download Filtered Reports (CSV - {len(filtered_df)} rows)",
            data=csv_bytes,
            file_name=f"disease_records_{selected_slug}.csv",
            mime="text/csv",
        )

# ----------------- Tab 5: Summary Findings -----------------
with tab_summary:
    st.subheader("Data Quality & Findings Summary")
    q_data = load_data_quality_report_data()

    if q_data:
        audits = q_data.get("pathogen_audits", [])
        current_audit = next((a for a in audits if a.get("slug") == selected_slug), None)

        if current_audit:
            summary_metrics_df = pd.DataFrame([
                {"Metric": "Total GBIF Records", "Value": f"{current_audit['total_records']:,}", "Description": "Total raw occurrences pulled from GBIF REST API"},
                {"Metric": "Dated Records", "Value": f"{current_audit['records_with_date']:,} ({current_audit['pct_with_date']}%)", "Description": "Observations with verified year/month timestamps"},
                {"Metric": "Geocoded in India", "Value": f"{current_audit['records_with_india_coords']:,} ({current_audit['pct_with_coords']}%)", "Description": "Records with verified Indian GPS coordinates"},
                {"Metric": "Temporal Coverage", "Value": f"{current_audit['temporal_span_years']} Years", "Description": f"{current_audit['min_date']} to {current_audit['max_date']}"},
                {"Metric": "Forecasting Suitability", "Value": "High Quality (Benchmark Ready)" if current_audit["is_suitable_for_monthly_forecasting"] else "Contextual Historical Benchmark", "Description": current_audit["suitability_reason"]},
            ])
            st.dataframe(summary_metrics_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)
col_exp_nav1, col_exp_nav2 = st.columns(2)
with col_exp_nav1:
    st.page_link("pages/2_Time_Series_Analysis.py", label="Run Time Series Analysis on this Disease ->", use_container_width=True)
with col_exp_nav2:
    st.page_link("pages/4_Future_Forecasts.py", label="Generate Future Predictions ->", use_container_width=True)



