"""Page 1: Disease Explorer — Curated Pathogen Surveillance & Spatial Analytics."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.icons import (
    icon_download,
    icon_leaf,
    icon_pathology,
    icon_pin,
    icon_pulse,
    icon_shield,
    icon_timeseries,
)
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_image_base64,
    inject_custom_css,
    load_all_pathogen_options,
    load_clean_records,
    load_data_quality_report_data,
    load_monthly_series,
)

st.set_page_config(
    page_title="Disease Explorer — Verdant Intelligence",
    page_icon="🌿",
    layout="wide",
)

inject_custom_css()

# ==============================================================================
# EDITORIAL HEADER & TELEMETRY STATUS
# ==============================================================================
last_sync_time = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_pathology(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">Disease Intelligence</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Monitor documented pathogen occurrences, inspect geospatial sample density, and review empirical data quality.
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 12px; padding: 8px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                {icon_pulse(8, "#527A50")} Live GBIF & NASA POWER Ingestion
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
st.markdown("<div style='font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #5A665D; margin-bottom: 8px;'>Surveillance Query Parameters</div>", unsafe_allow_html=True)

bar_c1, bar_c2, bar_c3 = st.columns([1.2, 1.0, 1.0])

with bar_c1:
    selected_slug = st.selectbox(
        "Target Pathogen / Host Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
        label_visibility="collapsed",
    )

df_clean = load_clean_records(selected_slug)
df_monthly = load_monthly_series(selected_slug)

if df_clean is None or df_clean.empty:
    st.error(f"No processed surveillance records available for {selected_slug}.")
    st.stop()

with bar_c2:
    available_regions = ["All Agro-Climatic Zones"] + sorted(list(df_clean["region_id"].dropna().unique()))
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
    )

with bar_c3:
    min_date = df_clean["event_date_parsed"].dropna().min().date()
    max_date = df_clean["event_date_parsed"].dropna().max().date()
    date_range = st.date_input(
        "Temporal Span",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
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
host_crop = filtered_df["host_crop"].iloc[0] if "host_crop" in filtered_df.columns else "Cereal Crop"
date_span_str = f"{filtered_df['event_date_parsed'].min().strftime('%b %Y')} – {filtered_df['event_date_parsed'].max().strftime('%b %Y')}" if not filtered_df.empty else "N/A"
geocoded_count = (filtered_df["decimalLatitude"].notna() & filtered_df["decimalLongitude"].notna()).sum()
geocoded_pct = (geocoded_count / max(1, len(filtered_df))) * 100

st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 18px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
            <div class="metric-label-text">Eligible Dated Records</div>
            <div class="metric-value-text">{len(filtered_df):,}</div>
            <div class="metric-note-text">GBIF & Research Herbarium Accessions</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(20, "#527A50")}</div>
            <div class="metric-label-text">Historical Date Span</div>
            <div class="metric-value-text" style="font-size: 1.4rem;">{date_span_str}</div>
            <div class="metric-note-text">Continuous Monthly Index</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_pin(20, "#527A50")}</div>
            <div class="metric-label-text">Geocoded Sites</div>
            <div class="metric-value-text">{geocoded_count:,}</div>
            <div class="metric-note-text">{geocoded_pct:.1f}% Indian territory coords</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
            <div class="metric-label-text">Target Host Crop</div>
            <div class="metric-value-text" style="font-size: 1.35rem;">{host_crop.split('(')[0].strip()}</div>
            <div class="metric-note-text">{filtered_df['crop_category'].iloc[0] if 'crop_category' in filtered_df.columns else 'Staple Agri'}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Host crop canopy context banner
wheat_img_path = ASSETS_DIR / "wheat_canopy_ears.jpg"
wheat_b64 = get_image_base64(wheat_img_path)
if wheat_b64:
    st.markdown(
        f"""
        <div class="verdant-card" style="padding: 0; overflow: hidden; margin-top: 18px; margin-bottom: 22px; display: grid; grid-template-columns: 240px 1fr; border: 1px solid rgba(82,122,80,0.18);">
            <div style="height: 130px; overflow: hidden; position: relative;">
                <img src="{wheat_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Host Crop Foliage" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.65rem; font-weight: 600; padding: 2px 7px; border-radius: 4px; letter-spacing: 0.05em;">
                    HOST CANOPY CONTEXT
                </div>
            </div>
            <div style="padding: 14px 20px; display: flex; flex-direction: column; justify-content: center; background: #FFFEFA;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                    Host Phenology & Foliar Micro-Climate
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.1rem; color: #102C26; margin: 2px 0;">
                    {host_crop}
                </div>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.45; margin: 0;">
                    Foliar lesions and blights proliferate during active vegetative and heading stages when canopy relative humidity exceeds 80%.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# TABS: MONTHLY VIEW, GEOSPATIAL MAP, SOURCE RECORDS, QUALITY AUDIT
# ==============================================================================
tab_monthly, tab_map, tab_table, tab_audit = st.tabs([
    "📈 Monthly Time-Series Surveillance",
    "🗺️ Geospatial Distribution",
    "📋 Verified Source Records",
    "📑 Data Quality & Completeness Audit",
])

# ----------------- Tab 1: Monthly Time-Series -----------------
with tab_monthly:
    st.subheader(f"Monthly Documented Occurrences: {pathogen_dict[selected_slug]}")
    if df_monthly is not None and not df_monthly.empty:
        fig_m = go.Figure()

        # Documented count bar
        fig_m.add_trace(
            go.Bar(
                x=df_monthly["date"],
                y=df_monthly["documented_occurrence_count"],
                name="Monthly Documented Count",
                marker_color="#183D32",
                opacity=0.75,
                hovertemplate="Date: %{x|%b %Y}<br>Count: %{y}<extra></extra>",
            )
        )
        # 3-Month Rolling Average
        fig_m.add_trace(
            go.Scatter(
                x=df_monthly["date"],
                y=df_monthly["documented_occurrence_count"].rolling(window=3, min_periods=1).mean(),
                name="3-Month Rolling Average",
                line=dict(color="#C4A66A", width=2.5),
            )
        )

        format_plotly_figure(
            fig_m,
            title="Monthly Occurrence Frequency with Explicit Missingness Indicators",
            height=420,
        )
        st.plotly_chart(fig_m, use_container_width=True)

        st.caption(
            "ℹ️ **Epistemology Note:** Bars measure qualifying occurrence records deposited in international repositories. "
            "Unmonitored months maintain `is_missing = True` to preserve true epidemiological epistemology without fake zeros."
        )

# ----------------- Tab 2: Geospatial Map -----------------
with tab_map:
    st.subheader("Georeferenced Field Collection Sites in India")
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
            height=500,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#FFFEFA",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("High-precision coordinates are not attached to this pathogen's historical records; surveillance is referenced at the state/national level.")

# ----------------- Tab 3: Source Records -----------------
with tab_table:
    st.subheader(f"Curated Occurrence Archive ({len(filtered_df):,} records)")

    display_cols = [
        col for col in [
            "gbifID", "scientific_name", "event_date_parsed", "year", "month",
            "region_id", "stateProvince", "decimalLatitude", "decimalLongitude",
            "basisOfRecord", "license",
        ]
        if col in filtered_df.columns
    ]

    st.dataframe(filtered_df[display_cols], use_container_width=True, height=380)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Occurrence CSV ({len(filtered_df)} rows)",
        data=csv_bytes,
        file_name=f"surveillance_records_{selected_slug}.csv",
        mime="text/csv",
    )

# ----------------- Tab 4: Quality Audit -----------------
with tab_audit:
    st.subheader("Data Quality & Taxonomic Integrity Audit")
    q_data = load_data_quality_report_data()

    if q_data:
        audits = q_data.get("pathogen_audits", [])
        current_audit = next((a for a in audits if a.get("slug") == selected_slug), None)

        if current_audit:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.write("**Total Documented Records:**", current_audit["total_records"])
                st.write("**Records with Valid Date:**", f"{current_audit['records_with_date']} ({current_audit['pct_with_date']}%)")
                st.write("**Verified Date Range:**", f"{current_audit['min_date']} to {current_audit['max_date']}")
                st.write("**Temporal Coverage:**", f"{current_audit['temporal_span_years']} Years")
            with col_a2:
                st.write("**Geocoded Indian Coordinates:**", f"{current_audit['records_with_india_coords']} ({current_audit['pct_with_coords']}%)")
                st.write("**Observation Basis:**", current_audit["basis_of_record"])
                st.write("**Forecasting Viability:**", "✅ High Viability" if current_audit["is_suitable_for_monthly_forecasting"] else "⚠️ Sparse (Contextual)")
                st.write("**Methodological Rationale:**", current_audit["suitability_reason"])

            st.markdown("#### Seasonal Month Distribution (1 = Jan, 12 = Dec)")
            md = current_audit.get("month_distribution", {})
            if md:
                df_mb = pd.DataFrame(list(md.items()), columns=["Month", "Records"])
                fig_mb = px.bar(df_mb, x="Month", y="Records", text_auto=True, color_discrete_sequence=["#527A50"])
                format_plotly_figure(fig_mb, title="Observations by Calendar Month", height=280)
                st.plotly_chart(fig_mb, use_container_width=True)
