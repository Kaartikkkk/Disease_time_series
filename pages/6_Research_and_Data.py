"""Page 6: Research & Data — Public Provenance, Species Roster, and Open Export Center."""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.icons import (
    icon_book,
    icon_database,
    icon_download,
    icon_leaf,
    icon_shield,
    icon_weather,
)
from src.reporting import generate_methodology_report
from src.ui_helpers import (
    inject_custom_css,
    render_app_sidebar,
    render_top_header,
)

st.set_page_config(
    page_title="Research & Data — Verdant Intelligence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="research_data")
render_top_header()

# Editorial Header
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_book(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">Research Data &amp; Open Access Repositories</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Verified biological specimen registries, NASA POWER climate telemetry, species taxonomy rosters, and exportable datasets.
            </p>
        </div>
        <div style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 8px; padding: 6px 14px; font-size: 0.82rem; color: #183D32; font-family: 'JetBrains Mono', monospace;">
            DATA CITATION · CC-BY 4.0 / CC0
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_sources, tab_roster, tab_exports = st.tabs([
    "📡 Data Sources & Provenance",
    "🌿 Pathogen Species Roster",
    "📥 Open Data Export Center",
])

# ----------------- Tab 1: Data Sources & Provenance -----------------
with tab_sources:
    st.markdown("### Public Data Provenance & API Specifications")
    st.markdown(
        "All telemetry ingested into Verdant Intelligence originates strictly from verified, open-access "
        "intergovernmental and scientific institutions with reproducible REST query parameters."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    {icon_leaf(22, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">GBIF Occurrence API</h3>
                </div>
                <div style="display: inline-block; background: rgba(82, 122, 80, 0.1); border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 12px;">
                    CC-BY 4.0 / CC0 Open Data
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5; margin-bottom: 12px;">
                    Primary biodiversity and plant pathogen occurrence repository maintained by the Global Biodiversity Information Facility Secretariat.
                </p>
                <div style="font-size: 0.8rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Endpoint:</strong> <code>api.gbif.org/v1/occurrence/search</code><br>
                    <strong>Parameters:</strong> <code>country=IN</code>, <code>year=2015,2026</code><br>
                    <strong>Taxon Matching:</strong> GBIF Backbone Taxonomy
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    {icon_weather(22, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">NASA POWER Climate</h3>
                </div>
                <div style="display: inline-block; background: rgba(82, 122, 80, 0.1); border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 12px;">
                    NASA Open Data Policy
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5; margin-bottom: 12px;">
                    Prediction of Worldwide Energy Resources agrometeorology service driven by MERRA-2 assimilation and GEOS-5 atmospheric models.
                </p>
                <div style="font-size: 0.8rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Endpoint:</strong> <code>power.larc.nasa.gov/api/temporal/monthly/point</code><br>
                    <strong>Features:</strong> <code>T2M</code>, <code>T2M_MAX</code>, <code>T2M_MIN</code>, <code>PRECTOTCORR</code>, <code>RH2M</code>, <code>T2MDEW</code><br>
                    <strong>Grid Resolution:</strong> 0.5° × 0.625°
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    {icon_shield(22, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">EPPO Global Database</h3>
                </div>
                <div style="display: inline-block; background: rgba(82, 122, 80, 0.1); border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 12px;">
                    European &amp; Med. Plant Protection
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5; margin-bottom: 12px;">
                    Official phytosanitary alert service and quarantine records confirming pest presence, geographical distribution, and host taxonomy in India.
                </p>
                <div style="font-size: 0.8rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Role:</strong> Qualitative Ground-Truth Validation<br>
                    <strong>Separation:</strong> Outbreak Event Date vs Publication Date<br>
                    <strong>Coverage:</strong> 10 Key Agricultural Taxa
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- Tab 2: Pathogen Species Roster -----------------
with tab_roster:
    st.markdown("### Ingested Indian Agricultural Pathogens Roster")
    st.markdown(
        "Standardized inventory of target crop pathogens analyzed in the continuous 2015–2026 time series."
    )
    taxa_table = [
        {"Pathogen Taxon": "Fusarium oxysporum", "Common Disease": "Wilt / Root Rot", "Primary Host Crops": "Tomato, Banana, Chickpea, Cotton", "Post-2015 Records": "1,180", "Status": "Active Series"},
        {"Pathogen Taxon": "Ralstonia solanacearum", "Common Disease": "Bacterial Wilt", "Primary Host Crops": "Potato, Tomato, Eggplant, Ginger", "Post-2015 Records": "640", "Status": "Active Series"},
        {"Pathogen Taxon": "Pyricularia oryzae (Magnaporthe)", "Common Disease": "Rice Blast", "Primary Host Crops": "Paddy / Rice, Finger Millet", "Post-2015 Records": "312", "Status": "Active Series"},
        {"Pathogen Taxon": "Bipolaris oryzae", "Common Disease": "Brown Spot of Rice", "Primary Host Crops": "Paddy / Rice", "Post-2015 Records": "184", "Status": "Active Series"},
        {"Pathogen Taxon": "Alternaria solani", "Common Disease": "Early Blight", "Primary Host Crops": "Potato, Tomato", "Post-2015 Records": "142", "Status": "Active Series"},
        {"Pathogen Taxon": "Phytophthora infestans", "Common Disease": "Late Blight", "Primary Host Crops": "Potato, Tomato", "Post-2015 Records": "128", "Status": "Active Series"},
        {"Pathogen Taxon": "Xanthomonas oryzae", "Common Disease": "Bacterial Blight", "Primary Host Crops": "Paddy / Rice", "Post-2015 Records": "115", "Status": "Active Series"},
        {"Pathogen Taxon": "Colletotrichum falcatum", "Common Disease": "Red Rot", "Primary Host Crops": "Sugarcane", "Post-2015 Records": "98", "Status": "Active Series"},
        {"Pathogen Taxon": "Magnaporthe grisea", "Common Disease": "Blast Disease Complex", "Primary Host Crops": "Millets, Grasses", "Post-2015 Records": "82", "Status": "Active Series"},
        {"Pathogen Taxon": "Puccinia graminis", "Common Disease": "Black Stem Rust", "Primary Host Crops": "Wheat, Barley", "Post-2015 Records": "0 (Controlled)", "Status": "Screened / Zero Post-2015"},
    ]
    st.dataframe(pd.DataFrame(taxa_table), width="stretch")

# ----------------- Tab 3: Open Data Export Center -----------------
with tab_exports:
    st.markdown("### Processed Datasets, Quality Audits & Formal Reports")
    st.markdown(
        "Download cleaned, standardized monthly series, validation benchmarking matrices, "
        "and data quality audit certificates for academic replication and field deployment."
    )

    ec1, ec2 = st.columns(2)

    with ec1:
        st.markdown(
            f"""
            <div class="verdant-card" style="margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_download(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.1rem; color: #102C26;">Unified Monthly Multi-Pathogen Time Series</h3>
                </div>
                <p style="font-size: 0.85rem; color: #5A665D; margin-bottom: 12px;">
                    Continuous monthly sequence (2015–2025) across all 10 Indian agricultural pathogens with zero-filled gaps, monthly totals, and regional breakdown.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        combined_path = Path("data/processed/india_plant_diseases_monthly.csv")
        if combined_path.exists():
            with open(combined_path, "rb") as f:
                st.download_button(
                    label="📥 Download Unified Monthly Dataset (CSV)",
                    data=f.read(),
                    file_name="india_plant_diseases_monthly.csv",
                    mime="text/csv",
                    width="stretch",
                )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 20px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_download(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.1rem; color: #102C26;">Data Quality &amp; Integrity Audit Certificate</h3>
                </div>
                <p style="font-size: 0.85rem; color: #5A665D; margin-bottom: 12px;">
                    JSON certificate cataloging completeness, missing date diagnostics, bounding box compliance, and schema verification scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        q_path = Path("data/reports/data_quality_report.json")
        if q_path.exists():
            with open(q_path, "rb") as f:
                st.download_button(
                    label="📥 Download Quality Audit Certificate (JSON)",
                    data=f.read(),
                    file_name="data_quality_report.json",
                    mime="application/json",
                    width="stretch",
                )

    with ec2:
        st.markdown(
            f"""
            <div class="verdant-card" style="margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_download(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.1rem; color: #102C26;">Scientific Methodology &amp; System Manifesto</h3>
                </div>
                <p style="font-size: 0.85rem; color: #5A665D; margin-bottom: 12px;">
                    Exhaustive markdown whitepaper covering data collection parameters, mathematical formulas, model hyperparameters, and citations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m_path = Path("data/reports/methodology_report.md")
        if m_path.exists():
            with open(m_path, "rb") as f:
                st.download_button(
                    label="📥 Download Methodology Whitepaper (Markdown)",
                    data=f.read(),
                    file_name="methodology_report.md",
                    mime="text/markdown",
                    width="stretch",
                )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 20px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_download(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.1rem; color: #102C26;">System Architecture &amp; Model Schema</h3>
                </div>
                <p style="font-size: 0.85rem; color: #5A665D; margin-bottom: 12px;">
                    Machine-readable schema describing full data processing pipeline, feature engineering definitions, and validation splits.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mj_path = Path("data/reports/methodology_report.json")
        if mj_path.exists():
            with open(mj_path, "rb") as f:
                st.download_button(
                    label="📥 Download System Architecture Specification (JSON)",
                    data=f.read(),
                    file_name="methodology_report.json",
                    mime="application/json",
                    width="stretch",
                )

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Verdant Intelligence · Plant Disease Outbreak Forecasting · Open Science & Reproducible Research")
