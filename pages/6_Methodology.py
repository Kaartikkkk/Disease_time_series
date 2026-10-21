"""Page 6: Scientific Methodology, Data Governance, Mathematical Formulations & Exports."""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.icons import (
    icon_download,
    icon_leaf,
    icon_model_compare,
    icon_shield,
    icon_timeseries,
    icon_weather,
)
from src.reporting import generate_methodology_report
from src.ui_helpers import inject_custom_css

st.set_page_config(
    page_title="Methodology & Governance — Verdant Intelligence",
    page_icon="📜",
    layout="wide",
)

inject_custom_css()

# Editorial Header
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_shield(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">Methodology, Data Governance & Exports</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Rigorous scientific architecture, public API endpoints, mathematical model formulations, and reproducible data exports.
            </p>
        </div>
        <div style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 8px; padding: 6px 14px; font-size: 0.82rem; color: #183D32; font-family: 'JetBrains Mono', monospace;">
            GOVERNANCE · CC-BY 4.0 / CC0
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

report_data = generate_methodology_report()

tab_sources, tab_methods, tab_limits, tab_exports = st.tabs([
    "📡 Data Sources & Attribution",
    "🔬 Statistical & Modeling Framework",
    "🛡️ Scientific Limitations & Responsible AI",
    "📥 Data Export Center",
])

# ----------------- Tab 1: Data Sources & Attribution -----------------
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
                    European & Med. Plant Protection
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

    st.markdown("---")
    st.markdown("#### Ingested Indian Pathogen Species Roster")
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
    st.dataframe(pd.DataFrame(taxa_table), use_container_width=True)

# ----------------- Tab 2: Modeling Framework -----------------
with tab_methods:
    st.markdown("### Econometric & Machine Learning Forecasting Architecture")

    f1, f2 = st.columns([1, 1])
    with f1:
        st.markdown(
            f"""
            <div class="verdant-card">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_timeseries(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">Target Variable Definition</h3>
                </div>
                <p style="font-size: 0.9rem; color: #404B43; line-height: 1.5;">
                    The forecasting target is strictly defined as monthly <code>documented_occurrence_count</code> ($y_t$):
                </p>
                <div style="background: rgba(16, 44, 38, 0.04); border-left: 3px solid #527A50; padding: 10px 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; margin: 10px 0;">
                    y_t = ∑ 1 for all occurrences i ∈ Month t
                </div>
                <p style="font-size: 0.84rem; color: #5A665D;">
                    This captures verified surveillance intensity and biological discoveries rather than unobserved absolute field incidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_model_compare(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">Classical Time-Series Suite</h3>
                </div>
                <ul style="font-size: 0.88rem; color: #404B43; padding-left: 20px; line-height: 1.6;">
                    <li><strong>Seasonal Naive:</strong> Baseline replicating the identical month from preceding annual cycle: $y_t = y_{{t-12}}$.</li>
                    <li><strong>ARIMA(p, d, q):</strong> Linear autoregression with difference order $d$ and moving average error innovations.</li>
                    <li><strong>SARIMA(p, d, q)(P, D, Q)₁₂:</strong> Incorporates 12-month annual seasonality reflecting India's Kharif (monsoon) and Rabi (winter) cropping calendars.</li>
                    <li><strong>Holt-Winters Exponential Smoothing:</strong> Additive Holt-Winters level, trend, and seasonal components with damped drift parameters.</li>
                    <li><strong>SARIMAX:</strong> Exogenous weather regressors ($T2M, PRECTOTCORR$) incorporated with strict historical lags ($t-1, t-2$) to avoid future lookahead bias.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:
        st.markdown(
            f"""
            <div class="verdant-card">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_shield(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">Validation & Chronological Splits</h3>
                </div>
                <p style="font-size: 0.9rem; color: #404B43; line-height: 1.5;">
                    To prevent synthetic data leakage, the evaluation strictly adheres to forward chronological splitting:
                </p>
                <div style="display: flex; gap: 8px; margin: 12px 0;">
                    <div style="flex: 7; background: rgba(82, 122, 80, 0.15); border: 1px solid #527A50; border-radius: 6px; padding: 8px; text-align: center; font-size: 0.8rem; font-weight: 600; color: #183D32;">
                        Train (70%)<br><span style="font-size: 0.72rem; font-weight: normal;">2015 – 2022</span>
                    </div>
                    <div style="flex: 1.5; background: rgba(196, 166, 106, 0.2); border: 1px solid #C4A66A; border-radius: 6px; padding: 8px; text-align: center; font-size: 0.8rem; font-weight: 600; color: #695221;">
                        Val (15%)<br><span style="font-size: 0.72rem; font-weight: normal;">2023</span>
                    </div>
                    <div style="flex: 1.5; background: rgba(16, 44, 38, 0.2); border: 1px solid #102C26; border-radius: 6px; padding: 8px; text-align: center; font-size: 0.8rem; font-weight: 600; color: #102C26;">
                        Test (15%)<br><span style="font-size: 0.72rem; font-weight: normal;">2024 – 2025</span>
                    </div>
                </div>
                <p style="font-size: 0.84rem; color: #5A665D;">
                    Zero cross-validation shuffling is applied; models are tested exclusively against future unseen horizons.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_download(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">Zero-Safe Error Metrics</h3>
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5;">
                    Plant disease series frequently exhibit zero monthly observations during dry non-host months. Standard MAPE divides by zero and produces infinity. We employ mathematically bounded metrics:
                </p>
                <ul style="font-size: 0.85rem; color: #404B43; padding-left: 20px; line-height: 1.6;">
                    <li><strong>sMAPE (Symmetric MAPE):</strong> Bounded in $[0\\%, 200\\%]$:
                        $$\\text{{sMAPE}} = \\frac{{100\\%}}{{n}} \\sum_{{t=1}}^n \\frac{{2 |y_t - \\hat{{y}}_t|}}{{|y_t| + |\\hat{{y}}_t|}}$$
                    </li>
                    <li><strong>WAPE (Weighted Absolute Percentage Error):</strong> Normalized by total aggregate observations:
                        $$\\text{{WAPE}} = \\frac{{\\sum |y_t - \\hat{{y}}_t|}}{{\\sum |y_t|}}$$
                    </li>
                    <li><strong>RMSE:</strong> Penalizes severe outbreak underestimation.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- Tab 3: Scientific Limitations -----------------
with tab_limits:
    st.markdown("### Scientific Safeguards & Responsible AI Guidelines")
    st.markdown(
        "To ensure responsible operational deployment, agricultural decision-makers must evaluate these "
        "time-series projections within established epidemiologic and computational boundaries."
    )

    st.markdown(
        """
        > [!IMPORTANT]
        > ### 1. Passive Surveillance & Reporting Sampling Bias
        > GBIF biological records aggregate specimen deposits from ICAR agricultural universities, State Agricultural 
        > Departments, and verified citizen scientists. A peak in documented occurrences can reflect an intensive 
        > regional field sampling expedition rather than an immediate biological outbreak. The model forecasts 
        > **documented pathogen discovery activity**, which acts as a leading indicator of regional epidemiological risk.

        > [!NOTE]
        > ### 2. Permissive Micro-Climate vs Direct Inoculum Availability
        > Agro-meteorological variables (surface temperature, dew point, relative humidity, precipitation) delineate 
        > **physiologically permissive infection windows**. However, actual disease manifestation requires three simultaneous 
        > components of the classical Disease Triangle: (1) viable virulent inoculum, (2) susceptible host crop phenology, 
        > and (3) permissive micro-climate. Weather correlations quantify environmental favorability, not absolute infection.

        > [!TIP]
        > ### 3. Absence of Wheat Stem Rust (*Puccinia graminis*) Post-2015
        > Ingested GBIF records show zero post-2015 occurrences for *Puccinia graminis* in India. This reflects 
        > substantial agricultural success: nationwide deployment of resistant wheat cultivars screened against stem rust 
        > (including Ug99 resistance initiatives by ICAR-IIWBR). The system accurately records this as zero observations 
        > rather than synthesizing artificial data.

        > [!WARNING]
        > ### 4. Forward Weather Climatology Scenarios
        > Forward forecasts extending past known NASA POWER satellite observations employ 10-year historical month-of-year 
        > climatological averages (±15% perturbation scenarios). They do not represent predictive long-range meteorological 
        > weather forecasts.
        """
    )

# ----------------- Tab 4: Export Center -----------------
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
                    use_container_width=True,
                )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 20px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_download(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.1rem; color: #102C26;">Data Quality & Integrity Audit Certificate</h3>
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
                    use_container_width=True,
                )

    with ec2:
        st.markdown(
            f"""
            <div class="verdant-card" style="margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_download(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.1rem; color: #102C26;">Scientific Methodology & System Manifesto</h3>
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
                    use_container_width=True,
                )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 20px; margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_download(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.1rem; color: #102C26;">System Architecture & Model Schema</h3>
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
                    use_container_width=True,
                )

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Verdant Intelligence · Plant Disease Outbreak Forecasting · Open Science & Reproducible Research")
