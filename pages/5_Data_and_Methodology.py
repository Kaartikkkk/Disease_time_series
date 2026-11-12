"""Page 6: About the Data — Provenance, How Predictions Work, Downloads & Responsible Use."""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.icons import (
    icon_book,
    icon_check_circle,
    icon_database,
    icon_download,
    icon_leaf,
    icon_model_compare,
    icon_settings,
    icon_shield,
    icon_table,
    icon_target,
    icon_timeseries,
    icon_weather,
)
from src.reporting import generate_methodology_report
from src.ui_helpers import (
    get_config_settings,
    inject_custom_css,
    is_advanced_mode,
    load_data_quality_report_data,
    render_app_sidebar,
    render_top_header,
)

st.set_page_config(
    page_title="Data & Methodology — Agricultural Disease Forecasting Using Time Series",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_app_sidebar(active_page="methodology")
render_top_header()

adv = is_advanced_mode()

# Header
title_text = "Data &amp; Methodology" if not adv else "Data Provenance, Econometric Architecture &amp; Governance"
subtitle_text = (
    "Learn where our plant disease and satellite weather data comes from, how forecasting models operate, and how to interpret predictions responsibly."
    if not adv
    else "Verified biodiversity accessions (GBIF), NASA POWER agrometeorology, mathematical model formulations, and reproducible data exports."
)
badge_text = "OPEN ACCESS · CC-BY 4.0 / CC0" if not adv else "PEER REVIEW ARCHITECTURE · CC-BY 4.0"

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div style="flex: 1 1 300px; min-width: 250px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                {icon_book(26, "#527A50")}
                <h1 style="font-size: clamp(1.6rem, 3.2vw, 2.2rem); margin: 0; color: #102C26; word-break: normal; hyphens: none; line-height: 1.2;">{title_text}</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0; line-height: 1.45;">
                {subtitle_text}
            </p>
        </div>
        <div style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 8px; padding: 6px 14px; font-size: 0.82rem; color: #183D32; font-family: 'JetBrains Mono', monospace; flex-shrink: 0; white-space: nowrap; align-self: flex-start;">
            {badge_text}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_sources, tab_prep, tab_methods, tab_eval, tab_downloads = st.tabs([
    "Data Sources & Coverage",
    "Preprocessing & Missing Values",
    "Model Methodology",
    "Evaluation & Responsible Use",
    "Downloads & References",
])

# ----------------- Tab 1: Data Sources & Coverage -----------------
with tab_sources:
    st.subheader("Public Data Sources & Field Surveillance")
    st.markdown("All occurrence records and satellite meteorological telemetry originate from verified open-access institutions.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    {icon_leaf(22, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">GBIF Pathogen Records</h3>
                </div>
                <div style="display: inline-block; background: rgba(82, 122, 80, 0.1); border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 12px;">
                    Global Biodiversity Facility · Open Data
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5; margin-bottom: 12px;">
                    Verified research plant pathogen records collected by agricultural universities and botanical herbaria.
                </p>
                <div style="font-size: 0.8rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Source:</strong> GBIF India (2015–2026)<br>
                    <strong>Verification:</strong> Taxonomically matched specimens
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
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">NASA Weather Data</h3>
                </div>
                <div style="display: inline-block; background: rgba(82, 122, 80, 0.1); border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 12px;">
                    NASA POWER Climate Service
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5; margin-bottom: 12px;">
                    Monthly temperature, precipitation, and relative humidity aligned with crop zones.
                </p>
                <div style="font-size: 0.8rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Source:</strong> NASA Langley Center<br>
                    <strong>Features:</strong> Temp, Rain, Relative Humidity (132 continuous months)
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
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">EPPO Plant Health Alert</h3>
                </div>
                <div style="display: inline-block; background: rgba(82, 122, 80, 0.1); border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 12px;">
                    European &amp; Med. Plant Protection
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5; margin-bottom: 12px;">
                    Quarantine lists and surveillance alerts used to confirm host associations and outbreak dates.
                </p>
                <div style="font-size: 0.8rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Role:</strong> Qualitative ground-truth confirmation<br>
                    <strong>Coverage:</strong> Major Indian staple and cash crops
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Monitored Plant Diseases & Dataset Coverage")
    settings = get_config_settings()
    pathogens_cfg = settings.get("pathogens", {})
    quality_report = load_data_quality_report_data() or {}
    audits_by_slug = {a.get("slug"): a for a in quality_report.get("pathogen_audits", [])}

    taxa_table = []
    for slug, info in pathogens_cfg.items():
        audit = audits_by_slug.get(slug, {})
        rec_count = audit.get("records_with_date", audit.get("total_records", "N/A"))
        taxa_table.append({
            "Common Disease Name": info.get("common_name", slug),
            "Host Crops": info.get("host_crop", "N/A"),
            "Scientific Name": info.get("scientific_name", ""),
            "Dated Records (2015–2025)": f"{rec_count:,}" if isinstance(rec_count, int) else str(rec_count),
            "Status": "Active Series" if audit.get("is_suitable_for_monthly_forecasting", True) else "Sparse / Contextual Series",
        })
    st.dataframe(pd.DataFrame(taxa_table), use_container_width=True, hide_index=True)

# ----------------- Tab 2: Preprocessing & Missing Values -----------------
with tab_prep:
    st.subheader("Data Preprocessing & Missing-Value Protocols")
    
    col_pr1, col_pr2 = st.columns(2)
    with col_pr1:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    {icon_settings(20, "#527A50")}
                    <h4 style="margin: 0; color: #102C26; font-size: 1.1rem;">Preprocessing Pipeline</h4>
                </div>
                <ul style="font-size: 0.86rem; color: #404B43; line-height: 1.6; padding-left: 20px; margin: 0;">
                    <li><strong>Temporal Aggregation:</strong> Parsed via ISO 8601 to calendar Month-Start (<code>MS</code>) frequency.</li>
                    <li><strong>Deduplication:</strong> Cryptographic hashing across ID, timestamp, and coordinates eliminates duplicates.</li>
                    <li><strong>Spatial Bounding:</strong> Filtered strictly within India (6.0°N–37.5°N, 68.0°E–97.5°E).</li>
                    <li><strong>Outlier Fences:</strong> Tukey fences ($Q_3 + 1.5 \\times \\text{{IQR}}$) identify anomalous sampling surges.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_pr2:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    {icon_shield(20, "#527A50")}
                    <h4 style="margin: 0; color: #102C26; font-size: 1.1rem;">Missing-Value Policy</h4>
                </div>
                <ul style="font-size: 0.86rem; color: #404B43; line-height: 1.6; padding-left: 20px; margin: 0;">
                    <li><strong>Explicit Tracking:</strong> Unmonitored intervals flagged with <code>is_missing = True</code> rather than assumed zero.</li>
                    <li><strong>Seasonal Reality:</strong> Off-season dry months naturally feature low reports due to absent foliage.</li>
                    <li><strong>Bias Prevention:</strong> Explicit tracking prevents downward bias in out-of-sample metrics.</li>
                    <li><strong>Index Regularity:</strong> Maintains continuous monthly sequence across the 132-month horizon.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- Tab 3: Model Methodology -----------------
with tab_methods:
    st.subheader("Model Architectures & Mathematical Foundations")
    
    methods_df = pd.DataFrame([
        {"Model": "ARIMA(1,1,1)", "Family": "Classical Econometrics", "Formulation": "d=1 differencing + AR(1) + MA(1)", "Primary Role": "Eliminates stochastic drift; models monthly persistence."},
        {"Model": "SARIMA(1,0,1)(1,0,1)[12]", "Family": "Seasonal Econometrics", "Formulation": "ARIMA + 12-month seasonal polynomial", "Primary Role": "Captures annual Kharif monsoon and Rabi winter surges."},
        {"Model": "Holt-Winters", "Family": "Exponential Smoothing", "Formulation": "Level (alpha) + Trend (beta) + Additive Season (gamma)", "Primary Role": "Adaptive exponential smoothing responding to recent shifts."},
        {"Model": "ARIMAX / SARIMAX", "Family": "Exogenous Econometrics", "Formulation": "SARIMA + Lagged (t-1) Temp, Rain, RH", "Primary Role": "Integrates satellite weather without lookahead data leakage."},
        {"Model": "Seasonal Naive", "Family": "Deterministic Benchmark", "Formulation": "y(t+h) = y(t+h-12)", "Primary Role": "Academic baseline carrying forward last year's same-month value."},
        {"Model": "Random Forest / GBDT", "Family": "Machine Learning Ensemble", "Formulation": "Ensemble of orthogonal decision splits", "Primary Role": "Models non-linear thresholds in humidity and temperature lags."},
    ])
    st.dataframe(methods_df, use_container_width=True, hide_index=True)

# ----------------- Tab 4: Evaluation & Responsible Use -----------------
with tab_eval:
    st.subheader("Validation Framework & Responsible Use")
    
    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    {icon_shield(20, "#527A50")}
                    <h4 style="margin: 0; color: #102C26; font-size: 1.1rem;">Chronological Testing Protocol</h4>
                </div>
                <p style="font-size: 0.86rem; color: #404B43; line-height: 1.55; margin-bottom: 10px;">
                    All models are evaluated strictly chronologically to eliminate future lookahead data leakage:
                </p>
                <ul style="font-size: 0.84rem; color: #5A665D; line-height: 1.6; padding-left: 20px; margin: 0;">
                    <li><strong>Training Set (70%):</strong> Historical baseline calibration (Jan 2015 – Dec 2022).</li>
                    <li><strong>Validation Set (15%):</strong> Hyperparameter tuning and lag order selection (2023).</li>
                    <li><strong>Test Set (15%):</strong> Strictly unseen final evaluation (2024–2025).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ev2:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                    {icon_target(20, "#527A50")}
                    <h4 style="margin: 0; color: #102C26; font-size: 1.1rem;">Zero-Safe Error Metrics</h4>
                </div>
                <p style="font-size: 0.86rem; color: #404B43; line-height: 1.55; margin-bottom: 10px;">
                    Because plant disease counts frequently drop to zero during dry off-season periods, standard MAPE is mathematically undefined:
                </p>
                <ul style="font-size: 0.84rem; color: #5A665D; line-height: 1.6; padding-left: 20px; margin: 0;">
                    <li><strong>sMAPE:</strong> Symmetric MAPE bounded in [0%, 200%], stable during zero-report months.</li>
                    <li><strong>WAPE:</strong> Weighted Absolute Percentage Error ($\\sum |e_t| / \\sum y_t$).</li>
                    <li><strong>RMSE & MAE:</strong> Standardized report count error metrics.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Responsible Agricultural Use Guidelines")
    
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
            <div class="verdant-card" style="border-left: 4px solid #102C26;">
                <h4 style="margin: 0 0 4px 0; color: #102C26; font-size: 0.98rem;">1. Documented Reports vs. Total Acreage</h4>
                <p style="font-size: 0.82rem; color: #404B43; line-height: 1.4; margin: 0;">
                    Counts reflect verified research accessions, not an absolute census of affected farm acreage.
                </p>
            </div>
            <div class="verdant-card" style="border-left: 4px solid #2563EB;">
                <h4 style="margin: 0 0 4px 0; color: #102C26; font-size: 0.98rem;">2. Weather Conditions vs. Infection</h4>
                <p style="font-size: 0.82rem; color: #404B43; line-height: 1.4; margin: 0;">
                    Permissive weather enables pathogen growth, but active disease requires susceptible crop phenology.
                </p>
            </div>
            <div class="verdant-card" style="border-left: 4px solid #527A50;">
                <h4 style="margin: 0 0 4px 0; color: #102C26; font-size: 0.98rem;">3. Cultivar Resistance Impact</h4>
                <p style="font-size: 0.82rem; color: #404B43; line-height: 1.4; margin: 0;">
                    Zero post-2015 reports for Wheat Stem Rust reflect successful nationwide adoption of resistant cultivars.
                </p>
            </div>
            <div class="verdant-card" style="border-left: 4px solid #C4A66A;">
                <h4 style="margin: 0 0 4px 0; color: #102C26; font-size: 0.98rem;">4. Predictive Uncertainty</h4>
                <p style="font-size: 0.82rem; color: #404B43; line-height: 1.4; margin: 0;">
                    Forecasts provide seasonal risk probabilities with 80%/95% confidence intervals, not deterministic guarantees.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- Tab 5: Downloads & References -----------------
with tab_downloads:
    st.subheader("Download Datasets & Artifacts")
    
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown(
            f"""
            <div class="verdant-card" style="margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    {icon_download(18, "#527A50")}
                    <h4 style="margin: 0; font-size: 1.05rem; color: #102C26;">Monthly Disease Dataset (CSV)</h4>
                </div>
                <p style="font-size: 0.84rem; color: #5A665D; margin-bottom: 10px;">
                    Continuous monthly records (2015–2025, 132 months) across 10 major Indian crop diseases with joined NASA POWER weather.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        combined_path = Path("data/processed/india_plant_diseases_monthly.csv")
        if combined_path.exists():
            with open(combined_path, "rb") as f:
                st.download_button(
                    label="Download Monthly Dataset (CSV)",
                    data=f.read(),
                    file_name="india_plant_diseases_monthly.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 18px; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    {icon_download(18, "#527A50")}
                    <h4 style="margin: 0; font-size: 1.05rem; color: #102C26;">Data Quality Audit (JSON)</h4>
                </div>
                <p style="font-size: 0.84rem; color: #5A665D; margin-bottom: 10px;">
                    Integrity checks, date verification percentages, and location bounding scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        q_path = Path("data/reports/data_quality_report.json")
        if q_path.exists():
            with open(q_path, "rb") as f:
                st.download_button(
                    label="Download Quality Certificate (JSON)",
                    data=f.read(),
                    file_name="data_quality_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

    with ec2:
        st.markdown(
            f"""
            <div class="verdant-card" style="margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    {icon_download(18, "#527A50")}
                    <h4 style="margin: 0; font-size: 1.05rem; color: #102C26;">Methodology Whitepaper (Markdown)</h4>
                </div>
                <p style="font-size: 0.84rem; color: #5A665D; margin-bottom: 10px;">
                    Full documentation detailing collection parameters, mathematical formulas, and scientific citations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m_path = Path("data/reports/methodology_report.md")
        if m_path.exists():
            with open(m_path, "rb") as f:
                st.download_button(
                    label="Download Methodology Whitepaper (Markdown)",
                    data=f.read(),
                    file_name="methodology_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 18px; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    {icon_download(18, "#527A50")}
                    <h4 style="margin: 0; font-size: 1.05rem; color: #102C26;">Architecture Specification (JSON)</h4>
                </div>
                <p style="font-size: 0.84rem; color: #5A665D; margin-bottom: 10px;">
                    Complete data pipeline definitions, feature engineering settings, and split indices.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        mj_path = Path("data/reports/methodology_report.json")
        if mj_path.exists():
            with open(mj_path, "rb") as f:
                st.download_button(
                    label="Download Architecture Spec (JSON)",
                    data=f.read(),
                    file_name="methodology_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Academic References & Literature")
    references = [
        "Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). John Wiley & Sons.",
        "Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts: Melbourne, Australia.",
        "GBIF Secretariat (2024). *GBIF Backbone Taxonomy*. Checklist dataset https://doi.org/10.15468/39omei accessed via GBIF.org.",
        "NASA POWER Project (2024). *Prediction Of Worldwide Energy Resources*. NASA Langley Research Center, https://power.larc.nasa.gov.",
        "EPPO (2024). *EPPO Global Database*. European and Mediterranean Plant Protection Organization, https://gd.eppo.int.",
    ]
    for ref in references:
        st.markdown(f"- <span style='font-size: 0.86rem; color: #404B43;'>{ref}</span>", unsafe_allow_html=True)

# Page Conclusion
st.markdown(
    f"""
    <div class="verdant-card" style="background: #F6F4EC; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 14px; padding: 20px 24px; margin-top: 24px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            {icon_target(18, "#183D32")}
            <h4 style="margin: 0; color: #183D32; font-size: 1.05rem;">Data & Methodology Conclusion</h4>
        </div>
        <p style="margin: 0; font-size: 0.88rem; color: #5A665D; line-height: 1.6;">
            Agricultural Disease Forecasting Using Time Series is built upon open-science data provenance, rigorous chronological validation, and zero-safe percentage error metrics. By uniting 11 years of verified GBIF occurrences with NASA POWER satellite meteorology, the platform provides an academically sound and reproducible forecasting framework for Indian agriculture.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

