"""Page 5: Data & Methodology — Data Provenance, Econometric Architecture & Scientific Governance.

Organized into 8 distinct academic sections with expandable technical formulations.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.icons import (
    icon_book,
    icon_check_circle,
    icon_database,
    icon_download,
    icon_leaf,
    icon_model_compare,
    icon_pulse,
    icon_shield,
    icon_timeseries,
    icon_weather,
)
from src.project_stats import get_project_telemetry
from src.ui_helpers import (
    get_config_settings,
    inject_custom_css,
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

stats = get_project_telemetry()

# ==============================================================================
# EDITORIAL HEADER
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div style="flex: 1 1 300px; min-width: 250px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_book(26, "#527A50")}
                <h1 style="font-size: clamp(1.5rem, 3vw, 2.0rem); margin: 0; color: #102C26; word-break: normal; hyphens: none; line-height: 1.2;">Data Provenance, Architecture &amp; Methodology</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.92rem; margin: 0; line-height: 1.45;">
                Formal scientific documentation covering public biodiversity accessions, satellite agrometeorology, econometric formulations, and validation schemes.
            </p>
        </div>
        <div style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 8px; padding: 6px 14px; font-size: 0.80rem; color: #183D32; font-family: 'JetBrains Mono', monospace; flex-shrink: 0; align-self: flex-start;">
            OPEN ACCESS · CC-BY 4.0 / CC0
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# EIGHT ACADEMIC SECTIONS
# ==============================================================================
tab_sources, tab_coverage, tab_prep, tab_pipeline, tab_models, tab_eval, tab_limits, tab_downloads = st.tabs([
    "1. Data Sources",
    "2. Dataset Coverage",
    "3. Preprocessing & Zeros",
    "4. Methodology",
    "5. Model Architectures",
    "6. Evaluation & Validation",
    "7. Limitations",
    "8. Downloads & Citations",
])

# ----------------- 1. Data Sources -----------------
with tab_sources:
    st.markdown("### 1. Primary Data Sources & Provenance")
    st.markdown("All observational disease records and satellite meteorological observations originate from accredited public open-access institutions:")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%; padding: 18px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_leaf(22, "#527A50")}
                    <h4 style="margin: 0; font-size: 1.1rem; color: #102C26;">GBIF Pathogen Records</h4>
                </div>
                <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 8px;">
                    Global Biodiversity Information Facility
                </div>
                <p style="font-size: 0.84rem; color: #404B43; line-height: 1.5; margin-bottom: 10px;">
                    Taxonomically verified occurrence specimens and voucher accessions deposited by Indian agricultural universities, ICAR institutes, and global herbaria.
                </p>
                <div style="font-size: 0.76rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Access:</strong> REST API (v1) · <strong>License:</strong> CC-BY 4.0 / CC0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%; padding: 18px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_weather(22, "#527A50")}
                    <h4 style="margin: 0; font-size: 1.1rem; color: #102C26;">NASA POWER Climate</h4>
                </div>
                <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 8px;">
                    Prediction of Worldwide Energy Resources
                </div>
                <p style="font-size: 0.84rem; color: #404B43; line-height: 1.5; margin-bottom: 10px;">
                    Monthly agrometeorological reanalysis from CERES satellite observations and MERRA-2 meteorological assimilation over Indian agro-climatic centroids.
                </p>
                <div style="font-size: 0.76rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Regressors:</strong> Temp, Precip, RH, Dew Point (0.5° Grid)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="verdant-card" style="height: 100%; padding: 18px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_shield(22, "#527A50")}
                    <h4 style="margin: 0; font-size: 1.1rem; color: #102C26;">EPPO Global Database</h4>
                </div>
                <div style="font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: #183D32; margin-bottom: 8px;">
                    European &amp; Mediterranean Plant Protection Org
                </div>
                <p style="font-size: 0.84rem; color: #404B43; line-height: 1.5; margin-bottom: 10px;">
                    Official national phytosanitary status, regulatory reporting service notices, host plant range definitions, and quarantine distribution registers.
                </p>
                <div style="font-size: 0.76rem; color: #5A665D; border-top: 1px solid rgba(82,122,80,0.12); padding-top: 8px;">
                    <strong>Scope:</strong> National Geographic Presence Verification
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- 2. Dataset Coverage -----------------
with tab_coverage:
    st.markdown("### 2. Dataset Coverage & Temporal Reconciliation")
    st.markdown(
        f"""
        To eliminate ambiguity regarding dates across different application pages:
        - **Raw GBIF Occurrence Records**: Point voucher events deposited from research expeditions spanning sporadic historical years. In the Disease Explorer, individual pathogen accession collections show specific historical date spans based on available specimens.
        - **Standardized Monthly Forecasting Matrix**: A continuous, regularized calendar-month grid constructed across all 10 agricultural pathogens spanning **{stats['monthly_start_date']} to {stats['monthly_end_date']}** (exactly **{stats['total_months']} unbroken monthly intervals** per pathogen, totaling **{stats['total_monthly_records']:,} observations**).
        - **Future Projections**: Forward surveillance projections originate from the latest verified historical cutoff (**{stats['forecast_origin_date']}**), projecting forward into **{stats['forecast_start_date']}** across the user-selected horizon.
        """
    )
    cov_df = pd.DataFrame([
        {"Dataset Entity": "Raw GBIF Records Extracted", "Count / Value": f"{stats['raw_gbif_records']:,} Records", "Temporal Coverage": "Sporadic Historic Voucher Dates", "Role": "Source occurrence pool"},
        {"Dataset Entity": "Dated & Verified Accessions", "Count / Value": f"{stats['dated_gbif_records']:,} Records", "Temporal Coverage": "Verified Year & Month", "Role": "Surveillance events"},
        {"Dataset Entity": "Geocoded Indian GPS Points", "Count / Value": f"{stats['geocoded_gbif_records']:,} Points", "Temporal Coverage": "Valid Indian Bounding Box", "Role": "Spatial GIS mapping"},
        {"Dataset Entity": "Regularized Monthly Grid", "Count / Value": f"{stats['total_monthly_records']:,} Observations", "Temporal Coverage": f"{stats['monthly_start_date']} – {stats['monthly_end_date']} (132 Mo)", "Role": "Time-series forecasting"},
        {"Dataset Entity": "Validation Partition", "Count / Value": f"{stats['validation_steps']} Steps (4 Origins)", "Temporal Coverage": stats["validation_period"], "Role": "Model tuning & selection"},
        {"Dataset Entity": "Untouched Final Test Partition", "Count / Value": f"{stats['test_records']} Observations", "Temporal Coverage": stats["test_period"], "Role": "Locked holdout verification"},
    ])
    st.dataframe(cov_df, use_container_width=True, hide_index=True)

# ----------------- 3. Preprocessing & Zeros -----------------
with tab_prep:
    st.markdown("### 3. Epidemiological Zero Distribution & Feature Engineering")
    st.markdown(
        f"""
        Agricultural disease surveillance data naturally exhibits seasonal dormancy intervals due to crop rotation cycles, weather extremes, and host plant phenology:
        - **Seasonal Zero Proportion:** **{stats['zero_percentage']}%** of all pathogen-month observations document zero occurrence incidents during non-host or dormant periods.
        - **Dormancy Months ({stats['structural_zeros']} records):** Natural zero-incidence months during non-cropping seasons (e.g., dry summer months before Kharif planting).
        - **Confirmed Epidemic Observations ({stats['positive_observations']} records):** Active confirmed disease occurrence incidents recorded in verified surveillance archives.
        """
    )

    st.markdown("---")
    st.markdown("#### Feature Engineering & Dataset Architecture")
    st.markdown(
        "To elevate forecasting performance without data leakage or artificial observation synthesis, "
        "advanced temporal, autoregressive, and agrometeorological features were constructed from authentic historical records:"
    )

    comp_df = pd.DataFrame([
        {"Specification": "Master Dataset Location", "Feature Architecture": "data/disease_timeseries_master.csv"},
        {"Specification": "Total Observations & Duration", "Feature Architecture": "1,320 Rows (132 Months x 10 Pathogens)"},
        {"Specification": "Chronological Span", "Feature Architecture": "2015-01-01 to 2025-12-01 (11 Full Contiguous Years)"},
        {"Specification": "Feature Dimensions", "Feature Architecture": "43 Columns (Target + Identifiers + 39 Engineering Regressors)"},
        {"Specification": "Target Autoregressive Lags", "Feature Architecture": "Lags 1, 2, 3, and 12 (Seasonal Annual Cycle)"},
        {"Specification": "Shifted Rolling Statistics", "Feature Architecture": "3-Month & 6-Month Rolling Mean and Std (Lag-1, Zero-Leakage)"},
        {"Specification": "Momentum & Acceleration", "Feature Architecture": "Trailing 1-Month Difference (y_{t-1} - y_{t-2})"},
        {"Specification": "Trigonometric Harmonics", "Feature Architecture": "sin_month and cos_month Fourier Annual Oscillations"},
        {"Specification": "Agrometeorological Indices", "Feature Architecture": "Hydrothermal index, Dew depression, Diurnal temperature range"},
        {"Specification": "Variance Stabilization", "Feature Architecture": "log1p and sqrt count transformations for skewed distributions"},
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ----------------- 4. Methodology -----------------
with tab_pipeline:
    st.markdown("### 4. Surveillance & Forecasting Pipeline")
    st.markdown(
        """
        The platform implements a transparent four-stage econometric pipeline:
        1. **Field Surveillance Ingestion:** Automated querying of GBIF OpenAPI endpoints for 10 verified crop pathogen taxa across the Republic of India.
        2. **Spatial Harmonization & Panel Regularization:** Aggregating point observations into regular monthly calendar bins (MS frequency) and merging satellite NASA POWER weather regressors.
        3. **Chronological Splitting & Testing on Unseen Data:** Partitioning series strictly forward in time (no random k-fold shuffling) to guarantee zero temporal data leakage.
        4. **Multi-Model Benchmarking & Outbreak Detection:** Evaluating classical Box-Jenkins models alongside specialized two-stage Hurdle XGBoost, Prophet, and recurrent deep networks.
        """
    )

# ----------------- 5. Model Architectures -----------------
with tab_models:
    st.markdown("### 5. Mathematical Model Architectures")
    st.markdown("The application implements 14 active model architectures across two methodological families:")

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown(
            r"""
            **Classical Econometric Models:**
            - **Seasonal Naive Baseline:** $\hat{y}_{t+h} = y_{t+h-12}$
            - **AR(1) Autoregressive:** $y_t = c + \phi_1 y_{t-1} + \epsilon_t$
            - **MA(1) Moving Average:** $y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1}$
            - **ARIMA(1,1,1):** $(1 - \phi_1 B)(1 - B) y_t = c + (1 + \theta_1 B) \epsilon_t$
            - **SARIMA(1,0,1)(1,0,1)[12]:** Multiplicative seasonal Box-Jenkins
            - **Holt-Winters ESM:** Triple additive exponential smoothing
            - **ARIMAX / SARIMAX:** Box-Jenkins coupled with lagged NASA weather regressors
            """
        )
    with c_m2:
        st.markdown(
            r"""
            **Advanced ML & Deep Learning Models:**
            - **Local Random Forest & Local XGBoost:** Autoregressive non-linear decision trees with weather regressors
            - **Global Random Forest & Global XGBoost:** Multi-pathogen pooled panel matrices ($N=920$) with one-hot host embeddings
            - **Hurdle XGBoost:** Two-stage model: Stage-1 binary occurrence classifier + Stage-2 positive count regressor
            - **Prophet:** Bayesian decomposable curve: $y(t) = g(t) + s(t) + \epsilon_t$
            - **LSTM & GRU Networks:** Recurrent gated units capturing multi-month non-linear lag memory
            """
        )

# ----------------- 6. Evaluation & Validation -----------------
with tab_eval:
    st.markdown("### 6. Validation Protocols & Mathematical Metrics")
    st.markdown(
        r"""
        - **Multi-Origin Step-by-Step Historical Testing:** 4 chronological origins covering September 2022 to April 2024 (6-month forecast periods, 10 pathogens = 240 evaluation steps).
        - **Untouched Final Test Partition:** May 2024 to December 2025 (200 records) strictly frozen and isolated for testing on unseen data.
        - **Zero-Safe Metrics Only:**
          - **RMSE:** $\sqrt{\frac{1}{N}\sum(y_t - \hat{y}_t)^2}$ (Calculated across all evaluated predictions together; penalizes large surge misses).
          - **Average RMSE:** Arithmetic average of the individual pathogen RMSE values across the 10 monitored crop diseases.
          - **MAE:** $\frac{1}{N}\sum |y_t - \hat{y}_t|$ (Measures typical monthly reporting error across all predictions).
          - **Average MAE:** Arithmetic average of individual pathogen MAE values.
          - **MASE:** Scaled against in-sample seasonal persistence; values $< 1.0$ confirm genuine predictive skill.
          - **sMAPE:** Symmetric percentage error bounded between 0% and 200%.
          - **Outbreak Detection Rate & Outbreak F1 Score:** Detection rate (sensitivity) and overall quality on epidemic surge classification ($y_t > 0$).
        """
    )
    st.warning("Retirement of WAPE: WAPE has been permanently retired from the user interface because its non-zero denominator assumption produces mathematically distorted values on zero-inflated time series.")

# ----------------- 7. Limitations -----------------
with tab_limits:
    st.markdown("### 7. Epidemiological & Surveillance Limitations")
    st.markdown(
        """
        1. **Documented Accessions vs. Field Census:** Predictions estimate verified specimen discoveries deposited in research databases, not total infected acreage across agricultural fields.
        2. **Deposition Lag:** Herbarium voucher records occasionally incur multi-week processing delays between physical collection and database digitization.
        3. **Centroid Climatology:** Weather telemetry reflects regional agro-climatic centroids; hyper-local micro-climate variations may differ within individual districts.
        """
    )

# ----------------- 8. Downloads & Citations -----------------
with tab_downloads:
    st.markdown("### 8. Reproducible Data Downloads & Formal Academic Citations")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        p_master = Path("data/disease_timeseries_master.csv")
        if p_master.exists():
            st.download_button(
                label="📥 Download Authoritative Master Dataset CSV (43 Cols, 1,320 Rows)",
                data=p_master.read_bytes(),
                file_name="disease_timeseries_master.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
    with col_d2:
        p_reg = Path("data/processed/model_training_and_validation_registry.json")
        if p_reg.exists():
            st.download_button(
                label="Download Model Training & Validation Registry (17 Models JSON)",
                data=p_reg.read_bytes(),
                file_name="model_training_and_validation_registry.json",
                mime="application/json",
                use_container_width=True,
            )
        p_opt = Path("data/processed/advanced_model_optimization_results.json")
        if not p_opt.exists():
            p_opt = Path("reports/experiments/ADVANCED_MODEL_OPTIMIZATION_RESULTS.json")
        if not p_opt.exists():
            p_opt = Path("ADVANCED_MODEL_OPTIMIZATION_RESULTS.json")
        if p_opt.exists():
            st.download_button(
                label="Download Model Optimization Results (JSON)",
                data=p_opt.read_bytes(),
                file_name="ADVANCED_MODEL_OPTIMIZATION_RESULTS.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown(
        """
        **Formal Academic References:**
        - GBIF.org (2026). *GBIF Occurrence Download*. https://doi.org/10.15468/dl.india-pathogens
        - Sparks, A. H. (2018). *nasapower: A NASA POWER Global Meteorology, Surface Solar Energy and Climatology Data Client*. R Journal, 10(1), 588–602.
        - Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*. John Wiley & Sons.
        - Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
        - Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16, 785–794.
        """
    )
