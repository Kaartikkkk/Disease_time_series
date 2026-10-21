"""Page 7: Scientific Methodology, Mathematical Formulations, and Responsible AI."""

from pathlib import Path
import streamlit as st

from src.icons import (
    icon_book,
    icon_model_compare,
    icon_settings,
    icon_shield,
    icon_timeseries,
)
from src.reporting import generate_methodology_report
from src.ui_helpers import (
    inject_custom_css,
    render_app_sidebar,
    render_top_header,
)

st.set_page_config(
    page_title="Methodology & Governance — Verdant Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="methodology")
render_top_header()

# Editorial Header
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_settings(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">Methodology &amp; Algorithmic Formulations</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Mathematical specifications, stationarity testing framework, chronological validation splits, and responsible AI safeguards.
            </p>
        </div>
        <div style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 8px; padding: 6px 14px; font-size: 0.82rem; color: #183D32; font-family: 'JetBrains Mono', monospace;">
            PEER REVIEW ARCHITECTURE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_math, tab_validation, tab_limits = st.tabs([
    "🔬 Mathematical Formulations & Models",
    "📊 Validation Protocols & Metrics",
    "🛡️ Scientific Safeguards & Responsible AI",
])

# ----------------- Tab 1: Mathematical Formulations & Models -----------------
with tab_math:
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
                    <li><strong>ARIMA(p, d, q):</strong> Linear autoregression with difference order $d$ and moving average error innovations:
                        $$\\phi(B)(1 - B)^d y_t = \\theta(B) \\epsilon_t$$
                    </li>
                    <li><strong>SARIMA(p, d, q)(P, D, Q)₁₂:</strong> Incorporates 12-month annual seasonality reflecting India's Kharif (monsoon) and Rabi (winter) cropping calendars:
                        $$\\Phi(B^{{12}})\\phi(B)(1 - B)^d (1 - B^{{12}})^D y_t = \\Theta(B^{{12}})\\theta(B) \\epsilon_t$$
                    </li>
                    <li><strong>Holt-Winters Exponential Smoothing:</strong> Additive Holt-Winters level, trend, and seasonal components with damped drift parameters.</li>
                    <li><strong>SARIMAX:</strong> Exogenous weather regressors ($T2M, PRECTOTCORR, RH2M$) incorporated with strict historical lags ($t-1, t-2$) to avoid future lookahead bias.</li>
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
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">Modern AI &amp; Non-Linear Forecasters</h3>
                </div>
                <ul style="font-size: 0.88rem; color: #404B43; padding-left: 20px; line-height: 1.6;">
                    <li><strong>Prophet (Additive Generalized Additive Model):</strong>
                        Decomposes trend, multiple seasonalities, and holiday effects with Bayesian uncertainty:
                        $$y(t) = g(t) + s(t) + h(t) + \\epsilon_t$$
                    </li>
                    <li><strong>LightGBM Gradient Boosted Decision Trees:</strong>
                        Gradient boosted decision trees trained on rolling lag features ($L_1, L_2, L_3, L_{{12}}$), rolling means, and climatic anomalies.
                    </li>
                    <li><strong>LSTM Recurrent Neural Networks:</strong>
                        Sequential Long Short-Term Memory network learning long-range temporal dependencies across multivariable agro-meteorological sequences.
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="verdant-card" style="margin-top: 16px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_timeseries(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">Stationarity &amp; Unit-Root Hypothesis Testing</h3>
                </div>
                <p style="font-size: 0.86rem; color: #404B43; line-height: 1.5;">
                    Dual-test verification protocol combining Augmented Dickey-Fuller (ADF) and KPSS tests:
                </p>
                <ul style="font-size: 0.84rem; color: #404B43; padding-left: 18px; line-height: 1.5;">
                    <li><strong>ADF Test:</strong> $H_0$: Unit root present (non-stationary). Rejection ($p < 0.05$) confirms mean-reverting stationarity.</li>
                    <li><strong>KPSS Test:</strong> $H_0$: Trend stationary. Non-rejection ($p > 0.05$) confirms absence of unit root.</li>
                    <li><strong>Automated Differencing ($d=1$):</strong> Applied when series exhibits deterministic drift or stochastic integration.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- Tab 2: Validation Protocols & Metrics -----------------
with tab_validation:
    st.markdown("### Chronological Benchmarking & Zero-Safe Metric Formulation")

    v1, v2 = st.columns([1, 1])
    with v1:
        st.markdown(
            f"""
            <div class="verdant-card">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_shield(20, "#527A50")}
                    <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">Chronological Split Protocol</h3>
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

    with v2:
        st.markdown(
            f"""
            <div class="verdant-card">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    {icon_model_compare(20, "#527A50")}
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
                    <li><strong>RMSE:</strong> Penalizes severe outbreak underestimation:
                        $$\\text{{RMSE}} = \\sqrt{{\\frac{{1}}{{n}} \\sum_{{t=1}}^n (y_t - \\hat{{y}}_t)^2}}$$
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------- Tab 3: Scientific Safeguards -----------------
with tab_limits:
    st.markdown("### Scientific Safeguards & Responsible AI Guidelines")
    st.markdown(
        "To ensure responsible operational deployment, agricultural decision-makers must evaluate these "
        "time-series projections within established epidemiologic and computational boundaries."
    )

    st.markdown(
        f"""
        <div class="verdant-card" style="margin-bottom: 16px; border-left: 4px solid #DC2626;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                {icon_shield(18, "#DC2626")}
                <h4 style="margin: 0; color: #102C26; font-size: 1.05rem;">1. Passive Surveillance &amp; Reporting Sampling Bias</h4>
            </div>
            <p style="font-size: 0.88rem; color: #404B43; line-height: 1.55; margin: 0;">
                GBIF biological records aggregate specimen deposits from ICAR agricultural universities, State Agricultural 
                Departments, and verified citizen scientists. A peak in documented occurrences can reflect an intensive 
                regional field sampling expedition rather than an immediate biological outbreak. The model forecasts 
                <strong>documented pathogen discovery activity</strong>, which acts as a leading indicator of regional epidemiological risk.
            </p>
        </div>

        <div class="verdant-card" style="margin-bottom: 16px; border-left: 4px solid #2563EB;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                {icon_model_compare(18, "#2563EB")}
                <h4 style="margin: 0; color: #102C26; font-size: 1.05rem;">2. Permissive Micro-Climate vs Direct Inoculum Availability</h4>
            </div>
            <p style="font-size: 0.88rem; color: #404B43; line-height: 1.55; margin: 0;">
                Agro-meteorological variables (surface temperature, dew point, relative humidity, precipitation) delineate 
                <strong>physiologically permissive infection windows</strong>. However, actual disease manifestation requires three simultaneous 
                components of the classical Disease Triangle: (1) viable virulent inoculum, (2) susceptible host crop phenology, 
                and (3) permissive micro-climate. Weather correlations quantify environmental favorability, not absolute infection.
            </p>
        </div>

        <div class="verdant-card" style="margin-bottom: 16px; border-left: 4px solid #166534;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                {icon_shield(18, "#166534")}
                <h4 style="margin: 0; color: #102C26; font-size: 1.05rem;">3. Absence of Wheat Stem Rust (<em>Puccinia graminis</em>) Post-2015</h4>
            </div>
            <p style="font-size: 0.88rem; color: #404B43; line-height: 1.55; margin: 0;">
                Ingested GBIF records show zero post-2015 occurrences for <em>Puccinia graminis</em> in India. This reflects 
                substantial agricultural success: nationwide deployment of resistant wheat cultivars screened against stem rust 
                (including Ug99 resistance initiatives by ICAR-IIWBR). The system accurately records this as zero observations 
                rather than synthesizing artificial data.
            </p>
        </div>

        <div class="verdant-card" style="margin-bottom: 16px; border-left: 4px solid #D97706;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                {icon_timeseries(18, "#D97706")}
                <h4 style="margin: 0; color: #102C26; font-size: 1.05rem;">4. Forward Weather Climatology Scenarios</h4>
            </div>
            <p style="font-size: 0.88rem; color: #404B43; line-height: 1.55; margin: 0;">
                Forward forecasts extending past known NASA POWER satellite observations employ 10-year historical month-of-year 
                climatological averages (&plusmn;15% perturbation scenarios). They do not represent predictive long-range meteorological 
                weather forecasts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Verdant Intelligence · Plant Disease Outbreak Forecasting · Open Science & Reproducible Research")
