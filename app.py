"""Plant Disease Outbreak Forecasting Using Time Series Analysis.

Advanced End-to-End Streamlit Application.
"""

import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Plant Disease Outbreak Forecasting",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.analysis.decomposition import decompose_series
from src.analysis.eda import (
    compute_acf_pacf,
    compute_eda_summary,
    compute_monthly_seasonality_matrix,
    compute_seasonal_subseries,
    compute_weather_disease_cross_correlation,
)
from src.analysis.stationarity import compute_rolling_statistics, test_stationarity
from src.data.aggregate_series import aggregate_time_series
from src.data.clean_data import clean_plant_disease_data
from src.data.load_data import load_dataset, load_all_available_datasets
from src.data.source_registry import DATASET_REGISTRY, list_all_datasets
from src.data.validate_schema import validate_canonical_schema, ensure_canonical_dataframe
from src.evaluation.backtesting import compare_all_models, backtest_single_model
from src.evaluation.diagnostics import analyze_residuals, compute_horizon_wise_metrics
from src.models.forecasting import generate_forecast
from src.models.model_registry import list_available_models
from src.risk.risk_assessment import evaluate_outbreak_risk
from src.ui.components import format_plotly_figure, inject_custom_css, render_metric_card
from src.utils.paths import REPORTS_DIR

# Inject custom CSS
inject_custom_css()


@st.cache_data(show_spinner=False)
def get_cached_dataset(dataset_id: str):
    return load_dataset(dataset_id, use_cache=True)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.markdown(
    """
    <div style="text-align: center; padding-bottom: 15px;">
        <h2 style="color: #059669; margin-bottom: 0px;">🌿 AgroCast AI</h2>
        <span style="font-size: 0.8rem; color: #64748B;">Plant Disease Outbreak Forecasting System</span>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGE_OPTIONS = [
    "Overview",
    "Data Explorer",
    "Time Series Analysis",
    "Forecasting",
    "Outbreak Risk",
    "Model Evaluation",
    "Data Management",
]

selected_page = st.sidebar.radio(
    "Navigation",
    PAGE_OPTIONS,
    index=0,
    format_func=lambda x: f"📌 {x}",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Active Dataset")

dataset_choices = {
    "agroscope_grapevine": "🍇 Grapevine Downy Mildew (Agroscope, Switzerland)",
    "wheat_blast_brazil": "🌾 Wheat Blast & FHB (Del Ponte et al., Brazil)",
    "citrus_ctv": "🍊 Citrus Tristeza Virus (Gottwald et al., Spain)",
    "tomato_tswv": "🍅 Tomato Spotted Wilt (Bald et al., Australia)",
    "synthetic_demo_benchmark": "🥔 Potato Late Blight (Daily Benchmark Demo)",
}

selected_dataset_id = st.sidebar.selectbox(
    "Select Agricultural Surveillance Data",
    options=list(dataset_choices.keys()),
    format_func=lambda k: dataset_choices[k],
    index=0,
)

# Load selected dataset
df_active, quality_report = get_cached_dataset(selected_dataset_id)
meta_active = DATASET_REGISTRY[selected_dataset_id]

# Sidebar provenance pill
if meta_active.is_synthetic:
    st.sidebar.markdown(
        '<div class="badge-synthetic" style="width: 100%; text-align: center; margin-bottom: 10px;">⚠️ SYNTHETIC BENCHMARK DEMO</div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div class="badge-real" style="width: 100%; text-align: center; margin-bottom: 10px;">✅ REAL PUBLIC SURVEILLANCE</div>',
        unsafe_allow_html=True,
    )

st.sidebar.caption(f"**Publisher**: {meta_active.publisher}")
st.sidebar.caption(f"**Target**: `{meta_active.target_variable}` ({meta_active.target_units})")
st.sidebar.caption(f"**Location**: {meta_active.geographic_coverage}")


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
def render_overview():
    st.title("🌿 Plant Disease Outbreak Forecasting System")
    st.markdown(
        """
        Welcome to **AgroCast AI**, an end-to-end epidemiological machine learning and time series forecasting
        platform designed to anticipate crop disease incidence, quantify outbreak risks, and identify climatic
        drivers across multi-crop agricultural ecosystems.
        """
    )

    # Provenance Banner
    if meta_active.is_synthetic:
        st.warning(
            f"**Demonstration Notice**: You are viewing `{meta_active.name}`. "
            "This benchmark is explicitly labeled as synthetic demonstration data generated to evaluate continuous daily forecasting."
        )
    else:
        st.success(
            f"**Real Surveillance Verification**: Currently loaded with empirical agricultural records from "
            f"**{meta_active.publisher}** covering `{meta_active.crop}` ({meta_active.disease})."
        )

    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Records", f"{len(df_active):,}", f"Cadence: {quality_report.inferred_frequency}")
    with col2:
        render_metric_card("Target Metric", meta_active.target_variable, meta_active.target_units)
    with col3:
        render_metric_card("Date Range", f"{quality_report.date_start[:4]} - {quality_report.date_end[:4]}", f"{quality_report.date_start} to {quality_report.date_end}")
    with col4:
        render_metric_card("Weather Covariates", "Included" if meta_active.has_weather_data else "None", "Temp, Humid, Rain")
    with col5:
        render_metric_card("Data Quality", "Validated", f"{quality_report.missing_target_count} missing rows dropped")

    st.markdown("### Surveillance Timeline Snapshot")
    fig = px.line(
        df_active,
        x="date",
        y="incidence_rate",
        color="crop" if df_active["crop"].nunique() > 1 else None,
        title=f"Historical Surveillance Records: {meta_active.crop} {meta_active.disease}",
        labels={"incidence_rate": f"Disease Target ({meta_active.target_units})", "date": "Observation Date"},
    )
    fig = format_plotly_figure(fig, f"Historical Surveillance Records: {meta_active.crop} {meta_active.disease}", yaxis_title=meta_active.target_units)
    st.plotly_chart(fig, use_container_width=True)

    # Core Capabilities
    st.markdown("### Architectural & Methodological Pillars")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            #### 1. Real Agricultural Data
            - Ingests verified surveillance from **Agroscope**, **Embrapa**, and **APS Press**.
            - Standardizes disparate schemas into a **canonical epidemiological model**.
            - Never naively converts missing survey dates into zero observations.
            """
        )
    with c2:
        st.markdown(
            """
            #### 2. Leakage-Free Forecasting
            - Strictly chronological splits: training data strictly precedes test horizons.
            - Compares baseline reference models against **ARIMA**, **SARIMA**, **Holt-Winters**, and **Gradient Boosters**.
            - Recursive multi-step forecasting without future weather leakage.
            """
        )
    with c3:
        st.markdown(
            """
            #### 3. Outbreak Risk Calibration
            - Distinguishes expected incidence from **probabilistic outbreak risk**.
            - Statistical percentile profiling (75th, 90th, 97.5th thresholds).
            - Tail-probability integration for actionable agronomic alert signals.
            """
        )


# ==============================================================================
# PAGE 2: DATA EXPLORER
# ==============================================================================
def render_explorer():
    st.title("🔍 Agricultural Disease Data Explorer")
    st.markdown("Inspect raw and processed records, audit missingness, and filter temporal subsets.")

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        available_crops = df_active["crop"].unique().tolist()
        sel_crops = st.multiselect("Filter Crops", available_crops, default=available_crops)
    with c2:
        available_locs = df_active["location_id"].unique().tolist() if "location_id" in df_active.columns else []
        sel_locs = st.multiselect("Filter Locations", available_locs, default=available_locs)
    with c3:
        min_d = df_active["date"].min().date()
        max_d = df_active["date"].max().date()
        date_range = st.date_input("Date Window", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    # Apply filters
    filtered_df = df_active.copy()
    if sel_crops:
        filtered_df = filtered_df[filtered_df["crop"].isin(sel_crops)]
    if sel_locs:
        filtered_df = filtered_df[filtered_df["location_id"].isin(sel_locs)]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= date_range[0]) & (filtered_df["date"].dt.date <= date_range[1])
        ]

    st.markdown(f"**Displaying {len(filtered_df):,} of {len(df_active):,} records.**")
    st.dataframe(filtered_df.head(200), use_container_width=True)

    # Download button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Canonical CSV",
        data=csv_bytes,
        file_name=f"{selected_dataset_id}_filtered.csv",
        mime="text/csv",
    )

    # Data Quality & Distribution Profiling
    st.markdown("---")
    st.subheader("Target Distribution & Statistical Profiling")
    eda_stats = compute_eda_summary(filtered_df, target_col="incidence_rate")

    col_stat1, col_stat2 = st.columns([1, 2])
    with col_stat1:
        st.markdown("#### Statistical Moments")
        if eda_stats:
            st.write(f"- **Mean**: {eda_stats['mean']:.2f}")
            st.write(f"- **Std Dev**: {eda_stats['std']:.2f}")
            st.write(f"- **Median**: {eda_stats['median']:.2f}")
            st.write(f"- **Min / Max**: {eda_stats['min']:.2f} / {eda_stats['max']:.2f}")
            st.write(f"- **IQR (25-75%)**: {eda_stats['iqr']:.2f}")
            st.write(f"- **Skewness**: {eda_stats['skewness']:.2f}")
            st.write(f"- **Zero Records**: {eda_stats['zero_count']} ({eda_stats['zero_percentage']}%)")
    with col_stat2:
        fig_dist = px.histogram(
            filtered_df,
            x="incidence_rate",
            nbins=35,
            marginal="box",
            title=f"Incidence Distribution: {meta_active.target_variable}",
            color_discrete_sequence=["#059669"],
        )
        fig_dist = format_plotly_figure(fig_dist, f"Incidence Distribution ({meta_active.target_units})", xaxis_title=meta_active.target_units)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Metadata & Limitations
    st.markdown("---")
    st.subheader("Data Provenance & Scientific Limitations")
    st.markdown(f"**Publisher / Institution**: {meta_active.publisher}")
    st.markdown(f"**Source URL / Reference**: [{meta_active.source_url}]({meta_active.source_url})")
    st.markdown(f"**License**: `{meta_active.license}`")
    st.markdown("**Documented Analytical Limitations**:")
    for lim in meta_active.limitations:
        st.markdown(f"- ⚠️ {lim}")


# ==============================================================================
# PAGE 3: TIME SERIES ANALYSIS
# ==============================================================================
def render_analysis():
    st.title("📈 Time Series Exploratory Data Analysis")
    st.markdown("Analyze longitudinal trends, seasonal cycles, stationarity, autocorrelation, and weather interactions.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Longitudinal Trends & Rolling Stats", "Decomposition (STL)", "Stationarity (ADF & KPSS)", "ACF/PACF & Weather Cross-Correlation"]
    )

    with tab1:
        st.subheader("Rolling Mean & Volatility Dynamics")
        w_size = st.slider("Rolling Window Size", min_value=3, max_value=60, value=14, step=1)
        rolling_df = compute_rolling_statistics(df_active["incidence_rate"], window=w_size)

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=df_active["date"], y=df_active["incidence_rate"], name="Observed", line=dict(color="#94A3B8", width=1), opacity=0.5))
        fig_roll.add_trace(go.Scatter(x=df_active["date"], y=rolling_df["rolling_mean"], name=f"{w_size}-Period Rolling Mean", line=dict(color="#059669", width=2.5)))
        fig_roll.add_trace(go.Scatter(
            x=df_active["date"],
            y=rolling_df["rolling_mean"] + rolling_df["rolling_std"],
            name="Upper Volatility (+1σ)",
            line=dict(color="#10B981", width=1, dash="dot"),
        ))
        fig_roll.add_trace(go.Scatter(
            x=df_active["date"],
            y=rolling_df["rolling_mean"] - rolling_df["rolling_std"],
            name="Lower Volatility (-1σ)",
            line=dict(color="#10B981", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.1)",
        ))
        fig_roll = format_plotly_figure(fig_roll, f"Observed Disease Series with {w_size}-Step Rolling Statistics", yaxis_title=meta_active.target_units)
        st.plotly_chart(fig_roll, use_container_width=True)

    with tab2:
        st.subheader("STL Time Series Decomposition")
        st.markdown("Decomposes series into $Y_t = \\text{Trend}_t + \\text{Seasonal}_t + \\text{Residual}_t$.")
        period_choice = st.selectbox("Decomposition Seasonal Period", options=[7, 30, 365], index=0 if len(df_active) < 300 else 2, format_func=lambda p: f"{p} intervals")

        decomp = decompose_series(df_active["incidence_rate"], period=period_choice, method="stl")
        if not decomp.is_valid:
            st.error(decomp.error_message)
        else:
            c1, c2 = st.columns(2)
            c1.metric("Seasonal Strength", f"{decomp.seasonal_strength:.4f}")
            c2.metric("Residual Variance", f"{decomp.residual_variance:.4f}")

            fig_dec = go.Figure()
            fig_dec.add_trace(go.Scatter(x=df_active["date"], y=decomp.observed, name="Observed", line=dict(color="black", width=1)))
            fig_dec.add_trace(go.Scatter(x=df_active["date"], y=decomp.trend, name="Trend Component", line=dict(color="#2563EB", width=2)))
            fig_dec.add_trace(go.Scatter(x=df_active["date"], y=decomp.seasonal, name="Seasonal Component", line=dict(color="#059669", width=1.5)))
            fig_dec.add_trace(go.Scatter(x=df_active["date"], y=decomp.resid, name="Residuals", line=dict(color="#DC2626", width=1), mode="markers"))
            fig_dec = format_plotly_figure(fig_dec, f"STL Decomposition (Period={period_choice})", yaxis_title=meta_active.target_units)
            st.plotly_chart(fig_dec, use_container_width=True)

    with tab3:
        st.subheader("Formal Stationarity Testing: ADF & KPSS")
        st.markdown(
            """
            - **Augmented Dickey-Fuller (ADF)** tests H0: Unit root present (*non-stationary*).
            - **KPSS** tests H0: Series is level/trend *stationary*.
            """
        )
        stat_res = test_stationarity(df_active["incidence_rate"])
        if stat_res.is_stationary:
            st.success(f"**Verdict**: {stat_res.verdict}")
        else:
            st.warning(f"**Verdict**: {stat_res.verdict}")

        st.info(stat_res.summary_explanation)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Augmented Dickey-Fuller Test")
            st.write(f"- **Test Statistic**: `{stat_res.adf_statistic:.4f}`")
            st.write(f"- **p-value**: `{stat_res.adf_pvalue:.4e}`")
            st.write(f"- **Lags Used**: `{stat_res.adf_used_lag}`")
            st.write("- **Critical Values**:")
            for k, v in stat_res.adf_critical_values.items():
                st.write(f"  - {k}: `{v:.3f}`")
        with c2:
            st.markdown("#### KPSS Test")
            st.write(f"- **Test Statistic**: `{stat_res.kpss_statistic:.4f}`")
            st.write(f"- **p-value**: `{stat_res.kpss_pvalue:.4f}`")
            st.write("- **Critical Values**:")
            for k, v in stat_res.kpss_critical_values.items():
                st.write(f"  - {k}: `{v:.3f}`")

    with tab4:
        st.subheader("Autocorrelation & Microclimatic Cross-Correlation")
        col_acf, col_ccf = st.columns(2)

        with col_acf:
            st.markdown("#### Autocorrelation Function (ACF)")
            acf_data = compute_acf_pacf(df_active["incidence_rate"], nlags=21)
            if acf_data.get("lags"):
                fig_acf = go.Figure()
                fig_acf.add_trace(go.Bar(x=acf_data["lags"], y=acf_data["acf"], name="ACF", marker_color="#3B82F6"))
                fig_acf.add_hline(y=acf_data["conf_bound"], line_dash="dash", line_color="red")
                fig_acf.add_hline(y=-acf_data["conf_bound"], line_dash="dash", line_color="red")
                fig_acf = format_plotly_figure(fig_acf, "Autocorrelation Function (ACF)", xaxis_title="Lag", yaxis_title="Correlation", height=350)
                st.plotly_chart(fig_acf, use_container_width=True)

        with col_ccf:
            st.markdown("#### Weather-Disease Cross-Correlation")
            ccf_data = compute_weather_disease_cross_correlation(df_active, max_lag=14)
            if ccf_data:
                fig_ccf = go.Figure()
                for var, lag_dict in ccf_data.items():
                    fig_ccf.add_trace(go.Scatter(x=list(lag_dict.keys()), y=list(lag_dict.values()), name=var, mode="lines+markers"))
                fig_ccf.add_hline(y=0, line_color="black")
                fig_ccf = format_plotly_figure(fig_ccf, "Cross-Correlation: Weather Leading Disease", xaxis_title="Days Leading", yaxis_title="Cross-Correlation", height=350)
                st.plotly_chart(fig_ccf, use_container_width=True)
            else:
                st.info("No paired continuous weather variables available for this dataset.")


# ==============================================================================
# PAGE 4: FORECASTING
# ==============================================================================
def render_forecasting():
    st.title("🔮 Time Series Forecasting Engine")
    st.markdown("Generate forward-looking predictions with confidence intervals across daily, weekly, and monthly frequencies.")

    # Control Bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        freq_option = st.selectbox("Forecast Frequency", options=["Daily", "Weekly", "Monthly"], index=0)
        freq_map = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}
        target_freq = freq_map[freq_option]
    with c2:
        horizon = st.slider("Forecast Horizon", min_value=3, max_value=30, value=14, step=1)
    with c3:
        all_models = list_available_models()
        model_choice = st.selectbox(
            "Forecasting Model",
            options=list(all_models.keys()),
            format_func=lambda m: all_models[m],
            index=0,
        )
    with c4:
        include_weather = st.checkbox("Include Weather Covariates", value=meta_active.has_weather_data)

    # Aggregate series if requested
    df_to_forecast = df_active.copy()
    if target_freq != "D":
        df_to_forecast = aggregate_time_series(df_to_forecast, target_freq=target_freq)

    if len(df_to_forecast) < 10:
        st.warning(f"Dataset has only {len(df_to_forecast)} observations after aggregation. Need >= 10 points.")
        return

    # Generate Forecast
    with st.spinner(f"Fitting {all_models[model_choice]} and generating {horizon}-{freq_option.lower()} forecast..."):
        try:
            fc_res = generate_forecast(
                df=df_to_forecast,
                model_name=model_choice,
                target_col="incidence_rate",
                horizon=horizon,
                frequency=target_freq,
                include_weather=include_weather,
            )
        except Exception as e:
            st.error(f"Forecasting error: {e}")
            return

    # Visual Display
    st.subheader(f"Forecast: {fc_res.model_name} ({horizon} steps)")

    recent_pts = st.slider("Historical Context Window (Recent Points)", min_value=14, max_value=min(365, len(df_to_forecast)), value=min(60, len(df_to_forecast)))
    hist_subset = df_to_forecast.tail(recent_pts)

    fig_fc = go.Figure()
    # Historical trace
    fig_fc.add_trace(go.Scatter(
        x=hist_subset["date"],
        y=hist_subset["incidence_rate"],
        name="Historical Observations",
        line=dict(color="#334155", width=2),
        mode="lines+markers",
    ))
    # Forecast trace
    fig_fc.add_trace(go.Scatter(
        x=fc_res.forecast_dates,
        y=fc_res.point_forecast,
        name=f"Forecast ({fc_res.model_name})",
        line=dict(color="#2563EB", width=3, dash="dash"),
        mode="lines+markers",
    ))
    # Prediction intervals
    fig_fc.add_trace(go.Scatter(
        x=fc_res.forecast_dates,
        y=fc_res.upper_bound,
        name="Upper 95% Bound",
        line=dict(color="#93C5FD", width=1, dash="dot"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_res.forecast_dates,
        y=fc_res.lower_bound,
        name="Lower 95% Bound",
        line=dict(color="#93C5FD", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(59, 130, 246, 0.15)",
    ))
    fig_fc = format_plotly_figure(fig_fc, f"{horizon}-Step Forward Disease Forecast", yaxis_title=meta_active.target_units)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast Summary & Feature Importance
    c_meta1, c_meta2 = st.columns([1, 1])
    with c_meta1:
        st.markdown("#### Forecast Metadata")
        st.write(f"- **Training Observations**: `{fc_res.training_points}`")
        st.write(f"- **Last Historical Date**: `{fc_res.last_historical_date.date()}`")
        st.write(f"- **Forecast Horizon**: `{fc_res.horizon} {freq_option.lower()} periods`")
        st.write(f"- **Execution Time**: `{fc_res.fit_time_seconds} s`")
        st.write(f"- **Methodology Note**: {fc_res.notes}")

    with c_meta2:
        if fc_res.feature_importances is not None and not fc_res.feature_importances.empty:
            st.markdown("#### Top Predictor Feature Importances")
            top_feats = fc_res.feature_importances.head(8)
            fig_imp = px.bar(
                top_feats,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Greens",
            )
            fig_imp.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.markdown("#### Statistical Model Parametric Details")
            st.info("Univariate statistical forecasters optimize internal parameters (AIC / state-space smoothing coefficients).")


# ==============================================================================
# PAGE 5: OUTBREAK RISK
# ==============================================================================
def render_risk():
    st.title("⚠️ Outbreak Risk Estimation & Agronomic Early Warning")
    st.markdown(
        """
        Translates raw disease forecasts into **probabilistic outbreak risk signals** and **actionable crop protection alerts**
        using historical percentile thresholds and predictive uncertainty.
        """
    )

    # Generate reference forecast for risk assessment
    df_demo = df_active if len(df_active) >= 14 else get_cached_dataset("synthetic_demo_benchmark")[0]
    fc_res = generate_forecast(df=df_demo, model_name="arima", horizon=14, frequency="D")

    risk_report = evaluate_outbreak_risk(
        forecast_dates=fc_res.forecast_dates,
        point_forecast=fc_res.point_forecast,
        lower_bound=fc_res.lower_bound,
        upper_bound=fc_res.upper_bound,
        train_history=df_demo["incidence_rate"],
        target_units=meta_active.target_units,
    )

    # Outbreak Status Banner
    banner_class = f"risk-{risk_report.peak_risk_level.lower().replace(' ', '-')}"
    st.markdown(
        f"""
        <div class="risk-banner {banner_class}">
            <h3 style="margin: 0; color: inherit;">🚨 {risk_report.overall_status}</h3>
            <p style="margin: 4px 0 0 0;">Peak estimated risk reaches <b>{risk_report.peak_risk_level}</b> on <b>{risk_report.peak_risk_date.date()}</b> (Forecast: {risk_report.max_forecast_value} {risk_report.thresholds.target_units}).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Threshold Reference
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    col_t1.metric("Baseline Median (50th %)", f"{risk_report.thresholds.baseline_level} {risk_report.thresholds.target_units}")
    col_t2.metric("Moderate Watch (75th %)", f"{risk_report.thresholds.moderate_threshold} {risk_report.thresholds.target_units}")
    col_t3.metric("High Alert (90th %)", f"{risk_report.thresholds.high_alert_threshold} {risk_report.thresholds.target_units}")
    col_t4.metric("Very High (97.5th %)", f"{risk_report.thresholds.very_high_threshold} {risk_report.thresholds.target_units}")

    # Probability Chart
    st.subheader("Daily Outbreak Probability P(Y > High Alert Threshold)")
    daily_df = pd.DataFrame([
        {
            "Date": d.date.strftime("%Y-%m-%d"),
            "Forecast": d.forecast_incidence,
            "Lower Bound": d.lower_ci,
            "Upper Bound": d.upper_ci,
            "Risk Level": d.risk_level,
            "Outbreak Probability (%)": d.outbreak_probability_pct,
            "Anomaly Z-Score": d.anomaly_z_score,
            "Recommended Action": d.recommended_action,
        }
        for d in risk_report.daily_assessments
    ])

    fig_prob = px.bar(
        daily_df,
        x="Date",
        y="Outbreak Probability (%)",
        color="Risk Level",
        color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444", "Very High": "#7F1D1D"},
        title="Estimated Probability of Exceeding High-Risk Outbreak Threshold",
    )
    fig_prob = format_plotly_figure(fig_prob, "Daily Outbreak Exceedance Probability (%)", yaxis_title="Probability (%)", height=320)
    st.plotly_chart(fig_prob, use_container_width=True)

    # Detailed Daily Table
    st.markdown("### Actionable Early-Warning Schedule")
    st.dataframe(daily_df, use_container_width=True)

    # Contributing drivers & scientific disclaimer
    st.markdown("---")
    c_drv1, c_drv2 = st.columns(2)
    with c_drv1:
        st.markdown("#### Contributing Environmental Drivers")
        for drv in risk_report.environmental_drivers:
            st.markdown(f"- 🌿 {drv}")
    with c_drv2:
        st.markdown("#### Scientific Advisory & Protocol")
        st.info(risk_report.disclaimer)


# ==============================================================================
# PAGE 6: MODEL EVALUATION
# ==============================================================================
def render_evaluation():
    st.title("🏆 Leakage-Free Model Leaderboard & Validation")
    st.markdown(
        """
        Rigorous out-of-sample benchmark comparison across chronological expanding evaluation folds.
        **No data leakage**: preprocessing and feature generation occur strictly within training windows.
        """
    )

    leaderboard_csv = REPORTS_DIR / "model_comparison.csv"
    if leaderboard_csv.exists():
        df_lb = pd.read_csv(leaderboard_csv)
        st.subheader("Official Multi-Model Benchmark Leaderboard")
        st.dataframe(df_lb, use_container_width=True)

        # Bar chart comparison
        fig_lb = px.bar(
            df_lb,
            x="Model",
            y="MAE",
            color="Model",
            title="Out-of-Sample MAE Across Models (Lower is Better)",
            labels={"MAE": f"Mean Absolute Error ({meta_active.target_units})"},
        )
        fig_lb = format_plotly_figure(fig_lb, "Out-of-Sample MAE Across Models", yaxis_title="MAE")
        st.plotly_chart(fig_lb, use_container_width=True)
    else:
        st.info("Run `python src/generate_evaluation_reports.py` to compile leaderboard.")

    # Live backtest on active dataset
    st.markdown("---")
    st.subheader("Run Live Expanding-Window Backtest on Active Dataset")
    c1, c2, c3 = st.columns(3)
    with c1:
        eval_model = st.selectbox("Model to Backtest", list(list_available_models().keys()), index=0)
    with c2:
        eval_horizon = st.slider("Backtest Horizon", min_value=3, max_value=21, value=7)
    with c3:
        eval_folds = st.slider("Number of Folds", min_value=2, max_value=5, value=3)

    if st.button("🚀 Execute Chronological Backtest"):
        with st.spinner("Executing expanding-window validation..."):
            try:
                rep = backtest_single_model(
                    model_name=eval_model,
                    df=df_active,
                    horizon=eval_horizon,
                    n_splits=eval_folds,
                )
                st.success(f"Backtest complete! Out-of-sample MAE: **{rep.mean_mae:.4f}**, RMSE: **{rep.mean_rmse:.4f}**")

                # Actual vs Predicted chart
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=rep.all_dates, y=rep.all_y_true, name="Actual Held-Out Observations", line=dict(color="black", width=2)))
                fig_bt.add_trace(go.Scatter(x=rep.all_dates, y=rep.all_y_pred, name="Model Predictions", line=dict(color="#2563EB", width=2, dash="dash")))
                fig_bt = format_plotly_figure(fig_bt, f"Actual vs Predicted on Chronological Test Windows ({rep.model_name})", yaxis_title=meta_active.target_units)
                st.plotly_chart(fig_bt, use_container_width=True)

                # Residual Diagnostics
                res_diag = analyze_residuals(rep.all_y_true, rep.all_y_pred)
                st.markdown("#### Residual Error Diagnostics")
                st.write(f"- **Mean Bias (Error)**: `{res_diag.get('mean_error_bias', 0.0)}`")
                st.write(f"- **Residual Std Dev**: `{res_diag.get('std_error', 0.0)}`")
                st.write(f"- **Ljung-Box p-value**: `{res_diag.get('ljung_box_pvalue', 'N/A')}`")
                st.info(res_diag.get("diagnostic_interpretation", ""))
            except Exception as e:
                st.error(f"Backtest failed: {e}")


# ==============================================================================
# PAGE 7: DATA MANAGEMENT
# ==============================================================================
def render_data_mgmt():
    st.title("⚙️ Data Management & Model Governance")
    st.markdown("Manage public data ingestion pipelines, inspect availability, or upload custom agricultural surveillance datasets.")

    # Source Registry Table
    st.subheader("Public Data Sources Registry")
    st.dataframe(list_all_datasets(), use_container_width=True)

    # Custom CSV Ingestion
    st.markdown("---")
    st.subheader("Upload Custom Agricultural Surveillance CSV")
    st.markdown(
        """
        Upload your own field observations. Required canonical columns:
        `date`, `crop`, `disease`, `incidence_rate`.
        Optional: `location_id`, `latitude`, `longitude`, `temperature`, `humidity`, `rainfall`.
        """
    )
    uploaded_file = st.file_uploader("Upload Agricultural CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            is_valid, errors = validate_canonical_schema(user_df)
            if not is_valid:
                st.error(f"Schema Validation Failed: {errors}")
            else:
                cleaned_user_df, user_report = clean_plant_disease_data(user_df)
                st.success(f"File validated successfully! {len(cleaned_user_df)} clean records parsed.")
                st.dataframe(cleaned_user_df.head(10), use_container_width=True)
                st.json(user_report.__dict__)
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

    # Evaluation artifacts download
    st.markdown("---")
    st.subheader("Download Evaluation Artifacts")
    c1, c2 = st.columns(2)
    with c1:
        lb_path = REPORTS_DIR / "model_comparison.csv"
        if lb_path.exists():
            with open(lb_path, "rb") as f:
                st.download_button("📥 Download Model Leaderboard CSV", f, file_name="model_comparison.csv", mime="text/csv")
    with c2:
        json_path = REPORTS_DIR / "evaluation_summary.json"
        if json_path.exists():
            with open(json_path, "rb") as f:
                st.download_button("📥 Download Evaluation Summary JSON", f, file_name="evaluation_summary.json", mime="application/json")


# ==============================================================================
# ROUTER
# ==============================================================================
if selected_page == "Overview":
    render_overview()
elif selected_page == "Data Explorer":
    render_explorer()
elif selected_page == "Time Series Analysis":
    render_analysis()
elif selected_page == "Forecasting":
    render_forecasting()
elif selected_page == "Outbreak Risk":
    render_risk()
elif selected_page == "Model Evaluation":
    render_evaluation()
elif selected_page == "Data Management":
    render_data_mgmt()
