"""Page 2: Time-Series Analysis & Econometric Stationarity Testing."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.eda import (
    assess_forecasting_suitability,
    compute_acf_pacf_values,
    compute_rolling_statistics,
    compute_series_summary,
    compute_weather_disease_correlations,
    detect_outliers,
    perform_seasonal_decomposition,
)
from src.icons import (
    icon_leaf,
    icon_model_compare,
    icon_pathology,
    icon_shield,
    icon_thermometer,
    icon_timeseries,
    icon_weather,
)
from src.stationarity import (
    analyze_differencing,
    perform_adf_test,
    perform_kpss_test,
    synthesize_stationarity_decision,
)
from src.ui_helpers import (
    format_plotly_figure,
    inject_custom_css,
    load_all_pathogen_options,
    load_monthly_series,
)

st.set_page_config(
    page_title="Time-Series Analysis — Verdant Intelligence",
    page_icon="📈",
    layout="wide",
)

inject_custom_css()

# Editorial Header
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_timeseries(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">Time-Series Analysis</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Rigorous statistical diagnostics, seasonal decomposition, ACF/PACF Bartlett limits, and formal ADF & KPSS unit-root testing.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pathogen_dict = load_all_pathogen_options()

# Top query toolbar
bar_c1, bar_c2 = st.columns([1.5, 1.0])
with bar_c1:
    selected_slug = st.selectbox(
        "Select Target Series",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
    )
with bar_c2:
    window_size = st.slider("Rolling Volatility Window (Months)", min_value=2, max_value=12, value=6, step=1)

df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"Monthly time series not found for {selected_slug}.")
    st.stop()

target_col = "documented_occurrence_count"
series_summary = compute_series_summary(df_monthly, target_col=target_col)
suitability = assess_forecasting_suitability(df_monthly, target_col=target_col, pathogen_name=pathogen_dict[selected_slug])

# 4 Premium KPI Cards
st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 14px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(20, "#527A50")}</div>
            <div class="metric-label-text">Total Months Indexed</div>
            <div class="metric-value-text">{series_summary['total_months']}</div>
            <div class="metric-note-text">Continuous MS Timeline</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
            <div class="metric-label-text">Active Surveillance</div>
            <div class="metric-value-text">{series_summary['active_months']} Mos</div>
            <div class="metric-note-text">{series_summary['pct_non_zero']:.1f}% non-zero density</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
            <div class="metric-label-text">Monthly Mean Count</div>
            <div class="metric-value-text">{series_summary['mean_monthly_occurrences']}</div>
            <div class="metric-note-text">Total: {series_summary['total_occurrences']:,} records</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_thermometer(20, "#527A50")}</div>
            <div class="metric-label-text">Volatility (Std Dev)</div>
            <div class="metric-value-text">{series_summary['std_monthly_occurrences']}</div>
            <div class="metric-note-text">Skew: {series_summary['skewness']} · Kurt: {series_summary['kurtosis']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# TABS
tab_trends, tab_decomp, tab_acf, tab_stat, tab_weather = st.tabs([
    "📈 Trends & Rolling Volatility",
    "🔬 Seasonal Decomposition",
    "📊 ACF & PACF Diagnostics",
    "⚖️ ADF & KPSS Stationarity Suite",
    "🌦️ Climate-Disease Associations",
])

# ----------------- Tab 1: Trends & Rolling -----------------
with tab_trends:
    st.subheader(f"Trajectory & {window_size}-Month Rolling Volatility Envelope")
    df_roll = compute_rolling_statistics(df_monthly, target_col=target_col, window=window_size)
    outliers_data = detect_outliers(df_monthly, target_col=target_col)

    fig_trend = go.Figure()

    # Actual Series
    fig_trend.add_trace(
        go.Scatter(
            x=df_monthly["date"],
            y=df_monthly[target_col],
            name="Documented Occurrences",
            mode="lines+markers",
            line=dict(color="#183D32", width=2),
            marker=dict(size=4),
        )
    )

    # Rolling Mean
    fig_trend.add_trace(
        go.Scatter(
            x=df_roll["date"],
            y=df_roll[f"rolling_mean_{window_size}m"],
            name=f"{window_size}-Month Rolling Mean",
            line=dict(color="#527A50", width=2.5),
        )
    )

    # Volatility Envelope
    upper_band = df_roll[f"rolling_mean_{window_size}m"] + df_roll[f"rolling_std_{window_size}m"]
    lower_band = np.maximum(0, df_roll[f"rolling_mean_{window_size}m"] - df_roll[f"rolling_std_{window_size}m"])

    fig_trend.add_trace(go.Scatter(x=df_roll["date"], y=upper_band, name="Upper Volatility (+1σ)", line=dict(color="#A9BEA0", width=1, dash="dot")))
    fig_trend.add_trace(go.Scatter(x=df_roll["date"], y=lower_band, name="Lower Volatility (-1σ)", line=dict(color="#A9BEA0", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(169, 190, 160, 0.15)"))

    format_plotly_figure(fig_trend, title=f"Historical Monthly Occurrences with {window_size}-Month Rolling Mean", height=420)
    st.plotly_chart(fig_trend, use_container_width=True)

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.info(f"**IQR Outlier Fence:** Upper fence = **{outliers_data['iqr_upper_fence']}** occurrences. Found **{outliers_data['iqr_outliers_count']}** outlier months.")
    with col_o2:
        st.write(f"**Forecasting Viability Assessment:** {suitability['recommendation']}")

# ----------------- Tab 2: Seasonal Decomposition -----------------
with tab_decomp:
    st.subheader("Classical Additive Seasonal Decomposition (Period s = 12 Months)")
    decomp_df = perform_seasonal_decomposition(df_monthly, target_col=target_col, period=12, model="additive")

    if decomp_df is not None:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["observed"], name="Observed Series", line=dict(color="#102C26", width=1.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["trend"], name="Macro Trend", line=dict(color="#527A50", width=2.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["seasonal"], name="Annual Seasonality (s=12)", line=dict(color="#C4A66A", width=1.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["residual"], name="Irregular Residual", line=dict(color="#C24134", width=1.0, dash="dot")))

        format_plotly_figure(fig_d, title="Additive Components: Observed = Trend + Seasonality + Residual", height=440)
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.warning("Insufficient observations for full 12-month annual seasonal decomposition.")

# ----------------- Tab 3: ACF & PACF -----------------
with tab_acf:
    st.subheader("Autocorrelation (ACF) & Partial Autocorrelation (PACF) Diagnostics")
    acf_pacf = compute_acf_pacf_values(df_monthly, target_col=target_col, nlags=20)

    if acf_pacf["lags"]:
        ci = acf_pacf["confidence_interval"]
        col_a, col_p = st.columns(2)

        with col_a:
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(x=acf_pacf["lags"], y=acf_pacf["acf"], name="ACF", marker_color="#183D32"))
            fig_a.add_hline(y=ci, line_dash="dash", line_color="#C24134", annotation_text="+95% Bartlett CI")
            fig_a.add_hline(y=-ci, line_dash="dash", line_color="#C24134", annotation_text="-95% Bartlett CI")
            format_plotly_figure(fig_a, title="Autocorrelation Function (ACF)", height=340)
            st.plotly_chart(fig_a, use_container_width=True)

        with col_p:
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=acf_pacf["lags"], y=acf_pacf["pacf"], name="PACF", marker_color="#527A50"))
            fig_p.add_hline(y=ci, line_dash="dash", line_color="#C24134", annotation_text="+95% Bartlett CI")
            fig_p.add_hline(y=-ci, line_dash="dash", line_color="#C24134", annotation_text="-95% Bartlett CI")
            format_plotly_figure(fig_p, title="Partial Autocorrelation Function (PACF)", height=340)
            st.plotly_chart(fig_p, use_container_width=True)

# ----------------- Tab 4: Stationarity Suite -----------------
with tab_stat:
    st.subheader("Augmented Dickey-Fuller (ADF) & KPSS Joint Hypothesis Suite")

    adf_res = perform_adf_test(df_monthly[target_col])
    kpss_res = perform_kpss_test(df_monthly[target_col])
    decision = synthesize_stationarity_decision(adf_res, kpss_res)

    col_adf, col_kpss = st.columns(2)

    with col_adf:
        st.markdown(
            f"""
            <div class="research-metric-card" style="border-left: 4px solid {'#527A50' if adf_res.get('is_stationary') else '#C24134'};">
                <div class="metric-label-text">Augmented Dickey-Fuller (ADF)</div>
                <div class="metric-value-text" style="font-size: 1.5rem; color: {'#527A50' if adf_res.get('is_stationary') else '#C24134'};">
                    {'STATIONARY' if adf_res.get('is_stationary') else 'NON-STATIONARY'}
                </div>
                <div style="font-size: 0.85rem; color: #5A665D; margin-top: 8px; line-height: 1.6;">
                    <b>Test Statistic:</b> <code>{adf_res.get('test_statistic')}</code><br>
                    <b>p-value:</b> <code>{adf_res.get('p_value')}</code><br>
                    <b>Critical Value (5%):</b> <code>{adf_res.get('critical_values', {}).get('5%')}</code><br>
                    <b>Lags Used:</b> {adf_res.get('used_lag')}
                </div>
                <div class="metric-note-text" style="color: {'#527A50' if adf_res.get('is_stationary') else '#C24134'};">
                    {adf_res.get('interpretation')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_kpss:
        st.markdown(
            f"""
            <div class="research-metric-card" style="border-left: 4px solid {'#527A50' if kpss_res.get('is_stationary') else '#C24134'};">
                <div class="metric-label-text">KPSS Stationarity Test</div>
                <div class="metric-value-text" style="font-size: 1.5rem; color: {'#527A50' if kpss_res.get('is_stationary') else '#C24134'};">
                    {'STATIONARY' if kpss_res.get('is_stationary') else 'NON-STATIONARY'}
                </div>
                <div style="font-size: 0.85rem; color: #5A665D; margin-top: 8px; line-height: 1.6;">
                    <b>Test Statistic:</b> <code>{kpss_res.get('test_statistic')}</code><br>
                    <b>p-value:</b> <code>{kpss_res.get('p_value')}</code><br>
                    <b>Critical Value (5%):</b> <code>{kpss_res.get('critical_values', {}).get('5%')}</code><br>
                    <b>Lags Used:</b> {kpss_res.get('lags_used')}
                </div>
                <div class="metric-note-text" style="color: {'#527A50' if kpss_res.get('is_stationary') else '#C24134'};">
                    {kpss_res.get('interpretation')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"### 🎯 Synthesized Econometric Classification: **{decision['classification']}**  \n"
        f"**Recommended Modeling Action:** {decision['action']}  \n"
        f"{decision['explanation']}"
    )

    diff_analysis = analyze_differencing(df_monthly[target_col])
    st.write(
        f"- **Level Variance (Original):** `{diff_analysis['original']['variance']}`  \n"
        f"- **First-Difference Variance (d=1):** `{diff_analysis['first_difference']['variance']}`  \n"
        f"- **Seasonal-Difference Variance (D=1):** `{diff_analysis['seasonal_difference']['variance']}`  \n"
        f"- **Variance Recommendation:** {diff_analysis['recommendation']}"
    )

# ----------------- Tab 5: Climate Association -----------------
with tab_weather:
    st.subheader("Cross-Correlation Across Distributed Climate Lags (t=0 to t-3 Months)")
    corr_res = compute_weather_disease_correlations(df_monthly, target_col=target_col, max_lag=3)

    if corr_res.get("correlations"):
        df_corr = pd.DataFrame(corr_res["correlations"])
        fig_corr = px.bar(
            df_corr,
            x="label",
            y="pearson_r",
            color="lag_months",
            barmode="group",
            color_discrete_sequence=["#183D32", "#527A50", "#C4A66A", "#2563EB"],
            labels={"pearson_r": "Pearson r", "label": "Climate Regressor", "lag_months": "Lag (Months)"},
            title="Pearson Correlation Across Distributed Lags",
        )
        format_plotly_figure(fig_corr, height=380)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption(f"⚠️ {corr_res['scientific_disclaimer']}")
