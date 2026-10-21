"""Page 2: Time-Series Analysis, Seasonality Heatmaps & Econometric Stationarity Testing."""

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
from src.forecasting import chronological_split
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
    render_app_sidebar,
    render_top_header,
)

st.set_page_config(
    page_title="Time-Series Analysis — Verdant Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="time_series")
render_top_header()

# ==============================================================================
# EDITORIAL HEADER
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 22px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_timeseries(28, "#527A50")}
                <h1 style="font-size: 2.3rem; margin: 0; color: #102C26;">Time-Series Analysis & Diagnostics</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Rigorous empirical diagnostics, annual seasonality heatmaps, ACF/PACF Bartlett confidence bands, and joint ADF/KPSS unit-root verification.
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 12px; padding: 8px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                Statistical Diagnostic Suite
            </div>
            <div style="font-size: 0.76rem; color: #183D32; font-weight: 600; margin-top: 2px;">
                ADF · KPSS · Bartlett CI · Classical Decomposition
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pathogen_dict = load_all_pathogen_options()

# Parameter Toolbar
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
    <div class="metric-grid" style="margin-top: 10px; margin-bottom: 24px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(20, "#527A50")}</div>
            <div class="metric-label-text">Total Months Indexed</div>
            <div class="metric-value-text">{series_summary['total_months']} Months</div>
            <div class="metric-note-text">Continuous Monthly Index</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
            <div class="metric-label-text">Surveillance Density</div>
            <div class="metric-value-text">{series_summary['active_months']} Active Mos</div>
            <div class="metric-note-text">{series_summary['pct_non_zero']:.1f}% non-zero activity density</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
            <div class="metric-label-text">Monthly Mean Count</div>
            <div class="metric-value-text">{series_summary['mean_monthly_occurrences']}</div>
            <div class="metric-note-text">Total: {series_summary['total_occurrences']:,} documented records</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_thermometer(20, "#C4A66A")}</div>
            <div class="metric-label-text">Historical Volatility</div>
            <div class="metric-value-text">σ = {series_summary['std_monthly_occurrences']}</div>
            <div class="metric-note-text">Skew: {series_summary['skewness']} · Kurtosis: {series_summary['kurtosis']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# MAIN DIAGNOSTIC WORKSPACE TABS
# ==============================================================================
tab_patterns, tab_season, tab_decomp, tab_acf, tab_stat, tab_split = st.tabs([
    "📈 Historical Patterns & Volatility",
    "🗓️ Annual Seasonality Heatmap",
    "🔬 Classical Decomposition",
    "📊 ACF & PACF Bartlett Limits",
    "⚖️ ADF & KPSS Stationarity Suite",
    "⏳ Chronological Validation Splits",
])

# ----------------- Tab 1: Historical Patterns -----------------
with tab_patterns:
    st.subheader(f"Historical Trajectory & {window_size}-Month Rolling Volatility Envelope")
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
            line=dict(color="#102C26", width=2),
            marker=dict(size=4, color="#102C26"),
            hovertemplate="<b>Occurrences:</b> %{y:.1f}<br><b>Date:</b> %{x|%b %Y}<extra></extra>",
        )
    )

    # Rolling Mean
    fig_trend.add_trace(
        go.Scatter(
            x=df_roll["date"],
            y=df_roll[f"rolling_mean_{window_size}m"],
            name=f"{window_size}-Month Rolling Mean",
            line=dict(color="#527A50", width=2.5),
            hovertemplate="<b>Rolling Mean:</b> %{y:.2f}<extra></extra>",
        )
    )

    # Volatility Envelope (+/- 1 sigma)
    upper_band = df_roll[f"rolling_mean_{window_size}m"] + df_roll[f"rolling_std_{window_size}m"]
    lower_band = np.maximum(0, df_roll[f"rolling_mean_{window_size}m"] - df_roll[f"rolling_std_{window_size}m"])

    fig_trend.add_trace(go.Scatter(x=df_roll["date"], y=upper_band, name="Upper Volatility (+1σ)", line=dict(color="#C4A66A", width=1, dash="dot"), hoverinfo="skip"))
    fig_trend.add_trace(go.Scatter(x=df_roll["date"], y=lower_band, name="Lower Volatility (-1σ)", line=dict(color="#C4A66A", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(196, 166, 106, 0.15)", hoverinfo="skip"))

    format_plotly_figure(fig_trend, title=f"Documented Monthly Occurrences with {window_size}-Month Rolling Volatility Envelope", height=420)
    fig_trend.update_layout(yaxis_title="Documented Occurrence Count", xaxis_title="Timeline")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Missing Month Bar Visualization
    st.markdown("#### Missingness & Surveillance Integrity Audit")
    if "is_missing" in df_monthly.columns:
        fig_miss = go.Figure()
        miss_colors = ["#C24134" if m else "#527A50" for m in df_monthly["is_missing"]]
        fig_miss.add_trace(
            go.Bar(
                x=df_monthly["date"],
                y=[1] * len(df_monthly),
                marker_color=miss_colors,
                name="Surveillance Status",
                hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Status:</b> %{text}<extra></extra>",
                text=["Unmonitored Interval (is_missing=True)" if m else "Verified Active Surveillance" for m in df_monthly["is_missing"]],
            )
        )
        format_plotly_figure(fig_miss, title="Timeline Surveillance Integrity (Green = Verified Active Surveillance | Red = Unobserved Interval)", height=160)
        fig_miss.update_layout(yaxis=dict(showticklabels=False, range=[0, 1.2]))
        st.plotly_chart(fig_miss, use_container_width=True)

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.info(f"**IQR Outlier Fence:** Upper statistical bound = **{outliers_data['iqr_upper_fence']}** occurrences. Found **{outliers_data['iqr_outliers_count']}** outlier months.")
    with col_o2:
        st.write(f"**Forecasting Viability Assessment:** {suitability['recommendation']}")

# ----------------- Tab 2: Annual Seasonality Heatmap -----------------
with tab_season:
    st.subheader("🗓️ Calendar Seasonality Heatmap (Month vs. Year)")
    st.markdown(
        "Visualizes intra-annual epidemic cycles across India's agricultural seasons. "
        "Intense clusters highlight typical Kharif monsoon (Jul–Nov) and Rabi winter (Dec–Apr) peaks."
    )

    df_heat = df_monthly.copy()
    df_heat["year"] = pd.to_datetime(df_heat["date"]).dt.year
    df_heat["month_num"] = pd.to_datetime(df_heat["date"]).dt.month
    df_heat["month_abbr"] = pd.to_datetime(df_heat["date"]).dt.strftime("%b")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_heat = df_heat.pivot_table(index="year", columns="month_abbr", values=target_col, aggfunc="sum").fillna(0)
    pivot_heat = pivot_heat.reindex(columns=[m for m in month_order if m in pivot_heat.columns])

    fig_heat = px.imshow(
        pivot_heat,
        labels=dict(x="Calendar Month", y="Year", color="Occurrences"),
        x=pivot_heat.columns,
        y=pivot_heat.index,
        color_continuous_scale=[
            (0.0, "#FFFEFA"),
            (0.2, "#E3EBE0"),
            (0.5, "#527A50"),
            (0.8, "#183D32"),
            (1.0, "#102C26"),
        ],
        aspect="auto",
    )
    format_plotly_figure(fig_heat, title=f"Annual Seasonality Heatmap: {pathogen_dict[selected_slug]}", height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

# ----------------- Tab 3: Seasonal Decomposition -----------------
with tab_decomp:
    st.subheader("Classical Additive Seasonal Decomposition (Period s = 12 Months)")
    decomp_df = perform_seasonal_decomposition(df_monthly, target_col=target_col, period=12, model="additive")

    if decomp_df is not None:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["observed"], name="Observed Series", line=dict(color="#102C26", width=1.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["trend"], name="Macro Secular Trend", line=dict(color="#527A50", width=2.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["seasonal"], name="Annual Seasonal Component (s=12)", line=dict(color="#C4A66A", width=1.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["residual"], name="Stochastic Irregular Residual", line=dict(color="#C24134", width=1.0, dash="dot")))

        format_plotly_figure(fig_d, title="Additive Decomposition: Observed = Trend + Seasonality + Residual", height=440)
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.warning("Insufficient observations for full 12-month annual seasonal decomposition.")

# ----------------- Tab 4: ACF & PACF -----------------
with tab_acf:
    st.subheader("Autocorrelation (ACF) & Partial Autocorrelation (PACF) Diagnostics")
    acf_pacf = compute_acf_pacf_values(df_monthly, target_col=target_col, nlags=20)

    if acf_pacf["lags"]:
        ci = acf_pacf["confidence_interval"]
        col_a, col_p = st.columns(2)

        with col_a:
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(x=acf_pacf["lags"], y=acf_pacf["acf"], name="ACF", marker_color="#183D32"))
            fig_a.add_hline(y=ci, line_dash="dash", line_color="#C4A66A", annotation_text="+95% Bartlett CI")
            fig_a.add_hline(y=-ci, line_dash="dash", line_color="#C4A66A", annotation_text="-95% Bartlett CI")
            format_plotly_figure(fig_a, title="Autocorrelation Function (ACF)", height=340)
            fig_a.update_layout(xaxis_title="Lag (Months)", yaxis_title="ACF")
            st.plotly_chart(fig_a, use_container_width=True)

        with col_p:
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=acf_pacf["lags"], y=acf_pacf["pacf"], name="PACF", marker_color="#527A50"))
            fig_p.add_hline(y=ci, line_dash="dash", line_color="#C4A66A", annotation_text="+95% Bartlett CI")
            fig_p.add_hline(y=-ci, line_dash="dash", line_color="#C4A66A", annotation_text="-95% Bartlett CI")
            format_plotly_figure(fig_p, title="Partial Autocorrelation Function (PACF)", height=340)
            fig_p.update_layout(xaxis_title="Lag (Months)", yaxis_title="PACF")
            st.plotly_chart(fig_p, use_container_width=True)

# ----------------- Tab 5: Stationarity Suite -----------------
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
                    <b>Null Hypothesis (H₀):</b> Unit root present (series is non-stationary)<br>
                    <b>Test Statistic:</b> <code>{adf_res.get('test_statistic')}</code><br>
                    <b>p-value:</b> <code>{adf_res.get('p_value')}</code><br>
                    <b>Critical Value (5%):</b> <code>{adf_res.get('critical_values', {}).get('5%')}</code><br>
                    <b>Lags Selected:</b> {adf_res.get('used_lag')}
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
                    <b>Null Hypothesis (H₀):</b> Series is trend-stationary (no unit root)<br>
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
        f"### 🎯 Synthesized Joint Stationarity Decision: **{decision['classification']}**  \n"
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

# ----------------- Tab 6: Chronological Validation Split -----------------
with tab_split:
    st.subheader("⏳ Chronological Train, Validation & Test Partitioning")
    st.markdown(
        "To prevent future lookahead data leakage, observations are strictly partitioned by chronological sequence. "
        "No temporal shuffling or k-fold random slicing is applied."
    )

    train_df, val_df, test_df = chronological_split(df_monthly, train_ratio=0.70, val_ratio=0.15)

    split_fig = go.Figure()
    split_fig.add_trace(go.Scatter(x=train_df["date"], y=train_df[target_col], name="Training Set (70%)", mode="lines+markers", line=dict(color="#102C26", width=2.5)))
    split_fig.add_trace(go.Scatter(x=val_df["date"], y=val_df[target_col], name="Validation Set (15%)", mode="lines+markers", line=dict(color="#C4A66A", width=2.5)))
    split_fig.add_trace(go.Scatter(x=test_df["date"], y=test_df[target_col], name="Test Set (15%)", mode="lines+markers", line=dict(color="#2563EB", width=2.5)))

    format_plotly_figure(split_fig, title="Chronological Validation Partitions (Zero Future Leakage)", height=380)
    split_fig.update_layout(yaxis_title="Documented Occurrences", xaxis_title="Timeline")
    st.plotly_chart(split_fig, use_container_width=True)

    st.markdown(
        f"""
        <div style="display: flex; gap: 12px; margin-top: 10px;">
            <div style="flex: 1; background: rgba(16, 44, 38, 0.08); border: 1px solid #102C26; border-radius: 8px; padding: 10px; text-align: center;">
                <strong>Train (70%):</strong> {train_df['date'].iloc[0].strftime('%b %Y')} – {train_df['date'].iloc[-1].strftime('%b %Y')} ({len(train_df)} Mos)
            </div>
            <div style="flex: 1; background: rgba(196, 166, 106, 0.15); border: 1px solid #C4A66A; border-radius: 8px; padding: 10px; text-align: center;">
                <strong>Validation (15%):</strong> {val_df['date'].iloc[0].strftime('%b %Y')} – {val_df['date'].iloc[-1].strftime('%b %Y')} ({len(val_df)} Mos)
            </div>
            <div style="flex: 1; background: rgba(37, 99, 235, 0.12); border: 1px solid #2563EB; border-radius: 8px; padding: 10px; text-align: center;">
                <strong>Test (15%):</strong> {test_df['date'].iloc[0].strftime('%b %Y')} – {test_df['date'].iloc[-1].strftime('%b %Y')} ({len(test_df)} Mos)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
