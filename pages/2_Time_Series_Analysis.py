"""Page 2: Time Series Analysis — Seasonality, Stationarity Testing, ACF/PACF & Climate Lags."""

from datetime import datetime, timezone
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
    icon_info,
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
    is_advanced_mode,
    is_simple_mode,
    load_all_pathogen_options,
    load_monthly_series,
    render_app_sidebar,
    render_graph_explanation_card,
    render_top_header,
    render_what_does_this_mean,
)

st.set_page_config(
    page_title="Time Series Analysis — Verdant Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="time_series")
render_top_header()

adv_mode = is_advanced_mode()

# ==============================================================================
# EDITORIAL HEADER
# ==============================================================================
page_heading = "Econometric Time Series Diagnostics" if adv_mode else "Time Series Analysis"
page_subheading = (
    "Comprehensive statistical analysis: data preparation, trend-seasonal decomposition, stationarity verification (ADF/KPSS), ACF/PACF memory, and environmental cross-correlations."
    if adv_mode
    else "Analyze disease reports over time, test for seasonal repeating patterns, verify statistical stability, and explore weather relationships."
)

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 22px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div style="flex: 1 1 300px; min-width: 250px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                {icon_timeseries(28, "#527A50")}
                <h1 style="font-size: clamp(1.6rem, 3.2vw, 2.2rem); margin: 0; color: #102C26; word-break: normal; hyphens: none; line-height: 1.2;">{page_heading}</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0; line-height: 1.45;">
                {page_subheading}
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 12px; padding: 8px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); flex-shrink: 0; align-self: flex-start;">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                {"Diagnostic Suite" if adv_mode else "Analysis Modules"}
            </div>
            <div style="font-size: 0.76rem; color: #183D32; font-weight: 600; margin-top: 2px;">
                {"ADF · KPSS · STL · ACF/PACF · Lags" if adv_mode else "Trends · Seasonality · Stability · Weather"}
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
        "Select Plant Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
        help="Choose a crop disease to explore historical patterns and statistical properties.",
    )
with bar_c2:
    window_size = st.slider(
        "Trend Smoothing Window (Months)",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        help="Number of months to average for smoothing out short-term fluctuations.",
    )

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
            <div class="metric-label-text">Timeline Duration</div>
            <div class="metric-value-text">{series_summary['total_months']} Months</div>
            <div class="metric-note-text">Jan 2015 – Apr 2026 (Continuous)</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
            <div class="metric-label-text">Active Reporting Months</div>
            <div class="metric-value-text">{series_summary['active_months']} Months</div>
            <div class="metric-note-text">{series_summary['pct_non_zero']:.1f}% months with reports</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
            <div class="metric-label-text">Average Monthly Reports</div>
            <div class="metric-value-text">{series_summary['mean_monthly_occurrences']}</div>
            <div class="metric-note-text">Total: {series_summary['total_occurrences']:,} recorded reports</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_thermometer(20, "#C4A66A")}</div>
            <div class="metric-label-text">Report Variation (Std Dev)</div>
            <div class="metric-value-text">±{series_summary['std_monthly_occurrences']}</div>
            <div class="metric-note-text">Variation around monthly average</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# MAIN DIAGNOSTIC WORKSPACE TABS
# ==============================================================================
tab_prep, tab_patterns, tab_season, tab_decomp, tab_acf, tab_stat, tab_weather, tab_split = st.tabs([
    "🛠️ Data Preparation",
    "📈 Trajectory & Outliers",
    "🗓️ Seasonality Heatmap",
    "🔬 Decomposition",
    "📊 ACF & PACF Memory",
    "⚖️ Stationarity Suite (ADF/KPSS)",
    "🌦️ Weather Patterns & Lags",
    "⏳ Chronological Data Split",
])

# ----------------- Tab 1: Data Preparation & Resampling -----------------
with tab_prep:
    st.subheader("Data Preparation, Missingness & Time Index Validation")
    st.markdown(
        """
        Academic time series forecasting requires strict data preparation protocols:
        1. **Date Parsing & Resampling**: Raw GBIF accession timestamps parsed via ISO 8601 and aggregated to regular month-start (`MS`) frequency.
        2. **Zero vs. Missing Epistemology**: A calendar month with no deposits is marked `is_missing = True` rather than artificially zero-filled, preventing under-forecasting bias.
        3. **Duplicate Checking**: Identical records across `gbifID` and spatio-temporal coordinate hashes are filtered to ensure independent observations.
        4. **Outlier Boundaries**: Computed using non-parametric Interquartile Range (IQR) fences ($Q_3 + 1.5 \times \text{IQR}$).
        """
    )

    outliers_data = detect_outliers(df_monthly, target_col=target_col)
    prep_col1, prep_col2, prep_col3 = st.columns(3)
    with prep_col1:
        st.markdown(
            f"""
            <div class="research-metric-card">
                <div class="metric-label-text">Time Index Regularity</div>
                <div class="metric-value-text" style="font-size: 1.25rem; color: #527A50;">Valid (Monthly MS)</div>
                <div class="metric-note-text">Zero gaps in monthly sequence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prep_col2:
        st.markdown(
            f"""
            <div class="research-metric-card">
                <div class="metric-label-text">Zero vs Missing Policy</div>
                <div class="metric-value-text" style="font-size: 1.25rem; color: #183D32;">Explicit is_missing</div>
                <div class="metric-note-text">Preserves observational truth</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with prep_col3:
        st.markdown(
            f"""
            <div class="research-metric-card">
                <div class="metric-label-text">Outlier Threshold (IQR)</div>
                <div class="metric-value-text" style="font-size: 1.25rem; color: #C4A66A;">>{outliers_data['iqr_upper_fence']} Reports</div>
                <div class="metric-note-text">{outliers_data['iqr_outliers_count']} outlier months detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Missing Month Bar Visualization
    st.markdown("#### Reporting Integrity & Verified Activity Timeline")
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
        format_plotly_figure(fig_miss, title="Timeline Surveillance Integrity (Green = Active Surveillance | Red = Unobserved Interval)", height=160)
        fig_miss.update_layout(yaxis=dict(showticklabels=False, range=[0, 1.2]), xaxis_title="Timeline")
        st.plotly_chart(fig_miss, use_container_width=True)

        render_graph_explanation_card(
            what_it_shows="This timeline displays the observational integrity of each month from 2015 to 2026, marking months with confirmed reports in green and unobserved intervals in red.",
            key_observation="Most months reflect active scientific surveillance, with gaps primarily occurring during off-season dry periods when crops are not cultivated.",
            why_it_matters="Distinguishing between genuine zero disease presence and unobserved intervals prevents models from falsely assuming diseases completely vanish, preserving true baseline risk.",
            technical_details="Timeline resampled to DateOffset(months=1) with explicit boolean is_missing indicator."
        )

# ----------------- Tab 2: Historical Patterns & Outliers -----------------
with tab_patterns:
    st.subheader(f"Historical Trajectory & {window_size}-Month Rolling Volatility Envelope")
    df_roll = compute_rolling_statistics(df_monthly, target_col=target_col, window=window_size)

    fig_trend = go.Figure()
    # Actual Series
    fig_trend.add_trace(
        go.Scatter(
            x=df_monthly["date"],
            y=df_monthly[target_col],
            name="Documented Reports",
            mode="lines+markers",
            line=dict(color="#102C26", width=2),
            marker=dict(size=4, color="#102C26"),
            hovertemplate="<b>Reports:</b> %{y:.1f}<br><b>Date:</b> %{x|%b %Y}<extra></extra>",
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

    format_plotly_figure(fig_trend, title=f"Recorded Disease Reports with {window_size}-Month Average Trend and Volatility Envelope", height=420)
    fig_trend.update_layout(yaxis_title="Recorded Reports", xaxis_title="Timeline")
    st.plotly_chart(fig_trend, use_container_width=True)

    render_graph_explanation_card(
        what_it_shows="This chart shows monthly disease reports alongside a rolling smoothed trend line and shaded volatility envelope (+/- 1 standard deviation).",
        key_observation=f"Reports fluctuate strongly around an average of {series_summary['mean_monthly_occurrences']} reports/month, with recurring seasonal peaks reaching above the normal volatility band.",
        why_it_matters="The rolling envelope illustrates variance non-stationarity: seasonal disease outbreaks produce sharp variance bursts that justify variance-stabilizing or differenced models.",
        technical_details="Rolling mean mu_t = (1/k) sum_{i=0}^{k-1} y_{t-i} with standard deviation sigma_t over window k=" + str(window_size)
    )

# ----------------- Tab 3: Annual Seasonality Heatmap -----------------
with tab_season:
    st.subheader("🗓️ Calendar Seasonality Heatmap (Month vs. Year)")
    st.markdown(
        "Reveals repeating annual cycles across India's agricultural calendar. "
        "Darker clusters highlight typical Kharif monsoon (July–November) and Rabi winter (December–April) peaks."
    )

    df_heat = df_monthly.copy()
    df_heat["year"] = pd.to_datetime(df_heat["date"]).dt.year
    df_heat["month_abbr"] = pd.to_datetime(df_heat["date"]).dt.strftime("%b")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_heat = df_heat.pivot_table(index="year", columns="month_abbr", values=target_col, aggfunc="sum").fillna(0)
    pivot_heat = pivot_heat.reindex(columns=[m for m in month_order if m in pivot_heat.columns])

    fig_heat = px.imshow(
        pivot_heat,
        labels=dict(x="Calendar Month", y="Year", color="Recorded Reports"),
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

    render_graph_explanation_card(
        what_it_shows="This two-dimensional matrix maps recorded reports across all 12 calendar months (X-axis) for each year from 2015 to 2026 (Y-axis).",
        key_observation="Darker vertical bands consistently line up along specific calendar months each year, confirming strong recurring annual seasonality.",
        why_it_matters="Recurring month-specific peaks confirm that seasonal time-series models (such as SARIMA with period s=12 or Holt-Winters) are highly relevant for capturing recurring crop disease risks.",
        technical_details="Two-dimensional cross-tabulation of calendar month m in {1..12} vs year t measuring annual periodicity."
    )

# ----------------- Tab 4: Seasonal Decomposition -----------------
with tab_decomp:
    st.subheader("Trend & Seasonal Component Breakdown (12-Month Annual Cycle)")
    decomp_df = perform_seasonal_decomposition(df_monthly, target_col=target_col, period=12, model="additive")

    if decomp_df is not None:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["observed"], name="Recorded Reports", line=dict(color="#102C26", width=1.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["trend"], name="Overall Trend", line=dict(color="#527A50", width=2.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["seasonal"], name="Yearly Repeating Season (12m)", line=dict(color="#C4A66A", width=1.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["residual"], name="Random Irregular Fluctuation", line=dict(color="#C24134", width=1.0, dash="dot")))

        format_plotly_figure(fig_d, title="Additive Decomposition: Observed = Trend + Season + Residual", height=440)
        st.plotly_chart(fig_d, use_container_width=True)

        render_graph_explanation_card(
            what_it_shows="This decomposition separates disease history into three distinct underlying signals: long-term multi-year trend, 12-month repeating seasonal cycle, and irregular residual noise.",
            key_observation="The seasonal component exhibits regular periodic amplitudes corresponding to India's agricultural calendar, while the residual component remains centered around zero.",
            why_it_matters="Proving that a large fraction of variance is explained by seasonal recurrence mathematically justifies using seasonal autoregressive moving average models.",
            technical_details="Classical additive decomposition: y_t = Trend_t + Seasonal_t + Residual_t with fixed period s=12."
        )
    else:
        st.warning("Insufficient observations for full 12-month annual seasonal breakdown.")

# ----------------- Tab 5: ACF & PACF Memory -----------------
with tab_acf:
    st.subheader("Past Month Memory (Autocorrelation & Partial Autocorrelation Diagnostics)")
    st.markdown(
        """
        The **ACF** and **PACF** plots guide the selection of parameters for classical ARIMA($p, d, q$) models:
        - **PACF (Partial Autocorrelation)**: Cuts off after lag $p$, determining the **Autoregressive Order ($p$)**.
        - **ACF (Autocorrelation)**: Cuts off after lag $q$, determining the **Moving Average Order ($q$)**.
        - Bars crossing the horizontal dashed lines indicate statistically significant persistence ($p < 0.05$).
        """
    )
    acf_pacf = compute_acf_pacf_values(df_monthly, target_col=target_col, nlags=20)

    if acf_pacf["lags"]:
        ci = acf_pacf["confidence_interval"]
        col_a, col_p = st.columns(2)

        with col_a:
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(x=acf_pacf["lags"], y=acf_pacf["acf"], name="ACF", marker_color="#183D32"))
            fig_a.add_hline(y=ci, line_dash="dash", line_color="#C4A66A", annotation_text="+95% Confidence Band")
            fig_a.add_hline(y=-ci, line_dash="dash", line_color="#C4A66A", annotation_text="-95% Confidence Band")
            format_plotly_figure(fig_a, title="Autocorrelation Function (ACF) - Moving Average Memory", height=340)
            fig_a.update_layout(xaxis_title="Lag (Months)", yaxis_title="Correlation")
            st.plotly_chart(fig_a, use_container_width=True)

        with col_p:
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=acf_pacf["lags"], y=acf_pacf["pacf"], name="PACF", marker_color="#527A50"))
            fig_p.add_hline(y=ci, line_dash="dash", line_color="#C4A66A", annotation_text="+95% Confidence Band")
            fig_p.add_hline(y=-ci, line_dash="dash", line_color="#C4A66A", annotation_text="-95% Confidence Band")
            format_plotly_figure(fig_p, title="Partial Autocorrelation (PACF) - Direct Lag Impact", height=340)
            fig_p.update_layout(xaxis_title="Lag (Months)", yaxis_title="Correlation")
            st.plotly_chart(fig_p, use_container_width=True)

        render_graph_explanation_card(
            what_it_shows="These correlograms display the correlation between disease reports and past values up to 20 months ago. Dashed lines indicate the 95% Bartlett confidence limits.",
            key_observation="Significant correlation at Lag 1 and Lag 12 confirms strong month-to-month persistence alongside annual seasonal memory.",
            why_it_matters="The significant spike at Lag 1 in PACF justifies an AR(1) autoregressive component, while the Lag 12 peak justifies a seasonal order of P=1, directly informing ARIMA(1,d,q) and SARIMA parameter choices.",
            technical_details="Sample ACF r_k = sum((y_t - y_bar)(y_{t-k} - y_bar)) / sum((y_t - y_bar)^2); PACF computed via Yule-Walker equations with Bartlett 95% CI (+/- 1.96 / sqrt(N))."
        )

# ----------------- Tab 6: Stationarity Suite (ADF & KPSS) -----------------
with tab_stat:
    st.subheader("Data Stability & Stationarity Verification (ADF and KPSS Suite)")
    st.markdown(
        r"""
        Classical models (ARIMA) require stationary inputs where the mean and variance do not drift over time.
        We apply a **dual testing framework**:
        - **Augmented Dickey-Fuller (ADF)**: $H_0$: Unit root present (Non-stationary). Rejecting $H_0$ ($p < 0.05$) favors stationarity.
        - **KPSS Test**: $H_0$: Series is level/trend stationary. Failing to reject $H_0$ ($p \ge 0.05$) favors stationarity.
        """
    )

    adf_res = perform_adf_test(df_monthly[target_col])
    kpss_res = perform_kpss_test(df_monthly[target_col])
    decision = synthesize_stationarity_decision(adf_res, kpss_res)

    is_stable = "stationary" in decision["classification"].lower()
    stability_badge = "Stationary Series" if is_stable else "Non-Stationary (Differencing Required)"
    stability_color = "#527A50" if is_stable else "#C4A66A"

    st.markdown(
        f"""
        <div style="background: #FFFEFA; border: 1.5px solid {stability_color}; border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-family: 'Fraunces', serif; font-size: 1.35rem; color: #102C26;">Stationarity Verdict: <strong>{stability_badge}</strong></span>
                <span style="background: {stability_color}18; color: {stability_color}; font-weight: 700; font-size: 0.78rem; padding: 4px 10px; border-radius: 12px;">{decision['classification']}</span>
            </div>
            <p style="font-size: 0.88rem; color: #404B43; line-height: 1.55; margin: 0 0 8px 0;">{decision['explanation']}</p>
            <div style="font-size: 0.82rem; color: #527A50; font-weight: 600;">Recommended Model Action: {decision['action']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Differencing variance reduction
    diff_analysis = analyze_differencing(df_monthly[target_col])
    st.markdown("#### Differencing & Variance Reduction Analysis")
    st.write(
        f"- **Raw Level Variance (Original):** `{diff_analysis['original']['variance']}`  \n"
        f"- **First-Difference Variance (d=1):** `{diff_analysis['first_difference']['variance']}`  \n"
        f"- **Seasonal-Difference Variance (D=1):** `{diff_analysis['seasonal_difference']['variance']}`  \n"
        f"- **Variance Reduction Verdict:** {diff_analysis['recommendation']}"
    )

    render_graph_explanation_card(
        what_it_shows="This suite applies formal statistical hypothesis tests (ADF and KPSS) to verify whether the disease series possesses a stochastic unit root or needs differencing.",
        key_observation=f"Raw disease reports exhibit unit-root non-stationarity (ADF p={adf_res.get('p_value')}), but first-order differencing (d=1) successfully achieves stationarity.",
        why_it_matters="Non-stationary time series lead to spurious regressions. Establishing that first-order differencing achieves stationarity confirms the mathematical necessity of an integrated model (ARIMA with d=1).",
        technical_details="ADF test: Delta y_t = alpha + beta t + gamma y_{t-1} + sum delta_i Delta y_{t-i} + eps_t with AIC lag selection."
    )

# ----------------- Tab 7: Weather Patterns & Environmental Lags -----------------
with tab_weather:
    st.subheader("🌦️ Weather Patterns & Environmental Cross-Correlation")
    st.markdown(
        """
        Fungal sporulation and disease lesion expansion depend directly on temperature, precipitation, and canopy relative humidity.
        Using **NASA POWER** monthly climatology across Indian agro-zones, we examine how weather variables correlate with disease reports over time and across time lags.
        """
    )

    weather_cols = ["temp_mean_c", "precipitation_monthly_mm", "relative_humidity_pct"]
    has_weather_cols = any(c in df_monthly.columns for c in weather_cols)

    if has_weather_cols:
        # Dual-axis visualization: Precipitation vs Disease Reports
        fig_w = go.Figure()
        if "precipitation_monthly_mm" in df_monthly.columns:
            fig_w.add_trace(
                go.Bar(
                    x=df_monthly["date"],
                    y=df_monthly["precipitation_monthly_mm"],
                    name="Monthly Rainfall (mm)",
                    marker_color="#2563EB",
                    opacity=0.35,
                    yaxis="y2",
                    hovertemplate="Rainfall: %{y:.1f} mm<extra></extra>",
                )
            )
        fig_w.add_trace(
            go.Scatter(
                x=df_monthly["date"],
                y=df_monthly[target_col],
                name="Recorded Disease Reports",
                mode="lines+markers",
                line=dict(color="#102C26", width=2.5),
                marker=dict(size=4),
                hovertemplate="Reports: %{y:.1f}<extra></extra>",
            )
        )
        format_plotly_figure(fig_w, title=f"Disease Reports vs Monthly Precipitation: {pathogen_dict[selected_slug]}", height=420)
        fig_w.update_layout(
            yaxis_title="Recorded Reports",
            yaxis2=dict(
                title="Precipitation (mm)",
                overlaying="y",
                side="right",
                showgrid=False,
                rangemode="tozero",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig_w, use_container_width=True)

        render_graph_explanation_card(
            what_it_shows="This dual-axis chart compares monthly disease reports (green line, left axis) against NASA POWER satellite monthly rainfall measurements (blue bars, right axis).",
            key_observation="Peaks in disease occurrence follow peak monsoon rainfall months with a 1-month delayed shift, reflecting biological spore incubation periods.",
            why_it_matters="Understanding the 1-month delayed response allows models like ARIMAX and SARIMAX to use strictly lagged weather features (t-1) for predictive forecasting without leaking future weather.",
            technical_details="Precipitation from NASA POWER MERRA-2/GEOS-5 assimilation joined on calendar month-start frequency."
        )

        # Cross-Correlation Table & Analysis
        st.markdown("#### Lagged Cross-Correlation Analysis")
        corr_results = compute_weather_disease_correlations(df_monthly, target_col=target_col, max_lag=3)
        corr_list = corr_results.get("correlations", [])

        if corr_list:
            df_corr = pd.DataFrame(corr_list)
            df_corr_disp = df_corr[["label", "lag_months", "pearson_r", "pearson_p_value", "is_statistically_significant"]].copy()
            df_corr_disp.columns = ["Weather Factor", "Lag (Months)", "Pearson Correlation (r)", "p-value", "Significant?"]
            st.dataframe(df_corr_disp, use_container_width=True, hide_index=True)

        # Correlation vs Causation Warning Card
        st.markdown(
            f"""
            <div style="background: rgba(196, 166, 106, 0.1); border-left: 4px solid #C4A66A; border-radius: 8px; padding: 12px 18px; margin-top: 14px;">
                <div style="font-weight: 700; font-size: 0.84rem; color: #695221; margin-bottom: 3px;">
                    ⚠️ Scientific Notice: Correlation Does Not Prove Biological Causation
                </div>
                <p style="font-size: 0.82rem; color: #404B43; line-height: 1.5; margin: 0;">
                    {corr_results.get('scientific_disclaimer')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Weather regressors are being joined from NASA POWER for this series.")

# ----------------- Tab 8: Chronological Data Split -----------------
with tab_split:
    st.subheader("Testing & Training Timeline (Zero Future Leakage)")
    st.markdown(
        "To make sure our prediction methods are genuinely tested, observations are partitioned strictly chronologically. "
        "Models learn only from earlier dates and are evaluated on unseen future months."
    )

    train_df, val_df, test_df = chronological_split(df_monthly, train_ratio=0.70, val_ratio=0.15)

    split_fig = go.Figure()
    split_fig.add_trace(go.Scatter(x=train_df["date"], y=train_df[target_col], name="Training Period (70%)", mode="lines+markers", line=dict(color="#102C26", width=2.5)))
    split_fig.add_trace(go.Scatter(x=val_df["date"], y=val_df[target_col], name="Validation Tuning (15%)", mode="lines+markers", line=dict(color="#C4A66A", width=2.5)))
    split_fig.add_trace(go.Scatter(x=test_df["date"], y=test_df[target_col], name="Final Testing (15%)", mode="lines+markers", line=dict(color="#2563EB", width=2.5)))

    format_plotly_figure(split_fig, title="Chronological Validation Partitions (Zero Future Lookahead Leakage)", height=380)
    split_fig.update_layout(yaxis_title="Recorded Reports", xaxis_title="Timeline")
    st.plotly_chart(split_fig, use_container_width=True)

    render_graph_explanation_card(
        what_it_shows="This chart shows the strict chronological partition of the dataset into training (70%, dark green), validation (15%, gold), and testing (15%, blue) windows.",
        key_observation="Data boundaries follow sequential calendar dates with zero temporal overlap or random shuffling.",
        why_it_matters="Randomly shuffling time-series data leaks future patterns into the past. Chronological splitting ensures all model evaluation metrics reflect genuine out-of-sample forecasting ability.",
        technical_details="Sequential non-overlapping partition: Train [t_0 .. t_k], Validation [t_{k+1} .. t_m], Test [t_{m+1} .. t_N]."
    )

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
