"""Page 2: Time Series Analysis — Seasonality, Stationarity Testing, ACF/PACF & Climate Lags.

Minimal, academic, and interactive interface for rigorous time-series diagnostics.
"""

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
    icon_activity,
    icon_bar_chart,
    icon_check_circle,
    icon_info,
    icon_leaf,
    icon_pulse,
    icon_shield,
    icon_thermometer,
    icon_timeseries,
    icon_weather,
)
from src.project_stats import get_project_telemetry
from src.stationarity import (
    analyze_differencing,
    perform_adf_test,
    perform_kpss_test,
    perform_ljung_box_test,
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
    page_title="Time Series Analysis — Agricultural Disease Forecasting Using Time Series",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_app_sidebar(active_page="timeseries")
render_top_header()

stats = get_project_telemetry()

# ==============================================================================
# 1. EDITORIAL HEADER
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div style="flex: 1 1 300px; min-width: 250px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_timeseries(26, "#527A50")}
                <h1 style="font-size: clamp(1.5rem, 3vw, 2.0rem); margin: 0; color: #102C26; word-break: normal; hyphens: none; line-height: 1.2;">Time Series Analysis &amp; Statistical Diagnostics</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.92rem; margin: 0; line-height: 1.45;">
                Longitudinal trajectory, stationarity tests (ADF/KPSS), seasonal decomposition, ACF/PACF memory, and NASA agrometeorological lags.
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 10px; padding: 6px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02); flex-shrink: 0; align-self: flex-start;">
            <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                Econometric Diagnostic Suite
            </div>
            <div style="font-size: 0.75rem; color: #183D32; font-weight: 600; margin-top: 1px;">
                Standard Grid: {stats['monthly_start_date']} – {stats['monthly_end_date']} ({stats['total_months']} Mo)
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pathogen_dict = load_all_pathogen_options()

# Parameter Toolbar
bar_c1, bar_c2 = st.columns([1.6, 1.0])
with bar_c1:
    selected_slug = st.selectbox(
        "Target Crop Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
        help="Select a crop disease series to run econometric diagnostics.",
    )
with bar_c2:
    window_size = st.slider(
        "Rolling Smoothing Window (Months)",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        help="Window size for moving average and standard deviation envelope.",
    )

df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"Monthly time series not found for {selected_slug}.")
    st.stop()

target_col = "documented_occurrence_count"
series_summary = compute_series_summary(df_monthly, target_col=target_col)
suitability = assess_forecasting_suitability(df_monthly, target_col=target_col, pathogen_name=pathogen_dict[selected_slug])
date_min_str = pd.to_datetime(df_monthly["date"]).min().strftime("%b %Y")
date_max_str = pd.to_datetime(df_monthly["date"]).max().strftime("%b %Y")
year_span_str = f"{pd.to_datetime(df_monthly['date']).min().year}–{pd.to_datetime(df_monthly['date']).max().year}"

# 4 Compact KPI Cards
st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 10px; margin-bottom: 20px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(18, "#527A50")}</div>
            <div class="metric-label-text">Timeline Span</div>
            <div class="metric-value-text">{series_summary['total_months']} Months</div>
            <div class="metric-note-text">{date_min_str} – {date_max_str} (Continuous)</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(18, "#527A50")}</div>
            <div class="metric-label-text">Active Reporting</div>
            <div class="metric-value-text">{series_summary['active_months']} Months</div>
            <div class="metric-note-text">{series_summary['pct_non_zero']:.1f}% Positive · {100.0 - series_summary['pct_non_zero']:.1f}% Zeros</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(18, "#527A50")}</div>
            <div class="metric-label-text">Monthly Mean</div>
            <div class="metric-value-text">{series_summary['mean_monthly_occurrences']}</div>
            <div class="metric-note-text">Total: {series_summary['total_occurrences']:,} Recorded Reports</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_thermometer(18, "#C4A66A")}</div>
            <div class="metric-label-text">Standard Deviation</div>
            <div class="metric-value-text">±{series_summary['std_monthly_occurrences']}</div>
            <div class="metric-note-text">Sample Standard Deviation</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CONSOLIDATED FIVE-TAB WORKSPACE (CHART FIRST, INTERPRETATION BELOW)
# ==============================================================================
tab_trend, tab_decomp, tab_stationarity, tab_acf, tab_weather = st.tabs([
    "1. Trajectory & Trend",
    "2. Seasonal Decomposition",
    "3. Stationarity & Differencing",
    "4. Autocorrelation (ACF / PACF)",
    "5. Weather Cross-Correlation",
])

# ----------------- Tab 1: Trajectory & Trend -----------------
with tab_trend:
    st.markdown(f"**Historical Occurrence Trajectory and Rolling Mean ({year_span_str})**")
    rolling_df = compute_rolling_statistics(df_monthly, window=window_size, target_col=target_col)
    
    fig_tr = go.Figure()
    fig_tr.add_trace(
        go.Bar(
            x=df_monthly["date"],
            y=df_monthly[target_col],
            name="Monthly Occurrences",
            marker_color="#183D32",
            opacity=0.75,
            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Reports:</b> %{y}<extra></extra>",
        )
    )
    fig_tr.add_trace(
        go.Scatter(
            x=rolling_df["date"],
            y=rolling_df[f"rolling_mean_{window_size}m"],
            name=f"{window_size}-Month Moving Average",
            line=dict(color="#C4A66A", width=2.5),
        )
    )
    format_plotly_figure(
        fig_tr,
        title=f"Monthly Reports and {window_size}-Month Moving Average Trend ({date_min_str} – {date_max_str})",
        height=360,
    )
    st.plotly_chart(fig_tr, use_container_width=True)
    
    st.caption(
        f"**Empirical Trend Interpretation:** The bar series highlights discrete monthly reporting pulses, while the gold curve tracks the {window_size}-month rolling baseline. "
        f"For {pathogen_dict[selected_slug]}, activity alternates between zero-count dormancy intervals and seasonal epidemic bursts."
    )
    
    with st.expander("📊 Outlier Detection & Distribution Bounds", expanded=False):
        outliers_data = detect_outliers(df_monthly, target_col=target_col)
        c_o1, c_o2, c_o3 = st.columns(3)
        with c_o1:
            st.metric("IQR Upper Fence", f"{outliers_data['iqr_upper_fence']:.1f} Reports")
        with c_o2:
            st.metric("Detected Outliers", f"{outliers_data['iqr_outliers_count']} Months")
        with c_o3:
            st.metric("Max Single-Month Spike", f"{df_monthly[target_col].max():.0f} Reports")
        st.write("Outliers represent genuine epidemiological outbreak surges rather than data errors. They reflect favorable climatic triggers during peak monsoonal incubation windows.")

# ----------------- Tab 2: Seasonal Decomposition -----------------
with tab_decomp:
    st.markdown(f"**Classical Seasonal-Trend-Loess (STL) Decomposition (Period = 12 Months)**")
    decomp_res = perform_seasonal_decomposition(df_monthly, period=12, model="additive", target_col=target_col)
    
    if decomp_res.get("success"):
        fig_d = go.Figure()
        dates = df_monthly["date"]
        fig_d.add_trace(go.Scatter(x=dates, y=decomp_res["observed"], name="Observed (Y_t)", line=dict(color="#102C26", width=1.5)))
        fig_d.add_trace(go.Scatter(x=dates, y=decomp_res["trend"], name="Trend (T_t)", line=dict(color="#C4A66A", width=2.0)))
        fig_d.add_trace(go.Scatter(x=dates, y=decomp_res["seasonal"], name="Seasonal (S_t, 12m)", line=dict(color="#047857", width=1.5)))
        fig_d.add_trace(go.Scatter(x=dates, y=decomp_res["residual"], name="Residual (e_t)", line=dict(color="#DC2626", width=1.2)))
        format_plotly_figure(fig_d, title=f"Additive Seasonal Decomposition: Y_t = Trend + Seasonal(12) + Residual", height=420)
        st.plotly_chart(fig_d, use_container_width=True)
        
        st.caption(
            "**Decomposition Interpretation:** The seasonal component (green) extracts repeating 12-month agrarian periodicity. "
            "Residuals isolate unexplained non-seasonal shocks, such as anomalous heavy rainfall events or localized reporting bursts."
        )
    else:
        st.info("Decomposition requires sufficient contiguous non-zero periods.")

# ----------------- Tab 3: Stationarity & Differencing -----------------
with tab_stationarity:
    st.markdown(f"**Stationarity Testing: ADF Unit-Root Test & KPSS Stationarity Test**")
    
    adf_res = perform_adf_test(df_monthly[target_col])
    kpss_res = perform_kpss_test(df_monthly[target_col])
    decision = synthesize_stationarity_decision(adf_res, kpss_res)
    
    # Synthesis Summary Table First
    stat_summary_df = pd.DataFrame([
        {
            "Test": "Augmented Dickey-Fuller (ADF)",
            "Null Hypothesis (H0)": "Unit Root Present (Non-Stationary)",
            "Test Statistic": f"{adf_res['test_statistic']:.4f}",
            "p-Value": f"{adf_res['p_value']:.4e}",
            "Critical 5%": f"{adf_res['critical_values']['5%']:.4f}",
            "Conclusion": "Reject H0 (Stationary)" if adf_res['is_stationary'] else "Fail to Reject (Non-Stationary)",
        },
        {
            "Test": "Kwiatkowski-Phillips-Schmidt-Shin (KPSS)",
            "Null Hypothesis (H0)": "Level Stationary",
            "Test Statistic": f"{kpss_res['test_statistic']:.4f}",
            "p-Value": f"{kpss_res['p_value']:.4f}",
            "Critical 5%": f"{kpss_res['critical_values']['5%']:.4f}",
            "Conclusion": "Fail to Reject (Stationary)" if kpss_res['is_stationary'] else "Reject H0 (Non-Stationary)",
        },
    ])
    st.dataframe(stat_summary_df, use_container_width=True, hide_index=True)
    
    # Differencing comparison chart
    diff_res = analyze_differencing(df_monthly[target_col])
    if diff_res.get("success"):
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(x=df_monthly["date"], y=df_monthly[target_col], name="Original Level (d=0)", line=dict(color="#102C26", width=1.5)))
        fig_diff.add_trace(go.Scatter(x=df_monthly["date"].iloc[1:], y=diff_res["first_diff"].dropna(), name="First Difference (d=1, Delta Y)", line=dict(color="#047857", width=1.5)))
        format_plotly_figure(fig_diff, title=f"Level vs. First Differenced Series (Variance: {diff_res['var_original']:.2f} -> {diff_res['var_first_diff']:.2f})", height=320)
        st.plotly_chart(fig_diff, use_container_width=True)
        
        st.caption(
            f"**Econometric Verdict:** Combined ADF/KPSS diagnostics classify this series as **{decision['joint_verdict']}**. "
            f"Taking first differences in data with many zero months increases artificial swings, explaining why ARIMA differencing degrades forecast accuracy on unseen data."
        )

    with st.expander("📐 Mathematical Formulations & Stationarity Theory", expanded=False):
        st.markdown(
            r"""
            **Augmented Dickey-Fuller (ADF) Regression:**
            $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t$$
            Testing $H_0: \gamma = 0$ (unit root exists) versus $H_1: \gamma < 0$ (mean-reverting stationary process).

            **KPSS Stationarity Formulation:**
            Decomposes $y_t = r_t + \beta t + \epsilon_t$, where $r_t = r_{t-1} + u_t$ is a random walk with variance $\sigma_u^2$.
            Testing $H_0: \sigma_u^2 = 0$ (level stationarity) versus $H_1: \sigma_u^2 > 0$ (random walk unit root).
            """
        )

# ----------------- Tab 4: Autocorrelation (ACF / PACF) -----------------
with tab_acf:
    st.markdown(f"**Sample Autocorrelation (ACF) and Partial Autocorrelation (PACF) up to Lag 24**")
    acf_pacf = compute_acf_pacf_values(df_monthly, target_col=target_col, nlags=24)
    
    if acf_pacf.get("success"):
        lags = list(range(len(acf_pacf["acf_values"])))
        ci_bound = acf_pacf["ci_upper"]
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(x=lags, y=acf_pacf["acf_values"], marker_color="#183D32", name="ACF"))
            fig_acf.add_hline(y=ci_bound, line_dash="dash", line_color="#C4A66A", annotation_text="95% CI")
            fig_acf.add_hline(y=-ci_bound, line_dash="dash", line_color="#C4A66A")
            format_plotly_figure(fig_acf, title="Autocorrelation Function (ACF)", height=320)
            st.plotly_chart(fig_acf, use_container_width=True)
            
        with col_a2:
            fig_pacf = go.Figure()
            fig_pacf.add_trace(go.Bar(x=lags, y=acf_pacf["pacf_values"], marker_color="#527A50", name="PACF"))
            fig_pacf.add_hline(y=ci_bound, line_dash="dash", line_color="#C4A66A", annotation_text="95% CI")
            fig_pacf.add_hline(y=-ci_bound, line_dash="dash", line_color="#C4A66A")
            format_plotly_figure(fig_pacf, title="Partial Autocorrelation Function (PACF)", height=320)
            st.plotly_chart(fig_pacf, use_container_width=True)
            
        st.caption(
            r"**Lag Structure Interpretation:** Significant spikes outside the 95% Bartlett bounds ($\pm 1.96 / \sqrt{N}$) indicate meaningful autoregressive memory. "
            "A sharp PACF cutoff at lag 1 supports an AR(1) or MA(1) specification, while elevation around lag 12 reflects annual monsoonal recurrence."
        )

# ----------------- Tab 5: Weather Cross-Correlation -----------------
with tab_weather:
    st.markdown(f"**NASA POWER Meteorological Telemetry & Lag Cross-Correlations**")
    corr_res = compute_weather_disease_correlations(df_monthly, target_col=target_col)
    
    if corr_res.get("success") and not corr_res["correlation_matrix"].empty:
        corr_df = corr_res["correlation_matrix"]
        fig_c = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=[[0, "#2563EB"], [0.5, "#FFFEFA"], [1, "#DC2626"]],
            title="Pearson Correlation: Weather Regressors vs. Disease Lags [t, t-1, t-2, t-3]",
        )
        format_plotly_figure(fig_c, title="Agrometeorological Cross-Correlation Matrix", height=380)
        st.plotly_chart(fig_c, use_container_width=True)
        
        st.caption(
            "**Bioclimatic Interpretation:** Relative humidity and monthly precipitation exhibit positive correlations at 1-month and 2-month lags, "
            "reflecting the biological incubation delay between monsoon leaf wetness and macroscopically observable foliar lesions."
        )
    else:
        st.info("Weather correlation requires valid climate regressor columns in the monthly dataset.")

    # Surveillance missingness integrity timeline
    if "is_missing" in df_monthly.columns:
        with st.expander("🔎 Surveillance Continuity & Missingness Quality", expanded=False):
            fig_miss = go.Figure()
            miss_colors = ["#DC2626" if m else "#527A50" for m in df_monthly["is_missing"]]
            fig_miss.add_trace(
                go.Bar(
                    x=df_monthly["date"],
                    y=[1] * len(df_monthly),
                    marker_color=miss_colors,
                    name="Surveillance Status",
                    hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Status:</b> %{text}<extra></extra>",
                    text=["Missing-Data Estimated Zero (is_missing=True)" if m else "Verified Active Surveillance" for m in df_monthly["is_missing"]],
                )
            )
            format_plotly_figure(fig_miss, title="Timeline Surveillance Integrity (Green = Active Surveillance | Red = Missing-Data Estimated Zero)", height=160)
            fig_miss.update_layout(yaxis=dict(showticklabels=False, range=[0, 1.2]), xaxis_title="Timeline")
            st.plotly_chart(fig_miss, use_container_width=True)
            st.caption("Green bars denote verified active surveillance months; red bars represent missing-data estimated zero intervals.")

# Navigation links
st.markdown("<br>", unsafe_allow_html=True)
try:
    c_bot1, c_bot2 = st.columns(2)
    with c_bot1:
        st.page_link("pages/3_Model_Comparison.py", label="Proceed to Model Comparison & Benchmark ->", use_container_width=True)
    with c_bot2:
        st.page_link("pages/4_Future_Forecasts.py", label="Generate Future Forecasts ->", use_container_width=True)
except Exception:
    pass
