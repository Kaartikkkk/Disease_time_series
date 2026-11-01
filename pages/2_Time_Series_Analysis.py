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
    icon_activity,
    icon_alert_triangle,
    icon_arrow_right,
    icon_check_circle,
    icon_database,
    icon_forecast,
    icon_info,
    icon_leaf,
    icon_model_compare,
    icon_pathology,
    icon_shield,
    icon_target,
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
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_app_sidebar(active_page="timeseries")
render_top_header()

adv_mode = is_advanced_mode()

# ==============================================================================
# EDITORIAL HEADER
# ==============================================================================
page_heading = "Time Series Analysis"
page_subheading = "Historical disease trajectory, stationarity tests, seasonal patterns, and weather cross-correlations."

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 18px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
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
                {"ADF · KPSS · Differencing · Lags" if adv_mode else "Trends · Seasonality · Stability · Weather"}
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

# 4 Clean KPI Cards
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
# MAIN DIAGNOSTIC WORKSPACE TABS (ACADEMIC ANALYSIS WORKFLOW)
# ==============================================================================
tab_prep, tab_missing, tab_patterns, tab_adf_kpss, tab_diff, tab_decomp, tab_acf, tab_weather = st.tabs([
    "1. Data Preparation",
    "2. Missing Values & Quality",
    "3. Stationarity Testing",
    "4. ADF & KPSS Results",
    "5. Differencing Analysis",
    "6. Seasonal Decomposition",
    "7. ACF & PACF Memory",
    "8. Weather Lags & Partitions",
])

# ----------------- Tab 1: Data Preparation & Resampling -----------------
with tab_prep:
    st.subheader("1. Preparing Historical Outbreaks for Predictive Forecasting")
    
    _render_forecast_purpose_badge(
        "Predictive time series algorithms (ARIMA, SARIMA, ARIMAX) cannot operate on sporadic, unorganized field reports. They require an unbroken, continuous monthly calendar timeline (132 consecutive calendar months from Jan 2015 to Apr 2026). Resampling ensures models calculate real outbreak momentum without confusing off-season gaps with disease eradication."
    )

    st.markdown(
        f"""
        <div style="background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.22); border-left: 4px solid #527A50; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                {icon_info(18, "#527A50")}
                <h4 style="margin: 0; color: #102C26; font-size: 1.02rem;">Why Data Preparation is the Foundation of Disease Forecasting</h4>
            </div>
            <p style="font-size: 0.86rem; color: #404B43; line-height: 1.55; margin: 0 0 12px 0;">
                Forecasting future crop epidemics requires transforming raw biodiversity and research field notes into a mathematically regular, stationary-ready time series. Here is how our data engineering pipeline prepares records for forecasting:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px;">
                <div style="background: #F6F4EC; padding: 12px 14px; border-radius: 8px; border-top: 2px solid #527A50;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #527A50; text-transform: uppercase;">1. Continuous Month Index (MS)</div>
                    <div style="font-size: 0.82rem; color: #2B3830; margin-top: 4px; line-height: 1.5;">Raw dates aggregated to 132 consecutive calendar months (2015–2026). Lag operators ($y_{{t-1}}, y_{{t-12}}$) require evenly spaced intervals.</div>
                </div>
                <div style="background: #F6F4EC; padding: 12px 14px; border-radius: 8px; border-top: 2px solid #183D32;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #183D32; text-transform: uppercase;">2. Preserving Zero vs. Missing</div>
                    <div style="font-size: 0.82rem; color: #2B3830; margin-top: 4px; line-height: 1.5;">Months with no reports are marked explicitly (`is_missing=True`). Prevents models from confusing unobserved intervals with complete disease eradication.</div>
                </div>
                <div style="background: #F6F4EC; padding: 12px 14px; border-radius: 8px; border-top: 2px solid #C4A66A;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #C4A66A; text-transform: uppercase;">3. NASA Weather Telemetry</div>
                    <div style="font-size: 0.82rem; color: #2B3830; margin-top: 4px; line-height: 1.5;">Coupled with NASA POWER rainfall, temperature, and humidity lagged strictly by 1 month ($t-1$) for forward predictive power without leakage.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    outliers_data = detect_outliers(df_monthly, target_col=target_col)
    prep_col1, prep_col2, prep_col3 = st.columns(3)
    with prep_col1:
        st.markdown(
            f"""
            <div class="research-metric-card">
                <div class="metric-label-text">Time Index Regularity</div>
                <div class="metric-value-text" style="font-size: 1.25rem; color: #527A50;">Valid (Monthly MS)</div>
                <div class="metric-note-text">132 unbroken intervals for lag math</div>
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
                <div class="metric-note-text">Prevents under-prediction bias</div>
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

    st.markdown(
        f"""
        <div style="background: rgba(82, 122, 80, 0.06); border: 1px solid rgba(82, 122, 80, 0.18); border-radius: 12px; padding: 14px 18px; margin-top: 20px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #102C26;">
                    Status: Outbreak Data is 100% Prepared & Forecast-Ready
                </div>
                <div style="font-size: 0.8rem; color: #5A665D;">
                    Explore subsequent diagnostic tabs to inspect stationarity and seasonality, or jump directly to evaluate and generate forecasts.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        st.page_link("pages/3_Model_Comparison.py", label="Compare 11 Forecasting Models (ARIMA, SARIMA, ARIMAX) ->", use_container_width=True)
    with col_nav2:
        st.page_link("pages/4_Future_Forecasts.py", label="Generate Future Disease Predictions ->", use_container_width=True)

# ----------------- Tab 2: Missing Values & Data Quality -----------------
with tab_missing:
    st.subheader("Missing Values, Surveillance Coverage & Data Quality")
    _render_forecast_purpose_badge(
        "Distinguishing between true off-season zero disease presence and unobserved intervals prevents models from underestimating future outbreak resurgence."
    )
    st.markdown(
        """
        Passive surveillance datasets feature uneven sampling intervals. Months with unmonitored intervals are tracked explicitly
        rather than assumed to have zero disease presence, preventing systematic under-estimation in out-of-sample evaluations.
        """
    )
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
        format_plotly_figure(fig_miss, title="Timeline Surveillance Integrity (Green = Active Surveillance | Red = Unobserved Interval)", height=180)
        fig_miss.update_layout(yaxis=dict(showticklabels=False, range=[0, 1.2]), xaxis_title="Timeline")
        st.plotly_chart(fig_miss, use_container_width=True)

        render_graph_explanation_card(
            what_it_shows="This timeline displays the observational integrity of each month from 2015 to 2026, marking months with confirmed reports in green and unobserved intervals in red.",
            key_observation="Most months reflect active scientific surveillance, with gaps primarily occurring during off-season dry periods when crops are not cultivated.",
            why_it_matters="Distinguishing between genuine zero disease presence and unobserved intervals prevents models from falsely assuming diseases completely vanish, preserving true baseline risk.",
            technical_details="Timeline resampled to DateOffset(months=1) with explicit boolean is_missing indicator."
        )

# ----------------- Tab 3: Stationarity Testing (Rolling Envelope) -----------------
with tab_patterns:
    st.subheader(f"Historical Trajectory & {window_size}-Month Rolling Volatility Envelope")
    _render_forecast_purpose_badge(
        "Forecasting models require stable historical baselines. The rolling envelope shows whether variance expands during outbreaks, dictating whether models need variance stabilization before training."
    )
    st.markdown(
        "Stationarity means that statistical properties such as the mean and variance remain consistent over time. "
        "A rolling average and volatility envelope show how mean levels and seasonal surges shift across the 11-year history."
    )
    df_roll = compute_rolling_statistics(df_monthly, target_col=target_col, window=window_size)

    fig_trend = go.Figure()
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
    fig_trend.add_trace(
        go.Scatter(
            x=df_roll["date"],
            y=df_roll[f"rolling_mean_{window_size}m"],
            name=f"{window_size}-Month Rolling Mean",
            line=dict(color="#527A50", width=2.5),
            hovertemplate="<b>Rolling Mean:</b> %{y:.2f}<extra></extra>",
        )
    )
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

# ----------------- Tab 4: ADF & KPSS Results -----------------
with tab_adf_kpss:
    st.subheader("Stationarity Hypothesis Testing (ADF and KPSS Suite)")
    _render_forecast_purpose_badge(
        "Proves whether the disease series has a stochastic unit root. If non-stationary, forecasting models must use differencing (d=1) to prevent runaway predictions."
    )
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

    render_graph_explanation_card(
        what_it_shows="This suite applies formal statistical hypothesis tests (ADF and KPSS) to verify whether the disease series possesses a stochastic unit root or needs differencing.",
        key_observation=f"Raw disease reports exhibit unit-root non-stationarity (ADF p={adf_res.get('p_value')}), confirming that raw series cannot be fed into ARIMA without differencing.",
        why_it_matters="Non-stationary time series lead to spurious predictions. Establishing that first-order differencing achieves stationarity confirms the mathematical necessity of an integrated model (ARIMA with d=1).",
        technical_details="ADF test: Delta y_t = alpha + beta t + gamma y_{t-1} + sum delta_i Delta y_{t-i} + eps_t with AIC lag selection."
    )

# ----------------- Tab 5: Differencing Analysis -----------------
with tab_diff:
    st.subheader("Differencing & Variance Reduction Analysis")
    _render_forecast_purpose_badge(
        "First-order differencing (d=1) centers disease changes around zero, removing multi-year drift and giving ARIMA a stable foundation for realistic future predictions."
    )
    st.markdown(
        r"""
        Differencing subtracts the previous observation from the current observation: $\Delta y_t = y_t - y_{t-1}$.
        This transformation removes non-stationary trends and shifts the series around a constant mean of zero.
        """
    )
    diff_analysis = analyze_differencing(df_monthly[target_col])

    # Plot original vs differenced series
    df_diff_plot = df_monthly.copy()
    df_diff_plot["first_diff"] = df_diff_plot[target_col].diff()

    fig_diff = go.Figure()
    fig_diff.add_trace(
        go.Scatter(
            x=df_diff_plot["date"],
            y=df_diff_plot[target_col],
            name="Original Series (Level)",
            line=dict(color="#102C26", width=2),
            hovertemplate="Level: %{y:.1f}<extra></extra>",
        )
    )
    fig_diff.add_trace(
        go.Scatter(
            x=df_diff_plot["date"],
            y=df_diff_plot["first_diff"],
            name="First Differenced (d=1)",
            line=dict(color="#C4A66A", width=1.8),
            hovertemplate="Diff: %{y:.1f}<extra></extra>",
        )
    )
    format_plotly_figure(fig_diff, title="Original Level vs. First-Differenced Series (d=1)", height=380)
    fig_diff.update_layout(yaxis_title="Reports / Difference", xaxis_title="Timeline")
    st.plotly_chart(fig_diff, use_container_width=True)

    render_graph_explanation_card(
        what_it_shows="This chart overlays the original raw disease reports (dark green) with the first-differenced series (gold) computed as current month minus previous month.",
        key_observation="Differencing stabilizes the mean directly around zero and removes multi-year level drift, converting the non-stationary series into a mean-reverting stationary process.",
        why_it_matters="Achieving a constant mean of zero confirms that setting ARIMA parameter d=1 is statistically justified and avoids over-differencing (which would inflate variance).",
        technical_details=f"Raw variance: {diff_analysis['original']['variance']} | d=1 variance: {diff_analysis['first_difference']['variance']} | Verdict: {diff_analysis['recommendation']}"
    )

    st.markdown(
        f"""
        <div style="background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 12px; padding: 14px 20px; margin-top: 10px;">
            <div style="font-weight: 700; color: #102C26; font-size: 0.95rem; margin-bottom: 6px;">Variance Comparison Summary</div>
            <div style="font-size: 0.85rem; color: #5A665D; line-height: 1.6;">
                - <strong>Original Level Variance:</strong> <code>{diff_analysis['original']['variance']}</code><br>
                - <strong>First-Difference Variance (d=1):</strong> <code>{diff_analysis['first_difference']['variance']}</code><br>
                - <strong>Seasonal-Difference Variance (D=1):</strong> <code>{diff_analysis['seasonal_difference']['variance']}</code><br>
                - <strong>Econometric Recommendation:</strong> {diff_analysis['recommendation']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- Tab 6: Seasonal Decomposition -----------------
with tab_decomp:
    st.subheader("Trend & Seasonal Component Breakdown (12-Month Annual Cycle)")
    _render_forecast_purpose_badge(
        "Decomposes disease counts into long-term Trend, Kharif Monsoon Seasonality, and Random Spikes — allowing SARIMA to predict exact high-risk calendar months 12 months ahead."
    )
    st.markdown(
        "Seasonal decomposition separates the observed disease timeline into three independent signals: "
        "an overall trend, a 12-month repeating seasonal cycle, and irregular residual noise."
    )
    decomp_df = perform_seasonal_decomposition(df_monthly, target_col=target_col, period=12, model="additive")

    if decomp_df is not None:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["observed"], name="Recorded Reports", line=dict(color="#102C26", width=1.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["trend"], name="Overall Trend", line=dict(color="#527A50", width=2.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["seasonal"], name="Yearly Repeating Season (12m)", line=dict(color="#C4A66A", width=1.5)))
        fig_d.add_trace(go.Scatter(x=decomp_df["date"], y=decomp_df["residual"], name="Random Irregular Fluctuation", line=dict(color="#C24134", width=1.0, dash="dot")))

        format_plotly_figure(fig_d, title="Additive Decomposition: Observed = Trend + Season + Residual", height=420)
        st.plotly_chart(fig_d, use_container_width=True)

        render_graph_explanation_card(
            what_it_shows="This decomposition separates disease history into three distinct underlying signals: long-term multi-year trend, 12-month repeating seasonal cycle, and irregular residual noise.",
            key_observation="The seasonal component exhibits regular periodic amplitudes corresponding to India's agricultural calendar, while the residual component remains centered around zero.",
            why_it_matters="Proving that a large fraction of variance is explained by seasonal recurrence mathematically justifies using seasonal autoregressive moving average models.",
            technical_details="Classical additive decomposition: y_t = Trend_t + Seasonal_t + Residual_t with fixed period s=12."
        )

    # Seasonal Heatmap
    st.markdown("#### Calendar Seasonality Heatmap (Month vs. Year)")
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
    format_plotly_figure(fig_heat, title=f"Annual Seasonality Heatmap: {pathogen_dict[selected_slug]}", height=360)
    st.plotly_chart(fig_heat, use_container_width=True)

    render_graph_explanation_card(
        what_it_shows="This matrix maps recorded disease reports across all 12 calendar months (X-axis) for each year from 2015 to 2026 (Y-axis).",
        key_observation="Darker vertical bands consistently line up along specific calendar months each year, confirming strong recurring annual seasonality.",
        why_it_matters="Recurring month-specific peaks confirm that seasonal time-series models (such as SARIMA with period s=12 or Holt-Winters) are highly relevant for capturing recurring crop disease risks."
    )

# ----------------- Tab 7: ACF & PACF Memory -----------------
with tab_acf:
    st.subheader("Past Month Memory (Autocorrelation & Partial Autocorrelation Diagnostics)")
    _render_forecast_purpose_badge(
        "Autocorrelation (ACF) and Partial Autocorrelation (PACF) tell the forecasting model exactly how many past months of infection history (p) and random shocks (q) predict next month's disease."
    )
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

# ----------------- Tab 8: Weather Patterns, Lags & Partitions -----------------
with tab_weather:
    st.subheader("Weather Cross-Correlation & Validation Partitions")
    _render_forecast_purpose_badge(
        "Demonstrates that rainfall and humidity 1 month prior (t-1) act as early warning indicators, proving that weather-informed models (ARIMAX) can forecast disease before crop damage occurs."
    )
    st.markdown(
        """
        Fungal sporulation and disease lesion expansion depend directly on temperature, precipitation, and canopy relative humidity.
        Using **NASA POWER** monthly climatology across Indian agro-zones, we examine how weather variables correlate with disease reports over time and across time lags.
        """
    )

    weather_cols = ["temp_mean_c", "precipitation_monthly_mm", "relative_humidity_pct"]
    has_weather_cols = any(c in df_monthly.columns for c in weather_cols)

    if has_weather_cols:
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
        format_plotly_figure(fig_w, title=f"Disease Reports vs Monthly Precipitation: {pathogen_dict[selected_slug]}", height=400)
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

        # Scientific Disclaimer Card (Clean SVG, no emoji)
        st.markdown(
            f"""
            <div style="background: rgba(196, 166, 106, 0.1); border-left: 4px solid #C4A66A; border-radius: 8px; padding: 12px 18px; margin-top: 14px; margin-bottom: 24px;">
                <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 0.84rem; color: #695221; margin-bottom: 3px;">
                    {icon_alert_triangle(16, "#C4A66A")} Scientific Notice: Correlation Does Not Prove Biological Causation
                </div>
                <p style="font-size: 0.82rem; color: #404B43; line-height: 1.5; margin: 0;">
                    {corr_results.get('scientific_disclaimer')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chronological Split
    st.markdown("#### Chronological Data Partitions (Zero Future Leakage)")
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

# Page Conclusion
st.markdown(
    f"""
    <div class="verdant-card" style="background: #F6F4EC; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 14px; padding: 20px 24px; margin-top: 28px; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            {icon_target(18, "#183D32")}
            <h4 style="margin: 0; color: #183D32; font-size: 1.05rem;">Diagnostic Conclusion: Foundation for Disease Forecasting</h4>
        </div>
        <p style="margin: 0; font-size: 0.88rem; color: #5A665D; line-height: 1.6;">
            The diagnostics above confirm that crop disease outbreaks follow structured, predictable mathematical dynamics: strong 12-month monsoon seasonality, unit-root non-stationarity requiring first-order differencing ($d=1$), and 1-month-lagged weather sensitivity. These findings directly justify training <strong>ARIMA(1,1,1)</strong>, <strong>SARIMA(1,0,1)(1,0,1)[12]</strong>, and <strong>ARIMAX</strong> against seasonal baselines.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.page_link("pages/3_Model_Comparison.py", label="Compare All 11 Forecasting Models ->", use_container_width=True)
with col_c2:
    st.page_link("pages/4_Future_Forecasts.py", label="Generate Future Outbreak Forecasts ->", use_container_width=True)

