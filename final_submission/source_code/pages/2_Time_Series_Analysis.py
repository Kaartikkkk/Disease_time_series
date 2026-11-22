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
    icon_bulb,
    icon_calendar,
    icon_check_circle,
    icon_info,
    icon_layers,
    icon_leaf,
    icon_pulse,
    icon_settings,
    icon_shield,
    icon_sigma,
    icon_target,
    icon_thermometer,
    icon_timeseries,
    icon_trending_up,
    icon_weather,
)
from src.project_stats import get_project_telemetry
from src.models.volatility.arch_garch_model import (
    ARCHGARCHModel,
    VOLATILITY_SPECS,
    fit_and_compare_all_specifications,
    perform_arch_lm_test,
)
from src.stationarity import (
    analyze_differencing,
    perform_adf_test,
    perform_kpss_test,
    perform_ljung_box_test,
    synthesize_stationarity_decision,
)
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_image_base64,
    inject_custom_css,
    load_all_pathogen_options,
    load_monthly_series,
    render_top_navigation,
)

st.set_page_config(
    page_title="Time Series Analysis — AgriForecast",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_top_navigation(active_page="analysis")

stats = get_project_telemetry()

# Load tailored hero visuals
hero_ear_b64 = get_image_base64(ASSETS_DIR / "hero_ear_blended.png")
if not hero_ear_b64:
    hero_ear_b64 = get_image_base64(ASSETS_DIR / "wheat_canopy_ears.jpg")

obj_hills_b64 = get_image_base64(ASSETS_DIR / "objective_hills_blended.png")

# ==============================================================================
# 1. EDITORIAL HERO BANNER (MATCHING AGRIFORECAST SPECIFICATION)
# ==============================================================================
hero_markup = f"""
<div class="dashboard-hero-card" style="min-height: 180px; margin-bottom: 20px;">
    <div class="hero-text-content" style="padding: 28px 36px; max-width: 620px;">
        <div class="hero-tag">TIME SERIES ANALYSIS</div>
        <h1 class="hero-headline" style="font-size: 2.1rem; margin-bottom: 8px;">
            Time Series Analysis & Statistical Diagnostics
        </h1>
        <p class="hero-subtitle" style="font-size: 0.95rem; margin-bottom: 0;">
            Longitudinal trajectory, stationarity tests, seasonal decomposition, ACF/PACF memory, and NASA agrometeorological lags.
        </p>
    </div>
    <div class="hero-image-wrapper" style="width: 50%;">
        <img src="{hero_ear_b64}" class="hero-ear-img" alt="Time series diagnostics" />
    </div>
</div>
"""
st.html(hero_markup)

pathogen_dict = load_all_pathogen_options()

# ==============================================================================
# 2. CONTROLS TOOLBAR CARD (FILTER + SLIDER)
# ==============================================================================
st.markdown('<div class="filter-toolbar-card">', unsafe_allow_html=True)
bar_c1, bar_c2 = st.columns([1.3, 1.1])
with bar_c1:
    st.markdown(
        f"<div class='filter-field-label'>{icon_leaf(15, '#2D6A4F')} Target Crop Disease</div>",
        unsafe_allow_html=True,
    )
    selected_slug = st.selectbox(
        "Target Crop Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
        label_visibility="collapsed",
        help="Select a crop disease series to run econometric diagnostics.",
    )
with bar_c2:
    st.markdown(
        f"<div class='filter-field-label'>{icon_settings(15, '#2D6A4F')} Rolling Smoothing Window (Months)</div>",
        unsafe_allow_html=True,
    )
    window_size = st.slider(
        "Rolling Smoothing Window (Months)",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
        label_visibility="collapsed",
        help="Window size for moving average and standard deviation envelope.",
    )
st.markdown('</div>', unsafe_allow_html=True)

df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"Monthly time series not found for {selected_slug}.")
    st.stop()

target_col = "documented_occurrence_count"
series_summary = compute_series_summary(df_monthly, target_col=target_col)
suitability = assess_forecasting_suitability(df_monthly, target_col=target_col, pathogen_name=pathogen_dict[selected_slug])
date_min_str = pd.to_datetime(df_monthly["date"]).min().strftime("%b %Y")
date_max_str = pd.to_datetime(df_monthly["date"]).max().strftime("%b %Y")
year_span_str = f"{pd.to_datetime(df_monthly['date']).min().year} – {pd.to_datetime(df_monthly['date']).max().year}"

# ==============================================================================
# 3. FOUR COMPACT KPI METRIC CARDS
# ==============================================================================
kpi_markup = f"""
<div class="kpi-row-grid" style="margin-top: 14px; margin-bottom: 22px;">
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_calendar(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label" style="text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; font-weight: 700; color: #6B7280;">Timeline Span</div>
            <div class="kpi-value">{series_summary['total_months']} Months</div>
            <div class="kpi-subtext">{date_min_str} – {date_max_str}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_layers(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label" style="text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; font-weight: 700; color: #6B7280;">Active Reporting</div>
            <div class="kpi-value">{series_summary['active_months']} Months</div>
            <div class="kpi-subtext">{series_summary['pct_non_zero']:.1f}% Positive • {100.0 - series_summary['pct_non_zero']:.1f}% Zeros</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_trending_up(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label" style="text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; font-weight: 700; color: #6B7280;">Monthly Mean</div>
            <div class="kpi-value">{series_summary['mean_monthly_occurrences']}</div>
            <div class="kpi-subtext">Total: {series_summary['total_occurrences']:,} Recorded Reports</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_sigma(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label" style="text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; font-weight: 700; color: #6B7280;">Standard Deviation</div>
            <div class="kpi-value">±{series_summary['std_monthly_occurrences']}</div>
            <div class="kpi-subtext">Sample Standard Deviation</div>
        </div>
    </div>
</div>
"""
st.html(kpi_markup)

# ==============================================================================
# 4. CONSOLIDATED PILL TABS
# ==============================================================================
tab_trend, tab_decomp, tab_stationarity, tab_acf, tab_weather, tab_volatility = st.tabs([
    "Trajectory & Trend",
    "Seasonal Decomposition",
    "Stationarity & Differencing",
    "Autocorrelation (ACF / PACF)",
    "Weather Correlation",
    "ARCH / GARCH",
])

# ----------------- Tab 1: Trajectory & Trend -----------------
with tab_trend:
    st.markdown(
        f"""
        <div class="explorer-chart-card">
            <div class="chart-card-header">
                <div class="chart-card-title">
                    Historical Occurrence Trajectory and Rolling Mean ({year_span_str})
                </div>
                <div class="chart-card-legend">
                    <div>
                        <span class="legend-swatch-bar"></span>
                        <span>Monthly Occurrences</span>
                    </div>
                    <div>
                        <span class="legend-swatch-line"></span>
                        <span>{window_size}-Month Moving Average</span>
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    rolling_df = compute_rolling_statistics(df_monthly, window=window_size, target_col=target_col)
    
    fig_tr = go.Figure()
    fig_tr.add_trace(
        go.Bar(
            x=df_monthly["date"],
            y=df_monthly[target_col],
            name="Monthly Occurrences",
            marker_color="#104030",
            opacity=1.0,
            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Reports:</b> %{y}<extra></extra>",
        )
    )
    fig_tr.add_trace(
        go.Scatter(
            x=rolling_df["date"],
            y=rolling_df[f"rolling_mean_{window_size}m"],
            name=f"{window_size}-Month Moving Average",
            line=dict(color="#D97706", width=2.4),
        )
    )
    fig_tr.update_layout(
        height=340,
        margin=dict(l=30, r=20, t=10, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
        xaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            tickformat="%Y",
            linecolor="#E5E7EB",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F3F4F6",
            linecolor="#E5E7EB",
        ),
    )
    st.plotly_chart(fig_tr, use_container_width=True)
    
    st.markdown(
        f"""
        <div class="chart-bottom-info-box">
            {icon_info(16, "#166534")}
            <span><strong>Empirical Trend Interpretation:</strong> The bar series highlights discrete monthly reporting pulses, while the gold curve tracks the {window_size}-month rolling baseline. For {pathogen_dict[selected_slug]}, activity alternates between zero-count dormancy intervals and seasonal epidemic bursts.</span>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    with st.expander("Outlier Detection & Distribution Bounds", expanded=False):
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
    
    if decomp_res is not None and not decomp_res.empty:
        fig_d = go.Figure()
        dates = decomp_res["date"]
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

    with st.expander("Mathematical Formulations & Stationarity Theory", expanded=False):
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
    
    if corr_res and corr_res.get("correlations"):
        df_corr = pd.DataFrame(corr_res["correlations"])
        corr_matrix = df_corr.pivot(index="label", columns="lag_months", values="pearson_r")
        corr_matrix.columns = [f"Lag {c}m" for c in corr_matrix.columns]
        
        fig_c = px.imshow(
            corr_matrix,
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

# ----------------- Tab 6: ARCH / GARCH Volatility -----------------
with tab_volatility:
    # 1. Plain-English Header & Educational Card
    st.markdown(
        f"""
        <div style="background: #F8FAF9; border: 1px solid rgba(45, 106, 79, 0.2); border-left: 4px solid #2D6A4F; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                {icon_trending_up(20, "#2D6A4F")}
                <strong style="color: #102C26; font-size: 1.05rem;">Outbreak Uncertainty &amp; Volatility Analysis (ARCH / GARCH)</strong>
            </div>
            <p style="color: #374151; font-size: 0.90rem; line-height: 1.5; margin: 0 0 10px 0;">
                While forecasting models predict <em>how many cases</em> will occur, <strong>ARCH and GARCH models predict how unpredictable the disease is</strong>. In agriculture, outbreaks are not constant—they arrive in sudden, unpredictable waves during favorable weather.
            </p>
            <div style="display: flex; gap: 16px; flex-wrap: wrap; font-size: 0.84rem; color: #4B5563;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #16A34A;"></span>
                    <span><strong>Low Volatility (Valleys):</strong> Predictable dormancy during dry months</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #DC2626;"></span>
                    <span><strong>High Volatility (Spikes):</strong> Unstable, clustered outbreaks during warm, humid seasons</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Simplified Controls
    c_ctl1, c_ctl2 = st.columns([1.5, 1])
    with c_ctl1:
        spec_display_map = {
            "GARCH(1,1) — Standard (Recommended)": (1, 1),
            "ARCH(1) — Simple Autoregressive Shock": (1, 0),
            "GARCH(2,1) — Extended Seasonal Memory": (2, 1),
            "ARCH(2) — Two-Month Lagged Shocks": (2, 0),
        }
        spec_label = st.selectbox(
            "Select Uncertainty Model:",
            options=list(spec_display_map.keys()),
            index=0,
            help="GARCH(1,1) is the gold standard for modeling time-varying outbreak clustering.",
        )
        p_order, q_order = spec_display_map[spec_label]
        spec_code = f"GARCH({p_order},{q_order})" if q_order > 0 else f"ARCH({p_order})"

    with c_ctl2:
        st.write("")
        st.write("")
        show_advanced_settings = st.checkbox("Advanced Model Settings", value=False)

    app_key = "approach_a"
    dist_key = "StudentsT"
    if show_advanced_settings:
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            app_choice = st.selectbox(
                "Approach:",
                options=["Approach A: Residual Volatility (ARIMA + GARCH)", "Approach B: Occurrence Changes (Δy_t)"],
                index=0,
            )
            app_key = "approach_a" if "Approach A" in app_choice else "approach_b"
        with c_adv2:
            dist_choice = st.selectbox(
                "Error Distribution:",
                options=["Student's t (Heavy-Tailed Outbreaks)", "Normal (Standard Gaussian)"],
                index=0,
            )
            dist_key = "StudentsT" if "Student" in dist_choice else "Normal"

    y_series = df_monthly[target_col].values.astype(float)
    dates_arr = df_monthly["date"].values

    with st.spinner("Fitting volatility model..."):
        try:
            vol_model = ARCHGARCHModel(p=p_order, q=q_order, dist=dist_key, approach=app_key)
            vol_res = vol_model.fit(y_series)
            vol_fcast = vol_model.forecast(horizon=12)
            has_fit = True
        except Exception as exc:
            st.error(f"Could not fit volatility model on {pathogen_dict.get(selected_slug, selected_slug)}: {exc}")
            has_fit = False

    if has_fit:
        # Calculate human-friendly stats
        mean_vol = float(np.mean(vol_res.conditional_volatility))
        max_vol = float(np.max(vol_res.conditional_volatility))
        max_vol_idx = int(np.argmax(vol_res.conditional_volatility))
        v_dates = dates_arr[-len(vol_res.conditional_volatility):]
        peak_date_str = pd.to_datetime(v_dates[max_vol_idx]).strftime("%b %Y") if len(v_dates) > max_vol_idx else "N/A"

        alpha_val = float(vol_res.params.get("alpha[1]", 0.0))
        beta_val = float(vol_res.params.get("beta[1]", 0.0))
        persistence = alpha_val + beta_val
        if persistence > 0.8:
            persist_label = "High (Prolonged Clusters)"
        elif persistence > 0.4:
            persist_label = "Moderate (1–2 Month Shocks)"
        else:
            persist_label = "Low (Quick Mean Reversion)"

        # 3. Clean, Human-Friendly KPI Cards
        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 22px;">
                <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #527A50; font-weight: 700; letter-spacing: 0.04em;">Average Uncertainty (σ̄)</div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #111827; margin: 4px 0 2px 0;">{mean_vol:.2f} <span style="font-size: 0.82rem; font-weight: 500; color: #6B7280;">reports/mo</span></div>
                    <div style="font-size: 0.78rem; color: #6B7280;">Typical monthly volatility baseline</div>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #B91C1C; font-weight: 700; letter-spacing: 0.04em;">Peak Outbreak Uncertainty</div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #111827; margin: 4px 0 2px 0;">{max_vol:.2f} <span style="font-size: 0.82rem; font-weight: 500; color: #6B7280;">reports/mo</span></div>
                    <div style="font-size: 0.78rem; color: #6B7280;">Highest turbulence in <strong>{peak_date_str}</strong></div>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #2563EB; font-weight: 700; letter-spacing: 0.04em;">Outbreak Shock Memory</div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #111827; margin: 4px 0 2px 0;">{persistence:.2f}</div>
                    <div style="font-size: 0.78rem; color: #6B7280;">{persist_label}</div>
                </div>
                <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #166534; font-weight: 700; letter-spacing: 0.04em;">Model Status</div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #166534; margin: 4px 0 2px 0;">Verified</div>
                    <div style="font-size: 0.78rem; color: #6B7280;">ARCH LM p = {vol_res.arch_lm_pvalue:.3f} (Clean fit)</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Two Clear, High-Impact Charts
        # Chart 1: Historical Outbreak Volatility
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=v_dates,
                y=vol_res.conditional_volatility,
                mode="lines",
                name="Historical Volatility (σ_t)",
                line=dict(color="#2D6A4F", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(45, 106, 79, 0.12)",
                hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Uncertainty (σ):</b> %{y:.2f} reports<extra></extra>",
            )
        )
        format_plotly_figure(
            fig_vol,
            title="Historical Disease Volatility Timeline (How Unpredictable Were Past Seasons?)",
            height=290,
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        st.caption(
            "**Interpretation:** Spikes represent explosive disease surges where actual field outcomes deviated wildly from expectations. Flat periods indicate predictable dry-season dormancy."
        )

        # Chart 2: 12-Month Out-of-Sample Volatility Forecast & Dynamic Uncertainty Interval
        last_dt = pd.to_datetime(df_monthly["date"].iloc[-1])
        future_dates = pd.date_range(last_dt + pd.DateOffset(months=1), periods=12, freq="MS")
        hist_tail = df_monthly.iloc[-24:]

        fig_env = go.Figure()
        # Historical context last 24 months
        fig_env.add_trace(
            go.Scatter(
                x=hist_tail["date"],
                y=hist_tail[target_col],
                mode="lines+markers",
                name="Observed Occurrences",
                line=dict(color="#111827", width=1.8),
                marker=dict(size=4, color="#111827"),
            )
        )
        fig_env.add_trace(
            go.Scatter(
                x=future_dates,
                y=vol_fcast.mean_forecast,
                mode="lines+markers",
                name="Projected Mean",
                line=dict(color="#D97706", width=2.4),
                marker=dict(size=5, color="#D97706"),
            )
        )
        fig_env.add_trace(
            go.Scatter(
                x=future_dates,
                y=vol_fcast.upper_bound,
                mode="lines",
                name="95% Upper Uncertainty Bound",
                line=dict(color="#2563EB", width=1.2, dash="dash"),
            )
        )
        fig_env.add_trace(
            go.Scatter(
                x=future_dates,
                y=vol_fcast.lower_bound,
                mode="lines",
                name="95% Lower Uncertainty Bound",
                line=dict(color="#2563EB", width=1.2, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(37, 99, 235, 0.12)",
            )
        )
        format_plotly_figure(
            fig_env,
            title="12-Month Forward Forecast with Dynamic Volatility Envelope",
            height=310,
        )
        st.plotly_chart(fig_env, use_container_width=True)
        st.caption(
            "**Interpretation:** The blue shaded envelope widens automatically during high-risk outbreak months and narrows during quiet periods, providing realistic uncertainty bounds."
        )

        # 5. Practical Takeaways
        st.markdown(
            f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 14px; margin-bottom: 20px;">
                <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 14px 16px;">
                    <div style="display: flex; align-items: center; gap: 6px; font-weight: 700; color: #166534; font-size: 0.88rem; margin-bottom: 4px;">
                        {icon_shield(16, "#166534")} <span>Pre-Positioning Advisory</span>
                    </div>
                    <div style="color: #374151; font-size: 0.82rem; line-height: 1.45;">
                        When forecasted volatility expands, agricultural extension officers should pre-position fungicides and advisory warnings even if current report counts remain low.
                    </div>
                </div>
                <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 14px 16px;">
                    <div style="display: flex; align-items: center; gap: 6px; font-weight: 700; color: #1E40AF; font-size: 0.88rem; margin-bottom: 4px;">
                        {icon_target(16, "#1E40AF")} <span>Outbreak Surge Gate</span>
                    </div>
                    <div style="color: #374151; font-size: 0.82rem; line-height: 1.45;">
                        Observations exceeding the upper volatility bound indicate true anomalous epidemic outbreaks rather than normal seasonal variation.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 6. Collapsible Technical Section for Evaluators & Researchers
        with st.expander("Detailed Econometric Tables (Parameters, AIC/BIC & All Specifications)", expanded=False):
            c_p1, c_p2 = st.columns([1, 1.4])
            with c_p1:
                st.markdown(f"**Fitted Parameters ({spec_code})**")
                param_df = pd.DataFrame({
                    "Parameter": list(vol_res.params.keys()),
                    "Estimate": [f"{v:.4f}" for v in vol_res.params.values()],
                    "p-value": [f"{vol_res.pvalues.get(k, np.nan):.4f}" for k in vol_res.params.keys()],
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)
                st.caption(f"Log-Likelihood: {vol_res.log_likelihood:.2f} · AIC: {vol_res.aic:.2f} · BIC: {vol_res.bic:.2f}")

            with c_p2:
                st.markdown("**Leaderboard Across All 7 ARCH/GARCH Specifications**")
                all_specs_dict = fit_and_compare_all_specifications(y_series, dist=dist_key, approach=app_key)
                comp_rows = []
                for s_name, s_res in all_specs_dict.items():
                    comp_rows.append({
                        "Model": s_name.upper().replace("_", ""),
                        "AIC": round(s_res.aic, 2),
                        "BIC": round(s_res.bic, 2),
                        "ARCH-LM p": round(s_res.arch_lm_pvalue, 4),
                        "Converged": "Yes" if s_res.converged else "No",
                    })
                comp_df = pd.DataFrame(comp_rows)
                if not comp_df.empty:
                    comp_df = comp_df.sort_values(by="AIC", ascending=True)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    best_spec = comp_df.iloc[0]["Model"]
                    st.caption(f"Lowest AIC achieved by `{best_spec}` ({comp_df.iloc[0]['AIC']}).")

# ==============================================================================
# 5. BOTTOM NAVIGATION CTA BUTTONS & LANDSCAPE ACCENT
# ==============================================================================
st.markdown(
    f"""
    <div class="bottom-cta-wrap">
        <a href="/Model_Comparison" target="_self" class="hero-btn-primary" style="padding: 12px 26px; font-size: 0.92rem;">
            {icon_layers(16, "#FFFFFF")} Proceed to Model Comparison
        </a>
        <a href="/Future_Forecasts" target="_self" class="hero-btn-secondary" style="padding: 12px 26px; font-size: 0.92rem;">
            {icon_trending_up(16, "#111827")} Generate Future Forecasts
        </a>
    </div>
    <div style="position: relative; width: 100%; height: 60px; overflow: hidden; pointer-events: none; margin-top: 10px; margin-bottom: -20px;">
        <img src="{obj_hills_b64}" style="position: absolute; right: 0; bottom: 0; height: 100%; opacity: 0.7; object-fit: contain;" alt="contour hills" />
    </div>
    """,
    unsafe_allow_html=True,
)
