"""Page 4: Forecast Intelligence — Professional Results Showcase & Monthly Disease Outlook."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecast_insights import (
    build_multi_disease_watch_matrix,
    classify_predicted_activity,
    compute_calendar_month_baselines,
    detect_forecast_trend,
    generate_monthly_disease_outlook,
    identify_peak_forecast_month,
)
from src.forecasting import (
    forecast_holt_winters,
    forecast_ml_autoregressor,
    forecast_prophet,
    forecast_sarimax,
    forecast_seasonal_naive,
    generate_future_weather_scenario,
    HAS_PROPHET,
)
from src.icons import (
    icon_arrow_right,
    icon_download,
    icon_forecast,
    icon_leaf,
    icon_model_compare,
    icon_pin,
    icon_pulse,
    icon_shield,
    icon_thermometer,
    icon_timeseries,
    icon_weather,
)
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_image_base64,
    inject_custom_css,
    load_all_pathogen_options,
    load_monthly_series,
    render_app_sidebar,
    render_top_header,
)

st.set_page_config(
    page_title="Forecast Intelligence — Verdant Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="forecasts")
render_top_header()

# ==============================================================================
# A. PAGE HEADER WITH DYNAMIC METADATA CHIPS
# ==============================================================================
pathogen_dict = load_all_pathogen_options()
run_timestamp = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_forecast(28, "#527A50")}
                <h1 style="font-size: 2.3rem; margin: 0; color: #102C26;">Forecast Intelligence</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Explore future documented pathogen occurrences, compare model projections, and understand forecast uncertainty.
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 12px; padding: 8px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                {icon_pulse(8, "#527A50")} Active Modeling Runtime
            </div>
            <div style="font-size: 0.75rem; color: #5A665D; margin-top: 2px;">
                Run Sync: <strong>{run_timestamp}</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# B. FORECAST CONFIGURATION PANEL (RESPONSIVE GRID)
# ==============================================================================
st.markdown("<div style='font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #5A665D; margin-bottom: 8px;'>Forecast Configuration & Control Parameters</div>", unsafe_allow_html=True)

with st.container():
    cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5 = st.columns([1.3, 1.0, 0.8, 1.3, 1.0])

    with cfg_c1:
        selected_slug = st.selectbox(
            "Target Pathogen / Host Disease",
            options=list(pathogen_dict.keys()),
            format_func=lambda k: pathogen_dict[k],
            index=0,
            key="fc_pathogen",
        )

    with cfg_c2:
        region_options = {
            "all": "All Agro-Climatic Zones",
            "national_composite": "National Composite",
            "odisha_bengal": "Eastern Coastal Belt",
            "punjab_haryana": "Northwest Indo-Gangetic",
            "tamil_nadu": "Southern Agricultural Zone",
            "maharashtra": "Western/Central Semi-Arid",
        }
        selected_region = st.selectbox(
            "Agro-Climatic Zone",
            options=list(region_options.keys()),
            format_func=lambda k: region_options[k],
            index=0,
            key="fc_region",
        )

    with cfg_c3:
        horizon_months = st.selectbox(
            "Horizon",
            options=[3, 6, 9, 12, 18, 24],
            index=1,  # default 6 months
            format_func=lambda h: f"{h} Months",
            key="fc_horizon",
        )

    available_models = [
        "SARIMA(1,0,1)(1,0,1)[12]",
        "SARIMAX (Seasonal + Weather Regressors)",
        "ARIMA(1,1,1)",
        "Holt-Winters Exponential Smoothing",
        "Seasonal Naive (Baseline)",
        "Hist Gradient Boosting (Autoregressive)",
    ]
    if HAS_PROPHET:
        available_models.append("Meta Prophet")

    with cfg_c4:
        chosen_model = st.selectbox(
            "Forecasting Architecture",
            options=available_models,
            index=0,
            key="fc_model",
        )

    with cfg_c5:
        ci_option = st.selectbox(
            "Prediction Intervals",
            options=["Both (80% & 95%)", "80% Interval Only", "95% Interval Only", "Point Forecast Only"],
            index=0,
            key="fc_ci",
        )

# Load real dataset
df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"No valid monthly time-series records found for {pathogen_dict.get(selected_slug, selected_slug)}. Please choose another series.")
    st.stop()

target_col = "documented_occurrence_count"
last_obs_date = pd.to_datetime(df_monthly["date"].max())
first_obs_date = pd.to_datetime(df_monthly["date"].min())
future_dates = pd.date_range(start=last_obs_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

# Compact Metadata Chips
st.markdown(
    f"""
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; margin-bottom: 22px;">
        <span style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #183D32;">
            <strong>Target:</strong> {pathogen_dict[selected_slug]}
        </span>
        <span style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #183D32;">
            <strong>Frequency:</strong> Monthly (MS)
        </span>
        <span style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #183D32;">
            <strong>Training Period:</strong> {first_obs_date.strftime('%b %Y')} – {last_obs_date.strftime('%b %Y')} ({len(df_monthly)} Mos)
        </span>
        <span style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #183D32;">
            <strong>Model:</strong> {chosen_model}
        </span>
        <span style="background: rgba(196, 166, 106, 0.15); border: 1px solid rgba(196, 166, 106, 0.35); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #695221;">
            <strong>Forecast Horizon:</strong> {future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Weather Regressor Scenario Generation
weather_cols = ["temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"]
has_weather = all(c in df_monthly.columns for c in weather_cols)
df_future_weather = generate_future_weather_scenario(df_monthly, horizon_months=horizon_months) if has_weather else None

# Run Forecast Execution with Robust Error Handling
with st.spinner(f"Estimating forward trajectory using {chosen_model}..."):
    try:
        y_full = df_monthly[target_col].astype(float)
        train_exog = df_monthly[weather_cols].bfill().fillna(0) if has_weather else None
        fut_exog = df_future_weather[weather_cols].bfill().fillna(0) if df_future_weather is not None else None

        if chosen_model == "Seasonal Naive (Baseline)":
            p_fut, lb_95, ub_95 = forecast_seasonal_naive(y_full, horizon=horizon_months, season_length=12)
        elif chosen_model == "SARIMA(1,0,1)(1,0,1)[12]":
            p_fut, lb_95, ub_95, _ = forecast_sarimax(y_full, horizon=horizon_months, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
        elif chosen_model == "SARIMAX (Seasonal + Weather Regressors)":
            p_fut, lb_95, ub_95, _ = forecast_sarimax(
                y_full, horizon=horizon_months, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), train_exog=train_exog, future_exog=fut_exog
            )
        elif chosen_model == "ARIMA(1,1,1)":
            p_fut, lb_95, ub_95, _ = forecast_sarimax(y_full, horizon=horizon_months, order=(1, 1, 1))
        elif chosen_model == "Holt-Winters Exponential Smoothing":
            p_fut, lb_95, ub_95, _ = forecast_holt_winters(y_full, horizon=horizon_months, seasonal_periods=12)
        elif chosen_model == "Meta Prophet":
            p_fut, lb_95, ub_95, _ = forecast_prophet(
                df_monthly, horizon=horizon_months, future_dates=future_dates,
                weather_cols=weather_cols if has_weather else None,
                future_weather_df=df_future_weather if has_weather else None,
            )
        else:
            p_fut, lb_95, ub_95, _ = forecast_ml_autoregressor(
                df_monthly, horizon=horizon_months, model_type="gradient_boosting",
                lags=[1, 2, 3, 12],
                weather_lag_cols=weather_cols if has_weather else None,
                future_weather_df=df_future_weather if has_weather else None,
            )
    except Exception as exc:
        st.error(f"Model convergence error for {chosen_model}: {exc}. Defaulting to Seasonal Naive baseline.")
        p_fut, lb_95, ub_95 = forecast_seasonal_naive(y_full, horizon=horizon_months, season_length=12)

# Seasonal Naive baseline comparison
p_baseline, _, _ = forecast_seasonal_naive(y_full, horizon=horizon_months, season_length=12)

# Non-negative clipping with statistical documentation
p_fut = np.maximum(0.0, p_fut)
lb_95 = np.maximum(0.0, lb_95)
ub_95 = np.maximum(p_fut, ub_95)

# Mathematically coherent 80% CI (Z=1.28 vs Z=1.96 for 95%)
ci_ratio = 1.28155 / 1.95996
half_width_95 = np.maximum(0.05, (ub_95 - lb_95) / 2.0)
half_width_80 = half_width_95 * ci_ratio
lb_80 = np.maximum(0.0, p_fut - half_width_80)
ub_80 = p_fut + half_width_80

df_forecast_out = pd.DataFrame({
    "date": future_dates,
    "forecast": np.round(p_fut, 2),
    "lower_80": np.round(lb_80, 2),
    "upper_80": np.round(ub_80, 2),
    "lower_95": np.round(lb_95, 2),
    "upper_95": np.round(ub_95, 2),
    "baseline_seasonal_naive": np.round(p_baseline, 2),
})

# ==============================================================================
# C. FORECAST SUMMARY CARDS (4 KPIS)
# ==============================================================================
latest_obs_val = float(df_monthly[target_col].dropna().iloc[-1])
endpoint_forecast = float(df_forecast_out["forecast"].iloc[-1])

# Historical baseline for the endpoint calendar month
cal_baselines = compute_calendar_month_baselines(df_monthly, target_col=target_col)
endpoint_cal_m = future_dates[-1].month
endpoint_hist_mean = cal_baselines.get(endpoint_cal_m, {}).get("mean", df_monthly[target_col].mean())
trend_diff = endpoint_forecast - endpoint_hist_mean
trend_sign = "+" if trend_diff >= 0 else ""
mean_ci_width = float(np.mean(df_forecast_out["upper_80"] - df_forecast_out["lower_80"]))

st.markdown(
    f"""
    <div class="metric-grid" style="margin-bottom: 24px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(20, "#527A50")}</div>
            <div class="metric-label-text">Latest Observed Count</div>
            <div class="metric-value-text">{latest_obs_val:.1f}</div>
            <div class="metric-note-text">Deposited in {last_obs_date.strftime('%b %Y')}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_forecast(20, "#183D32")}</div>
            <div class="metric-label-text">Horizon Endpoint Forecast</div>
            <div class="metric-value-text">{endpoint_forecast:.2f}</div>
            <div class="metric-note-text">Projected for {future_dates[-1].strftime('%b %Y')}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_model_compare(20, "#C4A66A")}</div>
            <div class="metric-label-text">Trend vs Historical Season</div>
            <div class="metric-value-text">{trend_sign}{trend_diff:.2f}</div>
            <div class="metric-note-text">vs {future_dates[-1].strftime('%B')} mean ({endpoint_hist_mean:.1f})</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
            <div class="metric-label-text">Uncertainty Spread (80% CI)</div>
            <div class="metric-value-text">±{mean_ci_width / 2.0:.2f}</div>
            <div class="metric-note-text">Mean prediction interval half-width</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# D. ADDITIONAL FEATURE: AI-STYLE FORECAST INSIGHTS & MONTHLY OUTLOOK
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="margin-bottom: 16px;">
        <div class="hero-eyebrow" style="margin-bottom: 4px;">Automated Narrative Analysis</div>
        <h2 style="font-size: 1.85rem; margin: 0 0 4px 0; color: #102C26;">Monthly Disease Outlook</h2>
        <p style="color: #5A665D; font-size: 0.92rem; margin: 0;">
            A clear view of predicted disease activity, emerging trends, and the months that may require closer monitoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Peak Forecast Month Callout Card
peak_info = identify_peak_forecast_month(df_forecast_out, pathogen_name=pathogen_dict[selected_slug])

st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, rgba(196, 166, 106, 0.12) 0%, rgba(255, 254, 250, 0.95) 100%); border: 1.5px solid #C4A66A; border-radius: 14px; padding: 18px 24px; margin-bottom: 22px; box-shadow: 0 4px 16px rgba(196, 166, 106, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: #C4A66A; color: #FFF; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                        PEAK FORECAST ACTIVITY
                    </span>
                    <span style="font-size: 0.82rem; color: #695221; font-weight: 600;">
                        {pathogen_dict[selected_slug]}
                    </span>
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.6rem; font-weight: 600; color: #102C26; margin: 6px 0 4px 0;">
                    {peak_info.get('peak_month', 'N/A')} · {peak_info.get('peak_value', 0.0):.2f} Documented Occurrences
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5; margin: 0; max-width: 820px;">
                    {peak_info.get('narrative', '')}
                </p>
            </div>
            <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(196, 166, 106, 0.3); border-radius: 10px; padding: 10px 16px;">
                <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: #695221;">80% Confidence Range</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600; color: #102C26;">
                    {peak_info.get('lower_80', 0.0):.2f} – {peak_info.get('upper_80', 0.0):.2f}
                </div>
                <div style="font-size: 0.72rem; color: #5A665D; margin-top: 2px;">Documented Records</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Responsive Monthly Outlook Cards Grid
monthly_outlooks = generate_monthly_disease_outlook(
    df_forecast=df_forecast_out,
    df_historical=df_monthly,
    pathogen_name=pathogen_dict[selected_slug],
    target_col=target_col,
)

# Render cards in 3-column responsive grid
grid_cols = st.columns(min(3, len(monthly_outlooks)))
for i, outlook in enumerate(monthly_outlooks):
    col = grid_cols[i % len(grid_cols)]
    with col:
        pct_str = f" ({outlook['change_pct']:+.1f}%)" if outlook['change_pct'] is not None else ""
        st.markdown(
            f"""
            <div class="verdant-card" style="margin-bottom: 16px; display: flex; flex-direction: column; justify-content: space-between;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-family: 'Fraunces', serif; font-size: 1.15rem; font-weight: 600; color: #102C26;">
                            {outlook['month_label']}
                        </span>
                        <span style="background: {outlook['activity_color']}15; border: 1px solid {outlook['activity_color']}40; color: {outlook['activity_color']}; font-size: 0.7rem; font-weight: 600; padding: 2px 7px; border-radius: 4px;">
                            {outlook['activity_level']}
                        </span>
                    </div>
                    <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px;">
                        <span style="font-family: 'Fraunces', serif; font-size: 1.8rem; font-weight: 600; color: #102C26;">
                            {outlook['predicted_count']:.2f}
                        </span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: {'#183D32' if outlook['change_abs'] <= 0 else '#C4A66A'};">
                            {outlook['trend_arrow']} {outlook['change_abs']:+.2f}{pct_str}
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: #5A665D; margin-bottom: 10px; font-family: 'JetBrains Mono', monospace;">
                        80% CI: [{outlook['lower_80']:.2f}, {outlook['upper_80']:.2f}] · 95% CI: [{outlook['lower_95']:.2f}, {outlook['upper_95']:.2f}]
                    </div>
                    <p style="font-size: 0.82rem; color: #404B43; line-height: 1.45; margin: 0;">
                        {outlook['narrative']}
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# 3. Month-by-Month Forecast Summary Table & Export
st.markdown("<div style='margin-top: 10px; margin-bottom: 6px; font-size: 1.1rem; font-family: Fraunces, serif; color: #102C26;'>Month-by-Month Forecast Summary</div>", unsafe_allow_html=True)

table_rows = []
for out in monthly_outlooks:
    pct_text = f"{out['change_pct']:+.1f}%" if out['change_pct'] is not None else "N/A"
    table_rows.append({
        "Month": out["month_label"],
        "Predicted Count": out["predicted_count"],
        "Change from Prior": f"{out['change_abs']:+.2f} ({pct_text})",
        "Trend": f"{out['trend_arrow']} {out['trend_label']}",
        "80% Uncertainty Band": f"[{out['lower_80']}, {out['upper_80']}]",
        "95% Uncertainty Band": f"[{out['lower_95']}, {out['upper_95']}]",
        "Activity Classification": out["activity_level"],
    })
df_summary_table = pd.DataFrame(table_rows)

st.dataframe(df_summary_table, use_container_width=True, hide_index=True)

csv_summary = df_summary_table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Monthly Forecast Summary (CSV)",
    data=csv_summary,
    file_name=f"monthly_outlook_{selected_slug}_{horizon_months}m.csv",
    mime="text/csv",
)

# 4. Multi-Disease Attention Summary ("Diseases to Watch")
with st.expander("🔍 Multi-Disease Attention Summary — Comparative Indian Pathogen Watchlist", expanded=False):
    st.markdown(
        "Ranked comparative overview of all 10 candidate plant pathogen series across India, evaluated "
        "by projected relative elevation above each pathogen's historical calendar baseline."
    )
    df_watch = build_multi_disease_watch_matrix(pathogen_dict, load_monthly_series, horizon_months=6)
    if not df_watch.empty:
        st.dataframe(
            df_watch[[
                "disease_name", "peak_month", "peak_val", "seasonal_baseline", 
                "relative_elevation", "activity_level", "data_quality", "total_records"
            ]].rename(columns={
                "disease_name": "Pathogen Series",
                "peak_month": "Projected Peak Month",
                "peak_val": "Peak Predicted Count",
                "seasonal_baseline": "Historical Baseline",
                "relative_elevation": "Relative Elevation (%)",
                "activity_level": "Activity Status",
                "data_quality": "Data Quality",
                "total_records": "Historical Records",
            }),
            use_container_width=True,
            hide_index=True,
        )
    st.caption("Epistemological note: Series are evaluated by relative seasonal deviation to ensure comparability across different taxonomic reporting intensities.")

# ==============================================================================
# E. MAIN FORECAST VISUALIZATION (CENTERPIECE TIME-SERIES CHART)
# ==============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="margin-bottom: 12px;">
        <h2 style="font-size: 1.85rem; margin: 0 0 4px 0; color: #102C26;">
            Time-Series Trajectory: {pathogen_dict[selected_slug]}
        </h2>
        <p style="color: #5A665D; font-size: 0.92rem; margin: 0;">
            Observed surveillance frequency, point projections, and nested 80%/95% prediction intervals.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

fig_f = go.Figure()

# Recent Historical Observations (last 24-36 months for clear visual focus)
hist_subset = df_monthly.tail(36).copy()

fig_f.add_trace(
    go.Scatter(
        x=hist_subset["date"],
        y=hist_subset[target_col],
        name="Historical Documented Occurrences",
        mode="lines+markers",
        line=dict(color="#102C26", width=2.5),
        marker=dict(size=6, color="#102C26"),
        hovertemplate="<b>Observed:</b> %{y:.1f}<br><b>Date:</b> %{x|%b %Y}<extra></extra>",
    )
)

# Render Uncertainty Intervals (Nested correctly)
if ci_option in ["Both (80% & 95%)", "95% Interval Only"]:
    fig_f.add_trace(
        go.Scatter(
            x=df_forecast_out["date"],
            y=df_forecast_out["upper_95"],
            mode="lines",
            line=dict(color="rgba(82, 122, 80, 0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_f.add_trace(
        go.Scatter(
            x=df_forecast_out["date"],
            y=df_forecast_out["lower_95"],
            name="95% Prediction Interval",
            mode="lines",
            line=dict(color="rgba(82, 122, 80, 0)", width=0),
            fill="tonexty",
            fillcolor="rgba(82, 122, 80, 0.12)",
            hovertemplate="<b>95% CI Lower:</b> %{y:.2f}<extra></extra>",
        )
    )

if ci_option in ["Both (80% & 95%)", "80% Interval Only"]:
    fig_f.add_trace(
        go.Scatter(
            x=df_forecast_out["date"],
            y=df_forecast_out["upper_80"],
            mode="lines",
            line=dict(color="rgba(82, 122, 80, 0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_f.add_trace(
        go.Scatter(
            x=df_forecast_out["date"],
            y=df_forecast_out["lower_80"],
            name="80% Prediction Interval",
            mode="lines",
            line=dict(color="rgba(82, 122, 80, 0)", width=0),
            fill="tonexty",
            fillcolor="rgba(82, 122, 80, 0.25)",
            hovertemplate="<b>80% CI Lower:</b> %{y:.2f}<extra></extra>",
        )
    )

# Baseline Comparison (Seasonal Naive)
fig_f.add_trace(
    go.Scatter(
        x=df_forecast_out["date"],
        y=df_forecast_out["baseline_seasonal_naive"],
        name="Seasonal Naive Baseline",
        mode="lines",
        line=dict(color="#C4A66A", width=1.75, dash="dot"),
        hovertemplate="<b>Seasonal Baseline:</b> %{y:.2f}<extra></extra>",
    )
)

# Forecast Point Line (On Top)
fig_f.add_trace(
    go.Scatter(
        x=df_forecast_out["date"],
        y=df_forecast_out["forecast"],
        name=f"Point Forecast ({chosen_model})",
        mode="lines+markers",
        line=dict(color="#527A50", width=3, dash="dash"),
        marker=dict(size=7, symbol="diamond", color="#183D32"),
        hovertemplate="<b>Forecast:</b> %{y:.2f}<br><b>Date:</b> %{x|%b %Y}<extra></extra>",
    )
)

# Vertical Boundary Line at Forecast Origin
fig_f.add_vline(
    x=last_obs_date,
    line_width=1.5,
    line_dash="dash",
    line_color="#5A665D",
    annotation_text="Forecast Origin",
    annotation_position="top left",
    annotation_font=dict(size=11, color="#5A665D", family="Plus Jakarta Sans"),
)

format_plotly_figure(
    fig_f,
    title=f"Multi-Step Documented Occurrence Projections with Forecast Origin ({future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')})",
    height=480,
)
fig_f.update_layout(
    yaxis_title="Documented Occurrence Count",
    xaxis_title="Surveillance Timeline (Monthly)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
)
st.plotly_chart(fig_f, use_container_width=True)

# ==============================================================================
# F. WEATHER CONTEXT & FORECAST INTERPRETATION PANEL
# ==============================================================================
col_w, col_i = st.columns([1, 1.2])

with col_w:
    st.markdown(
        f"""
        <div class="verdant-card" style="height: 100%;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                {icon_weather(22, "#527A50")}
                <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">Weather Context & Climatology</h3>
            </div>
            <p style="font-size: 0.86rem; color: #404B43; line-height: 1.5; margin-bottom: 12px;">
                Observed historical weather variables up to <strong>{last_obs_date.strftime('%B %Y')}</strong> originate from 
                <strong>NASA POWER</strong>. Forward forecasts beyond observed satellite telemetry assume 
                <strong>Baseline Climatology</strong> (10-year monthly historical means).
            </p>
            <div style="background: rgba(16, 44, 38, 0.04); border-left: 3px solid #527A50; padding: 10px 14px; font-size: 0.8rem; color: #183D32; margin-bottom: 12px;">
                <strong>Key Permissive Regimes:</strong> Fungal spore germination peaks when relative humidity exceeds 80% 
                and ambient temperatures range between 22°C–30°C.
            </div>
            <p style="font-size: 0.8rem; color: #5A665D; margin: 0;">
                Scientific Disclaimer: Weather correlations define physiologically permissive incubation windows, not direct biological causation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_i:
    st.markdown(
        f"""
        <div class="verdant-card" style="height: 100%;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                {icon_shield(22, "#527A50")}
                <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">How to Interpret This Forecast</h3>
            </div>
            <ul style="font-size: 0.85rem; color: #404B43; line-height: 1.55; padding-left: 18px; margin-bottom: 10px;">
                <li><strong>Modeled Target:</strong> The series reflects documented occurrence accessions deposited in research herbaria and public surveillance databases, not an absolute acreage census.</li>
                <li><strong>Prediction Intervals:</strong> The 80% and 95% bands convey uncertainty due to process noise and parameter estimation; they do not guarantee that observations cannot fall outside the cone.</li>
                <li><strong>Non-Negative Bounding:</strong> Count estimates are mathematically lower-bounded at zero ($y_t \\ge 0$).</li>
                <li><strong>Reporting Seasonality:</strong> Peaks often align with academic field collection campaigns and agricultural university diagnostic submissions during harvest months.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Operational harvest defense banner
golden_img_path = ASSETS_DIR / "golden_crop_field.jpg"
golden_b64 = get_image_base64(golden_img_path)
if golden_b64:
    st.markdown(
        f"""
        <div class="verdant-card" style="padding: 0; overflow: hidden; margin-top: 24px; margin-bottom: 24px; display: grid; grid-template-columns: 240px 1fr; border: 1px solid rgba(82,122,80,0.18);">
            <div style="height: 130px; overflow: hidden; position: relative;">
                <img src="{golden_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Harvest Season Protection" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.65rem; font-weight: 600; padding: 2px 7px; border-radius: 4px; letter-spacing: 0.05em;">
                    HARVEST DEFENSE
                </div>
            </div>
            <div style="padding: 14px 20px; display: flex; flex-direction: column; justify-content: center; background: #FFFEFA;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                    Operational Preventive Horizon
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.1rem; color: #102C26; margin: 2px 0;">
                    Preventative Spray & Scouting Triggers
                </div>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.45; margin: 0;">
                    When upper 80% prediction intervals exceed seasonal baselines, extension advisors should recommend prophylactic biocontrols before visible canopy necrosis occurs.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
