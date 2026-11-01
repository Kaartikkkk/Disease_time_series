"""Page 4: Future Predictions — Monthly Disease Outlook & Visual Predictions."""

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
    icon_info,
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
    get_disease_common_name,
    get_image_base64,
    get_pathogen_display_name,
    inject_custom_css,
    is_advanced_mode,
    is_simple_mode,
    load_friendly_pathogen_options,
    load_monthly_series,
    render_app_sidebar,
    render_top_header,
    render_what_does_this_mean,
)

st.set_page_config(
    page_title="Future Predictions — Verdant Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="forecasts")
render_top_header()

adv_mode = is_advanced_mode()
pathogen_dict = load_friendly_pathogen_options()
run_timestamp = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")

# ==============================================================================
# A. PAGE HEADER
# ==============================================================================
page_heading = "Forecast Intelligence" if adv_mode else "Future Disease Predictions"
page_subheading = (
    "Explore out-of-sample forward projections across candidate architectures, model confidence intervals, and seasonal climatology."
    if adv_mode
    else "See expected disease reports for the coming months and plan ahead with clear, easy-to-understand predictions."
)

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_forecast(28, "#527A50")}
                <h1 style="font-size: 2.3rem; margin: 0; color: #102C26;">{page_heading}</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                {page_subheading}
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 12px; padding: 8px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="display: flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                {icon_pulse(8, "#527A50")} {"Active Modeling Runtime" if adv_mode else "Prediction System Ready"}
            </div>
            <div style="font-size: 0.75rem; color: #5A665D; margin-top: 2px;">
                Updated: <strong>{run_timestamp}</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# B. FORECAST CONFIGURATION PANEL
# ==============================================================================
section_label = "Model Parameters & Prediction Settings" if adv_mode else "Select Disease, Region & Prediction Period"
st.markdown(f"<div style='font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #5A665D; margin-bottom: 8px;'>{section_label}</div>", unsafe_allow_html=True)

with st.container():
    cfg_c1, cfg_c2, cfg_c3, cfg_c4, cfg_c5 = st.columns([1.3, 1.0, 0.8, 1.3, 1.0])

    with cfg_c1:
        selected_slug = st.selectbox(
            "Plant Disease" if not adv_mode else "Target Pathogen Series",
            options=list(pathogen_dict.keys()),
            format_func=lambda k: pathogen_dict[k],
            index=0,
            key="fc_pathogen",
        )

    with cfg_c2:
        region_options = {
            "all": "All Agro-Climatic Zones" if adv_mode else "All Regions (National)",
            "national_composite": "National Composite" if adv_mode else "National Overview",
            "odisha_bengal": "Eastern Coastal Rice Belt (Odisha / WB)",
            "punjab_haryana": "Northwest Wheat Belt (Punjab / Haryana)",
            "tamil_nadu": "Southern Agricultural Region (Tamil Nadu)",
            "maharashtra": "Western / Central Region (Maharashtra)",
        }
        selected_region = st.selectbox(
            "Region" if not adv_mode else "Agro-Climatic Zone",
            options=list(region_options.keys()),
            format_func=lambda k: region_options[k],
            index=0,
            key="fc_region",
        )

    with cfg_c3:
        horizon_months = st.selectbox(
            "Prediction Period" if not adv_mode else "Forecast Horizon",
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

    def format_model_choice(m: str) -> str:
        if adv_mode:
            return m
        friendly_names = {
            "SARIMA(1,0,1)(1,0,1)[12]": "SARIMA (Seasonal Pattern)",
            "SARIMAX (Seasonal + Weather Regressors)": "SARIMAX (Seasonal + Weather Factors)",
            "ARIMA(1,1,1)": "ARIMA (Trend Following)",
            "Holt-Winters Exponential Smoothing": "Holt-Winters Smoothing",
            "Seasonal Naive (Baseline)": "Seasonal Benchmark (Previous Year)",
            "Hist Gradient Boosting (Autoregressive)": "Machine Learning (Gradient Boosting)",
            "Meta Prophet": "Prophet (Trend & Seasonality)",
        }
        return friendly_names.get(m, m)

    with cfg_c4:
        chosen_model = st.selectbox(
            "Prediction Method" if not adv_mode else "Forecasting Architecture",
            options=available_models,
            format_func=format_model_choice,
            index=0,
            key="fc_model",
        )

    with cfg_c5:
        ci_labels = {
            "Both (80% & 95%)": "Likely & Full Range (80% & 95%)" if not adv_mode else "Both (80% & 95%)",
            "80% Interval Only": "Likely Range Only (80%)" if not adv_mode else "80% Interval Only",
            "95% Interval Only": "Full Range Only (95%)" if not adv_mode else "95% Interval Only",
            "Point Forecast Only": "Predicted Value Only" if not adv_mode else "Point Forecast Only",
        }
        ci_option = st.selectbox(
            "Uncertainty Range" if not adv_mode else "Prediction Intervals",
            options=list(ci_labels.keys()),
            format_func=lambda k: ci_labels[k],
            index=0,
            key="fc_ci",
        )

# Load real dataset
df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"No monthly records found for {pathogen_dict.get(selected_slug, selected_slug)}. Please choose another plant disease.")
    st.stop()

target_col = "documented_occurrence_count"
last_obs_date = pd.to_datetime(df_monthly["date"].max())
first_obs_date = pd.to_datetime(df_monthly["date"].min())
future_dates = pd.date_range(start=last_obs_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

# Metadata Chips
st.markdown(
    f"""
    <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; margin-bottom: 22px;">
        <span style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #183D32;">
            <strong>Disease:</strong> {pathogen_dict[selected_slug]}
        </span>
        <span style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #183D32;">
            <strong>Frequency:</strong> Monthly
        </span>
        <span style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #183D32;">
            <strong>Historical Records:</strong> {first_obs_date.strftime('%b %Y')} – {last_obs_date.strftime('%b %Y')} ({len(df_monthly)} Months)
        </span>
        <span style="background: rgba(82, 122, 80, 0.08); border: 1px solid rgba(82, 122, 80, 0.2); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #183D32;">
            <strong>Method:</strong> {format_model_choice(chosen_model)}
        </span>
        <span style="background: rgba(196, 166, 106, 0.15); border: 1px solid rgba(196, 166, 106, 0.35); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: #695221;">
            <strong>Prediction Period:</strong> {future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')}
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
with st.spinner(f"Calculating monthly disease predictions using {format_model_choice(chosen_model)}..."):
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
        st.error(f"Calculation note for {chosen_model}: {exc}. Defaulting to seasonal baseline.")
        p_fut, lb_95, ub_95 = forecast_seasonal_naive(y_full, horizon=horizon_months, season_length=12)

# Seasonal Naive baseline comparison
p_baseline, _, _ = forecast_seasonal_naive(y_full, horizon=horizon_months, season_length=12)

# Non-negative clipping
p_fut = np.maximum(0.0, p_fut)
lb_95 = np.maximum(0.0, lb_95)
ub_95 = np.maximum(p_fut, ub_95)

# Coherent 80% CI (Z=1.28 vs Z=1.96 for 95%)
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

cal_baselines = compute_calendar_month_baselines(df_monthly, target_col=target_col)
endpoint_cal_m = future_dates[-1].month
endpoint_hist_mean = cal_baselines.get(endpoint_cal_m, {}).get("mean", df_monthly[target_col].mean())
trend_diff = endpoint_forecast - endpoint_hist_mean
trend_sign = "+" if trend_diff >= 0 else ""
mean_ci_width = float(np.mean(df_forecast_out["upper_80"] - df_forecast_out["lower_80"]))

card1_title = "Latest Recorded Reports" if not adv_mode else "Latest Observed Count"
card1_sub = f"Recorded in {last_obs_date.strftime('%b %Y')}" if not adv_mode else f"Deposited in {last_obs_date.strftime('%b %Y')}"

card2_title = "Predicted (End of Period)" if not adv_mode else "Horizon Endpoint Forecast"
card2_sub = f"Predicted for {future_dates[-1].strftime('%b %Y')}"

card3_title = "Trend vs Usual Season" if not adv_mode else "Trend vs Historical Season"
card3_sub = f"vs {future_dates[-1].strftime('%B')} average ({endpoint_hist_mean:.1f})"

card4_title = "Prediction Uncertainty" if not adv_mode else "Uncertainty Spread (80% CI)"
card4_sub = "Likely deviation from prediction" if not adv_mode else "Mean prediction interval half-width"

st.markdown(
    f"""
    <div class="metric-grid" style="margin-bottom: 24px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(20, "#527A50")}</div>
            <div class="metric-label-text">{card1_title}</div>
            <div class="metric-value-text">{latest_obs_val:.1f} reports</div>
            <div class="metric-note-text">{card1_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_forecast(20, "#183D32")}</div>
            <div class="metric-label-text">{card2_title}</div>
            <div class="metric-value-text">{endpoint_forecast:.1f} reports</div>
            <div class="metric-note-text">{card2_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_model_compare(20, "#C4A66A")}</div>
            <div class="metric-label-text">{card3_title}</div>
            <div class="metric-value-text">{trend_sign}{trend_diff:.1f}</div>
            <div class="metric-note-text">{card3_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#527A50")}</div>
            <div class="metric-label-text">{card4_title}</div>
            <div class="metric-value-text">±{mean_ci_width / 2.0:.1f} reports</div>
            <div class="metric-note-text">{card4_sub}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# D. MONTHLY DISEASE OUTLOOK (EVERYDAY PLAIN ENGLISH EXPLANATIONS)
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="margin-bottom: 16px;">
        <h2 style="font-size: 1.85rem; margin: 0 0 4px 0; color: #102C26;">Monthly Disease Outlook</h2>
        <p style="color: #5A665D; font-size: 0.92rem; margin: 0;">
            A clear month-by-month view of predicted disease activity, emerging trends, and which months may require closer monitoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Peak Forecast Month Callout Card
peak_info = identify_peak_forecast_month(df_forecast_out, pathogen_name=pathogen_dict[selected_slug])
peak_text = peak_info.get("simple_narrative") if not adv_mode else peak_info.get("narrative")

st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, rgba(196, 166, 106, 0.12) 0%, rgba(255, 254, 250, 0.95) 100%); border: 1.5px solid #C4A66A; border-radius: 14px; padding: 18px 24px; margin-bottom: 22px; box-shadow: 0 4px 16px rgba(196, 166, 106, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="background: #C4A66A; color: #FFF; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                        {"HIGHEST PREDICTED ACTIVITY" if not adv_mode else "PEAK FORECAST ACTIVITY"}
                    </span>
                    <span style="font-size: 0.82rem; color: #695221; font-weight: 600;">
                        {pathogen_dict[selected_slug]}
                    </span>
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.6rem; font-weight: 600; color: #102C26; margin: 6px 0 4px 0;">
                    {peak_info.get('peak_month', 'N/A')} · ~{peak_info.get('peak_value', 0.0):.1f} Recorded Reports
                </div>
                <p style="font-size: 0.88rem; color: #404B43; line-height: 1.5; margin: 0; max-width: 820px;">
                    {peak_text}
                </p>
            </div>
            <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(196, 166, 106, 0.3); border-radius: 10px; padding: 10px 16px;">
                <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: #695221;">
                    {"Likely Range (80%)" if not adv_mode else "80% Confidence Range"}
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600; color: #102C26;">
                    {peak_info.get('lower_80', 0.0):.1f} – {peak_info.get('upper_80', 0.0):.1f}
                </div>
                <div style="font-size: 0.72rem; color: #5A665D; margin-top: 2px;">Recorded Reports</div>
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
    region_name=region_options.get(selected_region, "India"),
)

# Render cards in 3-column responsive grid
grid_cols = st.columns(min(3, len(monthly_outlooks)))
for i, outlook in enumerate(monthly_outlooks):
    col = grid_cols[i % len(grid_cols)]
    with col:
        pct_str = f" ({outlook['change_pct']:+.1f}%)" if outlook['change_pct'] is not None else ""
        card_narrative = outlook.get("simple_explanation") if not adv_mode else outlook.get("narrative")
        
        ci_text = (
            f"Likely Range: {outlook['lower_80']:.1f} to {outlook['upper_80']:.1f} reports"
            if not adv_mode
            else f"80% CI: [{outlook['lower_80']:.2f}, {outlook['upper_80']:.2f}] · 95% CI: [{outlook['lower_95']:.2f}, {outlook['upper_95']:.2f}]"
        )
        
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
                            {outlook['predicted_count']:.1f}
                        </span>
                        <span style="font-size: 0.8rem; font-weight: 600; color: {'#183D32' if outlook['change_abs'] <= 0 else '#C4A66A'};">
                            {outlook['trend_arrow']} {outlook['change_abs']:+.1f}{pct_str}
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: #5A665D; margin-bottom: 10px;">
                        {ci_text}
                    </div>
                    <p style="font-size: 0.84rem; color: #404B43; line-height: 1.5; margin: 0;">
                        {card_narrative}
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# What Does This Mean card for Monthly Outlook
render_what_does_this_mean(
    simple_text="This section shows what the model predicts for each upcoming month and highlights whether disease activity is expected to rise, stay steady, or decline.",
    technical_details="Monthly values are estimated by fitting autoregressive and seasonal lags. The uncertainty range reflects natural variability in past data.",
)

# 3. Month-by-Month Forecast Summary Table & Export
st.markdown("<div style='margin-top: 14px; margin-bottom: 6px; font-size: 1.1rem; font-family: Fraunces, serif; color: #102C26;'>Month-by-Month Forecast Summary</div>", unsafe_allow_html=True)

table_rows = []
for out in monthly_outlooks:
    pct_text = f"{out['change_pct']:+.1f}%" if out['change_pct'] is not None else "N/A"
    if not adv_mode:
        table_rows.append({
            "Month": out["month_label"],
            "Predicted Reports": out["predicted_count"],
            "Change from Previous": f"{out['change_abs']:+.1f} ({pct_text})",
            "Trend": f"{out['trend_arrow']} {out['trend_label']}",
            "Likely Range (80%)": f"{out['lower_80']} – {out['upper_80']}",
            "Full Range (95%)": f"{out['lower_95']} – {out['upper_95']}",
            "Expected Activity": out["activity_level"],
        })
    else:
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
    label="📥 Download Forecast Table (CSV)",
    data=csv_summary,
    file_name=f"monthly_predictions_{selected_slug}_{horizon_months}m.csv",
    mime="text/csv",
)

# 4. Multi-Disease Attention Summary
with st.expander("🔍 Compare Predicted Activity Across All 10 Plant Diseases", expanded=False):
    st.markdown(
        "See how future predicted activity compares across all monitored plant diseases in India."
    )
    df_watch = build_multi_disease_watch_matrix(pathogen_dict, load_monthly_series, horizon_months=6)
    if not df_watch.empty:
        col_names = {
            "disease_name": "Plant Disease",
            "peak_month": "Peak Month",
            "peak_val": "Predicted Peak Reports",
            "seasonal_baseline": "Normal Season Average",
            "relative_elevation": "Above/Below Normal (%)",
            "activity_level": "Activity Level",
            "data_quality": "Data Reliability",
            "total_records": "Historical Reports",
        }
        st.dataframe(
            df_watch[[
                "disease_name", "peak_month", "peak_val", "seasonal_baseline", 
                "relative_elevation", "activity_level", "data_quality", "total_records"
            ]].rename(columns=col_names),
            use_container_width=True,
            hide_index=True,
        )
    st.caption("Note: Disease activity is compared relative to each disease's normal seasonal pattern for a fair national comparison.")

# ==============================================================================
# E. MAIN FORECAST VISUALIZATION (CENTERPIECE TIME-SERIES CHART)
# ==============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="margin-bottom: 12px;">
        <h2 style="font-size: 1.85rem; margin: 0 0 4px 0; color: #102C26;">
            {"Disease Predictions Over Time: " + pathogen_dict[selected_slug] if not adv_mode else "Time-Series Trajectory: " + pathogen_dict[selected_slug]}
        </h2>
        <p style="color: #5A665D; font-size: 0.92rem; margin: 0;">
            {"Past recorded reports, future predictions, and likely ranges of possible results." if not adv_mode else "Observed surveillance frequency, point projections, and nested 80%/95% prediction intervals."}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

fig_f = go.Figure()

# Recent Historical Observations (last 36 months)
hist_subset = df_monthly.tail(36).copy()

fig_f.add_trace(
    go.Scatter(
        x=hist_subset["date"],
        y=hist_subset[target_col],
        name="Past Recorded Reports" if not adv_mode else "Historical Documented Occurrences",
        mode="lines+markers",
        line=dict(color="#102C26", width=2.5),
        marker=dict(size=6, color="#102C26"),
        hovertemplate="<b>Recorded:</b> %{y:.1f} reports<br><b>Month:</b> %{x|%b %Y}<extra></extra>",
    )
)

# Uncertainty Intervals
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
            name="Full Range (95%)" if not adv_mode else "95% Prediction Interval",
            mode="lines",
            line=dict(color="rgba(82, 122, 80, 0)", width=0),
            fill="tonexty",
            fillcolor="rgba(82, 122, 80, 0.12)",
            hovertemplate="<b>95% Lower:</b> %{y:.2f} reports<extra></extra>",
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
            name="Likely Range (80%)" if not adv_mode else "80% Prediction Interval",
            mode="lines",
            line=dict(color="rgba(82, 122, 80, 0)", width=0),
            fill="tonexty",
            fillcolor="rgba(82, 122, 80, 0.25)",
            hovertemplate="<b>80% Lower:</b> %{y:.2f} reports<extra></extra>",
        )
    )

# Baseline Comparison (Seasonal Naive)
fig_f.add_trace(
    go.Scatter(
        x=df_forecast_out["date"],
        y=df_forecast_out["baseline_seasonal_naive"],
        name="Seasonal Benchmark (Past Year)" if not adv_mode else "Seasonal Naive Baseline",
        mode="lines",
        line=dict(color="#C4A66A", width=1.75, dash="dot"),
        hovertemplate="<b>Seasonal Benchmark:</b> %{y:.2f} reports<extra></extra>",
    )
)

# Forecast Point Line
fig_f.add_trace(
    go.Scatter(
        x=df_forecast_out["date"],
        y=df_forecast_out["forecast"],
        name=f"Predicted Reports ({format_model_choice(chosen_model)})",
        mode="lines+markers",
        line=dict(color="#527A50", width=3, dash="dash"),
        marker=dict(size=7, symbol="diamond", color="#183D32"),
        hovertemplate="<b>Prediction:</b> %{y:.2f} reports<br><b>Month:</b> %{x|%b %Y}<extra></extra>",
    )
)

# Vertical Boundary Line at Forecast Origin
fig_f.add_vline(
    x=last_obs_date,
    line_width=1.5,
    line_dash="dash",
    line_color="#5A665D",
    annotation_text="Today / Prediction Begins" if not adv_mode else "Forecast Origin",
    annotation_position="top left",
    annotation_font=dict(size=11, color="#5A665D", family="Plus Jakarta Sans"),
)

format_plotly_figure(
    fig_f,
    title=f"Predicted Disease Reports ({future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')})",
    height=480,
)
fig_f.update_layout(
    yaxis_title="Recorded Disease Reports",
    xaxis_title="Timeline (Month)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
)
st.plotly_chart(fig_f, use_container_width=True)

# What Does This Mean card for Disease Prediction Chart
render_what_does_this_mean(
    simple_text="This chart shows past disease reports and what the model predicts for the coming months. The shaded area shows a range of possible results. A wider range means the prediction is less precise.",
    technical_details="The model uses historical monthly rhythms to project future values. The 80% and 95% shaded bands represent statistical prediction intervals. Values are strictly bounded at zero.",
)

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
                <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">{"Weather Context & Seasonal Climate" if not adv_mode else "Weather Context & Climatology"}</h3>
            </div>
            <p style="font-size: 0.86rem; color: #404B43; line-height: 1.5; margin-bottom: 12px;">
                {"Past weather observations up to " + last_obs_date.strftime('%B %Y') + " come from NASA satellite telemetry. Future months assume typical 10-year average weather patterns for each season." if not adv_mode else "Observed historical weather variables up to " + last_obs_date.strftime('%B %Y') + " originate from NASA POWER. Forward forecasts beyond observed satellite telemetry assume Baseline Climatology (10-year monthly historical means)."}
            </p>
            <div style="background: rgba(16, 44, 38, 0.04); border-left: 3px solid #527A50; padding: 10px 14px; font-size: 0.8rem; color: #183D32; margin-bottom: 12px;">
                <strong>{"Key Weather Conditions:" if not adv_mode else "Key Permissive Regimes:"}</strong> {"Fungal and bacterial plant diseases often develop faster when humidity exceeds 80% and temperatures stay between 22°C and 30°C." if not adv_mode else "Fungal spore germination peaks when relative humidity exceeds 80% and ambient temperatures range between 22°C–30°C."}
            </div>
            <p style="font-size: 0.8rem; color: #5A665D; margin: 0;">
                {"Note: Favorable weather creates an environment where disease can spread more easily, but timely crop management can prevent damage." if not adv_mode else "Scientific Disclaimer: Weather correlations define physiologically permissive incubation windows, not direct biological causation."}
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
                <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">{"How to Understand This Forecast" if not adv_mode else "How to Interpret This Forecast"}</h3>
            </div>
            <ul style="font-size: 0.85rem; color: #404B43; line-height: 1.55; padding-left: 18px; margin-bottom: 10px;">
                <li><strong>{"Recorded Reports, Not Infected Acres:" if not adv_mode else "Modeled Target:"}</strong> {"These numbers represent documented disease occurrences from research herbaria and public surveillance databases, not the total number of affected fields or farms." if not adv_mode else "The series reflects documented occurrence accessions deposited in research herbaria and public surveillance databases, not an absolute acreage census."}</li>
                <li><strong>{"Predictions Have Natural Uncertainty:" if not adv_mode else "Prediction Intervals:"}</strong> {"Weather and biology vary from year to year. The shaded uncertainty range shows where actual results are expected to fall." if not adv_mode else "The 80% and 95% bands convey uncertainty due to process noise and parameter estimation."}</li>
                <li><strong>{"Predictions Cannot Be Negative:" if not adv_mode else "Non-Negative Bounding:"}</strong> {"Report counts are mathematically kept at zero or above." if not adv_mode else "Count estimates are mathematically lower-bounded at zero."}</li>
                <li><strong>{"Seasonal Timing:" if not adv_mode else "Reporting Seasonality:"}</strong> {"Peaks often happen around harvest or rainy months when researchers and extension workers conduct active field testing." if not adv_mode else "Peaks often align with academic field collection campaigns and agricultural university diagnostic submissions during harvest months."}</li>
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
                <img src="{golden_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Crop Protection Advice" />
                <div style="position: absolute; bottom: 8px; left: 10px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.65rem; font-weight: 600; padding: 2px 7px; border-radius: 4px; letter-spacing: 0.05em;">
                    CROP PROTECTION
                </div>
            </div>
            <div style="padding: 14px 20px; display: flex; flex-direction: column; justify-content: center; background: #FFFEFA;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                    {"Planning & Prevention Guidance" if not adv_mode else "Operational Preventive Horizon"}
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.1rem; color: #102C26; margin: 2px 0;">
                    {"Field Scouting & Early Advisory" if not adv_mode else "Preventative Spray & Scouting Triggers"}
                </div>
                <p style="font-size: 0.82rem; color: #5A665D; line-height: 1.45; margin: 0;">
                    {"When predicted disease activity rises above normal seasonal levels, farmers and extension officers can plan early field inspections and use approved preventive treatments before crop health is affected." if not adv_mode else "When upper 80% prediction intervals exceed seasonal baselines, extension advisors should recommend prophylactic biocontrols before visible canopy necrosis occurs."}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
