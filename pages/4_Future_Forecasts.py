"""Page 4: Future Forecasts — Out-of-Sample Surveillance Projections & Monthly Risk Outlook.

Structured to match exact visual mockup specification:
1. Shared Top Navigation Bar (AgriForecast standard)
2. Editorial Hero Card (Crop ear artwork on right)
3. Filter Toolbar Card (Disease dropdown, Model dropdown, Horizon slider)
4. Metadata Banner (Verified Cutoff & Forward Period with Origin badge)
5. Four Compact KPI Metric Cards (Last Count, Peak Month, Mean Projection, Uncertainty Spread)
6. Primary Forecast Chart (Historical trace, Cutoff line, Forecast trace & 95% Confidence Band)
7. Peak Activity Callout Box (80% Expected Range on right)
8. Month-by-Month Risk Outlook (Horizontal Row of Cards with Peak Month highlighted)
9. Detailed Forecast Table (Columns with Rank 1/Peak highlighted & badges)
10. Multi-Disease Watch Matrix & Bottom Action CTAs with Hill Contours
"""

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
    generate_future_weather_scenario,
    run_model_comparison,
)
from src.icons import (
    icon_arrow_right,
    icon_bar_chart,
    icon_calendar,
    icon_check_circle,
    icon_download,
    icon_forecast,
    icon_info,
    icon_layers,
    icon_leaf,
    icon_model_compare,
    icon_pulse,
    icon_shield,
    icon_target,
    icon_timeseries,
    icon_trending_up,
    icon_weather,
)
from src.models.ar_model import forecast_ar
from src.models.arima_model import forecast_arima
from src.models.arimax_model import forecast_arimax
from src.models.global_models import (
    forecast_global_random_forest,
    forecast_global_xgboost,
    forecast_hurdle_xgboost,
)
from src.models.gradient_boosting import forecast_gradient_boosting
from src.models.gru_model import forecast_gru
from src.models.holt_winters import forecast_holt_winters
from src.models.lstm_model import forecast_lstm
from src.models.ma_model import forecast_ma
from src.models.prophet_model import forecast_prophet
from src.models.random_forest import forecast_random_forest
from src.models.sarima_model import forecast_sarima
from src.models.sarimax_model import forecast_sarimax_model
from src.models.seasonal_naive import forecast_seasonal_naive
from src.models.xgboost_model import forecast_xgboost
from src.project_stats import get_project_telemetry
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
    page_title="Future Forecasts — AgriForecast",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_top_navigation(active_page="forecast")

stats = get_project_telemetry()
pathogen_dict = load_all_pathogen_options()

hero_ear_b64 = get_image_base64(ASSETS_DIR / "hero_ear_blended.png")
obj_hills_b64 = get_image_base64(ASSETS_DIR / "objective_hills_blended.png")

# ==============================================================================
# 1. EDITORIAL HERO BANNER (MATCHING SPECIFICATION)
# ==============================================================================
hero_markup = f"""
<div class="dashboard-hero-card" style="min-height: 180px; margin-bottom: 20px;">
    <div class="hero-text-content" style="padding: 28px 36px; max-width: 620px;">
        <div class="hero-tag">FUTURE FORECAST</div>
        <h1 class="hero-headline" style="font-size: 2.1rem; margin-bottom: 8px;">
            Future Disease Forecasts &amp; Surveillance Outlook
        </h1>
        <p class="hero-subtitle" style="font-size: 0.95rem; margin-bottom: 0;">
            Forward projections from verified historical data with 80% and 95% expected ranges.
        </p>
    </div>
    <div class="hero-image-wrapper" style="width: 50%;">
        <img src="{hero_ear_b64}" class="hero-ear-img" alt="Future forecast" />
    </div>
</div>
"""
st.html(hero_markup)

# ==============================================================================
# 2. CONTROLS TOOLBAR CARD
# ==============================================================================
st.markdown('<div class="filter-toolbar-card" style="margin-bottom: 16px;">', unsafe_allow_html=True)
col_c1, col_c2, col_c3 = st.columns([1.3, 1.4, 1.1])

with col_c1:
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
    )

df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"No monthly records found for {selected_slug}.")
    st.stop()

target_col = "documented_occurrence_count"
last_obs_date = pd.to_datetime(df_monthly["date"].max())
first_obs_date = pd.to_datetime(df_monthly["date"].min())

with col_c2:
    st.markdown(
        f"<div class='filter-field-label'>{icon_layers(15, '#2D6A4F')} Forecasting Model</div>",
        unsafe_allow_html=True,
    )
    model_choices = [
        "Hurdle XGBoost (Classifier + Regressor)",
        "LSTM (Sequence Neural Net)",
        "GRU (Gated Recurrent Unit)",
        "Global Random Forest (Multi-Pathogen)",
        "Global XGBoost (Multi-Pathogen)",
        "Prophet (Trend + Seasonality)",
        "MA(1)",
        "AR(1)",
        "Seasonal Naive (Baseline)",
        "Holt-Winters Exponential Smoothing",
        "ARIMA(1,1,1)",
        "SARIMA(1,0,1)(1,0,1)[12]",
        "ARIMAX (Weather Regressors)",
        "SARIMAX (Seasonal + Weather)",
        "Random Forest (Autoregressive)",
        "XGBoost (Autoregressive)",
    ]
    chosen_model = st.selectbox(
        "Forecasting Model Architecture",
        options=model_choices,
        index=0,
        label_visibility="collapsed",
        help="Select a calibrated model to project future disease occurrence.",
    )

with col_c3:
    st.markdown(
        f"<div class='filter-field-label'>{icon_calendar(15, '#2D6A4F')} Forecast Period (Months)</div>",
        unsafe_allow_html=True,
    )
    horizon_months = st.slider(
        "Forecast Period (Months)",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
        label_visibility="collapsed",
        help="Number of future months to project forward from the last observed date.",
    )
st.markdown('</div>', unsafe_allow_html=True)

future_dates = pd.date_range(start=last_obs_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

# ==============================================================================
# 3. METADATA BANNER (VERIFIED CUTOFF & FORWARD PERIOD)
# ==============================================================================
metadata_banner_html = f"""
<div style="background: #F4F8F5; border: 1px solid rgba(45, 106, 79, 0.2); border-radius: 10px; padding: 10px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
    <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #1B4332;">
        {icon_info(16, '#2D6A4F')}
        <span><strong>Verified Database Cutoff:</strong> {last_obs_date.strftime('%b %Y')} &nbsp;|&nbsp; <strong>Forward Forecast Period:</strong> {future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')} ({horizon_months} Months Ahead)</span>
    </div>
    <div style="background: #FFFFFF; border: 1px solid rgba(45, 106, 79, 0.25); color: #2D6A4F; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 9999px;">
        Origin: {last_obs_date.strftime('%b %Y')}
    </div>
</div>
"""
st.html(metadata_banner_html)

# Weather Regressor Scenario Generation
weather_cols = ["temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"]
has_weather = all(c in df_monthly.columns for c in weather_cols)
df_future_weather = generate_future_weather_scenario(df_monthly, horizon_months=horizon_months) if has_weather else None

# Run Forecast Execution
with st.spinner(f"Projecting forward occurrences using {chosen_model}..."):
    try:
        y_full = df_monthly[target_col].astype(float)
        train_exog = df_monthly[weather_cols].bfill().fillna(0) if has_weather else None
        fut_exog = df_future_weather[weather_cols].bfill().fillna(0) if df_future_weather is not None else None

        if chosen_model == "Seasonal Naive (Baseline)":
            p_fut, lb_95, ub_95 = forecast_seasonal_naive(y_full, horizon=horizon_months, season_length=12)
        elif chosen_model == "AR(1)":
            p_fut, lb_95, ub_95, _ = forecast_ar(y_full, horizon=horizon_months, p=1)
        elif chosen_model == "MA(1)":
            p_fut, lb_95, ub_95, _ = forecast_ma(y_full, horizon=horizon_months, q=1)
        elif chosen_model == "ARIMA(1,1,1)":
            p_fut, lb_95, ub_95, _ = forecast_arima(y_full, horizon=horizon_months, order=(1, 1, 1), auto_optimize=True)
        elif chosen_model == "SARIMA(1,0,1)(1,0,1)[12]":
            p_fut, lb_95, ub_95, _ = forecast_sarima(y_full, horizon=horizon_months, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), auto_optimize=True)
        elif chosen_model == "Holt-Winters Exponential Smoothing":
            p_fut, lb_95, ub_95, _ = forecast_holt_winters(y_full, horizon=horizon_months, seasonal_periods=12, auto_optimize=True)
        elif chosen_model == "ARIMAX (Weather Regressors)":
            p_fut, lb_95, ub_95, _ = forecast_arimax(
                y_full, horizon=horizon_months, train_exog=train_exog, future_exog=fut_exog, order=(1, 0, 1), auto_optimize=True
            )
        elif chosen_model == "SARIMAX (Seasonal + Weather)":
            p_fut, lb_95, ub_95, _ = forecast_sarimax_model(
                y_full, horizon=horizon_months, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), train_exog=train_exog, future_exog=fut_exog, auto_optimize=True
            )
        elif chosen_model == "Random Forest (Autoregressive)":
            p_fut, lb_95, ub_95, _ = forecast_random_forest(
                df_monthly, horizon=horizon_months, lags=[1, 2, 3, 12],
                weather_lag_cols=weather_cols if has_weather else None,
                future_weather_df=df_future_weather if has_weather else None,
            )
        elif chosen_model == "XGBoost (Autoregressive)":
            p_fut, lb_95, ub_95, _, _ = forecast_xgboost(
                df_monthly, horizon=horizon_months,
                weather_lag_cols=weather_cols if has_weather else None,
                future_weather_df=df_future_weather if has_weather else None,
            )
        elif chosen_model == "Prophet (Trend + Seasonality)":
            p_fut, lb_95, ub_95, _ = forecast_prophet(
                df_monthly, horizon=horizon_months, future_dates=future_dates
            )
        elif chosen_model == "LSTM (Sequence Neural Net)":
            p_fut, _, _, _, _ = forecast_lstm(df_monthly, horizon=horizon_months)
            lb_95 = p_fut.copy()
            ub_95 = p_fut.copy()
        elif chosen_model == "GRU (Gated Recurrent Unit)":
            p_fut, _, _, _, _ = forecast_gru(df_monthly, horizon=horizon_months)
            lb_95 = p_fut.copy()
            ub_95 = p_fut.copy()
        elif chosen_model == "Global XGBoost (Multi-Pathogen)":
            p_fut, lb_95, ub_95, _, _ = forecast_global_xgboost(
                df_monthly, horizon=horizon_months,
                future_weather_df=df_future_weather if has_weather else None,
            )
        elif chosen_model == "Global Random Forest (Multi-Pathogen)":
            p_fut, lb_95, ub_95, _, _ = forecast_global_random_forest(
                df_monthly, horizon=horizon_months,
                future_weather_df=df_future_weather if has_weather else None,
            )
        elif chosen_model == "Hurdle XGBoost (Classifier + Regressor)":
            p_fut, lb_95, ub_95, _, _ = forecast_hurdle_xgboost(
                df_monthly, horizon=horizon_months,
                future_weather_df=df_future_weather if has_weather else None,
            )
        else:
            p_fut, lb_95, ub_95, _ = forecast_gradient_boosting(
                df_monthly, horizon=horizon_months, lags=[1, 2, 3, 12],
                weather_lag_cols=weather_cols if has_weather else None,
                future_weather_df=df_future_weather if has_weather else None,
            )
    except Exception as exc:
        st.warning(f"Note: {chosen_model} calculation defaulted to seasonal persistence: {exc}")
        p_fut, lb_95, ub_95 = forecast_seasonal_naive(y_full, horizon=horizon_months, season_length=12)

# Non-negative clipping and intervals
p_fut = np.maximum(0.0, p_fut)
lb_95 = np.maximum(0.0, lb_95)
ub_95 = np.maximum(p_fut, ub_95)

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
})

peak_info = identify_peak_forecast_month(df_forecast_out, pathogen_name=pathogen_dict[selected_slug])

# ==============================================================================
# 4. FOUR COMPACT KPI METRIC CARDS
# ==============================================================================
latest_obs_val = float(df_monthly[target_col].dropna().iloc[-1])
mean_forecast = float(np.mean(df_forecast_out["forecast"]))
total_forecast = float(np.sum(df_forecast_out["forecast"]))
mean_ci_width = float(np.mean(df_forecast_out["upper_80"] - df_forecast_out["lower_80"]))

kpi_markup = f"""
<div class="kpi-row-grid" style="margin-top: 14px; margin-bottom: 22px;">
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_layers(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label" style="text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; font-weight: 700; color: #6B7280;">Last Recorded Count</div>
            <div class="kpi-value">{latest_obs_val:.1f} Reports</div>
            <div class="kpi-subtext">Observed in {last_obs_date.strftime('%b %Y')}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_trending_up(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label" style="text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; font-weight: 700; color: #6B7280;">Peak Forecast Month</div>
            <div class="kpi-value">{peak_info.get('peak_value', 0.0):.1f} Reports</div>
            <div class="kpi-subtext">Expected in {peak_info.get('peak_month', 'N/A')}</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_calendar(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label" style="text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; font-weight: 700; color: #6B7280;">Mean Monthly Projection</div>
            <div class="kpi-value">~{mean_forecast:.1f} / mo</div>
            <div class="kpi-subtext">Total: {total_forecast:.1f} Across {horizon_months} Mo</div>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon-badge">{icon_shield(20, "#2D6A4F")}</div>
        <div>
            <div class="kpi-label" style="text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.05em; font-weight: 700; color: #6B7280;">Uncertainty Spread</div>
            <div class="kpi-value">±{mean_ci_width / 2.0:.1f} Reports</div>
            <div class="kpi-subtext">80% Expected Range Half-Width</div>
        </div>
    </div>
</div>
"""
st.html(kpi_markup)

# ==============================================================================
# 5. PRIMARY FORECAST VISUALIZATION (WHITE CARD WITH EXACT LEGEND)
# ==============================================================================
st.markdown(
    f"""
    <div class="explorer-chart-card">
        <div class="chart-card-header">
            <div class="chart-card-title" style="display: flex; align-items: center; gap: 8px;">
                {icon_bar_chart(18, "#2D6A4F")}
                <span>Disease Occurrence Forecast: {pathogen_dict[selected_slug].split(' (')[0]} ({future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')})</span>
            </div>
            <div class="chart-card-legend" style="font-size: 0.78rem;">
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 3px; background: #16A34A; border-radius: 2px;"></span>
                    <span>Forecast ({chosen_model.split(' (')[0]})</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 10px; background: rgba(34, 197, 94, 0.2); border-radius: 2px;"></span>
                    <span>95% Expected Range</span>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="display: inline-block; width: 14px; height: 3px; background: #102C26; border-radius: 2px;"></span>
                    <span>Observed Reports</span>
                </div>
            </div>
        </div>
    """,
    unsafe_allow_html=True,
)

fig_f = go.Figure()

# Recent Historical Observations (last 24 months)
hist_subset = df_monthly.tail(24).copy()
fig_f.add_trace(
    go.Scatter(
        x=hist_subset["date"],
        y=hist_subset[target_col],
        name="Observed Reports",
        mode="lines+markers",
        line=dict(color="#102C26", width=2.4),
        marker=dict(size=5, color="#102C26"),
        hovertemplate="<b>Observed:</b> %{y:.1f}<br><b>Date:</b> %{x|%b %Y}<extra></extra>",
    )
)

# 95% Confidence Band
fig_f.add_trace(
    go.Scatter(
        x=pd.concat([pd.Series([last_obs_date]), pd.Series(df_forecast_out["date"])]),
        y=pd.concat([pd.Series([latest_obs_val]), df_forecast_out["upper_95"]]),
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig_f.add_trace(
    go.Scatter(
        x=pd.concat([pd.Series([last_obs_date]), pd.Series(df_forecast_out["date"])]),
        y=pd.concat([pd.Series([latest_obs_val]), df_forecast_out["lower_95"]]),
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(34, 197, 94, 0.16)",
        name="95% Expected Range",
    )
)

# Point Forecast Line
plot_dates = pd.concat([pd.Series([last_obs_date]), pd.Series(df_forecast_out["date"])])
plot_vals = pd.concat([pd.Series([latest_obs_val]), df_forecast_out["forecast"]])

fig_f.add_trace(
    go.Scatter(
        x=plot_dates,
        y=plot_vals,
        name=f"Forecast ({chosen_model.split(' (')[0]})",
        mode="lines+markers",
        line=dict(color="#16A34A", width=2.6, dash="dash"),
        marker=dict(size=6, color="#16A34A"),
        hovertemplate="<b>Forecast:</b> %{y:.1f}<br><b>Date:</b> %{x|%b %Y}<extra></extra>",
    )
)

# Vertical Cutoff line
fig_f.add_vline(
    x=last_obs_date,
    line_width=1.6,
    line_dash="dot",
    line_color="#D97706",
    annotation_text=f"Cutoff ({last_obs_date.strftime('%b %Y')})",
    annotation_position="top left",
    annotation_font=dict(size=10, color="#D97706"),
)

format_plotly_figure(
    fig_f,
    title="",
    height=340,
)
fig_f.update_layout(
    margin=dict(l=30, r=20, t=10, b=30),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", tickformat="%b %Y"),
    yaxis=dict(title="Reports", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
    showlegend=False,
)
st.plotly_chart(fig_f, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 6. CALLOUT CARD BELOW CHART (PEAK SURVEILLANCE & 80% RANGE)
# ==============================================================================
callout_html = f"""
<div style="background: #F4F8F5; border: 1px solid rgba(45, 106, 79, 0.2); border-radius: 12px; padding: 14px 22px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-top: 14px; margin-bottom: 24px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 36px; height: 36px; border-radius: 50%; background: #EBF5EE; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
            {icon_target(20, '#2D6A4F')}
        </div>
        <div>
            <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #6B7280; letter-spacing: 0.05em;">Peak Projected Surveillance Window</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #102C26; margin: 2px 0;">{peak_info.get('peak_month', 'N/A')} ~ {peak_info.get('peak_value', 0.0):.1f} Projected Reports</div>
            <div style="font-size: 0.82rem; color: #5A665D;">Highest predicted occurrence intensity during the next {horizon_months} months.</div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 10px; background: #EBF5EE; border: 1px solid rgba(45, 106, 79, 0.2); border-radius: 8px; padding: 8px 18px;">
        {icon_bar_chart(18, '#2D6A4F')}
        <div>
            <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #2D6A4F;">80% Expected Range</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #102C26; line-height: 1.1;">{peak_info.get('lower_80', 0.0):.1f} – {peak_info.get('upper_80', 0.0):.1f}</div>
        </div>
    </div>
</div>
"""
st.html(callout_html)

# ==============================================================================
# 7. MONTH-BY-MONTH RISK OUTLOOK (HORIZONTAL ROW OF CARDS)
# ==============================================================================
monthly_outlooks = generate_monthly_disease_outlook(
    df_forecast=df_forecast_out,
    df_historical=df_monthly,
    pathogen_name=pathogen_dict[selected_slug],
    target_col=target_col,
    region_name="India",
)

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        {icon_calendar(18, '#2D6A4F')}
        <h3 style="font-size: 1.2rem; font-weight: 700; color: #102C26; margin: 0;">Month-by-Month Risk Outlook</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

outlook_cards_html = []
for out in monthly_outlooks:
    is_peak = (out["month_label"] == peak_info.get("peak_month"))
    card_border = "border: 1.5px solid #2D6A4F; background: #F7FAF8;" if is_peak else "border: 1px solid rgba(82, 122, 80, 0.18); background: #FFFFFF;"
    
    # Delta Badge
    chg = out.get("change_abs", 0.0)
    if chg > 0:
        delta_markup = f"<span style='color: #166534; font-size: 0.72rem; font-weight: 700;'>▲ +{chg:.1f}</span>"
    elif chg < 0:
        delta_markup = f"<span style='color: #DC2626; font-size: 0.72rem; font-weight: 700;'>▼ {chg:.1f}</span>"
    else:
        delta_markup = f"<span style='color: #6B7280; font-size: 0.72rem; font-weight: 700;'>— 0.0</span>"
        
    # Status Pill
    level = out.get("activity_level", "Within typical activity")
    if "Above" in level:
        status_pill = "<span style='background: #FEF3C7; color: #92400E; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 9999px; display: inline-block;'>Above typical activity</span>"
    else:
        status_pill = "<span style='background: #EBF5EE; color: #166534; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 9999px; display: inline-block;'>Within typical activity</span>"
        
    card = f"""
    <div style="{card_border} border-radius: 12px; padding: 14px 16px; flex: 1 1 140px; min-width: 130px; box-shadow: 0 1px 4px rgba(0,0,0,0.02); text-align: center;">
        <div style="font-size: 0.80rem; font-weight: 700; color: #4B5563; margin-bottom: 6px;">{out['month_label']}</div>
        <div style="display: flex; align-items: baseline; justify-content: center; gap: 6px; margin-bottom: 8px;">
            <span style="font-size: 1.55rem; font-weight: 700; color: #102C26;">{out['predicted_count']:.1f}</span>
            {delta_markup}
        </div>
        <div>{status_pill}</div>
    </div>
    """
    outlook_cards_html.append(card)

row_markup = f"""
<div style="display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; margin-bottom: 24px;">
    {''.join(outlook_cards_html)}
</div>
"""
st.html(row_markup)

# ==============================================================================
# 8. DETAILED FORECAST TABLE (MATCHING SPECIFICATION)
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        {icon_layers(18, '#2D6A4F')}
        <h3 style="font-size: 1.2rem; font-weight: 700; color: #102C26; margin: 0;">Detailed Forecast ({future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')})</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

table_rows_html = []
for out in monthly_outlooks:
    is_peak = (out["month_label"] == peak_info.get("peak_month"))
    row_bg = "background-color: #EBF5EE; font-weight: 600;" if is_peak else "background-color: #FFFFFF;"
    
    pct_text = f"{out['change_pct']:+.1f}%" if out['change_pct'] is not None else "N/A"
    chg_abs = out.get("change_abs", 0.0)
    chg_str = f"+{chg_abs:.1f}" if chg_abs > 0 else f"{chg_abs:.1f}"
    
    trend_lbl = out.get("trend_label", "Broadly stable")
    if "Increas" in trend_lbl:
        trend_markup = f"<span style='color: #166534; font-weight: 600;'>▲ {trend_lbl}</span>"
    elif "Decreas" in trend_lbl:
        trend_markup = f"<span style='color: #DC2626; font-weight: 600;'>▼ {trend_lbl}</span>"
    else:
        trend_markup = f"<span style='color: #6B7280; font-weight: 600;'>▼ {trend_lbl}</span>"
        
    level = out.get("activity_level", "Within typical activity")
    if "Above" in level:
        risk_pill = "<span style='background: #FEF3C7; color: #92400E; font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 9999px;'>Above typical activity</span>"
    else:
        risk_pill = "<span style='background: #EBF5EE; color: #166534; font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 9999px;'>Within typical activity</span>"
        
    r_html = f"""
    <tr style="{row_bg} border-bottom: 1px solid #F1F5F2; height: 38px;">
        <td style="padding: 9px 12px; color: #102C26; font-weight: {'700' if is_peak else '500'};">{out['month_label']}</td>
        <td style="padding: 9px 12px; color: #102C26; font-weight: {'700' if is_peak else '500'};">{out['predicted_count']:.2f}</td>
        <td style="padding: 9px 12px; color: #4B5563;">{chg_str} ({pct_text})</td>
        <td style="padding: 9px 12px;">{trend_markup}</td>
        <td style="padding: 9px 12px; color: #4B5563;">{out['lower_80']:.1f} – {out['upper_80']:.2f}</td>
        <td style="padding: 9px 12px; color: #4B5563;">{out['lower_95']:.1f} – {out['upper_95']:.2f}</td>
        <td style="padding: 9px 12px;">{risk_pill}</td>
    </tr>
    """
    table_rows_html.append(r_html)

styled_forecast_table = f"""
<div style="border: 1px solid rgba(82, 122, 80, 0.18); border-radius: 10px; overflow: hidden; margin-bottom: 18px; background: #FFFFFF; box-shadow: 0 1px 4px rgba(0,0,0,0.02);">
    <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left;">
        <thead>
            <tr style="background-color: #F8FAF9; border-bottom: 1.5px solid rgba(82, 122, 80, 0.18); height: 40px;">
                <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">Month</th>
                <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">Predicted Reports</th>
                <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">Change from Prior</th>
                <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">Trend Direction</th>
                <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">80% Uncertainty Range</th>
                <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">95% Uncertainty Range</th>
                <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">Surveillance Risk</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows_html)}
        </tbody>
    </table>
</div>
"""
st.html(styled_forecast_table)

# Download CSV
export_rows = []
for out in monthly_outlooks:
    pct_text = f"{out['change_pct']:+.1f}%" if out['change_pct'] is not None else "N/A"
    export_rows.append({
        "Month": out["month_label"],
        "Predicted Reports": out["predicted_count"],
        "Change from Prior": f"{out['change_abs']:+.2f} ({pct_text})",
        "Trend Direction": f"{out['trend_arrow']} {out['trend_label']}",
        "80% Uncertainty Range": f"{out['lower_80']} – {out['upper_80']}",
        "95% Uncertainty Range": f"{out['lower_95']} – {out['upper_95']}",
        "Surveillance Risk": out["activity_level"],
    })
df_export = pd.DataFrame(export_rows)
csv_summary = df_export.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download Projected Forecast Table (CSV - {len(df_export)} Months)",
    data=csv_summary,
    file_name=f"disease_forecast_{selected_slug}_{horizon_months}m.csv",
    mime="text/csv",
)

# ==============================================================================
# 9. MULTI-DISEASE WATCH MATRIX (EXPANDER)
# ==============================================================================
with st.expander("🌐 National Multi-Disease Attention Matrix (10 Plant Pathogens)", expanded=False):
    st.markdown("Cross-pathogen surveillance summary ranking predicted activity against historical seasonal baselines:")
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

# ==============================================================================
# 10. BOTTOM ACTION CTA WITH HILL CONTOURS
# ==============================================================================
cta_markup = f"""
<div class="bottom-cta-wrap" style="position: relative; overflow: hidden; background: #FFFFFF; border: 1px solid rgba(82, 122, 80, 0.18); border-radius: 16px; padding: 24px; margin-top: 36px; display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;">
    <div style="position: absolute; right: 0; bottom: 0; height: 100%; width: 40%; pointer-events: none; opacity: 0.45; background-image: url('{obj_hills_b64}'); background-size: cover; background-position: right bottom; background-repeat: no-repeat;"></div>
    <div style="position: relative; z-index: 2; display: flex; gap: 16px; flex-wrap: wrap; width: 100%;">
        <div style="flex: 1; min-width: 200px;">
            <a href="/Model_Comparison" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px; background: #FFFFFF; border: 1.5px solid #2D6A4F; color: #2D6A4F; font-weight: 600; font-size: 0.92rem; padding: 12px 20px; border-radius: 10px;">
                    {icon_model_compare(16, "#2D6A4F")} Return to Model Comparison
                </div>
            </a>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <a href="/Data_and_Methodology" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px; background: #1B4332; color: #FFFFFF; font-weight: 600; font-size: 0.92rem; padding: 12px 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(27, 67, 50, 0.18);">
                    {icon_leaf(16, "#FFFFFF")} Review Scientific Methodology &rarr;
                </div>
            </a>
        </div>
    </div>
</div>
"""
st.html(cta_markup)
