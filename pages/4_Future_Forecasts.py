"""Page 4: Future Forecasts — Out-of-Sample Surveillance Projections & Monthly Risk Outlook.

Minimal, academic, and interactive interface for forward agricultural disease surveillance.
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
    icon_download,
    icon_forecast,
    icon_info,
    icon_leaf,
    icon_model_compare,
    icon_pulse,
    icon_shield,
    icon_timeseries,
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
    format_plotly_figure,
    inject_custom_css,
    load_all_pathogen_options,
    load_monthly_series,
    render_app_sidebar,
    render_top_header,
)

st.set_page_config(
    page_title="Future Forecasts — Agricultural Disease Forecasting Using Time Series",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_app_sidebar(active_page="predictions")
render_top_header()

stats = get_project_telemetry()
pathogen_dict = load_all_pathogen_options()

# ==============================================================================
# 1. EDITORIAL HEADER & DYNAMIC DATE TELEMETRY
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; margin-bottom: 16px; padding-bottom: 14px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div style="flex: 1 1 300px; min-width: 250px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_forecast(26, "#527A50")}
                <h1 style="font-size: clamp(1.5rem, 3vw, 2.0rem); margin: 0; color: #102C26; word-break: normal; hyphens: none; line-height: 1.2;">Future Disease Forecasts &amp; Surveillance Outlook</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.92rem; margin: 0; line-height: 1.45;">
                Forward projections originating from the latest verified historical observations with 80% and 95% expected ranges.
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 10px; padding: 6px 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.02); flex-shrink: 0; align-self: flex-start;">
            <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                Surveillance Origin
            </div>
            <div style="font-size: 0.75rem; color: #183D32; font-weight: 600; margin-top: 1px;">
                Latest Observed: <strong>{stats['forecast_origin_date']}</strong>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CONTROLS TOOLBAR
# ==============================================================================
col_c1, col_c2, col_c3 = st.columns([1.3, 1.3, 1.0])

with col_c1:
    selected_slug = st.selectbox(
        "Target Crop Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
    )

df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"No monthly records found for {selected_slug}.")
    st.stop()

target_col = "documented_occurrence_count"
last_obs_date = pd.to_datetime(df_monthly["date"].max())
first_obs_date = pd.to_datetime(df_monthly["date"].min())

with col_c2:
    model_choices = [
        "Hurdle XGBoost (Classifier + Regressor)",
        "GRU (Gated Recurrent Unit)",
        "Global Random Forest (Multi-Pathogen)",
        "Global XGBoost (Multi-Pathogen)",
        "Prophet (Trend + Seasonality)",
        "LSTM (Sequence Neural Net)",
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
        help="Select a calibrated model to project future disease occurrence.",
    )

with col_c3:
    horizon_months = st.slider(
        "Forecast Period (Months)",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
        help="Number of future months to project forward from the last observed date.",
    )

future_dates = pd.date_range(start=last_obs_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

# Prominent Dynamic Date Grounding Callout
st.markdown(
    f"""
    <div style="background: rgba(196, 166, 106, 0.12); border: 1px solid rgba(196, 166, 106, 0.35); border-radius: 8px; padding: 10px 16px; margin-bottom: 18px; font-size: 0.85rem; color: #4A3B12; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div>
            <strong>Verified Database Cutoff:</strong> {last_obs_date.strftime('%B %Y')} (End of Historical Surveillance Records) &nbsp;·&nbsp;
            <strong>Forward Forecast Period:</strong> <strong>{future_dates[0].strftime('%B %Y')} – {future_dates[-1].strftime('%B %Y')}</strong> ({horizon_months} Months Ahead)
        </div>
        <div style="font-size: 0.75rem; color: #695221; font-weight: 600;">
            Origin: {last_obs_date.strftime('%b %Y')} Cutoff
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
# 3. FOUR COMPACT KPI METRIC CARDS
# ==============================================================================
latest_obs_val = float(df_monthly[target_col].dropna().iloc[-1])
mean_forecast = float(np.mean(df_forecast_out["forecast"]))
total_forecast = float(np.sum(df_forecast_out["forecast"]))
mean_ci_width = float(np.mean(df_forecast_out["upper_80"] - df_forecast_out["lower_80"]))

st.markdown(
    f"""
    <div class="metric-grid" style="margin-bottom: 20px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(18, "#527A50")}</div>
            <div class="metric-label-text">Last Recorded Count</div>
            <div class="metric-value-text">{latest_obs_val:.1f} Reports</div>
            <div class="metric-note-text">Observed in {last_obs_date.strftime('%b %Y')}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_forecast(18, "#183D32")}</div>
            <div class="metric-label-text">Peak Forecast Month</div>
            <div class="metric-value-text">{peak_info.get('peak_value', 0.0):.1f} Reports</div>
            <div class="metric-note-text">Expected in {peak_info.get('peak_month', 'N/A')}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_model_compare(18, "#C4A66A")}</div>
            <div class="metric-label-text">Mean Monthly Projection</div>
            <div class="metric-value-text">~{mean_forecast:.1f} / mo</div>
            <div class="metric-note-text">Total: {total_forecast:.1f} Across {horizon_months} Mo</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(18, "#527A50")}</div>
            <div class="metric-label-text">Uncertainty Spread</div>
            <div class="metric-value-text">±{mean_ci_width / 2.0:.1f} Reports</div>
            <div class="metric-note-text">80% Expected Range Half-Width</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 4. PRIMARY FORECAST VISUALIZATION (PROMINENT CENTERPIECE)
# ==============================================================================
st.markdown(
    f"""
    <div style="margin-bottom: 10px;">
        <h3 style="font-family: 'Fraunces', serif; font-size: 1.35rem; color: #102C26; margin: 0 0 2px 0;">
            Forward Surveillance Projectory: {pathogen_dict[selected_slug]}
        </h3>
        <p style="color: #5A665D; font-size: 0.85rem; margin: 0;">
            Observed historical reports (pre-cutoff) and future projections ({future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')}) with expected ranges.
        </p>
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
        name="Observed Incidence",
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
        fillcolor="rgba(196, 166, 106, 0.12)",
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
        line=dict(color="#047857", width=2.6, dash="dash"),
        marker=dict(size=6, color="#047857"),
        hovertemplate="<b>Forecast:</b> %{y:.1f}<br><b>Date:</b> %{x|%b %Y}<extra></extra>",
    )
)

# Vertical line separating observation and forecast
fig_f.add_vline(
    x=last_obs_date,
    line_width=1.8,
    line_dash="dot",
    line_color="#C4A66A",
    annotation_text=f"Cutoff ({last_obs_date.strftime('%b %Y')})",
    annotation_position="top left",
)

format_plotly_figure(
    fig_f,
    title=f"Disease Occurrence Forecast ({future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')})",
    height=360,
)
st.plotly_chart(fig_f, use_container_width=True)

# ==============================================================================
# 5. PEAK ACTIVITY CALLOUT & MONTHLY OUTLOOK CARDS
# ==============================================================================
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, rgba(196, 166, 106, 0.12) 0%, rgba(255, 254, 250, 0.95) 100%); border: 1.5px solid #C4A66A; border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #695221; letter-spacing: 0.05em;">
                    PEAK PROJECTED SURVEILLANCE WINDOW
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 600; color: #102C26; margin: 4px 0 2px 0;">
                    {peak_info.get('peak_month', 'N/A')} · ~{peak_info.get('peak_value', 0.0):.1f} Projected Reports
                </div>
                <div style="font-size: 0.82rem; color: #5A665D;">
                    Highest predicted occurrence intensity during the forward {horizon_months}-month forecast period.
                </div>
            </div>
            <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(196, 166, 106, 0.3); border-radius: 8px; padding: 6px 14px;">
                <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #695221;">80% Expected Range</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; color: #102C26;">
                    {peak_info.get('lower_80', 0.0):.1f} – {peak_info.get('upper_80', 0.0):.1f}
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Monthly Outlook Cards Grid
monthly_outlooks = generate_monthly_disease_outlook(
    df_forecast=df_forecast_out,
    df_historical=df_monthly,
    pathogen_name=pathogen_dict[selected_slug],
    target_col=target_col,
    region_name="India",
)

st.markdown("<div style='font-size: 1.15rem; font-family: Fraunces, serif; color: #102C26; margin-bottom: 10px;'>Month-by-Month Risk Outlook</div>", unsafe_allow_html=True)

grid_cols = st.columns(min(3, len(monthly_outlooks)))
for i, outlook in enumerate(monthly_outlooks):
    col = grid_cols[i % len(grid_cols)]
    with col:
        pct_str = f" ({outlook['change_pct']:+.1f}%)" if outlook['change_pct'] is not None else ""
        st.markdown(
            f"""
            <div class="verdant-card" style="margin-bottom: 14px; padding: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-family: 'Fraunces', serif; font-size: 1.05rem; font-weight: 600; color: #102C26;">
                        {outlook['month_label']}
                    </span>
                    <span style="background: {outlook['activity_color']}15; border: 1px solid {outlook['activity_color']}40; color: {outlook['activity_color']}; font-size: 0.68rem; font-weight: 600; padding: 2px 6px; border-radius: 4px;">
                        {outlook['activity_level']}
                    </span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px;">
                    <span style="font-family: 'Fraunces', serif; font-size: 1.6rem; font-weight: 600; color: #102C26;">
                        {outlook['predicted_count']:.1f}
                    </span>
                    <span style="font-size: 0.78rem; font-weight: 600; color: {'#183D32' if outlook['change_abs'] <= 0 else '#C4A66A'};">
                        {outlook['trend_arrow']} {outlook['change_abs']:+.1f}{pct_str}
                    </span>
                </div>
                <div style="font-size: 0.72rem; color: #5A665D;">
                    80% Range: <strong>{outlook['lower_80']:.1f} – {outlook['upper_80']:.1f}</strong> reports
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# 6. SUMMARY TABLE & CSV EXPORT
# ==============================================================================
table_rows = []
for out in monthly_outlooks:
    pct_text = f"{out['change_pct']:+.1f}%" if out['change_pct'] is not None else "N/A"
    table_rows.append({
        "Month": out["month_label"],
        "Predicted Reports": out["predicted_count"],
        "Change from Prior": f"{out['change_abs']:+.1f} ({pct_text})",
        "Trend Direction": f"{out['trend_arrow']} {out['trend_label']}",
        "80% Uncertainty Range": f"{out['lower_80']} – {out['upper_80']}",
        "95% Uncertainty Range": f"{out['lower_95']} – {out['upper_95']}",
        "Surveillance Risk": out["activity_level"],
    })
df_summary_table = pd.DataFrame(table_rows)

st.dataframe(df_summary_table, use_container_width=True, hide_index=True)

csv_summary = df_summary_table.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"Download Projected Forecast Table (CSV - {len(df_summary_table)} Months)",
    data=csv_summary,
    file_name=f"disease_forecast_{selected_slug}_{horizon_months}m.csv",
    mime="text/csv",
)

# ==============================================================================
# 7. MULTI-DISEASE WATCH MATRIX & WEATHER SCENARIOS (EXPANDERS)
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

# Navigation
st.markdown("<br>", unsafe_allow_html=True)
try:
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        st.page_link("pages/3_Model_Comparison.py", label="Inspect Model Accuracy Leaderboard ->", use_container_width=True)
    with c_f2:
        st.page_link("pages/5_Data_and_Methodology.py", label="Review Scientific Methodology ->", use_container_width=True)
except Exception:
    pass
