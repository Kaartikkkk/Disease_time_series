"""Page 4: Forward Forecasting & Prediction Uncertainty Intervals."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    icon_download,
    icon_forecast,
    icon_leaf,
    icon_shield,
    icon_timeseries,
)
from src.ui_helpers import (
    ASSETS_DIR,
    format_plotly_figure,
    get_image_base64,
    inject_custom_css,
    load_all_pathogen_options,
    load_monthly_series,
)

st.set_page_config(
    page_title="Forecasts — Verdant Intelligence",
    page_icon="🔮",
    layout="wide",
)

inject_custom_css()

# Editorial Header
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_forecast(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">Out-of-Sample Forward Projections</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Generate forward forecasts of monthly documented occurrences with 80% and 95% uncertainty intervals.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pathogen_dict = load_all_pathogen_options()

# Top query toolbar
bar_c1, bar_c2, bar_c3 = st.columns([1.3, 0.9, 1.2])

with bar_c1:
    selected_slug = st.selectbox(
        "Target Pathogen / Disease",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
    )

with bar_c2:
    horizon_months = st.slider("Forecast Horizon (Months)", min_value=3, max_value=12, value=6, step=1)

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

with bar_c3:
    chosen_model = st.selectbox("Forecasting Model Architecture", options=available_models, index=0)

df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"Monthly series not found for {selected_slug}.")
    st.stop()

# Historical series dates
target_col = "documented_occurrence_count"
last_obs_date = df_monthly["date"].max()
future_dates = pd.date_range(start=last_obs_date + pd.DateOffset(months=1), periods=horizon_months, freq="MS")

# Generate weather scenario for future months
weather_cols = ["temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"]
has_weather = all(c in df_monthly.columns for c in weather_cols)
df_future_weather = generate_future_weather_scenario(df_monthly, horizon_months=horizon_months) if has_weather else None

# Run Forecast
with st.spinner(f"Computing forward forecast with {chosen_model}..."):
    y_full = df_monthly[target_col]
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

# Baseline Seasonal Naive
p_baseline, _, _ = forecast_seasonal_naive(y_full, horizon=horizon_months, season_length=12)

# Approximate 80% CI
half_width = (ub_95 - p_fut) * (1.28 / 1.96)
lb_80 = np.maximum(0.0, p_fut - half_width)
ub_80 = p_fut + half_width

df_forecast_out = pd.DataFrame({
    "date": future_dates,
    "forecast": np.round(p_fut, 2),
    "lower_80": np.round(lb_80, 2),
    "upper_80": np.round(ub_80, 2),
    "lower_95": np.round(lb_95, 2),
    "upper_95": np.round(ub_95, 2),
    "baseline_seasonal_naive": np.round(p_baseline, 2),
})

# Forecast Chart
st.subheader(f"📈 Forward Forecast Trajectory: {pathogen_dict[selected_slug]} ({future_dates[0].strftime('%b %Y')} – {future_dates[-1].strftime('%b %Y')})")

fig_f = go.Figure()

# Recent Historical Observations (last 24 months)
hist_subset = df_monthly.tail(24)
fig_f.add_trace(
    go.Scatter(
        x=hist_subset["date"],
        y=hist_subset[target_col],
        name="Historical Documented Occurrences",
        mode="lines+markers",
        line=dict(color="#102C26", width=2.5),
        marker=dict(size=6),
    )
)

# 95% Uncertainty Shading
fig_f.add_trace(go.Scatter(x=df_forecast_out["date"], y=df_forecast_out["upper_95"], mode="lines", line=dict(color="rgba(82, 122, 80, 0)", width=0), showlegend=False))
fig_f.add_trace(go.Scatter(x=df_forecast_out["date"], y=df_forecast_out["lower_95"], name="95% Prediction Interval", mode="lines", line=dict(color="rgba(82, 122, 80, 0)", width=0), fill="tonexty", fillcolor="rgba(82, 122, 80, 0.15)"))

# 80% Uncertainty Shading
fig_f.add_trace(go.Scatter(x=df_forecast_out["date"], y=df_forecast_out["upper_80"], mode="lines", line=dict(color="rgba(82, 122, 80, 0)", width=0), showlegend=False))
fig_f.add_trace(go.Scatter(x=df_forecast_out["date"], y=df_forecast_out["lower_80"], name="80% Prediction Interval", mode="lines", line=dict(color="rgba(82, 122, 80, 0)", width=0), fill="tonexty", fillcolor="rgba(82, 122, 80, 0.25)"))

# Forecast Point Line
fig_f.add_trace(
    go.Scatter(
        x=df_forecast_out["date"],
        y=df_forecast_out["forecast"],
        name=f"Point Forecast ({chosen_model})",
        mode="lines+markers",
        line=dict(color="#183D32", width=3, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
    )
)

# Baseline Comparison
fig_f.add_trace(
    go.Scatter(
        x=df_forecast_out["date"],
        y=df_forecast_out["baseline_seasonal_naive"],
        name="Seasonal Naive Baseline",
        mode="lines",
        line=dict(color="#C4A66A", width=2, dash="dot"),
    )
)

format_plotly_figure(
    fig_f,
    title=f"Multi-Step Forward Projections with 80% and 95% Confidence Intervals",
    height=460,
)
st.plotly_chart(fig_f, use_container_width=True)

# Scientific Assumption Notice
st.warning(
    "⚠️ **Methodological Assumption:** Forward forecasts beyond realized weather observations condition on "
    "**Baseline Climatology** (long-term monthly NASA POWER means). Projections represent qualifying **documented "
    "occurrences** deposited in surveillance databases, not an exact acreage count of infected farm plots."
)

# Harvest defense advisory card
golden_img_path = ASSETS_DIR / "golden_crop_field.jpg"
golden_b64 = get_image_base64(golden_img_path)
if golden_b64:
    st.markdown(
        f"""
        <div class="verdant-card" style="padding: 0; overflow: hidden; margin-top: 18px; margin-bottom: 22px; display: grid; grid-template-columns: 240px 1fr; border: 1px solid rgba(82,122,80,0.18);">
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

# Forecast Table
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("📋 Forecast Numerical Schedule")
st.dataframe(df_forecast_out, use_container_width=True, hide_index=True)

# Export
csv_f_bytes = df_forecast_out.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Forward Forecast CSV",
    data=csv_f_bytes,
    file_name=f"forecast_{selected_slug}_{horizon_months}m.csv",
    mime="text/csv",
)
