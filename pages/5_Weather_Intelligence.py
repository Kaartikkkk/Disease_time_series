"""Page 5: Weather Intelligence, Micro-Climate Associations & Scenario Simulation."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.eda import compute_weather_disease_correlations
from src.forecasting import forecast_sarimax, generate_future_weather_scenario
from src.icons import (
    icon_leaf,
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
    load_weather_series,
    render_app_sidebar,
    render_top_header,
)

st.set_page_config(
    page_title="Weather Intelligence — Verdant Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="weather")
render_top_header()

# Editorial Header
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_weather(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">Agro-Meteorological Intelligence</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                NASA POWER monthly climate observations across Indian agro-zones, empirical correlations, and scenario simulations.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Visual atmospheric banner
monsoon_img_path = ASSETS_DIR / "monsoon_weather_canopy.jpg"
monsoon_b64 = get_image_base64(monsoon_img_path)
if monsoon_b64:
    st.markdown(
        f"""
        <div class="verdant-card" style="padding: 0; overflow: hidden; margin-bottom: 24px; display: grid; grid-template-columns: 1.2fr 1fr; border: 1px solid rgba(82,122,80,0.2);">
            <div style="height: 200px; overflow: hidden; position: relative;">
                <img src="{monsoon_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Monsoon Precipitation Front" />
                <div style="position: absolute; bottom: 10px; left: 12px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    SATELLITE & CANOPY BOUNDARY LAYER
                </div>
            </div>
            <div style="padding: 18px 24px; display: flex; flex-direction: column; justify-content: center; background: #FFFEFA;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.06em; margin-bottom: 4px;">
                    Meteorological Driving Forces
                </div>
                <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #102C26;">Atmospheric Moisture & Disease Dynamics</h3>
                <p style="font-size: 0.84rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    Fungal sporulation and bacterial exudates require sustained canopy leaf wetness (&gt;10 hrs) and permissive temperatures. 
                    NASA POWER variables (T2M, PRECTOTCORR, RH2M) provide strictly lagged econometric regressors (t-1, t-2) to capture seasonal outbreak windows.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Region options
region_options = {
    "national_composite": "National Composite (Centroid)",
    "odisha_bengal": "Eastern Coastal Rice Belt (Odisha / Bengal)",
    "punjab_haryana": "Northwest Indo-Gangetic Plains (Punjab / Haryana)",
    "tamil_nadu": "Southern Agricultural Zone (Tamil Nadu / Deccan)",
    "maharashtra": "Western / Central Semi-Arid Zone (Maharashtra)",
}

pathogen_dict = load_all_pathogen_options()

# Top query toolbar
bar_c1, bar_c2 = st.columns([1.2, 1.2])
with bar_c1:
    selected_region = st.selectbox(
        "Select Agro-Climatic Zone",
        options=list(region_options.keys()),
        format_func=lambda k: region_options[k],
        index=0,
    )
with bar_c2:
    selected_pathogen = st.selectbox(
        "Associate with Pathogen Series",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
    )

df_weather = load_weather_series(selected_region)
df_monthly = load_monthly_series(selected_pathogen)

if df_weather is None or df_weather.empty:
    st.error(f"Weather series not found for {selected_region}.")
    st.stop()

# 4 Premium KPI Cards
st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 14px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_thermometer(20, "#527A50")}</div>
            <div class="metric-label-text">Mean Annual Temp</div>
            <div class="metric-value-text">{df_weather['temp_mean_c'].mean():.1f} °C</div>
            <div class="metric-note-text">NASA POWER 2m Air Temp</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_thermometer(20, "#DC2626")}</div>
            <div class="metric-label-text">Peak Summer Max</div>
            <div class="metric-value-text">{df_weather['temp_max_c'].max():.1f} °C</div>
            <div class="metric-note-text">Historical Extreme</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_weather(20, "#527A50")}</div>
            <div class="metric-label-text">Monthly Mean Rainfall</div>
            <div class="metric-value-text">{df_weather['precipitation_monthly_mm'].mean():.1f} mm</div>
            <div class="metric-note-text">Monsoon & Post-Monsoon</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
            <div class="metric-label-text">Relative Humidity</div>
            <div class="metric-value-text">{df_weather['relative_humidity_pct'].mean():.1f} %</div>
            <div class="metric-note-text">Dew point: {df_weather['dew_point_c'].mean():.1f} °C</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Meteorological Visualizations
st.subheader(f"📈 Regional Climate Trajectories: {region_options[selected_region]} (2015–2025)")

# Temperature plot
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["temp_max_c"], name="Max Temperature (°C)", line=dict(color="#C24134", width=1.5)))
fig_temp.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["temp_mean_c"], name="Mean Temperature (°C)", line=dict(color="#C4A66A", width=2.5)))
fig_temp.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["temp_min_c"], name="Min Temperature (°C)", line=dict(color="#2563EB", width=1.5)))
format_plotly_figure(fig_temp, title="Monthly Temperature Regimes (Mean, Max, Min)", height=340)
st.plotly_chart(fig_temp, use_container_width=True)

# Precipitation and Humidity plot
fig_precip = go.Figure()
fig_precip.add_trace(go.Bar(x=df_weather["date"], y=df_weather["precipitation_monthly_mm"], name="Monthly Rainfall (mm)", marker_color="#527A50", opacity=0.8))
fig_precip.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["relative_humidity_pct"], name="Relative Humidity (%)", yaxis="y2", line=dict(color="#102C26", width=2)))
fig_precip.update_layout(
    yaxis2=dict(title="Relative Humidity (%)", overlaying="y", side="right", range=[0, 100]),
    yaxis=dict(title="Precipitation (mm)"),
)
format_plotly_figure(fig_precip, title="Monthly Precipitation & Relative Humidity Dynamics", height=340)
st.plotly_chart(fig_precip, use_container_width=True)

# Dual-Axis Disease & Climate Alignment
if df_monthly is not None and not df_monthly.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"🌾 Dual-Axis Temporal Alignment: {pathogen_dict[selected_pathogen]} vs. Monsoon Rainfall")
    st.markdown(
        "Demonstrates how historical documented pathogen discoveries align temporally with monsoon rain spikes. "
        "Notice how peak occurrences typically follow active rainfall peaks by 1 to 2 months (infection incubation lag)."
    )

    # Merge by date
    df_merged = pd.merge(
        df_monthly[["date", "documented_occurrence_count"]],
        df_weather[["date", "precipitation_monthly_mm", "relative_humidity_pct", "temp_mean_c"]],
        on="date",
        how="inner",
    ).sort_values("date")

    if not df_merged.empty:
        fig_dual = go.Figure()

        # Monthly Rainfall on secondary axis
        fig_dual.add_trace(
            go.Bar(
                x=df_merged["date"],
                y=df_merged["precipitation_monthly_mm"],
                name="Monthly Precipitation (mm)",
                yaxis="y2",
                marker_color="rgba(169, 190, 160, 0.45)",
                hovertemplate="<b>Rainfall:</b> %{y:.1f} mm<extra></extra>",
            )
        )

        # Disease Occurrences on primary axis
        fig_dual.add_trace(
            go.Scatter(
                x=df_merged["date"],
                y=df_merged["documented_occurrence_count"],
                name=f"Documented Occurrences ({pathogen_dict[selected_pathogen].split('(')[0].strip()})",
                yaxis="y",
                mode="lines+markers",
                line=dict(color="#102C26", width=2.75),
                marker=dict(size=6, color="#102C26"),
                hovertemplate="<b>Occurrences:</b> %{y:.1f}<extra></extra>",
            )
        )

        fig_dual.update_layout(
            yaxis=dict(title="Documented Occurrence Count", side="left", showgrid=True),
            yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        format_plotly_figure(fig_dual, title="Co-Occurring Trajectories: Documented Pathogen Count vs Precipitation (2015–2025)", height=400)
        st.plotly_chart(fig_dual, use_container_width=True)

# Cross Correlation
st.markdown("<br>", unsafe_allow_html=True)
st.subheader(f"🔬 Empirical Cross-Correlation: Climate Factors vs {pathogen_dict[selected_pathogen]}")

if df_monthly is not None and not df_monthly.empty:
    corrs = compute_weather_disease_correlations(df_monthly, target_col="documented_occurrence_count", max_lag=3)
    if corrs.get("correlations"):
        df_c = pd.DataFrame(corrs["correlations"])
        fig_c = px.bar(
            df_c,
            x="label",
            y="pearson_r",
            color="lag_months",
            barmode="group",
            color_discrete_sequence=["#183D32", "#527A50", "#C4A66A", "#2563EB"],
            labels={"pearson_r": "Pearson r", "label": "Climate Regressor", "lag_months": "Lag (Months)"},
            title="Pearson Correlation Across Distributed Lags",
        )
        format_plotly_figure(fig_c, height=360)
        st.plotly_chart(fig_c, use_container_width=True)

    st.warning(
        "⚠️ **Causality Boundary:** Empirical correlation does NOT demonstrate direct biological causation. "
        "Elevated rainfall and humidity condition macro-environments for fungal sporulation, but physical infection "
        "requires viable spore density, canopy wetness, and host crop tissue susceptibility."
    )

# Interactive Scenario Simulation
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🔮 Interactive Climate Scenario Simulation (SARIMAX Conditioning)")
st.markdown("Simulate projected occurrence tendencies under hypothetical micro-climatic shifts.")

sc_col1, sc_col2, sc_col3 = st.columns(3)
with sc_col1:
    scenario_choice = st.selectbox(
        "Pre-Configured Scenario",
        options=["Baseline Climatology", "Elevated Monsoon (+25% Rain)", "Heatwave & Drought (+2.5°C, -30% Rain)", "Custom Scenario"],
        index=0,
    )
with sc_col2:
    temp_shift = st.slider("Temperature Shift (ΔT °C)", min_value=-3.0, max_value=5.0, value=0.0, step=0.5)
with sc_col3:
    precip_mult = st.slider("Rainfall Multiplier (x)", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

if scenario_choice == "Elevated Monsoon (+25% Rain)":
    temp_shift = 0.5
    precip_mult = 1.25
elif scenario_choice == "Heatwave & Drought (+2.5°C, -30% Rain)":
    temp_shift = 2.5
    precip_mult = 0.70

if df_monthly is not None:
    horizon = 6
    df_fut_weather = generate_future_weather_scenario(
        df_monthly,
        horizon_months=horizon,
        scenario_type=scenario_choice,
        temp_shift_c=temp_shift,
        precip_factor=precip_mult,
    )

    weather_cols = ["temp_mean_c_lag1", "precipitation_monthly_mm_lag1", "relative_humidity_pct_lag1"]
    y_target = df_monthly["documented_occurrence_count"]
    train_exog = df_monthly[weather_cols].bfill().fillna(0)
    fut_exog = df_fut_weather[weather_cols].bfill().fillna(0)

    # SARIMAX forecast under scenario
    p_scen, lb_scen, ub_scen, _ = forecast_sarimax(
        y_target,
        horizon=horizon,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        train_exog=train_exog,
        future_exog=fut_exog,
    )

    # Baseline comparison (no shift)
    df_base_weather = generate_future_weather_scenario(df_monthly, horizon_months=horizon, scenario_type="Baseline")
    fut_base_exog = df_base_weather[weather_cols].bfill().fillna(0)
    p_base, _, _, _ = forecast_sarimax(
        y_target,
        horizon=horizon,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        train_exog=train_exog,
        future_exog=fut_base_exog,
    )

    fig_scen = go.Figure()
    fig_scen.add_trace(go.Scatter(x=df_fut_weather["date"], y=p_base, name="Baseline Climatology", line=dict(color="#5A665D", width=2, dash="dot")))
    fig_scen.add_trace(go.Scatter(x=df_fut_weather["date"], y=p_scen, name=f"Scenario: {scenario_choice}", mode="lines+markers", line=dict(color="#102C26", width=3)))
    format_plotly_figure(fig_scen, title=f"SARIMAX Weather-Informed Forecast: {scenario_choice}", height=380)
    st.plotly_chart(fig_scen, use_container_width=True)

    st.info(
        f"**Scenario Telemetry:** ΔT = <code>{temp_shift:+.1f} °C</code> | Rainfall Multiplier = <code>{precip_mult:.2f}x</code>.  \n"
        "SARIMAX utilizes strictly lagged exogenous regressors to condition forecasts on environmental suitability."
    )
