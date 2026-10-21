"""Page 5: Weather Insights — Climate Associations, Rainfall Alignment & Scenario Simulation."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.eda import compute_weather_disease_correlations
from src.forecasting import forecast_sarimax, generate_future_weather_scenario
from src.icons import (
    icon_info,
    icon_leaf,
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
    load_weather_series,
    render_app_sidebar,
    render_top_header,
    render_what_does_this_mean,
)

st.set_page_config(
    page_title="Weather Insights — Verdant Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="weather")
render_top_header()

adv_mode = is_advanced_mode()
pathogen_dict = load_friendly_pathogen_options()

# Editorial Header
page_heading = "Agro-Meteorological Intelligence" if adv_mode else "Weather Insights & Climate Patterns"
page_subheading = (
    "NASA POWER monthly climate observations across Indian agro-zones, empirical correlations, and scenario simulations."
    if adv_mode
    else "Understand how temperature, rainfall, and humidity influence plant disease patterns across India."
)

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_weather(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">{page_heading}</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                {page_subheading}
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
    banner_badge = "SATELLITE & CANOPY BOUNDARY LAYER" if adv_mode else "SATELLITE WEATHER DATA"
    banner_label = "Meteorological Driving Forces" if adv_mode else "How Weather Shapes Disease"
    banner_title = "Atmospheric Moisture & Disease Dynamics" if adv_mode else "Rainfall, Moisture & Disease Risk"
    banner_desc = (
        "Fungal sporulation and bacterial exudates require sustained canopy leaf wetness (>10 hrs) and permissive temperatures. "
        "NASA POWER variables (T2M, PRECTOTCORR, RH2M) provide strictly lagged econometric regressors (t-1, t-2) to capture seasonal outbreak windows."
        if adv_mode
        else "Warm temperatures and high humidity create ideal conditions for fungal spores to germinate and spread. "
        "By comparing monthly weather records from NASA satellites with past disease reports, we can see how rainy seasons influence disease timing."
    )
    st.markdown(
        f"""
        <div class="verdant-card" style="padding: 0; overflow: hidden; margin-bottom: 24px; display: grid; grid-template-columns: 1.2fr 1fr; border: 1px solid rgba(82,122,80,0.2);">
            <div style="height: 200px; overflow: hidden; position: relative;">
                <img src="{monsoon_b64}" style="width: 100%; height: 100%; object-fit: cover;" alt="Monsoon Weather Canopy" />
                <div style="position: absolute; bottom: 10px; left: 12px; background: rgba(16, 44, 38, 0.85); backdrop-filter: blur(8px); color: #FFF; font-size: 0.68rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">
                    {banner_badge}
                </div>
            </div>
            <div style="padding: 18px 24px; display: flex; flex-direction: column; justify-content: center; background: #FFFEFA;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.06em; margin-bottom: 4px;">
                    {banner_label}
                </div>
                <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #102C26;">{banner_title}</h3>
                <p style="font-size: 0.84rem; color: #5A665D; line-height: 1.5; margin: 0;">
                    {banner_desc}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Region options
region_options = {
    "national_composite": "National Average (All India)" if not adv_mode else "National Composite (Centroid)",
    "odisha_bengal": "Eastern Coastal Rice Belt (Odisha / WB)",
    "punjab_haryana": "Northwest Wheat Belt (Punjab / Haryana)",
    "tamil_nadu": "Southern Agricultural Region (Tamil Nadu)",
    "maharashtra": "Western / Central Region (Maharashtra)",
}

# Top query toolbar
bar_c1, bar_c2 = st.columns([1.2, 1.2])
with bar_c1:
    selected_region = st.selectbox(
        "Select Region" if not adv_mode else "Select Agro-Climatic Zone",
        options=list(region_options.keys()),
        format_func=lambda k: region_options[k],
        index=0,
    )
with bar_c2:
    selected_pathogen = st.selectbox(
        "Select Plant Disease" if not adv_mode else "Associate with Pathogen Series",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
    )

df_weather = load_weather_series(selected_region)
df_monthly = load_monthly_series(selected_pathogen)

if df_weather is None or df_weather.empty:
    st.error(f"Weather data not found for {region_options[selected_region]}.")
    st.stop()

# 4 Premium KPI Cards
kpi1_label = "Average Temperature" if not adv_mode else "Mean Annual Temp"
kpi1_sub = "Year-round average (NASA)" if not adv_mode else "NASA POWER 2m Air Temp"

kpi2_label = "Peak Summer Temperature" if not adv_mode else "Peak Summer Max"
kpi2_sub = "Hottest recorded month" if not adv_mode else "Historical Extreme"

kpi3_label = "Average Monthly Rainfall" if not adv_mode else "Monthly Mean Rainfall"
kpi3_sub = "Monsoon and seasonal rain" if not adv_mode else "Monsoon & Post-Monsoon"

kpi4_label = "Average Humidity" if not adv_mode else "Relative Humidity"
kpi4_sub = f"Dew point: {df_weather['dew_point_c'].mean():.1f} °C"

st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 14px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_thermometer(20, "#527A50")}</div>
            <div class="metric-label-text">{kpi1_label}</div>
            <div class="metric-value-text">{df_weather['temp_mean_c'].mean():.1f} °C</div>
            <div class="metric-note-text">{kpi1_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_thermometer(20, "#DC2626")}</div>
            <div class="metric-label-text">{kpi2_label}</div>
            <div class="metric-value-text">{df_weather['temp_max_c'].max():.1f} °C</div>
            <div class="metric-note-text">{kpi2_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_weather(20, "#527A50")}</div>
            <div class="metric-label-text">{kpi3_label}</div>
            <div class="metric-value-text">{df_weather['precipitation_monthly_mm'].mean():.1f} mm</div>
            <div class="metric-note-text">{kpi3_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_leaf(20, "#527A50")}</div>
            <div class="metric-label-text">{kpi4_label}</div>
            <div class="metric-value-text">{df_weather['relative_humidity_pct'].mean():.1f} %</div>
            <div class="metric-note-text">{kpi4_sub}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Meteorological Visualizations
chart_title_prefix = "Regional Climate Trends" if not adv_mode else "Regional Climate Trajectories"
st.markdown(f"<div style='font-size: 1.25rem; font-family: Fraunces, serif; color: #102C26; margin-top: 22px; margin-bottom: 6px;'>📈 {chart_title_prefix}: {region_options[selected_region]} (2015–2025)</div>", unsafe_allow_html=True)

# Temperature plot
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["temp_max_c"], name="Max Temperature (°C)", line=dict(color="#C24134", width=1.5)))
fig_temp.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["temp_mean_c"], name="Average Temperature (°C)", line=dict(color="#C4A66A", width=2.5)))
fig_temp.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["temp_min_c"], name="Min Temperature (°C)", line=dict(color="#2563EB", width=1.5)))
format_plotly_figure(fig_temp, title="Monthly Temperatures (Average, Maximum, Minimum)", height=340)
fig_temp.update_layout(yaxis_title="Temperature (°C)", xaxis_title="Month")
st.plotly_chart(fig_temp, use_container_width=True)

# Precipitation and Humidity plot
fig_precip = go.Figure()
fig_precip.add_trace(go.Bar(x=df_weather["date"], y=df_weather["precipitation_monthly_mm"], name="Monthly Rainfall (mm)", marker_color="#527A50", opacity=0.8))
fig_precip.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["relative_humidity_pct"], name="Relative Humidity (%)", yaxis="y2", line=dict(color="#102C26", width=2)))
fig_precip.update_layout(
    yaxis2=dict(title="Relative Humidity (%)", overlaying="y", side="right", range=[0, 100]),
    yaxis=dict(title="Rainfall (mm)"),
    xaxis=dict(title="Month"),
)
format_plotly_figure(fig_precip, title="Monthly Rainfall & Humidity Levels", height=340)
st.plotly_chart(fig_precip, use_container_width=True)

# What Does This Mean card for weather charts
render_what_does_this_mean(
    simple_text="This section explores how temperature and rainfall patterns are related to recorded disease activity. Rainy, humid seasons often precede spikes in fungal disease.",
    technical_details="NASA POWER meteorological observations capture seasonal variation in temperature, precipitation, and moisture that define biological incubation windows.",
)

# Dual-Axis Disease & Climate Alignment
if df_monthly is not None and not df_monthly.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    align_heading = "🌾 Comparing Disease Reports with Monsoon Rainfall" if not adv_mode else f"🌾 Dual-Axis Temporal Alignment: {pathogen_dict[selected_pathogen]} vs. Monsoon Rainfall"
    st.markdown(f"<div style='font-size: 1.25rem; font-family: Fraunces, serif; color: #102C26; margin-bottom: 6px;'>{align_heading}</div>", unsafe_allow_html=True)
    
    align_desc = (
        "Notice how disease reports often rise 1 to 2 months after heavy rainfall. "
        "Rain and high humidity help fungal spores multiply and spread, leading to higher disease sightings a few weeks later."
        if not adv_mode
        else "Demonstrates how historical documented pathogen discoveries align temporally with monsoon rain spikes. "
        "Peak occurrences typically follow active rainfall peaks by 1 to 2 months (infection incubation lag)."
    )
    st.markdown(align_desc)

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
                name="Monthly Rainfall (mm)",
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
                name=f"Recorded Reports ({pathogen_dict[selected_pathogen]})",
                yaxis="y",
                mode="lines+markers",
                line=dict(color="#102C26", width=2.75),
                marker=dict(size=6, color="#102C26"),
                hovertemplate="<b>Recorded Reports:</b> %{y:.1f}<extra></extra>",
            )
        )

        fig_dual.update_layout(
            yaxis=dict(title="Recorded Disease Reports", side="left", showgrid=True),
            yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right", showgrid=False),
            xaxis=dict(title="Month"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        format_plotly_figure(fig_dual, title=f"Disease Reports vs Rainfall Over Time: {pathogen_dict[selected_pathogen]}", height=400)
        st.plotly_chart(fig_dual, use_container_width=True)

        render_what_does_this_mean(
            simple_text="This chart shows that disease reports often increase 1 to 2 months after heavy rainfall, because rain and humidity create moist conditions where spores germinate and spread.",
            technical_details="A dual-axis view comparing continuous biological incidence against precipitation pulses to inspect seasonal phase shifts and biological incubation lag.",
        )

# Cross Correlation
st.markdown("<br>", unsafe_allow_html=True)
corr_heading = f"🔬 How Weather Factors Relate to {pathogen_dict[selected_pathogen]}" if not adv_mode else f"🔬 Empirical Cross-Correlation: Climate Factors vs {pathogen_dict[selected_pathogen]}"
st.markdown(f"<div style='font-size: 1.25rem; font-family: Fraunces, serif; color: #102C26; margin-bottom: 6px;'>{corr_heading}</div>", unsafe_allow_html=True)

if df_monthly is not None and not df_monthly.empty:
    corrs = compute_weather_disease_correlations(df_monthly, target_col="documented_occurrence_count", max_lag=3)
    if corrs.get("correlations"):
        df_c = pd.DataFrame(corrs["correlations"])
        
        # Friendly labels in Simple View
        if not adv_mode:
            df_c["label"] = df_c["label"].replace({
                "temp_mean_c": "Temperature",
                "precipitation_monthly_mm": "Rainfall",
                "relative_humidity_pct": "Humidity",
                "dew_point_c": "Dew Point",
            })
            lag_names = {0: "Same Month", 1: "1 Month Earlier", 2: "2 Months Earlier", 3: "3 Months Earlier"}
            df_c["lag_label"] = df_c["lag_months"].map(lag_names)
            
            fig_c = px.bar(
                df_c,
                x="label",
                y="pearson_r",
                color="lag_label",
                barmode="group",
                color_discrete_sequence=["#183D32", "#527A50", "#C4A66A", "#2563EB"],
                labels={"pearson_r": "Relationship Strength", "label": "Weather Factor", "lag_label": "Timing"},
                title=f"Weather Relationship Strength Across Different Months ({pathogen_dict[selected_pathogen]})",
            )
        else:
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

    warn_text = (
        "⚠️ **Important Note:** Weather creates favorable conditions for disease to spread, but weather alone doesn't guarantee a disease outbreak. "
        "Crop resistance, farm practices, and timely field care all play major roles."
        if not adv_mode
        else "⚠️ **Causality Boundary:** Empirical correlation does NOT demonstrate direct biological causation. "
        "Elevated rainfall and humidity condition macro-environments for fungal sporulation, but physical infection "
        "requires viable spore density, canopy wetness, and host crop tissue susceptibility."
    )
    st.warning(warn_text)

# Interactive Scenario Simulation
st.markdown("<br>", unsafe_allow_html=True)
scen_title = "🔮 What-If Weather Scenarios" if not adv_mode else "🔮 Interactive Climate Scenario Simulation (SARIMAX Conditioning)"
st.markdown(f"<div style='font-size: 1.25rem; font-family: Fraunces, serif; color: #102C26; margin-bottom: 6px;'>{scen_title}</div>", unsafe_allow_html=True)
st.markdown(
    "See how future disease predictions might change under different weather conditions (like heavier rainfall or warmer temperatures)."
    if not adv_mode
    else "Simulate projected occurrence tendencies under hypothetical micro-climatic shifts."
)

sc_col1, sc_col2, sc_col3 = st.columns(3)
with sc_col1:
    scenario_choice = st.selectbox(
        "Weather Scenario" if not adv_mode else "Pre-Configured Scenario",
        options=[
            "Normal Seasonal Weather",
            "Heavier Monsoon (+25% Rain)",
            "Hot & Dry (+2.5°C, -30% Rain)",
            "Custom Weather Settings",
        ] if not adv_mode else [
            "Baseline Climatology",
            "Elevated Monsoon (+25% Rain)",
            "Heatwave & Drought (+2.5°C, -30% Rain)",
            "Custom Scenario",
        ],
        index=0,
    )
with sc_col2:
    temp_shift = st.slider("Temperature Change (°C)", min_value=-3.0, max_value=5.0, value=0.0, step=0.5)
with sc_col3:
    precip_mult = st.slider("Rainfall Multiplier (x)", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

# Map options
if scenario_choice in ["Heavier Monsoon (+25% Rain)", "Elevated Monsoon (+25% Rain)"]:
    temp_shift = 0.5
    precip_mult = 1.25
elif scenario_choice in ["Hot & Dry (+2.5°C, -30% Rain)", "Heatwave & Drought (+2.5°C, -30% Rain)"]:
    temp_shift = 2.5
    precip_mult = 0.70

if df_monthly is not None:
    horizon = 6
    df_fut_weather = generate_future_weather_scenario(
        df_monthly,
        horizon_months=horizon,
        scenario_type="Baseline" if "Normal" in scenario_choice or "Baseline" in scenario_choice else scenario_choice,
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
    fig_scen.add_trace(go.Scatter(x=df_fut_weather["date"], y=p_base, name="Normal Seasonal Prediction", line=dict(color="#5A665D", width=2, dash="dot")))
    fig_scen.add_trace(go.Scatter(x=df_fut_weather["date"], y=p_scen, name=f"Scenario: {scenario_choice}", mode="lines+markers", line=dict(color="#102C26", width=3)))
    format_plotly_figure(fig_scen, title=f"What-If Disease Prediction: {scenario_choice}", height=380)
    fig_scen.update_layout(yaxis_title="Predicted Disease Reports", xaxis_title="Month")
    st.plotly_chart(fig_scen, use_container_width=True)

    info_desc = (
        f"**Scenario Settings:** Temperature Change = <code>{temp_shift:+.1f} °C</code> | Rainfall Multiplier = <code>{precip_mult:.2f}x</code>.  \n"
        "This what-if simulation adjusts the model's weather inputs to show how disease predictions might rise or fall under changing climate conditions."
        if not adv_mode
        else f"**Scenario Telemetry:** ΔT = <code>{temp_shift:+.1f} °C</code> | Rainfall Multiplier = <code>{precip_mult:.2f}x</code>.  \n"
        "SARIMAX utilizes strictly lagged exogenous regressors to condition forecasts on environmental suitability."
    )
    st.info(info_desc)
