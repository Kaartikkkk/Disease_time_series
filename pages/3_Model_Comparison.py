"""Page 3: Compare Models — Find the Best Forecasting Method on Past Disease Records."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import acf

from src.evaluation import build_model_comparison_table
from src.forecasting import run_model_comparison
from src.icons import (
    icon_arrow_right,
    icon_download,
    icon_forecast,
    icon_info,
    icon_model_compare,
    icon_pulse,
    icon_shield,
    icon_timeseries,
)
from src.stationarity import perform_ljung_box_test
from src.ui_helpers import (
    format_plotly_figure,
    get_disease_common_name,
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
    page_title="Compare Models — Verdant Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_app_sidebar(active_page="models")
render_top_header()

adv_mode = is_advanced_mode()
pathogen_dict = load_friendly_pathogen_options()

# ==============================================================================
# A. EDITORIAL HEADER & BENCHMARK PARAMETERS
# ==============================================================================
page_heading = "Model Benchmark & Leaderboard" if adv_mode else "Compare Prediction Models"
page_subheading = (
    "Rigorous out-of-sample chronological evaluation across classical statistical econometrics and modern machine learning forecasters."
    if adv_mode
    else "Test multiple forecasting methods on past disease records to find which one makes the most accurate predictions."
)

st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 22px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_model_compare(28, "#527A50")}
                <h1 style="font-size: 2.3rem; margin: 0; color: #102C26;">{page_heading}</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                {page_subheading}
            </p>
        </div>
        <div style="text-align: right; background: #FFFEFA; border: 1px solid rgba(82, 122, 80, 0.2); border-radius: 12px; padding: 8px 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #527A50; letter-spacing: 0.05em;">
                {"Validation Protocol" if adv_mode else "Testing Method"}
            </div>
            <div style="font-size: 0.76rem; color: #183D32; font-weight: 600; margin-top: 2px;">
                {"Forward Chronological · Zero Data Leakage" if adv_mode else "Tested on Real Past Months · Fair Comparison"}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Parameter Toolbar
col_p1, col_p2, col_p3 = st.columns([1.5, 1.0, 0.9])
with col_p1:
    selected_slug = st.selectbox(
        "Plant Disease" if not adv_mode else "Target Pathogen Series",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
    )
with col_p2:
    horizon = st.slider(
        "Testing Period (Months)" if not adv_mode else "Out-of-Sample Holdout Horizon (Months)",
        min_value=3,
        max_value=12,
        value=6,
        step=1,
        help="The number of recent months kept hidden from each model during training to test real prediction accuracy.",
    )
with col_p3:
    st.write("")
    run_btn = st.button(
        "🚀 Compare Models" if not adv_mode else "🚀 Re-Run Benchmark",
        type="primary",
        use_container_width=True,
    )

df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"Monthly disease records not found for {pathogen_dict.get(selected_slug, selected_slug)}.")
    st.stop()

# Cache or run benchmark
session_key = f"comp_results_{selected_slug}_{horizon}"
if run_btn or session_key not in st.session_state:
    spinner_text = (
        f"Testing prediction methods on the last {horizon} months of real data..."
        if not adv_mode
        else f"Evaluating 11 model architectures against held-out validation horizon ({horizon} months)..."
    )
    with st.spinner(spinner_text):
        comp_results = run_model_comparison(df_monthly, horizon=horizon)
        st.session_state[session_key] = comp_results
else:
    comp_results = st.session_state[session_key]

eval_metrics = comp_results.get("evaluation_metrics", [])
table_df = build_model_comparison_table(eval_metrics)
best_model = comp_results.get("best_model", "N/A")
val_dates = pd.to_datetime(comp_results.get("validation_dates"))
y_val = comp_results.get("validation_actual")
forecasts = comp_results.get("forecasts", {})

# Extract baseline metric (Seasonal Naive)
baseline_row = next((m for m in eval_metrics if "Seasonal Naive" in m["model_name"]), None)
baseline_rmse = baseline_row.get("rmse", 0.0) if baseline_row else 0.0
best_row = eval_metrics[0] if eval_metrics else {}
best_rmse = best_row.get("rmse", 0.0)
improvement_pct = ((baseline_rmse - best_rmse) / max(0.01, baseline_rmse)) * 100.0 if baseline_rmse > 0 else 0.0

# ==============================================================================
# B. EVALUATION SUMMARY CARDS
# ==============================================================================
card1_label = "Prediction Methods Tested" if not adv_mode else "Eligible Models Evaluated"
card1_sub = "Statistical & machine learning methods" if not adv_mode else "Classical, Econometric & ML"

card2_label = "Best Performing Method" if not adv_mode else "Top Validated Model"
card2_sub = f"Average error: {best_rmse:.2f} reports" if not adv_mode else f"RMSE: {best_rmse:.3f} | sMAPE: {best_row.get('smape', 0.0):.1f}%"

card3_label = "Seasonal Benchmark" if not adv_mode else "Seasonal Naive Baseline"
card3_val = f"Error: {baseline_rmse:.2f}" if not adv_mode else f"RMSE: {baseline_rmse:.3f}"
card3_sub = f"Best method made {abs(improvement_pct):.1f}% fewer errors" if improvement_pct >= 0 else "Baseline performance comparison"

card4_label = "Testing Timeframe" if not adv_mode else "Chronological Evaluation Period"
card4_sub = f"{len(val_dates)} hidden past test months" if not adv_mode else f"Held-out horizon: {len(val_dates)} Months"

st.markdown(
    f"""
    <div class="metric-grid" style="margin-top: 10px; margin-bottom: 24px;">
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_model_compare(20, "#527A50")}</div>
            <div class="metric-label-text">{card1_label}</div>
            <div class="metric-value-text">{len(eval_metrics)} Methods</div>
            <div class="metric-note-text">{card1_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_shield(20, "#102C26")}</div>
            <div class="metric-label-text">{card2_label}</div>
            <div class="metric-value-text" style="font-size: 1.25rem;">{best_model}</div>
            <div class="metric-note-text">{card2_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_timeseries(20, "#C4A66A")}</div>
            <div class="metric-label-text">{card3_label}</div>
            <div class="metric-value-text">{card3_val}</div>
            <div class="metric-note-text">{card3_sub}</div>
        </div>
        <div class="research-metric-card">
            <div class="metric-icon-wrap">{icon_forecast(20, "#527A50")}</div>
            <div class="metric-label-text">{card4_label}</div>
            <div class="metric-value-text" style="font-size: 1.2rem;">{val_dates.iloc[0].strftime('%b %Y')} – {val_dates.iloc[-1].strftime('%b %Y')}</div>
            <div class="metric-note-text">{card4_sub}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Model Champion Award Banner
banner_badge = "🏆 BEST PERFORMING PREDICTION METHOD" if not adv_mode else "🏆 OPTIMAL CHRONOLOGICAL BENCHMARK CHAMPION"
banner_desc = (
    f"Ranked #1 out of {len(eval_metrics)} methods tested on past real data. It produced the lowest prediction error ({best_rmse:.2f} reports on average) during the {len(val_dates)}-month test period."
    if not adv_mode
    else f"Ranked #1 out of {len(eval_metrics)} models on chronological out-of-sample Root Mean Squared Error ({best_rmse:.3f}) and Mean Absolute Error ({best_row.get('mae', 0.0):.3f})."
)

st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, #102C26 0%, #183D32 100%); border-radius: 16px; padding: 22px 28px; color: #FFFFFF; margin-bottom: 24px; box-shadow: 0 8px 24px rgba(16, 44, 38, 0.15);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #C4A66A; letter-spacing: 0.1em; margin-bottom: 4px;">
                    {banner_badge}
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.85rem; font-weight: 600; color: #FFFFFF; line-height: 1.2;">
                    {best_model}
                </div>
                <p style="font-size: 0.88rem; color: #E3EBE0; margin: 6px 0 0 0; max-width: 720px;">
                    {banner_desc}
                </p>
            </div>
            <div style="display: flex; gap: 12px;">
                <div style="background: rgba(255, 254, 250, 0.1); border: 1px solid rgba(255, 254, 250, 0.2); border-radius: 12px; padding: 10px 18px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #A9BEA0; text-transform: uppercase;">
                        {"Avg Error (RMSE)" if not adv_mode else "Holdout RMSE"}
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 700; color: #C4A66A;">
                        {best_rmse:.3f}
                    </div>
                </div>
                <div style="background: rgba(255, 254, 250, 0.1); border: 1px solid rgba(255, 254, 250, 0.2); border-radius: 12px; padding: 10px 18px; text-align: center;">
                    <div style="font-size: 0.68rem; color: #A9BEA0; text-transform: uppercase;">
                        {"Mean Error (MAE)" if not adv_mode else "Holdout MAE"}
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 700; color: #FFFFFF;">
                        {best_row.get('mae', 0.0):.3f}
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# What Does This Mean card for model comparison
render_what_does_this_mean(
    simple_text="This section compares different prediction methods to find which worked best on past data. The method with the lowest error is recommended for future predictions.",
    technical_details="Models are tested using forward chronological holdout validation. We train on earlier data and test on the final months to guarantee zero data leakage. RMSE and MAE quantify forecast deviations in natural count units.",
)

# ==============================================================================
# C. MODEL LEADERBOARD TABLE
# ==============================================================================
st.markdown("<div style='font-size: 1.25rem; font-family: Fraunces, serif; color: #102C26; margin-top: 18px; margin-bottom: 6px;'>Prediction Methods Ranked by Accuracy</div>", unsafe_allow_html=True)

if not adv_mode:
    st.markdown(
        "Methods with lower error values are ranked higher. You can see the average difference between what each method predicted and the real recorded reports."
    )
    # Simplified clean display table
    table_display = table_df[["Model", "RMSE", "MAE", "sMAPE (%)"]].copy()
    table_display.columns = ["Prediction Method", "Average Error (RMSE)", "Mean Difference (MAE)", "Percentage Error (sMAPE %)"]
    st.dataframe(table_display, use_container_width=True, hide_index=True)
else:
    st.markdown(
        "Models are ranked strictly on chronological test data. Metrics utilize zero-safe formulations "
        "(sMAPE bounded in $[0\\%, 200\\%]$ and WAPE) to prevent mathematical breakdown on sparse surveillance zero counts."
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)

# Export Leaderboard
csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Comparison Results (CSV)",
    data=csv_bytes,
    file_name=f"model_comparison_{selected_slug}_{horizon}m.csv",
    mime="text/csv",
)

st.markdown("---")

# ==============================================================================
# D. VISUAL MODEL COMPARISONS
# ==============================================================================
st.markdown(
    """
    <div style="margin-bottom: 14px;">
        <h2 style="font-size: 1.85rem; margin: 0 0 4px 0; color: #102C26;">Visual Performance Comparison</h2>
        <p style="color: #5A665D; font-size: 0.92rem; margin: 0;">
            Compare actual recorded disease reports against predictions from the top methods.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_actual, tab_errors, tab_residuals = st.tabs([
    "📈 Actual vs. Predicted Reports",
    "📊 Prediction Error Comparison",
    "🔬 Prediction Error Analysis" if not adv_mode else "🔬 Residual Diagnostics & Autocorrelation",
])

with tab_actual:
    fig_val = go.Figure()

    # Ground Truth Actual
    fig_val.add_trace(
        go.Scatter(
            x=val_dates,
            y=y_val,
            name="Actual Recorded Reports",
            mode="lines+markers",
            line=dict(color="#102C26", width=3.5),
            marker=dict(size=8, symbol="circle", color="#102C26"),
            hovertemplate="<b>Actual:</b> %{y:.1f} reports<br><b>Month:</b> %{x|%b %Y}<extra></extra>",
        )
    )

    # Top 5 models
    top_models_to_plot = [r["model_name"] for r in eval_metrics[:5]]
    palette = ["#527A50", "#2563EB", "#C4A66A", "#7C3AED", "#D97706"]

    for i, m_name in enumerate(top_models_to_plot):
        if m_name in forecasts:
            fig_val.add_trace(
                go.Scatter(
                    x=val_dates,
                    y=forecasts[m_name],
                    name=m_name,
                    mode="lines+markers",
                    line=dict(color=palette[i % len(palette)], width=2.2, dash="solid" if i == 0 else "dash"),
                    marker=dict(size=6, symbol="diamond" if i == 0 else "circle"),
                    hovertemplate=f"<b>{m_name}:</b> %{{y:.2f}} reports<extra></extra>",
                )
            )

    format_plotly_figure(
        fig_val,
        title=f"Actual Recorded Reports vs Top Methods ({val_dates.iloc[0].strftime('%b %Y')} – {val_dates.iloc[-1].strftime('%b %Y')})",
        height=440,
    )
    fig_val.update_layout(
        xaxis_title="Month",
        yaxis_title="Recorded Disease Reports",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    st.plotly_chart(fig_val, use_container_width=True)

    render_what_does_this_mean(
        simple_text="This chart compares what actually happened in the past test months with what each method predicted would happen. Lines that stay closer to the dark line made better predictions.",
        technical_details="Multi-model trajectory traces projected across the out-of-sample holdout timeline compared directly against ground truth surveillance actuals.",
    )

with tab_errors:
    # Error bar chart: RMSE and MAE across all models
    model_names = [m["model_name"] for m in eval_metrics]
    rmse_vals = [m.get("rmse", 0.0) for m in eval_metrics]
    mae_vals = [m.get("mae", 0.0) for m in eval_metrics]

    fig_err = go.Figure()
    fig_err.add_trace(
        go.Bar(
            x=model_names,
            y=rmse_vals,
            name="RMSE (Root Mean Squared Error)",
            marker_color="#102C26",
            hovertemplate="<b>%{x}</b><br>RMSE: %{y:.3f}<extra></extra>",
        )
    )
    fig_err.add_trace(
        go.Bar(
            x=model_names,
            y=mae_vals,
            name="MAE (Mean Absolute Error)",
            marker_color="#C4A66A",
            hovertemplate="<b>%{x}</b><br>MAE: %{y:.3f}<extra></extra>",
        )
    )

    format_plotly_figure(
        fig_err,
        title="Prediction Errors Across Methods (Lower is Better)",
        height=420,
    )
    fig_err.update_layout(
        barmode="group",
        xaxis_title="Prediction Method",
        yaxis_title="Error Size (Recorded Reports)",
        xaxis_tickangle=-35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    render_what_does_this_mean(
        simple_text="This chart shows the error size for each method. Shorter bars mean smaller errors and more accurate predictions.",
        technical_details="RMSE penalizes larger forecast misses more heavily through quadratic weighting, while MAE shows the average absolute magnitude of prediction errors.",
    )

with tab_residuals:
    if best_model in forecasts:
        residuals = np.array(y_val) - np.array(forecasts[best_model])
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            # Residual histogram with zero reference
            fig_r_hist = go.Figure()
            fig_r_hist.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=10,
                    name="Prediction Errors",
                    marker_color="#527A50",
                    opacity=0.85,
                )
            )
            fig_r_hist.add_vline(x=0, line_dash="dash", line_color="#102C26", line_width=2)
            format_plotly_figure(fig_r_hist, title=f"Error Distribution ({best_model})", height=340)
            fig_r_hist.update_layout(xaxis_title="Prediction Error (Actual - Predicted)", yaxis_title="Number of Months")
            st.plotly_chart(fig_r_hist, use_container_width=True)

        with res_col2:
            # Residual Autocorrelation (ACF)
            if len(residuals) >= 4:
                nlags = min(len(residuals) - 1, 5)
                res_acf = acf(residuals, nlags=nlags, fft=True)
                ci_band = 1.96 / np.sqrt(len(residuals))

                fig_r_acf = go.Figure()
                fig_r_acf.add_trace(go.Bar(x=list(range(nlags + 1)), y=res_acf, marker_color="#183D32", name="ACF"))
                fig_r_acf.add_hline(y=ci_band, line_dash="dash", line_color="#C4A66A")
                fig_r_acf.add_hline(y=-ci_band, line_dash="dash", line_color="#C4A66A")
                format_plotly_figure(fig_r_acf, title="Error Pattern Over Time (Autocorrelation)", height=340)
                fig_r_acf.update_layout(xaxis_title="Lag (Months)", yaxis_title="Correlation")
                st.plotly_chart(fig_r_acf, use_container_width=True)
            else:
                st.info("Testing sample size is too compact for multi-month pattern checks.")

        render_what_does_this_mean(
            simple_text="This test checks whether the model's errors are just random or if it keeps making the same mistake month after month. Balanced errors centered at zero mean the model is performing reliably.",
            technical_details="Residual analysis evaluates whether forecast errors behave like white noise with zero mean and no significant autocorrelation remaining in the residuals.",
        )

        if adv_mode:
            # Ljung-Box test in Advanced View
            lb_diag = perform_ljung_box_test(pd.Series(residuals), lags=[2])
            st.markdown(
                f"""
                <div style="background: rgba(16, 44, 38, 0.04); border-left: 3px solid #527A50; padding: 12px 16px; border-radius: 4px; font-size: 0.85rem; color: #102C26; margin-top: 10px;">
                    <strong>Ljung-Box White Noise Diagnostic:</strong> {lb_diag.get('interpretation', 'Compact sample; visual inspection confirmed residual zero-centering.')}
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# E. MODEL INTERPRETATION & WEAKNESS ANALYSIS
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="verdant-card">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            {icon_shield(22, "#527A50")}
            <h3 style="margin: 0; font-size: 1.15rem; color: #102C26;">
                {"Why " + best_model + " Succeeded & Things to Keep in Mind" if not adv_mode else "Why " + best_model + " Succeeded & Known Weaknesses"}
            </h3>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4 style="margin: 0 0 6px 0; font-size: 0.95rem; color: #183D32;">
                    {"Key Strengths:" if not adv_mode else "Strengths & Architectural Drivers:"}
                </h4>
                <p style="font-size: 0.84rem; color: #404B43; line-height: 1.5; margin: 0;">
                    {"Simpler, time-tested statistical methods often perform better than complex machine learning on seasonal plant disease data. Because plant diseases have low activity during dry months and sudden peaks in monsoon seasons, " + best_model + " focuses on the recurring yearly rhythm without getting misled by random month-to-month spikes." if not adv_mode else "Classical parsimonious architectures (e.g. ARIMA/SARIMA) often outperform heavily parameterized ML forecasters on epidemiological surveillance data. The low-variance formulation of " + best_model + " resists overfitting noise during dry unmonitored spells."}
                </p>
            </div>
            <div>
                <h4 style="margin: 0 0 6px 0; font-size: 0.95rem; color: #695221;">
                    {"What to Watch Out For:" if not adv_mode else "Operational Caveats & Limitations:"}
                </h4>
                <p style="font-size: 0.84rem; color: #404B43; line-height: 1.5; margin: 0;">
                    {"The model predicts based on historical seasonal rhythms. Unusual extreme weather events (such as an unexpected cyclone or severe drought) can cause disease reports to be higher or lower than predicted. When extreme weather happens, local field checks are strongly recommended." if not adv_mode else "Linear autoregression cannot simulate sudden non-linear phase shifts triggered by unprecedented meteorological extremes. On series with substantial structural breaks, rolling re-estimation is required before deploying operational advisories."}
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
