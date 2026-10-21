"""Page 3: Model Benchmark & Chronological Validation Suite."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluation import build_model_comparison_table
from src.forecasting import run_model_comparison
from src.icons import (
    icon_arrow_right,
    icon_download,
    icon_forecast,
    icon_model_compare,
    icon_shield,
    icon_timeseries,
)
from src.stationarity import perform_ljung_box_test
from src.ui_helpers import (
    format_plotly_figure,
    inject_custom_css,
    load_all_pathogen_options,
    load_monthly_series,
)

st.set_page_config(
    page_title="Model Benchmark — Verdant Intelligence",
    page_icon="⚖️",
    layout="wide",
)

inject_custom_css()

# Editorial Header
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid rgba(82, 122, 80, 0.16);">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                {icon_model_compare(26, "#527A50")}
                <h1 style="font-size: 2.2rem; margin: 0; color: #102C26;">Model Benchmark & Validation</h1>
            </div>
            <p style="color: #5A665D; font-size: 0.95rem; margin: 0;">
                Side-by-side chronological evaluation of 11 model families without lookahead bias.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pathogen_dict = load_all_pathogen_options()

# Top query toolbar
bar_c1, bar_c2, bar_c3 = st.columns([1.5, 1.0, 0.8])
with bar_c1:
    selected_slug = st.selectbox(
        "Select Target Series",
        options=list(pathogen_dict.keys()),
        format_func=lambda k: pathogen_dict[k],
        index=0,
    )
with bar_c2:
    horizon = st.slider("Validation Horizon (Months)", min_value=3, max_value=12, value=6, step=1)
with bar_c3:
    st.write("")
    run_btn = st.button("🚀 Re-Run Benchmark", type="primary", use_container_width=True)

df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"Monthly series not found for {selected_slug}.")
    st.stop()

# Cache or run comparison
session_key = f"comp_results_{selected_slug}_{horizon}"
if run_btn or session_key not in st.session_state:
    with st.spinner("Executing chronological out-of-sample benchmarking across 11 models..."):
        comp_results = run_model_comparison(df_monthly, horizon=horizon)
        st.session_state[session_key] = comp_results
else:
    comp_results = st.session_state[session_key]

eval_metrics = comp_results.get("evaluation_metrics", [])
table_df = build_model_comparison_table(eval_metrics)
best_model = comp_results.get("best_model", "N/A")
val_dates = comp_results.get("validation_dates")
y_val = comp_results.get("validation_actual")
forecasts = comp_results.get("forecasts", {})

# Top Banner: Recommended Best Model
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, #102C26 0%, #183D32 100%); border-radius: 18px; padding: 24px 28px; color: #FFFFFF; margin-bottom: 24px; box-shadow: 0 8px 24px rgba(16, 44, 38, 0.15);">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: #C4A66A; letter-spacing: 0.1em; margin-bottom: 4px;">
                    🏆 Recommended Optimal Model (Out-of-Sample Validation)
                </div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.85rem; font-weight: 600; color: #FFFFFF; line-height: 1.2;">
                    {best_model}
                </div>
                <div style="font-size: 0.85rem; color: #D1FAE5; margin-top: 6px;">
                    Selected strictly based on lowest Root Mean Squared Error (RMSE) on chronological holdout data.
                </div>
            </div>
            <div style="text-align: right; background: rgba(255, 254, 250, 0.1); border: 1px solid rgba(255, 254, 250, 0.2); border-radius: 14px; padding: 12px 20px;">
                <div style="font-size: 0.72rem; color: #A9BEA0; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                    Holdout RMSE
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #C4A66A;">
                    {eval_metrics[0].get('rmse', 'N/A') if eval_metrics else 'N/A'}
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Comparison Table
st.subheader("📊 Model Comparison & Diagnostic Performance Table")
st.dataframe(table_df, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# Actual vs Predicted Plot
st.subheader(f"📈 Out-of-Sample Predictions vs Actual Validation Occurrences ({len(y_val)} Months)")

fig_val = go.Figure()

# Actual Values
fig_val.add_trace(
    go.Scatter(
        x=val_dates,
        y=y_val,
        name="Actual Documented Occurrences",
        mode="lines+markers",
        line=dict(color="#102C26", width=3),
        marker=dict(size=8, symbol="circle"),
    )
)

# Top 4 models
top_models_to_plot = [r["model_name"] for r in eval_metrics[:4]]
palette = ["#527A50", "#2563EB", "#C4A66A", "#7C3AED"]

for i, m_name in enumerate(top_models_to_plot):
    if m_name in forecasts:
        fig_val.add_trace(
            go.Scatter(
                x=val_dates,
                y=forecasts[m_name],
                name=f"{m_name}",
                mode="lines+markers",
                line=dict(color=palette[i % len(palette)], width=2, dash="dash" if i > 0 else "solid"),
                marker=dict(size=6),
            )
        )

format_plotly_figure(
    fig_val,
    title=f"Holdout Validation Horizon ({val_dates.iloc[0].strftime('%b %Y')} to {val_dates.iloc[-1].strftime('%b %Y')})",
    height=440,
)
st.plotly_chart(fig_val, use_container_width=True)

# Residual Diagnostics for Top Model
st.markdown("<br>", unsafe_allow_html=True)
st.subheader(f"🔍 Residual Diagnostics for Recommended Model: {best_model}")

if best_model in forecasts:
    residuals = y_val - forecasts[best_model]
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=val_dates, y=residuals, name="Residual (Actual - Predicted)", marker_color="#C24134"))
        fig_res.add_hline(y=0, line_dash="dash", line_color="#5A665D")
        format_plotly_figure(fig_res, title="Validation Residual Errors Over Time", height=320)
        st.plotly_chart(fig_res, use_container_width=True)

    with col_r2:
        lb_diag = perform_ljung_box_test(pd.Series(residuals), lags=[2, 4])
        st.markdown("#### Residual Autocorrelation Test (Ljung-Box)")
        if "error" not in lb_diag:
            st.write(f"- **Null Hypothesis:** {lb_diag.get('null_hypothesis')}")
            for res_lag in lb_diag.get("results_by_lag", []):
                st.write(f"- **Lag {res_lag['lag']}:** p-value = `{res_lag['p_value']}` | White Noise = `{res_lag['is_white_noise']}`")
            st.info(f"**Diagnostic Conclusion:** {lb_diag.get('interpretation')}")
        else:
            st.info("Validation horizon is compact; residual visual distribution inspected.")

# CSV Download
st.markdown("<br>", unsafe_allow_html=True)
csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Comparative Evaluation CSV",
    data=csv_bytes,
    file_name=f"model_benchmark_{selected_slug}.csv",
    mime="text/csv",
)
