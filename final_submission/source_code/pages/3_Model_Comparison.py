"""Page 3: Model Comparison — Benchmark Classical Econometrics & Modern Forecasters.

Structured to match exact visual mockup specification:
1. Shared Top Navigation Bar (AgriForecast standard)
2. Editorial Hero Card (Crop ear artwork on right)
3. Filter Toolbar Card (Disease dropdown, Protocol dropdown, Horizon slider, Re-Run button)
4. Overall Winner & Key Takeaways Card (3-column layout: Winner, Key Takeaways, RMSE Score)
5. Classical & Advanced Model Pills
6. Classical Statistical Forecasting Models (Table with Rank 1 highlighted + Horizontal Bar Chart)
7. Advanced & Modern Forecasting Models (Table with Rank 1 highlighted + Horizontal Bar Chart)
8. Ground-Truth Overlay & Head-to-Head Benchmark (Expandable comparison section + CSV download)
9. Bottom Action CTAs with Landscape Hill Contours
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.evaluation import build_model_comparison_table
from src.forecasting import chronological_split, run_model_comparison, run_rolling_walk_forward_evaluation
from src.icons import (
    icon_arrow_right,
    icon_bar_chart,
    icon_calendar,
    icon_check,
    icon_check_circle,
    icon_download,
    icon_forecast,
    icon_info,
    icon_layers,
    icon_leaf,
    icon_model_compare,
    icon_pulse,
    icon_settings,
    icon_shield,
    icon_target,
    icon_timeseries,
    icon_trending_up,
    icon_trophy,
    icon_weather,
)
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
    page_title="Model Comparison — AgriForecast",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_top_navigation(active_page="models")

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
        <div class="hero-tag">MODEL COMPARISON</div>
        <h1 class="hero-headline" style="font-size: 2.1rem; margin-bottom: 8px;">
            Model Comparison &amp; Benchmark Leaderboard
        </h1>
        <p class="hero-subtitle" style="font-size: 0.95rem; margin-bottom: 0;">
            Fair, strictly chronological testing on unseen data comparing classical statistical models against modern machine-learning and deep-learning architectures.
        </p>
    </div>
    <div class="hero-image-wrapper" style="width: 50%;">
        <img src="{hero_ear_b64}" class="hero-ear-img" alt="Model comparison" />
    </div>
</div>
"""
st.html(hero_markup)

# ==============================================================================
# 2. CONTROLS TOOLBAR CARD (FILTER + PROTOCOL + SLIDER + BUTTON)
# ==============================================================================
st.markdown('<div class="filter-toolbar-card" style="margin-bottom: 20px;">', unsafe_allow_html=True)
col_p1, col_p2, col_p3, col_p4 = st.columns([1.3, 1.3, 1.2, 0.9])
with col_p1:
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
with col_p2:
    st.markdown(
        f"<div class='filter-field-label'>{icon_settings(15, '#2D6A4F')} Validation Protocol</div>",
        unsafe_allow_html=True,
    )
    eval_protocol = st.selectbox(
        "Validation Protocol",
        options=[
            "Fixed Chronological Holdout",
            "Multi-Origin Validation (Robustness Analysis)",
            "Operational Walk-Forward (1-3 Mo Ahead)",
        ],
        index=0,
        label_visibility="collapsed",
        help="Fixed Holdout evaluates models on a single chronological holdout window. Multi-Origin Validation tests models across 4 chronological origins covering Sep 2022 to Apr 2024.",
    )
with col_p3:
    st.markdown(
        f"<div class='filter-field-label'>{icon_calendar(15, '#2D6A4F')} Testing Period (Months)</div>",
        unsafe_allow_html=True,
    )
    horizon = st.slider(
        "Testing Period (Months)",
        min_value=3,
        max_value=20,
        value=6,
        step=1,
        label_visibility="collapsed",
        help="Number of unseen future months to evaluate in the chronological test window.",
    )
with col_p4:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("Re-Run Evaluation", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

is_multi_origin = "Multi-Origin" in eval_protocol
is_walk_forward = "Walk-Forward" in eval_protocol and not is_multi_origin

# Multi-Origin Delegation
if is_multi_origin:
    st.markdown(
        f"""
        <div style="background: rgba(82, 122, 80, 0.08); padding: 10px 16px; border-radius: 8px; border: 1px solid rgba(82, 122, 80, 0.2); margin-bottom: 16px; font-size: 0.85rem; color: #102C26;">
            <strong>Multi-Origin &amp; Untouched Holdout Benchmark:</strong> Standardized multi-origin walk-forward validation across the 20-month validation period (<strong>{stats['validation_period']}</strong>) and the 20-month untouched final holdout (<strong>May 2024 – Dec 2025</strong>). 
            Evaluates 18 model architectures and ensembles across all 10 plant pathogens with full robustness diagnostics.
        </div>
        """,
        unsafe_allow_html=True,
    )
    from src.multi_origin_ui import render_multi_origin_dashboard
    render_multi_origin_dashboard(selected_slug, True, pathogen_dict)
    st.stop()

# ==============================================================================
# 3. DATA LOAD & EVALUATION EXECUTION
# ==============================================================================
df_monthly = load_monthly_series(selected_slug)
if df_monthly is None or df_monthly.empty:
    st.error(f"Monthly series not found for {selected_slug}.")
    st.stop()


@st.cache_data(show_spinner=False)
def get_cached_fixed_holdout(slug: str, h: int) -> dict:
    df_series = load_monthly_series(slug)
    return run_model_comparison(df_series, horizon=h)


@st.cache_data(show_spinner=False)
def get_cached_walk_forward(slug: str, h: int, step: int, wtype: str, thresh: bool, tau: float, log1p: bool) -> dict:
    df_series = load_monthly_series(slug)
    return run_rolling_walk_forward_evaluation(
        df_series,
        n_test_steps=h,
        step_ahead=step,
        window_type=wtype,
        use_threshold=thresh,
        tau=tau,
        use_log1p=log1p,
    )


with st.spinner("Executing out-of-sample forecast evaluation..."):
    if is_walk_forward:
        comp_results = get_cached_walk_forward(selected_slug, horizon, 1, "expanding", False, 0.5, False)
    else:
        comp_results = get_cached_fixed_holdout(selected_slug, horizon)

eval_metrics = comp_results.get("evaluation_metrics", [])
val_dates = pd.to_datetime(comp_results.get("validation_dates"))
y_val = np.array(comp_results.get("validation_actual", []))
forecasts = comp_results.get("forecasts", {})

# Partition into Classical and Advanced collections
classical_keywords = ["naive", "seasonal naive", "ar(", "ma(", "arima", "sarima", "holt-winters", "arimax", "sarimax"]


def is_classical(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in classical_keywords)


classical_rows = [m for m in eval_metrics if is_classical(m["model_name"])]
advanced_rows = [m for m in eval_metrics if not is_classical(m["model_name"])]

df_classical = build_model_comparison_table(classical_rows)
df_advanced = build_model_comparison_table(advanced_rows)

best_classical = df_classical.iloc[0] if not df_classical.empty else None
best_advanced = df_advanced.iloc[0] if not df_advanced.empty else None

c_m1_name = str(best_classical["Model"]) if best_classical is not None else "N/A"
c_m1_rmse = float(best_classical["RMSE"]) if best_classical is not None else 0.0
c_m1_mae = float(best_classical["MAE"]) if best_classical is not None else 0.0

a_m1_name = str(best_advanced["Model"]) if best_advanced is not None else "N/A"
a_m1_rmse = float(best_advanced["RMSE"]) if best_advanced is not None else 0.0
a_m1_mae = float(best_advanced["MAE"]) if best_advanced is not None else 0.0

rmse_gain_pct = ((c_m1_rmse - a_m1_rmse) / max(0.01, c_m1_rmse)) * 100.0 if c_m1_rmse > 0 else 0.0
overall_champ = a_m1_name if a_m1_rmse <= c_m1_rmse else c_m1_name
winner_rmse = a_m1_rmse if overall_champ == a_m1_name else c_m1_rmse

val_start_str = val_dates.iloc[0].strftime("%b %Y") if len(val_dates) > 0 else "N/A"
val_end_str = val_dates.iloc[-1].strftime("%b %Y") if len(val_dates) > 0 else "N/A"
train_len = len(df_monthly) - len(val_dates)
train_start_str = pd.to_datetime(df_monthly["date"].min()).strftime("%b %Y")
train_end_str = pd.to_datetime(df_monthly["date"].iloc[train_len - 1]).strftime("%b %Y") if train_len > 0 else "N/A"

# ==============================================================================
# 3. OVERALL WINNER & KEY TAKEAWAYS CARD (MATCHING SPECIFICATION)
# ==============================================================================
winner_card_html = f"""
<div style="background: #F7FAF8; border: 1px solid rgba(45, 106, 79, 0.22); border-radius: 14px; padding: 22px 26px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1 1 260px; min-width: 240px;">
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #2D6A4F; letter-spacing: 0.05em; display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                {icon_trophy(16, "#2D6A4F")} <span>OVERALL WINNER</span>
            </div>
            <h2 style="font-size: 1.6rem; font-weight: 700; color: #102C26; margin: 4px 0 6px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                {overall_champ}
            </h2>
            <div style="font-size: 0.85rem; color: #5A665D;">
                Best overall performance on unseen data for {pathogen_dict[selected_slug]} ({val_start_str} – {val_end_str}).
            </div>
        </div>
        <div style="flex: 1.5 1 340px; min-width: 280px; border-left: 1px solid rgba(45, 106, 79, 0.15); padding-left: 20px;">
            <div style="margin-bottom: 6px;">
                <span style="background: rgba(45, 106, 79, 0.1); color: #2D6A4F; font-size: 0.70rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em;">KEY TAKEAWAYS</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #102C26; margin-top: 6px;">
                {icon_check(14, "#2D6A4F")}
                <span><strong>Lower error:</strong> RMSE {winner_rmse:.2f} (best among all models)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #102C26; margin-top: 4px;">
                {icon_check(14, "#2D6A4F")}
                <span><strong>Captures complex patterns:</strong> Handles non-linear and seasonal behavior</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #102C26; margin-top: 4px;">
                {icon_check(14, "#2D6A4F")}
                <span><strong>More reliable forecasts:</strong> Reduces false spikes and improves stability</span>
            </div>
        </div>
        <div style="background: #FFFFFF; border: 1px solid rgba(45, 106, 79, 0.2); border-radius: 10px; padding: 14px 26px; text-align: center; min-width: 105px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; color: #6B7280; letter-spacing: 0.05em;">RMSE</div>
            <div style="font-size: 1.95rem; font-weight: 700; color: #2D6A4F; line-height: 1.1; margin-top: 2px;">
                {winner_rmse:.2f}
            </div>
        </div>
    </div>
</div>
"""
st.html(winner_card_html)

# ==============================================================================
# ==============================================================================
# 5. HELPER FUNCTION TO RENDER EXACT STYLED TABLES
# ==============================================================================
def render_styled_table(df_table, is_classical_section=True, best_classical_rmse=None, best_classical_name="Classical"):
    rows_html = []
    comp_header = "Improvement vs. Baseline" if is_classical_section else f"vs. Best Classical ({best_classical_name})"
    
    for idx, r in df_table.iterrows():
        rank_num = idx + 1
        model_name = str(r.get("Model", ""))
        try:
            rmse_val = f"{float(r.get('RMSE', 0.0)):.3f}"
            rmse_num = float(r.get("RMSE", 0.0))
        except Exception:
            rmse_val = str(r.get("RMSE", ""))
            rmse_num = None
        try:
            mae_val = f"{float(r.get('MAE', 0.0)):.3f}"
        except Exception:
            mae_val = str(r.get("MAE", ""))
        try:
            smape_val = f"{float(str(r.get('sMAPE (%)', 0.0)).replace('%', '')):.2f}%" if "sMAPE (%)" in r else "N/A"
        except Exception:
            smape_val = str(r.get("sMAPE (%)", "N/A"))
        try:
            mase_val = f"{float(r.get('MASE', 0.0)):.3f}" if "MASE" in r and r.get("MASE") != "N/A" else "N/A"
        except Exception:
            mase_val = str(r.get("MASE", "N/A"))
            
        if is_classical_section:
            raw_imp = r.get("Improvement vs Baseline", "0.0%")
            if not raw_imp or raw_imp == "N/A":
                raw_imp = "0.0%"
            col_markup = f"<span style='color: {'#166534' if '+' in str(raw_imp) else '#4B5563'}; font-weight: {'600' if '+' in str(raw_imp) else '400'};'>{raw_imp}</span>"
        else:
            if best_classical_rmse and best_classical_rmse > 0 and rmse_num is not None:
                gain_pct = ((best_classical_rmse - rmse_num) / best_classical_rmse) * 100.0
                if gain_pct > 0.05:
                    col_markup = f"<span style='color: #166534; font-weight: 700; background: rgba(22, 101, 52, 0.08); padding: 2px 7px; border-radius: 4px;'>+{gain_pct:.1f}% vs {best_classical_name}</span>"
                elif gain_pct < -0.05:
                    col_markup = f"<span style='color: #991B1B; font-weight: 600; background: rgba(153, 27, 27, 0.06); padding: 2px 7px; border-radius: 4px;'>{gain_pct:.1f}% vs {best_classical_name}</span>"
                else:
                    col_markup = f"<span style='color: #4B5563; font-weight: 500;'>0.0% (Par)</span>"
            else:
                col_markup = "<span style='color: #6B7280;'>N/A</span>"
            
        bg_style = "background-color: #EBF5EE; font-weight: 600;" if rank_num == 1 else "background-color: #FFFFFF;"
        row_markup = f"""
        <tr style="{bg_style} border-bottom: 1px solid #F1F5F2; height: 38px;">
            <td style="padding: 9px 12px; color: #102C26; width: 45px; font-weight: {'700' if rank_num == 1 else '500'};">{rank_num}</td>
            <td style="padding: 9px 12px; color: #102C26; font-weight: {'700' if rank_num == 1 else '500'};">{model_name}</td>
            <td style="padding: 9px 12px; color: #102C26; font-weight: {'700' if rank_num == 1 else '500'};">{rmse_val}</td>
            <td style="padding: 9px 12px; color: #4B5563;">{mae_val}</td>
            <td style="padding: 9px 12px; color: #4B5563;">{smape_val}</td>
            <td style="padding: 9px 12px; color: #4B5563;">{mase_val}</td>
            <td style="padding: 9px 12px;">{col_markup}</td>
        </tr>
        """
        rows_html.append(row_markup)
        
    return f"""
    <div style="border: 1px solid rgba(82, 122, 80, 0.18); border-radius: 10px; overflow: hidden; margin-top: 10px; margin-bottom: 20px; background: #FFFFFF; box-shadow: 0 1px 4px rgba(0,0,0,0.02);">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left;">
            <thead>
                <tr style="background-color: #F8FAF9; border-bottom: 1.5px solid rgba(82, 122, 80, 0.18); height: 40px;">
                    <th style="padding: 9px 12px; color: #4B5563; font-weight: 600; width: 45px;">#</th>
                    <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">Model</th>
                    <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">RMSE &darr;</th>
                    <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">MAE</th>
                    <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">sMAPE (%)</th>
                    <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">MASE</th>
                    <th style="padding: 9px 12px; color: #4B5563; font-weight: 600;">{comp_header}</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """

# ==============================================================================
# 6. INTERACTIVE PILL TABS: CLASSICAL & ADVANCED MODELS (ZERO-SCROLL VIEW TOGGLE WITH SVG ICONS)
# ==============================================================================
# Read query param for initial state
q_view = str(st.query_params.get("view", "classical")).lower()
opt_classical = ":material/bar_chart: Classical Models"
opt_advanced = ":material/layers: Advanced Models"
opt_both = ":material/compare_arrows: Compare Both"

default_pill = opt_classical
if "adv" in q_view:
    default_pill = opt_advanced
elif "both" in q_view:
    default_pill = opt_both

# Inject custom CSS to make Streamlit pills look beautiful and match AgriForecast branding
st.markdown(
    """
    <style>
    /* Custom styling for Model Category Pills */
    div[data-testid="stPills"] {
        margin-top: 4px !important;
        margin-bottom: 22px !important;
        gap: 12px !important;
    }
    div[data-testid="stPill"] {
        border-radius: 9999px !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        padding: 6px 18px !important;
        border: 1.5px solid rgba(0,0,0,0.12) !important;
        background: #F8F9FA !important;
        color: #374151 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stPill"]:hover {
        background: #EBF5EE !important;
        border-color: #1B4332 !important;
        color: #1B4332 !important;
    }
    div[data-testid="stPill"][aria-selected="true"],
    div[data-testid="stPill"][data-selected="true"] {
        background-color: #1B4332 !important;
        border-color: #1B4332 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(27,67,50,0.22) !important;
    }
    div[data-testid="stPill"][aria-selected="true"] span,
    div[data-testid="stPill"][data-selected="true"] span {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

selected_pill = st.pills(
    label="Model Family Navigation",
    options=[opt_classical, opt_advanced, opt_both],
    default=default_pill,
    selection_mode="single",
    label_visibility="collapsed",
    key="model_comparison_pills_selector",
)

if not selected_pill or "Classical" in selected_pill:
    view_param = "classical"
elif "Advanced" in selected_pill:
    view_param = "advanced"
else:
    view_param = "both"

# Silently sync query param without page reload
try:
    st.query_params["view"] = view_param
except Exception:
    pass

# ----------------- SECTION 1: CLASSICAL STATISTICAL MODELS -----------------
if view_param in ["classical", "both"]:
    st.markdown('<div id="classical-section">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                {icon_bar_chart(20, "#102C26")}
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #102C26; margin: 0;">Classical Statistical Forecasting Models</h3>
            </div>
            <span style="font-size: 0.78rem; font-weight: 600; color: #2D6A4F; background: rgba(45, 106, 79, 0.1); padding: 3px 9px; border-radius: 9999px;">
                Milestone 2 Econometric Suite ({len(df_classical)} Models)
            </span>
        </div>
        <div style="font-size: 0.86rem; color: #5A665D; margin-bottom: 8px;">
            Univariate Box-Jenkins econometrics (AR, MA, ARIMA, SARIMA), Holt-Winters exponential smoothing, and seasonal persistence baselines.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not df_classical.empty:
        st.html(render_styled_table(df_classical, is_classical_section=True))
        
        # Horizontal Bar Chart
        c_chart_df = df_classical.copy().sort_values(by="RMSE", ascending=True)
        fig_c_bar = px.bar(
            c_chart_df,
            x="RMSE",
            y="Model",
            orientation="h",
            text="RMSE",
            color_discrete_sequence=["#527A50"],
        )
        fig_c_bar.update_layout(
            title=dict(text="<b>Classical Models: RMSE on Unseen Data (Lower is Better)</b>", font=dict(size=14, color="#102C26"), x=0.01),
            xaxis=dict(title="RMSE (Reports)", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
            yaxis=dict(title="", showgrid=False, automargin=True),
            margin=dict(l=10, r=40, t=40, b=30),
            height=270,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_c_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_c_bar, use_container_width=True)
    else:
        st.info("No classical models evaluated for this configuration.")

    st.markdown('</div>', unsafe_allow_html=True)
    if view_param == "both":
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

# ----------------- SECTION 2: ADVANCED & MODERN MODELS -----------------
if view_param in ["advanced", "both"]:
    st.markdown('<div id="advanced-section">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                {icon_layers(20, "#102C26")}
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #102C26; margin: 0;">Advanced &amp; Modern Forecasting Models</h3>
            </div>
            <span style="font-size: 0.78rem; font-weight: 600; color: #166534; background: rgba(22, 101, 52, 0.1); padding: 3px 9px; border-radius: 9999px;">
                Milestone 3 ML/DL Suite ({len(df_advanced)} Models)
            </span>
        </div>
        <div style="font-size: 0.86rem; color: #5A665D; margin-bottom: 8px;">
            Two-stage Hurdle XGBoost, Prophet Bayesian curves, multi-pathogen pooled ensembles, and deep recurrent neural networks (GRU &amp; LSTM), benchmarked directly against top classical baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not df_advanced.empty:
        st.html(render_styled_table(
            df_advanced,
            is_classical_section=False,
            best_classical_rmse=c_m1_rmse,
            best_classical_name=c_m1_name,
        ))
        
        # Horizontal Bar Chart
        a_chart_df = df_advanced.copy().sort_values(by="RMSE", ascending=True)
        fig_a_bar = px.bar(
            a_chart_df,
            x="RMSE",
            y="Model",
            orientation="h",
            text="RMSE",
            color_discrete_sequence=["#527A50"],
        )
        fig_a_bar.update_layout(
            title=dict(text="<b>Advanced Models: RMSE on Unseen Data (Lower is Better)</b>", font=dict(size=14, color="#102C26"), x=0.01),
            xaxis=dict(title="RMSE (Reports)", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
            yaxis=dict(title="", showgrid=False, automargin=True),
            margin=dict(l=10, r=40, t=40, b=30),
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_a_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_a_bar, use_container_width=True)
    else:
        st.info("No advanced models evaluated for this configuration.")

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 7. GROUND-TRUTH OVERLAY & HEAD-TO-HEAD BENCHMARK (EXPANDABLE DETAIL)
# ==============================================================================
st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
with st.expander("Ground-Truth Overlay & Head-to-Head Comparison Matrix", expanded=False):
    # Actual vs Predicted Plot
    fig_val = go.Figure()
    fig_val.add_trace(
        go.Scatter(
            x=val_dates,
            y=y_val,
            name="Actual Ground Truth (Verified)",
            mode="lines+markers",
            line=dict(color="#102C26", width=3.5),
            marker=dict(size=8, symbol="circle", color="#102C26"),
            hovertemplate="<b>Actual:</b> %{y:.1f} reports<br><b>Date:</b> %{x|%b %Y}<extra></extra>",
        )
    )

    models_to_plot = []
    if best_classical is not None:
        models_to_plot.append((best_classical["Model"], "#527A50", "Top Classical"))
    if best_advanced is not None:
        models_to_plot.append((best_advanced["Model"], "#7C3AED", "Top Advanced"))

    for m_name, color, label_prefix in models_to_plot:
        matched_key = next((k for k in forecasts if k.split(" (")[0].lower() in m_name.lower() or m_name.lower() in k.lower()), None)
        if matched_key and matched_key in forecasts:
            fig_val.add_trace(
                go.Scatter(
                    x=val_dates,
                    y=forecasts[matched_key],
                    name=f"{label_prefix}: {m_name.split('(')[0].strip()}",
                    mode="lines+markers",
                    line=dict(color=color, width=2.5, dash="dash"),
                    marker=dict(size=6),
                    hovertemplate=f"<b>{m_name}:</b> %{{y:.1f}} reports<br><b>Date:</b> %{{x|%b %Y}}<extra></extra>",
                )
            )

    format_plotly_figure(
        fig_val,
        title=f"Predictions vs. Actual Reports ({val_start_str} – {val_end_str})",
        height=360,
    )
    fig_val.update_layout(
        xaxis_title="Evaluation Date (Month)",
        yaxis_title="Monthly Verified Reports",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_val, use_container_width=True)

    # Side-by-Side Comparison Plot
    if not df_classical.empty and not df_advanced.empty:
        h2h_rows = [
            {"Model": f"Classical: {c_m1_name.split('(')[0].strip()}", "Metric": "RMSE (Overall Error)", "Value": c_m1_rmse, "Category": "Classical"},
            {"Model": f"Advanced: {a_m1_name.split('(')[0].strip()}", "Metric": "RMSE (Overall Error)", "Value": a_m1_rmse, "Category": "Advanced"},
            {"Model": f"Classical: {c_m1_name.split('(')[0].strip()}", "Metric": "MAE (Typical Error)", "Value": c_m1_mae, "Category": "Classical"},
            {"Model": f"Advanced: {a_m1_name.split('(')[0].strip()}", "Metric": "MAE (Typical Error)", "Value": a_m1_mae, "Category": "Advanced"},
        ]
        df_h2h = pd.DataFrame(h2h_rows)
        fig_h2h = px.bar(
            df_h2h,
            x="Metric",
            y="Value",
            color="Category",
            barmode="group",
            text=[f"{v:.2f}" for v in df_h2h["Value"]],
            color_discrete_map={"Classical": "#527A50", "Advanced": "#166534"},
        )
        format_plotly_figure(fig_h2h, title="Head-to-Head Error Comparison: Classical vs. Advanced", height=300)
        fig_h2h.update_layout(yaxis_title="Prediction Error (Reports)", xaxis_title="")
        st.plotly_chart(fig_h2h, use_container_width=True)

    # Direct Comparison Matrix Table
    comparison_matrix_rows = [
        {
            "Evaluation Metric": "Overall Accuracy (RMSE)",
            "Best Classical": f"{c_m1_name} ({c_m1_rmse:.2f})",
            "Best Advanced": f"{a_m1_name} ({a_m1_rmse:.2f})",
            "Improvement": f"{rmse_gain_pct:+.1f}% RMSE",
            "Winner": "Advanced" if rmse_gain_pct > 0 else "Classical",
            "Plain-English Explanation": "Modern architectures avoid zero-differencing instability while capturing non-linear weather interactions",
        },
        {
            "Evaluation Metric": "Zero-Count Handling (71.8% Zero Months)",
            "Best Classical": "Constrained linear differencing",
            "Best Advanced": "Hurdle two-stage or neural gating",
            "Improvement": "Preserves natural zero boundaries",
            "Winner": "Advanced",
            "Plain-English Explanation": "Classical differencing yields negative counts; modern architectures naturally bound predictions >= 0",
        },
    ]
    df_comp_matrix = pd.DataFrame(comparison_matrix_rows)
    st.dataframe(df_comp_matrix, use_container_width=True, hide_index=True)

    # Download Button
    full_table = build_model_comparison_table(eval_metrics)
    csv_bytes = full_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Complete Evaluation Leaderboard (CSV)",
        data=csv_bytes,
        file_name=f"model_comparison_{selected_slug}_{horizon}m.csv",
        mime="text/csv",
    )

# ==============================================================================
# 8. BOTTOM ACTION CTA WITH HILL CONTOURS
# ==============================================================================
cta_markup = f"""
<div class="bottom-cta-wrap" style="position: relative; overflow: hidden; background: #FFFFFF; border: 1px solid rgba(82, 122, 80, 0.18); border-radius: 16px; padding: 24px; margin-top: 36px; display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;">
    <div style="position: absolute; right: 0; bottom: 0; height: 100%; width: 40%; pointer-events: none; opacity: 0.45; background-image: url('{obj_hills_b64}'); background-size: cover; background-position: right bottom; background-repeat: no-repeat;"></div>
    <div style="position: relative; z-index: 2; display: flex; gap: 16px; flex-wrap: wrap; width: 100%;">
        <div style="flex: 1; min-width: 200px;">
            <a href="/Time_Series_Analysis" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px; background: #FFFFFF; border: 1.5px solid #2D6A4F; color: #2D6A4F; font-weight: 600; font-size: 0.92rem; padding: 12px 20px; border-radius: 10px;">
                    {icon_timeseries(16, "#2D6A4F")} Return to Time Series Diagnostics
                </div>
            </a>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <a href="/Future_Forecasts" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px; background: #1B4332; color: #FFFFFF; font-weight: 600; font-size: 0.92rem; padding: 12px 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(27, 67, 50, 0.18);">
                    {icon_forecast(16, "#FFFFFF")} Generate Future Forecasts &rarr;
                </div>
            </a>
        </div>
    </div>
</div>
"""
st.html(cta_markup)
