"""UI helper components, custom styling injection, badges, and Plotly figure stylers."""

import plotly.graph_objects as go
import streamlit as st


def inject_custom_css():
    """Inject modern, responsive CSS styling for headers, metric cards, badges, and tabs."""
    css = """
    <style>
    /* Global font & smooth transitions */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Badges */
    .badge-real {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 9999px;
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    .badge-synthetic {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 9999px;
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    /* Alert callouts */
    .risk-banner {
        padding: 16px 20px;
        border-radius: 10px;
        margin: 14px 0;
        font-weight: 500;
    }
    .risk-low {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        color: #064E3B;
    }
    .risk-moderate {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        color: #78350F;
    }
    .risk-high {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        color: #7F1D1D;
    }
    .risk-very-high {
        background-color: #450A0A;
        border-left: 5px solid #DC2626;
        color: #FEF2F2;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def format_plotly_figure(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "Date",
    yaxis_title: str = "Value",
    height: int = 420,
) -> go.Figure:
    """Apply consistent, professional styling to Plotly figures."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "family": "Arial, sans-serif"},
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        template="plotly_white",
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.6)",
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E2E8F0")
    return fig


def render_metric_card(title: str, value: str, subtitle: str = ""):
    """Render a styled HTML KPI metric card."""
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
