"""Shared UI Components, Styling, and Data Loaders for Verdant Intelligence."""

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.icons import (
    icon_alert_triangle,
    icon_arrow_right,
    icon_bar_chart,
    icon_bell,
    icon_book,
    icon_chevron_down,
    icon_database,
    icon_download,
    icon_forecast,
    icon_home,
    icon_india_map,
    icon_info,
    icon_layers,
    icon_leaf,
    icon_logo,
    icon_model_compare,
    icon_pathology,
    icon_pin,
    icon_pulse,
    icon_search,
    icon_settings,
    icon_shield,
    icon_sprout,
    icon_sun,
    icon_thermometer,
    icon_timeseries,
    icon_trending_up,
    icon_weather,
)

PROCESSED_DIR = Path("data/processed")
RAW_GBIF_DIR = Path("data/raw/gbif")
RAW_WEATHER_DIR = Path("data/raw/weather")
REPORTS_DIR = Path("data/reports")
ASSETS_DIR = Path("assets")


def get_image_base64(image_path: Path) -> str:
    """Read an image file and return base64 string for seamless HTML embedding."""
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = image_path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def inject_custom_css():
    """Inject Verdant Intelligence editorial design system into Streamlit."""
    st.markdown(
        """
        <style>
        /* =====================================================================
           GOOGLE FONTS & BASE DESIGN TOKENS
           ===================================================================== */
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --forest-deep: #102C26;
            --evergreen: #183D32;
            --botanical: #527A50;
            --sage-muted: #A9BEA0;
            --sage-light: #E3EBE0;
            --warm-ivory: #F6F4EC;
            --paper-white: #FFFEFA;
            --charcoal: #202822;
            --charcoal-muted: #5A665D;
            --muted-gold: #C4A66A;
            --gold-light: #FDF9F0;
            --border-subtle: rgba(82, 122, 80, 0.16);
            --shadow-subtle: 0 4px 20px rgba(16, 44, 38, 0.05);
            --shadow-float: 0 12px 32px rgba(16, 44, 38, 0.09);
        }

        /* Base Typography & Background */
        .stApp {
            background-color: var(--warm-ivory);
            color: var(--charcoal);
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-image: radial-gradient(rgba(82, 122, 80, 0.04) 1px, transparent 1px);
            background-size: 28px 28px;
        }

        /* Top App Padding & Header Hide */
        header[data-testid="stHeader"] {
            background-color: rgba(246, 244, 236, 0.85);
            backdrop-filter: blur(10px);
            z-index: 99;
        }

        /* Typography Hierarchy */
        h1, h2, h3, .editorial-heading {
            font-family: 'Fraunces', 'DM Serif Display', Georgia, serif !important;
            color: var(--forest-deep) !important;
            letter-spacing: -0.025em;
            font-feature-settings: "liga" 1, "kern" 1;
        }

        h1 {
            font-size: 2.5rem !important;
            font-weight: 600 !important;
            line-height: 1.18 !important;
        }

        h2 {
            font-size: 1.85rem !important;
            font-weight: 500 !important;
            line-height: 1.25 !important;
        }

        h3 {
            font-size: 1.35rem !important;
            font-weight: 500 !important;
        }

        p, div, label, span {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--charcoal);
        }

        /* Navigation Bar */
        .verdant-navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 254, 250, 0.85);
            backdrop-filter: blur(14px);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 14px 24px;
            margin-bottom: 28px;
            box-shadow: var(--shadow-subtle);
        }
        .verdant-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: var(--forest-deep);
        }
        .verdant-brand-title {
            font-family: 'Fraunces', serif;
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--forest-deep);
            line-height: 1;
        }
        .verdant-brand-sub {
            font-size: 0.72rem;
            color: var(--botanical);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 600;
        }
        .verdant-nav-links {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .verdant-nav-link {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--charcoal-muted);
            text-decoration: none;
            transition: color 0.2s ease;
        }
        .verdant-nav-link:hover {
            color: var(--forest-deep);
        }

        /* Hero Split Section */
        .hero-container {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 36px;
            align-items: center;
            background: linear-gradient(135deg, rgba(255, 254, 250, 0.95) 0%, rgba(246, 244, 236, 0.8) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 24px;
            padding: 44px 48px;
            margin-bottom: 36px;
            box-shadow: var(--shadow-float);
            position: relative;
            overflow: hidden;
        }
        .hero-left {
            position: relative;
            z-index: 2;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--gold-light);
            border: 1px solid rgba(196, 166, 106, 0.35);
            color: #8C6F2D;
            padding: 5px 13px;
            border-radius: 30px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 18px;
        }
        .hero-headline {
            font-family: 'Fraunces', serif !important;
            font-size: 2.75rem !important;
            font-weight: 600 !important;
            line-height: 1.12 !important;
            color: var(--forest-deep) !important;
            margin-bottom: 18px;
            letter-spacing: -0.03em;
        }
        .hero-lead {
            font-size: 1.05rem;
            line-height: 1.6;
            color: var(--charcoal-muted);
            margin-bottom: 28px;
            max-width: 540px;
        }
        .hero-actions {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
        }
        .btn-primary-cta {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--forest-deep);
            color: var(--paper-white) !important;
            font-weight: 600;
            font-size: 0.92rem;
            padding: 12px 24px;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(16, 44, 38, 0.2);
        }
        .btn-primary-cta:hover {
            background: var(--evergreen);
            transform: translateY(-2px);
            box-shadow: 0 8px 18px rgba(16, 44, 38, 0.25);
            color: #FFFFFF !important;
        }
        .btn-secondary-cta {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: transparent;
            color: var(--forest-deep) !important;
            font-weight: 600;
            font-size: 0.92rem;
            padding: 11px 20px;
            border-radius: 12px;
            border: 1px solid rgba(16, 44, 38, 0.25);
            text-decoration: none;
            transition: all 0.2s ease;
        }
        .btn-secondary-cta:hover {
            background: rgba(16, 44, 38, 0.05);
            border-color: var(--forest-deep);
        }

        /* Telemetry status badge */
        .telemetry-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.78rem;
            color: var(--charcoal-muted);
            font-weight: 500;
        }
        .telemetry-status strong {
            color: var(--botanical);
        }

        /* Hero Right Visual Stack */
        .hero-right {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .hero-visual-frame {
            width: 100%;
            height: 380px;
            border-radius: 20px;
            overflow: hidden;
            border: 2px solid var(--paper-white);
            box-shadow: 0 16px 36px rgba(16, 44, 38, 0.14);
            position: relative;
        }
        .hero-visual-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 6s ease;
        }
        .hero-visual-frame:hover .hero-visual-img {
            transform: scale(1.03);
        }
        .hero-overlay-card {
            position: absolute;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(255, 254, 250, 0.92);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(169, 190, 160, 0.35);
            border-radius: 14px;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        }

        /* Research Metric Cards */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 36px;
        }
        .research-metric-card {
            background: var(--paper-white);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: var(--shadow-subtle);
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }
        .research-metric-card::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--botanical) 0%, var(--muted-gold) 100%);
            opacity: 0;
            transition: opacity 0.25s ease;
        }
        .research-metric-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-float);
        }
        .research-metric-card:hover::after {
            opacity: 1;
        }
        .metric-icon-wrap {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: var(--warm-ivory);
            color: var(--botanical);
            margin-bottom: 12px;
        }
        .metric-label-text {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--charcoal-muted);
            margin-bottom: 4px;
        }
        .metric-value-text {
            font-family: 'Fraunces', serif;
            font-size: 1.85rem;
            font-weight: 600;
            color: var(--forest-deep);
            line-height: 1.1;
        }
        .metric-note-text {
            font-size: 0.78rem;
            color: var(--botanical);
            margin-top: 6px;
            font-weight: 500;
        }

        /* Process Steps Grid */
        .process-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 28px 0 36px 0;
            position: relative;
        }
        .process-card {
            background: var(--paper-white);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 24px 20px;
            box-shadow: var(--shadow-subtle);
            position: relative;
            transition: all 0.25s ease;
        }
        .process-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-float);
            border-color: var(--sage-muted);
        }
        .process-num {
            font-family: 'Fraunces', serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--muted-gold);
            margin-bottom: 12px;
            line-height: 1;
        }
        .process-title {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.98rem;
            font-weight: 600;
            color: var(--forest-deep);
            margin-bottom: 8px;
        }
        .process-desc {
            font-size: 0.82rem;
            line-height: 1.5;
            color: var(--charcoal-muted);
        }

        /* Model Capabilities Grid */
        .model-cards-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 24px 0 36px 0;
        }
        .model-card {
            background: var(--paper-white);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 24px 22px;
            box-shadow: var(--shadow-subtle);
            transition: all 0.25s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .model-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-float);
            border-color: var(--botanical);
        }
        .model-badge {
            display: inline-block;
            background: rgba(82, 122, 80, 0.12);
            color: var(--botanical);
            font-size: 0.72rem;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 10px;
        }
        .model-card-title {
            font-family: 'Fraunces', serif;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--forest-deep);
            margin-bottom: 8px;
        }
        .model-card-desc {
            font-size: 0.82rem;
            color: var(--charcoal-muted);
            line-height: 1.5;
            margin-bottom: 16px;
        }
        .model-stat-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-top: 14px;
            border-top: 1px dashed var(--sage-light);
            font-size: 0.78rem;
            color: var(--charcoal-muted);
        }
        .model-stat-val {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--forest-deep);
        }

        /* Final CTA Banner */
        .final-cta-section {
            background: linear-gradient(135deg, var(--forest-deep) 0%, var(--evergreen) 100%);
            border-radius: 24px;
            padding: 48px;
            color: var(--paper-white);
            text-align: center;
            margin: 44px 0 28px 0;
            box-shadow: 0 16px 40px rgba(16, 44, 38, 0.2);
            position: relative;
            overflow: hidden;
        }
        .final-cta-title {
            font-family: 'Fraunces', serif !important;
            font-size: 2.3rem !important;
            color: #FFFFFF !important;
            margin-bottom: 14px;
        }
        .final-cta-lead {
            font-size: 1rem;
            color: #D1FAE5;
            max-width: 620px;
            margin: 0 auto 28px auto;
            line-height: 1.6;
        }
        .btn-gold-cta {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: var(--muted-gold);
            color: var(--forest-deep) !important;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 13px 28px;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(196, 166, 106, 0.35);
        }
        .btn-gold-cta:hover {
            background: #D4B67A;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(196, 166, 106, 0.45);
        }

        /* Editorial Footer */
        .verdant-footer {
            border-top: 1px solid var(--border-subtle);
            padding: 32px 0;
            margin-top: 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--charcoal-muted);
        }

        /* Keyframes */
        @keyframes pulseGlow {
            0% { transform: scale(0.95); opacity: 0.6; }
            50% { transform: scale(1.6); opacity: 0; }
            100% { transform: scale(0.95); opacity: 0.6; }
        }

        /* Accessibility: Reduced Motion */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation: none !important;
                transition: none !important;
            }
        }

        /* Hide default Streamlit sidebar page list */
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNavItems"],
        [data-testid="stSidebarNavSeparator"],
        section[data-testid="stSidebar"] > div:first-child > div:first-child > ul,
        section[data-testid="stSidebar"] nav {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }

        .sidebar-brand-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 22px;
            padding: 0 4px;
            text-decoration: none;
        }
        .sidebar-logo-circle {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #13382C;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 4px 10px rgba(19, 56, 44, 0.25);
        }
        .sidebar-brand-title {
            font-family: 'Fraunces', serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #102C26;
            line-height: 1.1;
            letter-spacing: -0.01em;
        }
        .sidebar-brand-subtitle {
            font-size: 0.58rem;
            font-weight: 700;
            color: #527A50;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-top: 3px;
        }

        /* Custom Styling for st.page_link in Sidebar */
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] a {
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            padding: 9px 14px !important;
            border-radius: 10px !important;
            color: #4A554E !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            text-decoration: none !important;
            transition: all 0.15s ease !important;
            border: none !important;
            background: transparent !important;
            margin-bottom: 2px !important;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] a:hover {
            background: rgba(82, 122, 80, 0.08) !important;
            color: #102C26 !important;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] a[aria-current="page"],
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] a.st-emotion-cache-16g04fd {
            background: #13382C !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(19, 56, 44, 0.2) !important;
        }
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] a[aria-current="page"] p,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] a[aria-current="page"] span,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] a.st-emotion-cache-16g04fd p,
        [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] a.st-emotion-cache-16g04fd span {
            color: #FFFFFF !important;
        }

        .sidebar-callout-card {
            background: linear-gradient(135deg, #13382C 0%, #1F4D3C 100%);
            border-radius: 14px;
            padding: 12px 14px;
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 22px;
            box-shadow: 0 6px 16px rgba(19, 56, 44, 0.15);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .callout-thumb {
            width: 44px;
            height: 44px;
            border-radius: 8px;
            overflow: hidden;
            flex-shrink: 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .callout-thumb img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .callout-content {
            flex: 1;
        }
        .callout-text {
            font-size: 0.78rem;
            font-weight: 600;
            color: #FFFFFF;
            line-height: 1.35;
        }
        .callout-arrow-btn {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: #2D6A4F;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            text-decoration: none !important;
            transition: transform 0.2s ease;
        }
        .callout-arrow-btn:hover {
            transform: scale(1.1);
            background: #398363;
        }

        .sidebar-footer {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 6px 4px;
        }
        .sidebar-footer-text {
            font-size: 0.72rem;
            color: #7D8A82;
            line-height: 1.4;
            font-weight: 500;
        }

        /* Top App Bar */
        .app-top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 22px;
            padding-bottom: 4px;
        }
        .search-pill {
            display: flex;
            align-items: center;
            gap: 10px;
            background: #FFFFFF;
            border: 1px solid #EAE6DC;
            border-radius: 24px;
            padding: 8px 18px;
            width: 440px;
            max-width: 100%;
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        }
        .search-placeholder {
            font-size: 0.84rem;
            color: #8A968E;
            flex: 1;
        }
        .search-kbd {
            background: #F6F4EC;
            border: 1px solid #E2DDD0;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            color: #7D8A82;
            padding: 2px 7px;
            font-family: -apple-system, sans-serif;
        }
        .top-util-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .top-icon-btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #FFFFFF;
            border: 1px solid #EAE6DC;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .top-icon-btn:hover {
            background: #F6F4EC;
            border-color: #527A50;
        }
        .user-profile-pill {
            display: flex;
            align-items: center;
            gap: 9px;
            cursor: pointer;
            padding: 3px 10px;
            border-radius: 20px;
            background: #FFFFFF;
            border: 1px solid #EAE6DC;
            transition: background 0.15s ease;
        }
        .user-profile-pill:hover {
            background: #F8F7F2;
        }
        .user-avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #13382C;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.82rem;
        }
        .user-name {
            font-size: 0.86rem;
            font-weight: 600;
            color: #102C26;
        }

        /* Overview Panoramic Hero Banner */
        .dashboard-hero-banner {
            position: relative;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 12px 36px rgba(16, 44, 38, 0.12);
            margin-bottom: 28px;
            background-size: cover;
            background-position: center;
        }
        .dashboard-hero-overlay {
            position: relative;
            background: linear-gradient(90deg, rgba(14, 38, 32, 0.95) 0%, rgba(14, 38, 32, 0.88) 45%, rgba(14, 38, 32, 0.35) 80%, rgba(14, 38, 32, 0.15) 100%);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 42px 46px;
            gap: 40px;
        }
        .hero-left-content {
            max-width: 580px;
            z-index: 2;
        }
        .hero-main-title {
            font-family: 'Fraunces', Georgia, serif !important;
            font-size: 2.85rem !important;
            font-weight: 600 !important;
            color: #FFFFFF !important;
            line-height: 1.12 !important;
            margin: 0 0 16px 0 !important;
            letter-spacing: -0.02em !important;
            text-shadow: 0 2px 14px rgba(0,0,0,0.45) !important;
        }
        .hero-main-title span {
            color: #FAF7F0 !important;
        }
        .hero-glass-card {
            width: 330px;
            flex-shrink: 0;
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.15);
            z-index: 2;
        }

        /* Feature Stat Cards Grid */
        .feature-cards-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 32px;
        }
        .feature-card {
            background: #FFFFFF;
            border: 1px solid #EAE6DC;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 2px 10px rgba(16, 44, 38, 0.03);
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 125px;
            text-decoration: none !important;
        }
        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(16, 44, 38, 0.08);
            border-color: #527A50;
        }
        .feature-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .feature-icon-circle {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .feature-arrow-btn {
            width: 26px;
            height: 26px;
            border-radius: 50%;
            border: 1px solid #EAE6DC;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #7D8A82;
            transition: all 0.2s ease;
        }
        .feature-card:hover .feature-arrow-btn {
            background: #102C26;
            border-color: #102C26;
            color: #FFFFFF;
        }
        .feature-card-stat {
            font-size: 1.35rem;
            font-weight: 700;
            color: #102C26;
            line-height: 1.1;
            margin-bottom: 4px;
        }
        .feature-card-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: #202822;
            margin-bottom: 3px;
        }
        .feature-card-sub {
            font-size: 0.78rem;
            color: #6B7280;
            line-height: 1.35;
        }

        /* Key Insights Section */
        .insights-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }
        .insights-title-wrap {
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        .insights-accent-bar {
            width: 3px;
            height: 24px;
            background: #527A50;
            border-radius: 2px;
            margin-top: 4px;
        }
        .insights-heading {
            font-family: 'Fraunces', serif;
            font-size: 1.3rem;
            font-weight: 600;
            color: #102C26;
            line-height: 1.1;
        }
        .insights-subtitle {
            font-size: 0.82rem;
            color: #6B7280;
            margin-top: 3px;
        }
        .filter-pill-group {
            display: flex;
            align-items: center;
            gap: 4px;
            background: #FFFFFF;
            border: 1px solid #EAE6DC;
            border-radius: 20px;
            padding: 3px 5px;
        }
        .filter-pill {
            padding: 4px 12px;
            border-radius: 14px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #5A665D;
            text-decoration: none !important;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .filter-pill.active {
            background: #102C26;
            color: #FFFFFF !important;
        }

        .insight-cards-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
            margin-bottom: 36px;
        }
        .insight-card {
            background: #FFFFFF;
            border: 1px solid #EAE6DC;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 2px 10px rgba(16, 44, 38, 0.03);
            transition: all 0.2s ease;
            display: flex;
            align-items: flex-start;
            gap: 14px;
            text-decoration: none !important;
        }
        .insight-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(16, 44, 38, 0.08);
            border-color: #527A50;
        }
        .insight-icon-circle {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .insight-card-content {
            flex: 1;
        }
        .insight-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .insight-card-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: #102C26;
        }
        .insight-badge {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 10px;
        }
        .insight-card-desc {
            font-size: 0.82rem;
            color: #4B5563;
            line-height: 1.45;
        }
        .insight-arrow-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: 1px solid #EAE6DC;
            color: #7D8A82;
            align-self: center;
            flex-shrink: 0;
            transition: all 0.2s ease;
        }
        .insight-card:hover .insight-arrow-btn {
            background: #102C26;
            border-color: #102C26;
            color: #FFFFFF;
        }

        /* What Does This Mean? Explanatory Card */
        .what-does-this-mean-card {
            background: #FFFEFA;
            border: 1px solid rgba(82, 122, 80, 0.22);
            border-left: 4px solid var(--botanical);
            border-radius: 12px;
            padding: 14px 18px;
            margin: 16px 0 22px 0;
            box-shadow: 0 2px 10px rgba(16, 44, 38, 0.04);
        }
        .what-mean-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.84rem;
            font-weight: 700;
            color: var(--forest-deep);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .what-mean-body {
            font-size: 0.88rem;
            line-height: 1.55;
            color: var(--charcoal);
            margin: 0;
        }
        .what-mean-tech {
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px dashed rgba(82, 122, 80, 0.2);
            font-size: 0.8rem;
            color: var(--charcoal-muted);
            line-height: 1.5;
        }
        .mode-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: rgba(82, 122, 80, 0.12);
            color: var(--forest-deep);
            border: 1px solid rgba(82, 122, 80, 0.25);
            margin-bottom: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def is_advanced_mode() -> bool:
    """Return True if user toggled to Advanced View (for professors/evaluators), False for Simple View."""
    return st.session_state.get("view_mode", "Simple View") == "Advanced View"


def is_simple_mode() -> bool:
    """Return True if in Simple View (default for farmers/students)."""
    return not is_advanced_mode()


def render_what_does_this_mean(simple_text: str, technical_details: Optional[str] = None):
    """Render a dedicated, elegant 'What Does This Mean?' explanation card."""
    tech_html = ""
    if is_advanced_mode() and technical_details:
        tech_html = f"""
        <div class="what-mean-tech">
            <strong>Technical Note / Formulation:</strong> {technical_details}
        </div>
        """
    card_html = f"""
    <div class="what-does-this-mean-card">
        <div class="what-mean-header">
            {icon_info(16, "#527A50")}
            <span>What Does This Mean?</span>
        </div>
        <p class="what-mean-body">{simple_text}</p>
        {tech_html}
    </div>
    """
    st.html(card_html)


def render_app_sidebar(active_page: str = "overview"):
    """Render the cohesive left sidebar matching the Verdant Intelligence platform mockup."""
    sprout_img_path = ASSETS_DIR / "seedling_sprout.jpg"
    sprout_b64 = get_image_base64(sprout_img_path)

    # Initialize viewing mode if not set
    if "view_mode" not in st.session_state:
        st.session_state["view_mode"] = "Simple View"

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand-header">
                <div class="sidebar-logo-circle">
                    {icon_leaf(20, "#FFFFFF")}
                </div>
                <div>
                    <div class="sidebar-brand-title">Verdant Intelligence</div>
                    <div class="sidebar-brand-subtitle">PLANT DISEASE FORECASTING · INDIA</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Simple View (Default) vs Advanced View Toggle
        current_mode = st.session_state.get("view_mode", "Simple View")
        selected_mode = st.radio(
            "Viewing Mode",
            options=["Simple View", "Advanced View"],
            index=0 if current_mode == "Simple View" else 1,
            help="Simple View uses everyday English for farmers and students. Advanced View displays statistical models (ARIMA/SARIMAX), equations, ADF/KPSS tests, and diagnostic plots for researchers.",
            key="global_view_mode_selector",
        )
        st.session_state["view_mode"] = selected_mode

        mode_badge = "🌿 Simple View" if selected_mode == "Simple View" else "🔬 Advanced View"
        st.markdown(
            f"""
            <div class="mode-status-pill">
                {mode_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/1_Disease_Explorer.py", label="Disease Records", icon="🍃")
        st.page_link("pages/2_Time_Series_Analysis.py", label="Disease Trends", icon="📊")
        st.page_link("pages/3_Model_Comparison.py", label="Compare Models", icon="📑")
        st.page_link("pages/4_Forecasts.py", label="Future Predictions", icon="📈")
        st.page_link("pages/5_Weather_Intelligence.py", label="Weather Insights", icon="☁️")
        st.page_link("pages/6_About_the_Data.py", label="About the Data", icon="📖")

        st.markdown(
            f"""
            <div style="margin-top: 24px;"></div>
            <div class="sidebar-callout-card">
                <div class="callout-thumb">
                    <img src="{sprout_b64}" alt="Sprout" />
                </div>
                <div class="callout-content">
                    <div class="callout-text">Building healthier crops for a stronger India.</div>
                </div>
                <a href="/Forecasts" target="_self" class="callout-arrow-btn">
                    {icon_arrow_right(13, "#FFFFFF")}
                </a>
            </div>
            <div class="sidebar-footer">
                <div class="sidebar-footer-map">
                    {icon_india_map(36, "#8A968E")}
                </div>
                <div class="sidebar-footer-text">
                    <div>India Focused</div>
                    <div>Data Driven</div>
                    <div>Farmer Empowered</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_top_header():
    """Render the top search and user profile bar matching the platform mockup."""
    top_header_markup = (
        f'<div class="app-top-bar">'
        f'<div class="search-pill">'
        f'<span class="search-icon">{icon_search(16, "#7D8A82")}</span>'
        f'<span class="search-placeholder">Search crops, diseases, states, or insights...</span>'
        f'<span class="search-kbd">⌘ K</span>'
        f'</div>'
        f'<div class="top-util-group">'
        f'<div class="top-icon-btn" title="Light / Dark mode">{icon_sun(17, "#5A665D")}</div>'
        f'<div class="top-icon-btn" title="Notifications">{icon_bell(17, "#5A665D", with_dot=True)}</div>'
        f'<div class="user-profile-pill">'
        f'<div class="user-avatar">K</div>'
        f'<span class="user-name">Kartik</span>'
        f'<span class="user-chevron">{icon_chevron_down(12, "#7D8A82")}</span>'
        f'</div></div></div>'
    )
    st.html(top_header_markup)





@st.cache_data
def get_config_settings() -> Dict[str, Any]:
    """Load settings from config/settings.yaml."""
    config_file = Path("config/settings.yaml")
    if not config_file.exists():
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def get_pathogen_display_name(slug: str, force_scientific: bool = False) -> str:
    """Return friendly display label: Common name first, with scientific name in parentheses in Advanced Mode."""
    settings = get_config_settings()
    pathogens = settings.get("pathogens", {})
    info = pathogens.get(slug, {})
    common = info.get("common_name", slug.replace("_", " ").title())
    scientific = info.get("scientific_name", "")
    if is_advanced_mode() or force_scientific:
        return f"{common} ({scientific})" if scientific else common
    return common


@st.cache_data
def get_disease_common_name(slug: str) -> str:
    """Return common everyday disease name (e.g. 'Rice Blast')."""
    settings = get_config_settings()
    info = settings.get("pathogens", {}).get(slug, {})
    return info.get("common_name", slug.replace("_", " ").title())


@st.cache_data
def get_disease_scientific_name(slug: str) -> str:
    """Return scientific organism name (e.g. 'Pyricularia oryzae')."""
    settings = get_config_settings()
    info = settings.get("pathogens", {}).get(slug, {})
    return info.get("scientific_name", slug)


@st.cache_data
def load_all_pathogen_options() -> Dict[str, str]:
    """Return dictionary of pathogen slug -> formatted display label with familiar common name first."""
    settings = get_config_settings()
    pathogens = settings.get("pathogens", {})
    return {
        slug: f"{info.get('common_name')} ({info.get('scientific_name')})"
        for slug, info in pathogens.items()
    }


@st.cache_data
def load_friendly_pathogen_options() -> Dict[str, str]:
    """Return dictionary of pathogen slug -> common name in Simple View, or common (scientific) in Advanced View."""
    settings = get_config_settings()
    pathogens = settings.get("pathogens", {})
    return {
        slug: get_pathogen_display_name(slug)
        for slug in pathogens.keys()
    }


@st.cache_data
def load_monthly_series(slug: str) -> Optional[pd.DataFrame]:
    """Load processed continuous monthly series for a pathogen."""
    path = PROCESSED_DIR / f"monthly_{slug}.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


@st.cache_data
def load_clean_records(slug: str) -> Optional[pd.DataFrame]:
    """Load cleaned occurrence records for a pathogen."""
    path = PROCESSED_DIR / f"clean_{slug}_records.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        if "event_date_parsed" in df.columns:
            df["event_date_parsed"] = pd.to_datetime(df["event_date_parsed"])
        return df
    return None


@st.cache_data
def load_data_quality_report_data() -> Optional[Dict[str, Any]]:
    """Load data quality report JSON."""
    path = REPORTS_DIR / "data_quality_report.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_weather_series(region_id: str) -> Optional[pd.DataFrame]:
    """Load NASA POWER monthly weather series for a region."""
    path = RAW_WEATHER_DIR / f"nasa_power_{region_id}_monthly.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def format_plotly_figure(
    fig: go.Figure,
    title: str = "",
    height: int = 420,
    show_legend: bool = True,
) -> go.Figure:
    """Apply high-contrast editorial styling to Plotly charts."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.02,
            "xanchor": "left",
            "yanchor": "top",
            "font": {
                "size": 15,
                "family": "Fraunces, DM Serif Display, Georgia, serif",
                "color": "#102C26",
            },
        },
        template="plotly_white",
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FFFEFA",
        height=height,
        margin=dict(l=40, r=40, t=55, b=40),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, family="Plus Jakarta Sans"),
        ),
        hoverlabel=dict(
            bgcolor="#102C26",
            font_size=12,
            font_family="Plus Jakarta Sans",
            font_color="#FFFFFF",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E3EBE0",
        linecolor="#A9BEA0",
        tickfont=dict(size=11, family="Plus Jakarta Sans", color="#5A665D"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E3EBE0",
        linecolor="#A9BEA0",
        tickfont=dict(size=11, family="Plus Jakarta Sans", color="#5A665D"),
    )
    return fig
