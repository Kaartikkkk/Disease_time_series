"""Verdant Intelligence — Custom SVG Icon System.

Consistent stroke-based iconography tailored for agricultural intelligence,
climate-tech, and scientific time-series platforms.
All icons are encoded into base64 data URIs inside <img> tags to prevent
Streamlit's DOMPurify / st.html sanitization from stripping raw <svg> tags.
"""

import base64
from typing import Optional


def _to_img(svg_str: str, size: int, css_class: str = "", style: str = "") -> str:
    """Encode SVG string to base64 data URI image element."""
    b64 = base64.b64encode(svg_str.strip().encode("utf-8")).decode("utf-8")
    cls_attr = f' class="{css_class}"' if css_class else ""
    default_style = f"display: inline-block; vertical-align: middle; width: {size}px; height: {size}px; {style}".strip()
    return f'<img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}" style="{default_style}"{cls_attr} alt="icon" />'


def _res_col(color: str, default: str = "#102C26") -> str:
    """Resolve currentColor into an explicit hex color for standalone SVG images."""
    if color == "currentColor":
        return default
    return color


def icon_logo(size: int = 24, color: str = "#13382C", css_class: str = "") -> str:
    """Verdant Intelligence brand mark: germinating embryo inside a climate circle."""
    c = _res_col(color, "#13382C")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="10" stroke="{c}" stroke-width="1.75" opacity="0.25"/>
  <path d="M12 2C6.48 2 2 6.48 2 12C2 15.5 3.8 18.58 6.55 20.35C8.1 16.5 10.5 13.5 14 12C15.5 11.36 17.5 11 19.5 11.5C19.83 10.05 20 8.55 20 7C20 4.24 17.76 2 15 2H12Z" fill="{c}" fill-opacity="0.15"/>
  <path d="M12 21V11C12 7.5 9 5 6 5C6 9 8 13 12 15M12 15C14.5 13 17 11.5 19.5 12" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_leaf(size: int = 20, color: str = "#527A50", css_class: str = "") -> str:
    """Delicate botanical leaf with primary vein."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M19.5 4.5C15.5 3.5 8.5 5.5 5 11C2.5 15 3.5 19 6.5 20C9.5 21 15 19.5 18 15C21 10.5 20.5 5.5 19.5 4.5Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M5.5 19.5C8 16 12 12 19 5" stroke="{c}" stroke-width="1.75" stroke-linecap="round"/>
  <path d="M11 13L13.5 15.5M14 10L17 12" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_pathology(size: int = 20, color: str = "#DC2626", css_class: str = "") -> str:
    """Pathogen spore / microscopic lesion cluster."""
    c = _res_col(color, "#DC2626")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="5" stroke="{c}" stroke-width="1.75"/>
  <circle cx="12" cy="12" r="2" fill="{c}"/>
  <circle cx="6" cy="8" r="2" stroke="{c}" stroke-width="1.5"/>
  <circle cx="18" cy="8" r="2" stroke="{c}" stroke-width="1.5"/>
  <circle cx="7" cy="17" r="1.75" stroke="{c}" stroke-width="1.5"/>
  <circle cx="17" cy="17" r="1.75" stroke="{c}" stroke-width="1.5"/>
  <path d="M10 8L7.5 8M14 8L16.5 8M10 15L8 16M14 15L16 16" stroke="{c}" stroke-width="1.25" stroke-linecap="round" opacity="0.6"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_weather(size: int = 20, color: str = "#2563EB", css_class: str = "") -> str:
    """Rain cloud with precipitation vectors."""
    c = _res_col(color, "#2563EB")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M7 16H6.5C4.01 16 2 13.99 2 11.5C2 9.17 3.77 7.26 6.08 7.03C6.73 4.14 9.3 2 12.35 2C15.97 2 18.94 4.8 19.14 8.4C20.78 8.94 22 10.48 22 12.3C22 14.51 20.21 16.3 18 16.3H17" stroke="{c}" stroke-width="1.75" stroke-linecap="round"/>
  <path d="M8 19L7 22M12 19L11 22M16 19L15 22" stroke="{c}" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_thermometer(size: int = 20, color: str = "#D97706", css_class: str = "") -> str:
    """Slender scientific temperature sensor."""
    c = _res_col(color, "#D97706")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M14 14.76V5C14 3.9 13.1 3 12 3C10.9 3 10 3.9 10 5V14.76C8.8 15.65 8 17.22 8 19C8 21.21 9.79 23 12 23C14.21 23 16 21.21 16 19C16 17.22 15.2 15.65 14 14.76Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="12" cy="19" r="2.5" fill="{c}"/>
  <path d="M12 8V16.5" stroke="{c}" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_timeseries(size: int = 20, color: str = "#527A50", css_class: str = "") -> str:
    """Analytical time-series waveform."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M3 19H21M3 5V19" stroke="{c}" stroke-width="1.75" stroke-linecap="round"/>
  <path d="M4 14L8 8L12 13L16 6L20 11" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="8" cy="8" r="1.5" fill="{c}"/>
  <circle cx="16" cy="6" r="1.5" fill="{c}"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_forecast(size: int = 20, color: str = "#166534", css_class: str = "") -> str:
    """Forward trajectory / predictive projection."""
    c = _res_col(color, "#166534")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M3 17L9 11L13 15L21 7" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M16 7H21V12" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M13 19C15 17 18 16 21 16" stroke="{c}" stroke-width="1.5" stroke-dasharray="2 3" stroke-linecap="round" opacity="0.7"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_model_compare(size: int = 20, color: str = "#9333EA", css_class: str = "") -> str:
    """Econometric balance scale for model benchmarking."""
    c = _res_col(color, "#9333EA")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 3V21M3 7L12 5L21 7M6 14C4.5 14 3 13 3 10L6 7L9 10C9 13 7.5 14 6 14ZM18 16C16.5 16 15 15 15 12L18 9L21 12C21 15 19.5 16 18 16Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M8 21H16" stroke="{c}" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_shield(size: int = 20, color: str = "#527A50", css_class: str = "") -> str:
    """Verification shield with botanical / checkmark cue."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2L4 5.5V11.5C4 16.5 7.4 21.2 12 22.5C16.6 21.2 20 16.5 20 11.5V5.5L12 2Z" stroke="{c}" stroke-width="1.75" stroke-linejoin="round"/>
  <path d="M9 12L11 14L15 10" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_pin(size: int = 20, color: str = "#527A50", css_class: str = "") -> str:
    """Geographic precision pin for Indian agro-zones."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 21C16 16.5 19 13.5 19 9.5C19 5.36 15.87 2 12 2C8.13 2 5 5.36 5 9.5C5 13.5 8 16.5 12 21Z" stroke="{c}" stroke-width="1.75" stroke-linejoin="round"/>
  <circle cx="12" cy="9.5" r="3" stroke="{c}" stroke-width="1.5"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_download(size: int = 20, color: str = "#527A50", css_class: str = "") -> str:
    """Export and data archive download tray."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 3V15M12 15L7.5 10.5M12 15L16.5 10.5" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M4 17V19C4 20.1 4.9 21 6 21H18C19.1 21 20 20.1 20 19V17" stroke="{c}" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_pulse(size: int = 12, color: str = "#527A50", css_class: str = "") -> str:
    """Animated status pulse indicating live verified API telemetry."""
    return f"""<span style="display: inline-flex; align-items: center; justify-content: center; width: {size}px; height: {size}px; position: relative;" class="{css_class}">
  <span style="position: absolute; width: 100%; height: 100%; border-radius: 50%; background-color: {color}; opacity: 0.4; animation: pulseGlow 2s infinite ease-out;"></span>
  <span style="width: {size * 0.55:.1f}px; height: {size * 0.55:.1f}px; border-radius: 50%; background-color: {color};"></span>
</span>"""


def icon_arrow_right(size: int = 16, color: str = "#7D8A82", css_class: str = "") -> str:
    """Sleek directional arrow."""
    c = _res_col(color, "#7D8A82")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M5 12H19M19 12L13 6M19 12L13 18" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_home(size: int = 20, color: str = "#102C26", css_class: str = "") -> str:
    """Home / overview icon."""
    c = _res_col(color, "#102C26")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M3 10.5L12 3L21 10.5V20C21 20.55 20.55 21 20 21H15V14H9V21H4C3.45 21 3 20.55 3 20V10.5Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_layers(size: int = 20, color: str = "#527A50", css_class: str = "") -> str:
    """Stacked layers icon for Model Comparison."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M2 17L12 22L22 17" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M2 12L12 17L22 12" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_search(size: int = 18, color: str = "#7D8A82", css_class: str = "") -> str:
    """Search magnifying glass icon."""
    c = _res_col(color, "#7D8A82")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="11" cy="11" r="7" stroke="{c}" stroke-width="1.75"/>
  <path d="M16.5 16.5L21 21" stroke="{c}" stroke-width="2" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_sun(size: int = 18, color: str = "#5A665D", css_class: str = "") -> str:
    """Sun light/dark theme toggle icon."""
    c = _res_col(color, "#5A665D")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="4.5" stroke="{c}" stroke-width="1.75"/>
  <path d="M12 2V4M12 20V22M4 12H2M22 12H20M5.64 5.64L7.05 7.05M16.95 16.95L18.36 18.36M5.64 18.36L7.05 16.95M16.95 7.05L18.36 5.64" stroke="{c}" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_bell(size: int = 18, color: str = "#5A665D", with_dot: bool = True, css_class: str = "") -> str:
    """Notification bell icon with indicator dot."""
    c = _res_col(color, "#5A665D")
    dot_svg = '<circle cx="18" cy="5" r="3" fill="#E53935" stroke="#FFFFFF" stroke-width="1.5"/>' if with_dot else ''
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M18 8C18 6.4 17.37 4.88 16.24 3.76C15.12 2.63 13.59 2 12 2C10.4 2 8.88 2.63 7.76 3.76C6.63 4.88 6 6.4 6 8C6 15 3 17 3 17H21C21 17 18 15 18 8Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M13.73 21C13.55 21.3 13.3 21.56 13 21.73C12.7 21.9 12.35 22 12 22C11.64 22 11.3 21.9 11 21.73C10.7 21.56 10.45 21.3 10.27 21" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  {dot_svg}
</svg>"""
    return _to_img(svg, size, css_class)


def icon_chevron_down(size: int = 14, color: str = "#7D8A82", css_class: str = "") -> str:
    """Chevron down arrow icon."""
    c = _res_col(color, "#7D8A82")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M6 9L12 15L18 9" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_sprout(size: int = 20, color: str = "#166534", css_class: str = "") -> str:
    """Sprout / seedling growth icon."""
    c = _res_col(color, "#166534")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 22V13M12 13C12 9 7 9 7 9C7 14 12 13 12 13ZM12 13C12 9 17 9 17 9C17 14 12 13 12 13Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M3 22H21" stroke="{c}" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_database(size: int = 20, color: str = "#2563EB", css_class: str = "") -> str:
    """Database / data source cylinders icon."""
    c = _res_col(color, "#2563EB")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="12" cy="5" rx="8" ry="3" stroke="{c}" stroke-width="1.75"/>
  <path d="M4 5V12C4 13.66 7.58 15 12 15C16.42 15 20 13.66 20 12V5" stroke="{c}" stroke-width="1.75"/>
  <path d="M4 12V19C4 20.66 7.58 22 12 22C16.42 22 20 20.66 20 19V12" stroke="{c}" stroke-width="1.75"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_bar_chart(size: int = 20, color: str = "#9333EA", css_class: str = "") -> str:
    """Modern vertical bar chart icon."""
    c = _res_col(color, "#9333EA")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M18 20V10M12 20V4M6 20V14" stroke="{c}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_alert_triangle(size: int = 20, color: str = "#DC2626", css_class: str = "") -> str:
    """Warning alert triangle icon."""
    c = _res_col(color, "#DC2626")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M10.29 3.86L1.82 18C1.64 18.3 1.55 18.65 1.55 19C1.55 19.53 1.76 20.04 2.14 20.41C2.51 20.79 3.02 21 3.55 21H20.45C20.98 21 21.49 20.79 21.86 20.41C22.24 20.04 22.45 19.53 22.45 19C22.45 18.65 22.36 18.3 22.18 18L13.71 3.86C13.52 3.55 13.25 3.3 12.92 3.14C12.59 2.97 12.23 2.88 12 2.88C11.77 2.88 11.41 2.97 11.08 3.14C10.75 3.3 10.48 3.55 10.29 3.86Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M12 9V13M12 17H12.01" stroke="{c}" stroke-width="2" stroke-linecap="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_trending_up(size: int = 20, color: str = "#166534", css_class: str = "") -> str:
    """Trending upward trajectory arrow."""
    c = _res_col(color, "#166534")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M23 6L13.5 15.5L8.5 10.5L1 18" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M17 6H23V12" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_book(size: int = 20, color: str = "#527A50", css_class: str = "") -> str:
    """Open book icon for research & data."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 19.5C4 18.84 4.26 18.2 4.73 17.73C5.2 17.26 5.84 17 6.5 17H20M4 19.5C4 20.16 4.26 20.8 4.73 21.27C5.2 21.74 5.84 22 6.5 22H20V2H6.5C5.84 2 5.2 2.26 4.73 2.73C4.26 3.2 4 3.84 4 4.5V19.5Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_settings(size: int = 20, color: str = "#527A50", css_class: str = "") -> str:
    """Gear / sliders methodology icon."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="3" stroke="{c}" stroke-width="1.75"/>
  <path d="M19.4 15A1.65 1.65 0 0 0 19.73 16.82L19.85 16.94C20.18 17.27 20.36 17.72 20.36 18.19C20.36 18.66 20.18 19.11 19.85 19.44C19.52 19.77 19.07 19.95 18.6 19.95C18.13 19.95 17.68 19.77 17.35 19.44L17.23 19.32A1.65 1.65 0 0 0 15.41 19A1.65 1.65 0 0 0 14.33 20.5V21C14.33 21.53 14.12 22.04 13.74 22.41C13.37 22.79 12.86 23 12.33 23C11.8 23 11.29 22.79 10.92 22.41C10.54 22.04 10.33 21.53 10.33 21V20.5A1.65 1.65 0 0 0 9.25 19A1.65 1.65 0 0 0 7.43 19.32L7.31 19.44C6.98 19.77 6.53 19.95 6.06 19.95C5.59 19.95 5.14 19.77 4.81 19.44C4.48 19.11 4.3 18.66 4.3 18.19C4.3 17.72 4.48 17.27 4.81 16.94L4.93 16.82A1.65 1.65 0 0 0 5.26 15A1.65 1.65 0 0 0 3.76 13.92H3.26C2.73 13.92 2.22 13.71 1.84 13.33C1.47 12.96 1.26 12.45 1.26 11.92C1.26 11.39 1.47 10.88 1.84 10.51C2.22 10.13 2.73 9.92 3.26 9.92H3.76A1.65 1.65 0 0 0 5.26 8.84A1.65 1.65 0 0 0 4.93 7.02L4.81 6.9C4.48 6.57 4.3 6.12 4.3 5.65C4.3 5.18 4.48 4.73 4.81 4.4C5.14 4.07 5.59 3.89 6.06 3.89C6.53 3.89 6.98 4.07 7.31 4.4L7.43 4.52A1.65 1.65 0 0 0 9.25 4.84A1.65 1.65 0 0 0 10.33 3.34V2.84C10.33 2.31 10.54 1.8 10.92 1.43C11.29 1.05 11.8 0.84 12.33 0.84C12.86 0.84 13.37 1.05 13.74 1.43C14.12 1.8 14.33 2.31 14.33 2.84V3.34A1.65 1.65 0 0 0 15.41 4.84A1.65 1.65 0 0 0 17.23 4.52L17.35 4.4C17.68 4.07 18.13 3.89 18.6 3.89C19.07 3.89 19.52 4.07 19.85 4.4C20.18 4.73 20.36 5.18 20.36 5.65C20.36 6.12 20.18 6.57 19.85 6.9L19.73 7.02A1.65 1.65 0 0 0 19.4 8.84A1.65 1.65 0 0 0 20.9 9.92H21.4C21.93 9.92 22.44 10.13 22.82 10.51C23.19 10.88 23.4 11.39 23.4 11.92C23.4 12.45 23.19 12.96 22.82 13.33C22.44 13.71 21.93 13.92 21.4 13.92H20.9A1.65 1.65 0 0 0 19.4 15Z" stroke="{c}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_india_map(size: int = 40, color: str = "#8A968E", css_class: str = "") -> str:
    """India geographical silhouette contour."""
    c = _res_col(color, "#8A968E")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 100 120" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M42 5L48 10L55 8L60 14L55 20L62 25L58 35L65 40L75 35L85 42L80 48L72 45L65 52L68 62L60 68L62 78L55 85L50 95L48 108L44 115L40 105L35 92L32 80L28 72L22 68L25 58L20 52L18 42L25 38L30 28L35 25L32 18L38 12Z" stroke="{c}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.35"/>
</svg>"""
    return _to_img(svg, size, css_class)


def icon_info(size: int = 16, color: str = "#527A50", css_class: str = "") -> str:
    """Information circle icon for tooltips and contextual explanations."""
    c = _res_col(color, "#527A50")
    svg = f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="12" r="10" stroke="{c}" stroke-width="1.75"/>
  <line x1="12" y1="16" x2="12" y2="11" stroke="{c}" stroke-width="2" stroke-linecap="round"/>
  <circle cx="12" cy="7.5" r="1.25" fill="{c}"/>
</svg>"""
    return _to_img(svg, size, css_class)

