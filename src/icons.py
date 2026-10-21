"""Verdant Intelligence — Custom SVG Icon System.

Consistent stroke-based iconography tailored for agricultural intelligence,
climate-tech, and scientific time-series platforms.
Style attributes:
- Geometry: Clean, restrained, organic geometric curves
- Stroke width: 1.75px default
- Default sizing: 20px / 24px
- Palette integration: uses currentColor or explicit design system tokens
"""

from typing import Optional


def icon_logo(size: int = 24, color: str = "currentColor", css_class: str = "") -> str:
    """Verdant Intelligence brand mark: germinating embryo inside a climate circle."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <circle cx="12" cy="12" r="10" stroke="{color}" stroke-width="1.75" opacity="0.25"/>
  <path d="M12 2C6.48 2 2 6.48 2 12C2 15.5 3.8 18.58 6.55 20.35C8.1 16.5 10.5 13.5 14 12C15.5 11.36 17.5 11 19.5 11.5C19.83 10.05 20 8.55 20 7C20 4.24 17.76 2 15 2H12Z" fill="{color}" fill-opacity="0.15"/>
  <path d="M12 21V11C12 7.5 9 5 6 5C6 9 8 13 12 15M12 15C14.5 13 17 11.5 19.5 12" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def icon_leaf(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Delicate botanical leaf with primary vein."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M19.5 4.5C15.5 3.5 8.5 5.5 5 11C2.5 15 3.5 19 6.5 20C9.5 21 15 19.5 18 15C21 10.5 20.5 5.5 19.5 4.5Z" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M5.5 19.5C8 16 12 12 19 5" stroke="{color}" stroke-width="1.75" stroke-linecap="round"/>
  <path d="M11 13L13.5 15.5M14 10L17 12" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""


def icon_pathology(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Pathogen spore / microscopic lesion cluster."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <circle cx="12" cy="12" r="5" stroke="{color}" stroke-width="1.75"/>
  <circle cx="12" cy="12" r="2" fill="{color}"/>
  <circle cx="6" cy="8" r="2" stroke="{color}" stroke-width="1.5"/>
  <circle cx="18" cy="8" r="2" stroke="{color}" stroke-width="1.5"/>
  <circle cx="7" cy="17" r="1.75" stroke="{color}" stroke-width="1.5"/>
  <circle cx="17" cy="17" r="1.75" stroke="{color}" stroke-width="1.5"/>
  <path d="M10 8L7.5 8M14 8L16.5 8M10 15L8 16M14 15L16 16" stroke="{color}" stroke-width="1.25" stroke-linecap="round" opacity="0.6"/>
</svg>"""


def icon_weather(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Rain cloud with precipitation vectors."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M7 16H6.5C4.01 16 2 13.99 2 11.5C2 9.17 3.77 7.26 6.08 7.03C6.73 4.14 9.3 2 12.35 2C15.97 2 18.94 4.8 19.14 8.4C20.78 8.94 22 10.48 22 12.3C22 14.51 20.21 16.3 18 16.3H17" stroke="{color}" stroke-width="1.75" stroke-linecap="round"/>
  <path d="M8 19L7 22M12 19L11 22M16 19L15 22" stroke="{color}" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""


def icon_thermometer(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Slender scientific temperature sensor."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M14 14.76V5C14 3.9 13.1 3 12 3C10.9 3 10 3.9 10 5V14.76C8.8 15.65 8 17.22 8 19C8 21.21 9.79 23 12 23C14.21 23 16 21.21 16 19C16 17.22 15.2 15.65 14 14.76Z" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="12" cy="19" r="2.5" fill="{color}"/>
  <path d="M12 8V16.5" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""


def icon_timeseries(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Analytical time-series waveform."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M3 19H21M3 5V19" stroke="{color}" stroke-width="1.75" stroke-linecap="round"/>
  <path d="M4 14L8 8L12 13L16 6L20 11" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="8" cy="8" r="1.5" fill="{color}"/>
  <circle cx="16" cy="6" r="1.5" fill="{color}"/>
</svg>"""


def icon_forecast(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Forward trajectory / predictive projection."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M3 17L9 11L13 15L21 7" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M16 7H21V12" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M13 19C15 17 18 16 21 16" stroke="{color}" stroke-width="1.5" stroke-dasharray="2 3" stroke-linecap="round" opacity="0.7"/>
</svg>"""


def icon_model_compare(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Econometric balance scale for model benchmarking."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M12 3V21M3 7L12 5L21 7M6 14C4.5 14 3 13 3 10L6 7L9 10C9 13 7.5 14 6 14ZM18 16C16.5 16 15 15 15 12L18 9L21 12C21 15 19.5 16 18 16Z" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M8 21H16" stroke="{color}" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""


def icon_shield(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Verification shield with botanical / checkmark cue."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M12 2L4 5.5V11.5C4 16.5 7.4 21.2 12 22.5C16.6 21.2 20 16.5 20 11.5V5.5L12 2Z" stroke="{color}" stroke-width="1.75" stroke-linejoin="round"/>
  <path d="M9 12L11 14L15 10" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def icon_pin(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Geographic precision pin for Indian agro-zones."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M12 21C16 16.5 19 13.5 19 9.5C19 5.36 15.87 2 12 2C8.13 2 5 5.36 5 9.5C5 13.5 8 16.5 12 21Z" stroke="{color}" stroke-width="1.75" stroke-linejoin="round"/>
  <circle cx="12" cy="9.5" r="3" stroke="{color}" stroke-width="1.5"/>
</svg>"""


def icon_download(size: int = 20, color: str = "currentColor", css_class: str = "") -> str:
    """Export and data archive download tray."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M12 3V15M12 15L7.5 10.5M12 15L16.5 10.5" stroke="{color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M4 17V19C4 20.1 4.9 21 6 21H18C19.1 21 20 20.1 20 19V17" stroke="{color}" stroke-width="1.75" stroke-linecap="round"/>
</svg>"""


def icon_pulse(size: int = 12, color: str = "#527A50", css_class: str = "") -> str:
    """Animated status pulse indicating live verified API telemetry."""
    return f"""<span style="display: inline-flex; align-items: center; justify-content: center; width: {size}px; height: {size}px; position: relative;" class="{css_class}">
  <span style="position: absolute; width: 100%; height: 100%; border-radius: 50%; background-color: {color}; opacity: 0.4; animation: pulseGlow 2s infinite ease-out;"></span>
  <span style="width: {size * 0.55:.1f}px; height: {size * 0.55:.1f}px; border-radius: 50%; background-color: {color};"></span>
</span>"""


def icon_arrow_right(size: int = 16, color: str = "currentColor", css_class: str = "") -> str:
    """Sleek directional arrow."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" class="{css_class}">
  <path d="M5 12H19M19 12L13 6M19 12L13 18" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""
