"""Outbreak risk categorization, uncertainty-aware probability estimation, and agronomic guidance."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from src.risk.thresholds import OutbreakThresholds, compute_empirical_outbreak_thresholds


@dataclass
class DailyRiskAssessment:
    date: pd.Timestamp
    forecast_incidence: float
    lower_ci: float
    upper_ci: float
    risk_level: str  # 'Low', 'Moderate', 'High', 'Very High'
    risk_color: str
    outbreak_probability_pct: float  # Estimated P(Y > high_threshold | forecast, uncertainty)
    anomaly_z_score: float
    recommended_action: str


@dataclass
class OverallRiskReport:
    overall_status: str
    peak_risk_level: str
    peak_risk_date: pd.Timestamp
    max_forecast_value: float
    thresholds: OutbreakThresholds
    daily_assessments: List[DailyRiskAssessment]
    environmental_drivers: List[str]
    disclaimer: str


def evaluate_outbreak_risk(
    forecast_dates: List[pd.Timestamp],
    point_forecast: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    train_history: pd.Series,
    target_units: str = "percentage (%)",
    custom_high_threshold: Optional[float] = None,
    weather_df: Optional[pd.DataFrame] = None,
) -> OverallRiskReport:
    """Evaluate outbreak risk across the forecast horizon with uncertainty quantification.

    METHODOLOGICAL DISTINCTIONS:
    1. Forecasted Incidence: Expected point value (e.g., 34.5% or spore copies/m³).
    2. Outbreak Probability: P(Y > threshold) computed by integrating the tail of
       the predictive error distribution: 1 - NormalCDF((threshold - y_hat) / sigma_hat).
    3. Categorical Risk Level: Low, Moderate, High (Alert), or Very High based on documented percentiles.
    """
    thresholds = compute_empirical_outbreak_thresholds(train_history, target_units=target_units)
    if custom_high_threshold is not None:
        thresholds.high_alert_threshold = float(custom_high_threshold)

    train_vals = pd.to_numeric(train_history, errors="coerce").dropna().values
    hist_mean = float(np.mean(train_vals)) if len(train_vals) > 0 else 0.0
    hist_std = float(np.std(train_vals)) if len(train_vals) > 1 and np.std(train_vals) > 1e-4 else 1.0

    assessments: List[DailyRiskAssessment] = []
    peak_level = "Low"
    level_hierarchy = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
    colors = {
        "Low": "#10B981",       # Emerald green
        "Moderate": "#F59E0B",  # Amber orange
        "High": "#EF4444",      # Red
        "Very High": "#7F1D1D",  # Dark Crimson
    }

    peak_date = forecast_dates[0] if len(forecast_dates) > 0 else pd.Timestamp.now()
    max_val = -float("inf")

    for i, date in enumerate(forecast_dates):
        y_hat = float(point_forecast[i])
        lb = float(lower_bound[i])
        ub = float(upper_bound[i])

        if y_hat > max_val:
            max_val = y_hat
            peak_date = date

        # Effective forecast standard error implied by (ub - lb) ~ 3.92 * sigma
        sigma_hat = max(0.5, (ub - lb) / 3.92)

        # Statistical Outbreak Probability: P(Y > high_alert_threshold)
        z_thresh = (thresholds.high_alert_threshold - y_hat) / sigma_hat
        prob_outbreak = float((1.0 - stats.norm.cdf(z_thresh)) * 100.0)

        # Baseline-relative anomaly Z-score
        z_anomaly = (y_hat - hist_mean) / hist_std

        # Categorize
        if y_hat >= thresholds.very_high_threshold or prob_outbreak >= 75.0:
            level = "Very High"
            action = "Urgent: Immediate field verification and emergency prophylactic intervention recommended."
        elif y_hat >= thresholds.high_alert_threshold or prob_outbreak >= 45.0:
            level = "High"
            action = "Alert: High risk of rapid epidemiological spread. Conduct rigorous crop scouting."
        elif y_hat >= thresholds.moderate_threshold or prob_outbreak >= 20.0:
            level = "Moderate"
            action = "Watch: Elevated incidence potential. Monitor local weather and canopy humidity."
        else:
            level = "Low"
            action = "Normal: Disease incidence within historical baseline bounds. Maintain standard scouting."

        if level_hierarchy[level] > level_hierarchy[peak_level]:
            peak_level = level

        assessments.append(
            DailyRiskAssessment(
                date=pd.to_datetime(date),
                forecast_incidence=round(y_hat, 2),
                lower_ci=round(lb, 2),
                upper_ci=round(ub, 2),
                risk_level=level,
                risk_color=colors[level],
                outbreak_probability_pct=round(prob_outbreak, 1),
                anomaly_z_score=round(z_anomaly, 2),
                recommended_action=action,
            )
        )

    # Detect contributing environmental drivers if weather is available
    drivers = []
    if weather_df is not None and not weather_df.empty:
        if "humidity" in weather_df.columns:
            mean_hum = float(weather_df["humidity"].tail(14).mean())
            if mean_hum > 75.0:
                drivers.append(f"Elevated relative humidity (14-day mean: {mean_hum:.1f}%) accelerates spore germination.")
        if "temperature" in weather_df.columns:
            mean_temp = float(weather_df["temperature"].tail(14).mean())
            if 15.0 <= mean_temp <= 25.0:
                drivers.append(f"Favorable thermal regime (mean {mean_temp:.1f}°C) matches optimal fungal pathogen development.")
        if "rainfall" in weather_df.columns:
            rain_sum = float(weather_df["rainfall"].tail(14).sum())
            if rain_sum > 20.0:
                drivers.append(f"Recent cumulative rainfall ({rain_sum:.1f} mm) provides persistent leaf surface wetness.")

    if not drivers:
        drivers.append("Risk driven primarily by autoregressive pathogen load and historical seasonal momentum.")

    disclaimer = (
        "SCIENTIFIC ADVISORY: This outbreak risk assessment is an indicative analytical signal produced "
        "by mathematical time-series modeling. It does NOT substitute for in-field visual diagnosis, molecular "
        "pathology assays, or guidance from local agricultural extension agronomists."
    )

    return OverallRiskReport(
        overall_status=f"{peak_level} Outbreak Risk Expected",
        peak_risk_level=peak_level,
        peak_risk_date=pd.to_datetime(peak_date),
        max_forecast_value=round(max_val, 2),
        thresholds=thresholds,
        daily_assessments=assessments,
        environmental_drivers=drivers,
        disclaimer=disclaimer,
    )
