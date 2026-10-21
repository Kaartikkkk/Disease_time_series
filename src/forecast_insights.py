"""Verdant Intelligence — Programmatic Forecast Insights & Monthly Outlook Engine.

Translates econometric and time-series model projections into clear, data-driven,
interpretable scientific narratives and structured insight cards.

Core Capabilities:
1. compute_calendar_month_baselines: Historical mean/std for each calendar month (Jan-Dec)
2. classify_predicted_activity: Contextual activity level based on calendar-month distributions
3. detect_forecast_trend: Month-over-month absolute/relative change and directional classification
4. identify_peak_forecast_month: Pinpoints peak or elevated plateau window with honest caveats
5. generate_monthly_disease_outlook: Programmatic structured monthly cards and narrative sentences
6. build_multi_disease_watch_matrix: Multi-pathogen attention summary ranking
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ==============================================================================
# 1. CALENDAR-MONTH HISTORICAL BASELINES
# ==============================================================================

def compute_calendar_month_baselines(
    df_historical: pd.DataFrame,
    target_col: str = "documented_occurrence_count",
) -> Dict[int, Dict[str, float]]:
    """Compute historical distribution statistics for each calendar month (1..12).
    
    Returns:
        dict mapping month (1..12) -> {'mean': float, 'std': float, 'median': float, 'count': int, 'p75': float, 'p90': float}
    """
    if df_historical.empty or target_col not in df_historical.columns:
        return {}

    df = df_historical.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df["cal_month"] = df["date_dt"].dt.month

    # If is_missing column exists, only use non-missing rows if possible, or all
    if "is_missing" in df.columns:
        valid_rows = df[~df["is_missing"].fillna(False)]
        if len(valid_rows) >= 12:
            df = valid_rows

    baselines = {}
    for m in range(1, 13):
        m_vals = df[df["cal_month"] == m][target_col].dropna().values
        if len(m_vals) == 0:
            baselines[m] = {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "count": 0,
                "p75": 0.0,
                "p90": 0.0,
            }
        else:
            baselines[m] = {
                "mean": float(np.mean(m_vals)),
                "std": float(np.std(m_vals, ddof=1)) if len(m_vals) > 1 else 0.0,
                "median": float(np.median(m_vals)),
                "count": int(len(m_vals)),
                "p75": float(np.percentile(m_vals, 75)),
                "p90": float(np.percentile(m_vals, 90)),
            }
    return baselines


# ==============================================================================
# 2. ACTIVITY LEVEL CLASSIFICATION
# ==============================================================================

def classify_predicted_activity(
    pred_val: float,
    cal_month: int,
    baselines: Dict[int, Dict[str, float]],
    min_historical_samples: int = 3,
) -> Tuple[str, str, str]:
    """Classify forecast activity relative to historical baseline for the same calendar month.
    
    Categories:
    - Below typical activity (Green)
    - Within typical activity (Sage / Neutral)
    - Above typical activity (Amber)
    - Substantially above typical activity (Deep Amber / Muted Red)
    - Insufficient evidence (Muted Grey)
    
    Returns:
        (category_name, badge_color, plain_explanation)
    """
    if cal_month not in baselines or baselines[cal_month]["count"] < min_historical_samples:
        return (
            "Insufficient evidence",
            "#6B7280",
            f"Fewer than {min_historical_samples} historical observations available for this calendar month to establish a baseline.",
        )

    b = baselines[cal_month]
    mean_val = b["mean"]
    std_val = b["std"] if b["std"] > 0 else 0.5

    # Upper thresholds
    thresh_substantial = mean_val + 1.5 * std_val
    thresh_above = mean_val + 0.5 * std_val
    thresh_below = max(0.0, mean_val - 0.5 * std_val)

    if pred_val > thresh_substantial and pred_val >= 2.0:
        return (
            "Substantially above typical activity",
            "#B91C1C",
            f"Forecast ({pred_val:.1f}) exceeds historical {datetime(2000, cal_month, 1).strftime('%B')} average ({mean_val:.1f}) by over 1.5 standard deviations.",
        )
    elif pred_val > thresh_above:
        return (
            "Above typical activity",
            "#C4A66A",
            f"Forecast ({pred_val:.1f}) is moderately elevated above typical {datetime(2000, cal_month, 1).strftime('%B')} average ({mean_val:.1f}).",
        )
    elif pred_val < thresh_below and mean_val >= 1.0:
        return (
            "Below typical activity",
            "#527A50",
            f"Forecast ({pred_val:.1f}) is lower than typical {datetime(2000, cal_month, 1).strftime('%B')} average ({mean_val:.1f}).",
        )
    else:
        return (
            "Within typical activity",
            "#183D32",
            f"Forecast ({pred_val:.1f}) aligns with typical historical {datetime(2000, cal_month, 1).strftime('%B')} range (mean: {mean_val:.1f}).",
        )


# ==============================================================================
# 3. TREND DETECTION
# ==============================================================================

def detect_forecast_trend(
    current_val: float,
    previous_val: float,
    stability_threshold: float = 0.35,
) -> Tuple[str, str, float, Optional[float]]:
    """Detect direction and magnitude of change from previous month.
    
    Returns:
        (trend_label, trend_arrow, absolute_change, percentage_change_or_none)
    """
    abs_change = current_val - previous_val

    pct_change = None
    if previous_val > 0.05:
        pct_change = (abs_change / previous_val) * 100.0

    if abs_change > stability_threshold:
        return ("Increasing", "↗", round(abs_change, 2), round(pct_change, 1) if pct_change is not None else None)
    elif abs_change < -stability_threshold:
        return ("Decreasing", "↘", round(abs_change, 2), round(pct_change, 1) if pct_change is not None else None)
    else:
        return ("Broadly stable", "→", round(abs_change, 2), round(pct_change, 1) if pct_change is not None else None)


# ==============================================================================
# 4. PEAK MONTH IDENTIFICATION
# ==============================================================================

def identify_peak_forecast_month(
    df_forecast: pd.DataFrame,
    pathogen_name: str = "Target Pathogen",
) -> Dict[str, Any]:
    """Identify the peak forecast month or an elevated plateau window.
    
    Returns structured peak information with scientific phrasing.
    """
    if df_forecast.empty:
        return {}

    df = df_forecast.copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    max_idx = df["forecast"].idxmax()
    peak_row = df.loc[max_idx]
    max_val = float(peak_row["forecast"])
    peak_date = peak_row["date_dt"]

    # Check for near-peak plateau (within 10% of maximum)
    plateau_rows = df[(df["forecast"] >= (max_val * 0.90)) & (df["forecast"] > 0.1)]
    is_plateau = len(plateau_rows) >= 2 and max_val >= 1.0

    month_str = peak_date.strftime("%B %Y")
    lower_80 = float(peak_row.get("lower_80", max_val * 0.7))
    upper_80 = float(peak_row.get("upper_80", max_val * 1.3))

    avg_other = float(df["forecast"].drop(max_idx).mean()) if len(df) > 1 else max_val

    if is_plateau:
        months_list = [d.strftime("%b %Y") for d in plateau_rows["date_dt"]]
        narrative = (
            f"The model projects an elevated surveillance activity window spanning {', '.join(months_list)}, "
            f"reaching an empirical peak of {max_val:.2f} documented occurrences in {month_str} "
            f"(80% prediction interval: {lower_80:.2f}–{upper_80:.2f}). "
            f"This represents a multi-month period of sustained epidemiological favorability rather than an isolated spike."
        )
    elif max_val <= 0.1:
        narrative = (
            f"Throughout the entire forecast horizon, predicted documented occurrences remain near zero "
            f"(maximum {max_val:.2f} in {month_str}). This reflects non-conducive seasonal periods or suppressed inoculum in historical records."
        )
    else:
        diff_str = f"+{max_val - avg_other:.2f}" if max_val > avg_other else "0.00"
        narrative = (
            f"The highest predicted documented occurrence count in this forecast horizon is expected in {month_str} "
            f"at {max_val:.2f} occurrences (80% CI: {lower_80:.2f}–{upper_80:.2f}), exceeding other forecast months by an average of {diff_str}. "
            f"Note: This measures documented specimen discoveries deposited in research databases, not an absolute census of infected acreage."
        )

    return {
        "peak_month": month_str,
        "peak_date": peak_date,
        "peak_value": round(max_val, 2),
        "lower_80": round(lower_80, 2),
        "upper_80": round(upper_80, 2),
        "is_plateau": is_plateau,
        "avg_other": round(avg_other, 2),
        "narrative": narrative,
    }


# ==============================================================================
# 5. STRUCTURED MONTHLY OUTLOOK GENERATOR
# ==============================================================================

def generate_monthly_disease_outlook(
    df_forecast: pd.DataFrame,
    df_historical: pd.DataFrame,
    pathogen_name: str,
    target_col: str = "documented_occurrence_count",
) -> List[Dict[str, Any]]:
    """Build a comprehensive list of monthly outlook objects for the UI.
    
    Each item contains:
    - month_label: e.g. "October 2026"
    - short_date: e.g. "Oct '26"
    - predicted_count: float
    - previous_count: float
    - change_abs: float
    - change_pct: Optional[float]
    - trend_label: "Increasing" / "Decreasing" / "Broadly stable"
    - trend_arrow: "↗" / "↘" / "→"
    - uncertainty_80: (lower_80, upper_80)
    - uncertainty_95: (lower_95, upper_95)
    - activity_level: str
    - activity_color: str
    - activity_note: str
    - narrative: str
    """
    if df_forecast.empty:
        return []

    baselines = compute_calendar_month_baselines(df_historical, target_col=target_col)
    last_hist_val = float(df_historical[target_col].dropna().iloc[-1]) if not df_historical.empty else 0.0

    outlooks = []
    prev_val = last_hist_val

    for idx, row in df_forecast.reset_index(drop=True).iterrows():
        dt = pd.to_datetime(row["date"])
        p_val = float(row["forecast"])
        l80 = float(row.get("lower_80", max(0.0, p_val * 0.7)))
        u80 = float(row.get("upper_80", p_val * 1.3))
        l95 = float(row.get("lower_95", max(0.0, p_val * 0.5)))
        u95 = float(row.get("upper_95", p_val * 1.5))

        trend_label, trend_arrow, abs_change, pct_change = detect_forecast_trend(p_val, prev_val)
        activity_label, activity_color, activity_note = classify_predicted_activity(p_val, dt.month, baselines)

        # Build clean narrative
        month_full = dt.strftime("%B %Y")
        month_name = dt.strftime("%B")
        hist_mean = baselines.get(dt.month, {}).get("mean", 0.0)

        change_phrase = ""
        if idx == 0:
            change_phrase = f"from the latest historical observation of {prev_val:.1f}"
        else:
            change_phrase = f"compared with {trend_label.lower()} trajectory from preceding month ({prev_val:.1f})"

        narrative = (
            f"Documented surveillance activity for {pathogen_name} is forecast to be {trend_label.lower()} {change_phrase}. "
            f"The model estimates {p_val:.2f} documented occurrences (80% prediction interval: {l80:.2f}–{u80:.2f}). "
            f"Historical observations for {month_name} average {hist_mean:.1f} occurrences, placing this horizon as '{activity_label}'. "
            f"Consider monitoring subsequent diagnostic accessions and local agro-weather conditions."
        )

        outlooks.append({
            "month_label": month_full,
            "short_date": dt.strftime("%b %y"),
            "date": dt,
            "cal_month": dt.month,
            "predicted_count": round(p_val, 2),
            "previous_count": round(prev_val, 2),
            "change_abs": abs_change,
            "change_pct": pct_change,
            "trend_label": trend_label,
            "trend_arrow": trend_arrow,
            "lower_80": round(l80, 2),
            "upper_80": round(u80, 2),
            "lower_95": round(l95, 2),
            "upper_95": round(u95, 2),
            "activity_level": activity_label,
            "activity_color": activity_color,
            "activity_note": activity_note,
            "narrative": narrative,
        })

        prev_val = p_val

    return outlooks


# ==============================================================================
# 6. MULTI-DISEASE ATTENTION SUMMARY ("DISEASES TO WATCH")
# ==============================================================================

def build_multi_disease_watch_matrix(
    all_pathogen_options: Dict[str, str],
    load_monthly_fn: Any,
    horizon_months: int = 6,
) -> pd.DataFrame:
    """Generate comparative overview across all eligible pathogen series.
    
    Ranks series by relative seasonal elevation over historical baseline.
    """
    rows = []

    for slug, label in all_pathogen_options.items():
        try:
            df = load_monthly_fn(slug)
            if df is None or df.empty or len(df) < 12:
                continue

            # Compute historical stats
            total_records = int(df["documented_occurrence_count"].sum())
            baselines = compute_calendar_month_baselines(df)
            overall_mean = float(df["documented_occurrence_count"].mean())

            # Recent 6 months trend
            recent_mean = float(df["documented_occurrence_count"].tail(6).mean())

            # Simple Seasonal Naive baseline projection for watch matrix
            last_12 = df["documented_occurrence_count"].tail(12).values
            if len(last_12) >= 12:
                proj_next_6 = last_12[:6]
                peak_val = float(np.max(proj_next_6))
                peak_month_idx = int(np.argmax(proj_next_6))
            else:
                peak_val = recent_mean
                peak_month_idx = 0

            last_date = pd.to_datetime(df["date"].max())
            peak_date = last_date + pd.DateOffset(months=peak_month_idx + 1)

            # Seasonal deviation
            cal_m = peak_date.month
            m_base = baselines.get(cal_m, {}).get("mean", overall_mean)
            rel_dev = (peak_val - m_base) / max(0.5, m_base)

            # Quality badge
            if total_records >= 500:
                q_badge = "High Coverage"
            elif total_records >= 50:
                q_badge = "Adequate Series"
            elif total_records > 0:
                q_badge = "Sparse Surveillance"
            else:
                q_badge = "Screened / Zero Post-2015"

            # Activity status
            act_label, act_color, _ = classify_predicted_activity(peak_val, cal_m, baselines)

            rows.append({
                "slug": slug,
                "disease_name": label,
                "peak_month": peak_date.strftime("%B %Y"),
                "peak_val": round(peak_val, 2),
                "seasonal_baseline": round(m_base, 2),
                "relative_elevation": round(rel_dev * 100, 1),
                "activity_level": act_label,
                "activity_color": act_color,
                "data_quality": q_badge,
                "total_records": total_records,
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df_watch = pd.DataFrame(rows)
    # Sort by relative elevation and peak value
    df_watch = df_watch.sort_values(by=["relative_elevation", "peak_val"], ascending=[False, False]).reset_index(drop=True)
    return df_watch
