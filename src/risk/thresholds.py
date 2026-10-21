"""Threshold determination and statistical outbreak calibration methodology."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class OutbreakThresholds:
    baseline_level: float
    moderate_threshold: float
    high_alert_threshold: float
    very_high_threshold: float
    methodology: str
    target_units: str


def compute_empirical_outbreak_thresholds(
    train_history: pd.Series,
    target_units: str = "percentage (%)",
    moderate_percentile: float = 75.0,
    high_percentile: float = 90.0,
    very_high_percentile: float = 97.5,
) -> OutbreakThresholds:
    """Derive scientifically defensible, data-driven outbreak thresholds strictly from training history.

    Methodology:
    - Baseline: Historical median (50th percentile)
    - Moderate Risk: 75th percentile of historical incidence
    - High Risk (Epidemic Alert): 90th percentile (statistical anomaly threshold)
    - Very High Risk: 97.5th percentile (extreme outbreak event)

    Prevents test leakage by computing thresholds exclusively on pre-forecast training history.
    """
    clean_vals = pd.to_numeric(train_history, errors="coerce").dropna().values
    if len(clean_vals) == 0:
        return OutbreakThresholds(
            baseline_level=0.0,
            moderate_threshold=10.0,
            high_alert_threshold=25.0,
            very_high_threshold=50.0,
            methodology="Default fallback heuristics (no valid training data)",
            target_units=target_units,
        )

    p50 = float(np.percentile(clean_vals, 50.0))
    p_mod = float(np.percentile(clean_vals, moderate_percentile))
    p_high = float(np.percentile(clean_vals, high_percentile))
    p_vhigh = float(np.percentile(clean_vals, very_high_percentile))

    # Guard against identical percentiles if data is sparse or heavy with zeros
    if p_mod <= p50:
        p_mod = p50 + (np.std(clean_vals) * 0.5)
    if p_high <= p_mod:
        p_high = p_mod + (np.std(clean_vals) * 1.0)
    if p_vhigh <= p_high:
        p_vhigh = p_high + (np.std(clean_vals) * 1.5)

    return OutbreakThresholds(
        baseline_level=round(p50, 2),
        moderate_threshold=round(p_mod, 2),
        high_alert_threshold=round(p_high, 2),
        very_high_threshold=round(p_vhigh, 2),
        methodology=(
            f"Historical Percentile Profiling on Training Data (N={len(clean_vals)}): "
            f"Moderate >= {moderate_percentile}th, High >= {high_percentile}th, Very High >= {very_high_percentile}th."
        ),
        target_units=target_units,
    )
