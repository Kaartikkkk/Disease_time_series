"""Calendar and cyclical trigonometric feature engineering."""

import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
    frequency: str = "D",
) -> pd.DataFrame:
    """Extract deterministic calendar and cyclical trigonometric features without future leakage.

    Features generated:
    - month (1-12)
    - day_of_year (1-366)
    - day_of_week (0-6)
    - week_of_year (1-53)
    - quarter (1-4)
    - is_weekend (0 or 1)
    - sin_dayofyear, cos_dayofyear: annual cyclical components
    - sin_month, cos_month: monthly cyclical components
    - sin_dayofweek, cos_dayofweek: weekly cyclical components
    """
    if df.empty or date_col not in df.columns:
        return df

    df_feat = df.copy()
    dates = pd.to_datetime(df_feat[date_col])

    df_feat["month"] = dates.dt.month
    df_feat["quarter"] = dates.dt.quarter

    # Weekly/annual features depending on frequency
    if frequency in ["D", "DAILY"]:
        df_feat["day_of_year"] = dates.dt.dayofyear
        df_feat["day_of_week"] = dates.dt.dayofweek
        df_feat["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)

        # Annual cycle
        df_feat["sin_dayofyear"] = np.sin(2 * np.pi * df_feat["day_of_year"] / 365.25)
        df_feat["cos_dayofyear"] = np.cos(2 * np.pi * df_feat["day_of_year"] / 365.25)

        # Weekly cycle
        df_feat["sin_dayofweek"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7.0)
        df_feat["cos_dayofweek"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7.0)

    elif frequency in ["W", "WEEKLY", "W-SUN"]:
        df_feat["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        df_feat["sin_weekofyear"] = np.sin(2 * np.pi * df_feat["week_of_year"] / 52.18)
        df_feat["cos_weekofyear"] = np.cos(2 * np.pi * df_feat["week_of_year"] / 52.18)

    # Monthly cycle
    df_feat["sin_month"] = np.sin(2 * np.pi * df_feat["month"] / 12.0)
    df_feat["cos_month"] = np.cos(2 * np.pi * df_feat["month"] / 12.0)

    return df_feat
