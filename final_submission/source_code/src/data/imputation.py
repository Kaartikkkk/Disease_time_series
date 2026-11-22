"""Training-Aware Missing Data Imputation & Preprocessing Engine.

Ensures that all missing value handling adheres strictly to causal time-series principles:
1. All imputers and statistical parameters are fitted SOLELY on historical training observations.
2. Zero future information leakage: future validation/test rows never inform training statistics.
3. Imputation strategies:
   - Temporal Forward-Fill with Training Median fallback (primary causal strategy)
   - Training-only Median Imputation
   - Time-based Linear Interpolation (with boundary training median)
   - Model-Native Missing Value Preservation (for XGBoost and HistGradientBoosting)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrainingAwareImputer:
    """Strictly causal imputer fitted exclusively on historical training data."""

    def __init__(
        self,
        strategy: str = "forward_fill_median",
        feature_cols: Optional[List[str]] = None,
        date_col: str = "date",
    ) -> None:
        """Initialize imputer.

        Args:
            strategy: One of 'forward_fill_median', 'training_median',
                      'linear_interpolation', or 'model_native'.
            feature_cols: List of column names to impute. If None, all numeric columns are selected.
            date_col: Column name containing observation dates.
        """
        self.strategy = strategy
        self.feature_cols = feature_cols
        self.date_col = date_col
        self.is_fitted = False

        self.training_medians_: Dict[str, float] = {}
        self.training_means_: Dict[str, float] = {}
        self.calendar_month_medians_: Dict[str, Dict[int, float]] = {}
        self.last_training_values_: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "TrainingAwareImputer":
        """Compute imputation parameters strictly from training dataframe.

        Args:
            train_df: Historical training observations (date <= cutoff).

        Returns:
            Fitted imputer instance.
        """
        if self.feature_cols is None:
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            # Do not impute target or time index columns
            excluded = {"documented_occurrence_count", "year", "month", "quarter", "trend_index"}
            self.feature_cols = [c for c in numeric_cols if c not in excluded]

        df_sorted = train_df.sort_values(self.date_col).copy() if self.date_col in train_df.columns else train_df.copy()

        has_month = "month" in df_sorted.columns or (
            self.date_col in df_sorted.columns and pd.api.types.is_datetime64_any_dtype(df_sorted[self.date_col])
        )

        if not has_month and self.date_col in df_sorted.columns:
            df_sorted["_month_temp"] = pd.to_datetime(df_sorted[self.date_col]).dt.month
            month_col = "_month_temp"
        elif "month" in df_sorted.columns:
            month_col = "month"
        else:
            month_col = None

        for col in self.feature_cols:
            if col in df_sorted.columns:
                series = df_sorted[col].dropna()
                med_val = float(series.median()) if not series.empty else 0.0
                mean_val = float(series.mean()) if not series.empty else 0.0
                self.training_medians_[col] = med_val
                self.training_means_[col] = mean_val

                # Last known training value for boundary forward-filling
                last_valid = series.iloc[-1] if not series.empty else med_val
                self.last_training_values_[col] = float(last_valid)

                # Calendar month medians for seasonal imputation
                if month_col and month_col in df_sorted.columns:
                    month_grp = df_sorted.groupby(month_col)[col].median().to_dict()
                    self.calendar_month_medians_[col] = {int(k): float(v) for k, v in month_grp.items()}

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation to a dataframe (training, validation, or test partition).

        Args:
            df: Target dataframe containing missing values.

        Returns:
            Cleaned dataframe with missing values handled according to strategy.
        """
        if not self.is_fitted:
            raise RuntimeError("TrainingAwareImputer must be fitted on training data before calling transform()!")

        df_out = df.copy()

        if self.strategy == "model_native":
            # Retain NaNs in input features; algorithms like XGBoost handle them natively
            return df_out

        if self.strategy == "training_median":
            for col in self.feature_cols or []:
                if col in df_out.columns:
                    fill_val = self.training_medians_.get(col, 0.0)
                    df_out[col] = df_out[col].fillna(fill_val)
            return df_out

        if self.strategy == "linear_interpolation":
            for col in self.feature_cols or []:
                if col in df_out.columns:
                    # Time-based linear interpolation
                    s = df_out[col].interpolate(method="linear", limit_direction="forward")
                    # Fill leading boundary NaNs with training median
                    fill_val = self.training_medians_.get(col, 0.0)
                    df_out[col] = s.fillna(fill_val)
            return df_out

        # Default strategy: 'forward_fill_median'
        for col in self.feature_cols or []:
            if col in df_out.columns:
                # 1. Forward fill along observation timeline
                s = df_out[col].ffill()
                # 2. Leading boundary NaNs filled with training median
                fill_val = self.training_medians_.get(col, 0.0)
                df_out[col] = s.fillna(fill_val)

        return df_out

    def transform_eval_with_training_boundary(
        self,
        eval_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform an out-of-sample evaluation partition, causally carrying forward the last known training observation.

        Ensures that if the first month of the evaluation period is missing, it receives
        the final valid observation from the training period rather than a future value.
        """
        if not self.is_fitted:
            raise RuntimeError("Imputer must be fitted before transforming evaluation partition!")

        df_out = eval_df.copy()
        if self.strategy == "model_native":
            return df_out

        for col in self.feature_cols or []:
            if col in df_out.columns:
                # Prepend the last known training value at boundary index -1 for forward-filling
                last_train_val = self.last_training_values_.get(col, self.training_medians_.get(col, 0.0))
                vals = [last_train_val] + list(df_out[col].values)
                s_prepended = pd.Series(vals)

                if self.strategy == "linear_interpolation":
                    s_prepended = s_prepended.interpolate(method="linear", limit_direction="forward")
                else:
                    s_prepended = s_prepended.ffill()

                # Remove prepended anchor
                df_out[col] = s_prepended.iloc[1:].values
                df_out[col] = df_out[col].fillna(self.training_medians_.get(col, 0.0))

        return df_out


def fit_and_impute_chronological_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    strategy: str = "forward_fill_median",
) -> Tuple[pd.DataFrame, pd.DataFrame, TrainingAwareImputer]:
    """Helper to fit imputer exclusively on train_df and transform both train_df and test_df causally.

    Args:
        train_df: Historical training data.
        test_df: Out-of-sample test or validation data.
        feature_cols: Environmental/meteorological features requiring imputation.
        strategy: Imputation strategy.

    Returns:
        Tuple of (clean_train_df, clean_test_df, fitted_imputer).
    """
    imputer = TrainingAwareImputer(strategy=strategy, feature_cols=feature_cols)
    imputer.fit(train_df)
    clean_train = imputer.transform(train_df)
    clean_test = imputer.transform_eval_with_training_boundary(test_df)
    return clean_train, clean_test, imputer

