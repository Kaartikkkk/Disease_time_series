"""Two-Stage Hurdle / Zero-Inflated Model for Intermittent Disease Surveillance.

Designed specifically for zero-heavy disease incidence series (e.g. 75-85% zero months).

Statistical Formulation:
- Stage 1 (Occurrence Probability):
    Models the binary probability of an outbreak:
    P(Y_t > 0) = sigma(beta_0 + beta_1 * sin(2*pi*m/12) + beta_2 * cos(2*pi*m/12) + gamma * lags + delta * weather)
    Estimated via L2-penalized Logistic Regression.

- Stage 2 (Positive Count Intensity):
    Models the expected report magnitude given an outbreak occurred:
    E[Y_t | Y_t > 0] = exp(alpha_0 + alpha_1 * x_t)
    Estimated via regularized log-linear Ridge regression on non-zero observations.

- Combined Expected Monthly Count:
    E[Y_t] = P(Y_t > 0) * E[Y_t | Y_t > 0]
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


class HurdleFitResult:
    """Container for Hurdle model fitted parameters and diagnostic outputs."""

    def __init__(
        self,
        classifier: LogisticRegression,
        regressor: Optional[Ridge],
        scaler: StandardScaler,
        feature_names: List[str],
        pos_mean: float,
        param_summary: str,
        prob_forecasts: np.ndarray,
        intensity_forecasts: np.ndarray,
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.scaler = scaler
        self.feature_names = feature_names
        self.pos_mean = pos_mean
        self.param_summary = param_summary
        self.prob_forecasts = prob_forecasts
        self.intensity_forecasts = intensity_forecasts


def forecast_hurdle(
    train_y: pd.Series | np.ndarray,
    horizon: int = 12,
    train_exog: Optional[pd.DataFrame | np.ndarray] = None,
    future_exog: Optional[pd.DataFrame | np.ndarray] = None,
    train_dates: Optional[pd.Series | pd.DatetimeIndex | List[Any]] = None,
    future_dates: Optional[pd.Series | pd.DatetimeIndex | List[Any]] = None,
    c_logistic: float = 0.5,
    alpha_ridge: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, HurdleFitResult]:
    """Execute two-stage hurdle forecasting on zero-inflated disease count series.

    Args:
        train_y: Historical training target values.
        horizon: Number of forward periods to predict.
        train_exog: Optional exogenous weather regressors for training period.
        future_exog: Optional exogenous weather regressors for forecast horizon.
        train_dates: Optional timestamps corresponding to train_y for exact month extraction.
        future_dates: Optional timestamps corresponding to future horizon.
        c_logistic: Inverse regularization strength for Stage 1 Logistic Regression.
        alpha_ridge: L2 penalty parameter for Stage 2 positive intensity regression.

    Returns:
        Tuple of (point_forecasts, lower_bound_95, upper_bound_95, HurdleFitResult).
    """
    y_tr = np.asarray(train_y, dtype=float)
    n = len(y_tr)

    # 1. Extract calendar months
    if train_dates is not None and len(train_dates) == n:
        m_tr = pd.to_datetime(train_dates).dt.month.values
    else:
        m_tr = np.array([(i % 12) + 1 for i in range(n)])

    y_bin = (y_tr > 0).astype(int)

    # 2. Build feature matrices
    sin_m = np.sin(2.0 * np.pi * m_tr / 12.0)
    cos_m = np.cos(2.0 * np.pi * m_tr / 12.0)

    # Lags (1, 2, 12) with zero/mean padding for initial values
    l1 = np.roll(y_tr, 1)
    l1[0] = y_tr[0]
    l2 = np.roll(y_tr, 2)
    l2[:2] = y_tr[0]
    l12 = np.roll(y_tr, 12) if n >= 12 else np.roll(y_tr, 1)
    if n >= 12:
        l12[:12] = y_tr[0]

    feature_cols = ["sin_month", "cos_month", "lag_1", "lag_2", "lag_12"]
    base_features = [sin_m, cos_m, l1, l2, l12]

    # Incorporate exogenous weather features if provided
    has_weather = train_exog is not None and future_exog is not None
    if has_weather:
        w_tr = np.asarray(train_exog, dtype=float)
        if w_tr.ndim == 1:
            w_tr = w_tr.reshape(-1, 1)
        w_cols = (
            list(train_exog.columns)
            if hasattr(train_exog, "columns")
            else [f"weather_{i}" for i in range(w_tr.shape[1])]
        )
        feature_cols.extend(w_cols)
        X_tr = np.column_stack([*base_features, w_tr])
    else:
        X_tr = np.column_stack(base_features)

    # Fit standardizer
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)

    # 3. Stage 1: Outbreak Occurrence Classifier
    # Handle edge case where training set has only 0s or only 1s (single class)
    pos_count = int(np.sum(y_bin))
    if pos_count == 0:
        # All zeros in training
        preds = np.zeros(horizon)
        return (
            preds,
            np.zeros(horizon),
            np.zeros(horizon),
            HurdleFitResult(None, None, scaler, feature_cols, 0.0, "Degenerate Zero Series", preds, preds),
        )

    if pos_count == n:
        # All positive values in training: P(Y > 0) = 1.0
        clf: Optional[LogisticRegression] = None
    else:
        clf = LogisticRegression(C=c_logistic, solver="lbfgs", random_state=42)
        clf.fit(X_tr_sc, y_bin)

    # 4. Stage 2: Positive Report Intensity
    pos_mask = (y_bin > 0)
    pos_y = y_tr[pos_mask]
    pos_mean = float(np.mean(pos_y))

    reg: Optional[Ridge] = None
    if pos_count >= 5:
        reg = Ridge(alpha=alpha_ridge)
        # Log scale for positive target to enforce strictly positive intensity
        reg.fit(X_tr_sc[pos_mask], np.log(np.maximum(1.0, pos_y)))

    # 5. Multi-Step Recursive Rollout
    if future_dates is not None and len(future_dates) >= horizon:
        m_fut = pd.to_datetime(future_dates).dt.month.values[:horizon]
    else:
        last_m = m_tr[-1]
        m_fut = np.array([((last_m + step - 1) % 12) + 1 for step in range(1, horizon + 1)])

    if has_weather:
        w_fut = np.asarray(future_exog, dtype=float)
        if w_fut.ndim == 1:
            w_fut = w_fut.reshape(-1, 1)

    sim_y = list(y_tr)
    point_forecasts: List[float] = []
    prob_forecasts: List[float] = []
    intensity_forecasts: List[float] = []

    for h in range(horizon):
        curr_m = m_fut[h]
        s_m = np.sin(2.0 * np.pi * curr_m / 12.0)
        c_m = np.cos(2.0 * np.pi * curr_m / 12.0)
        cur_l1 = sim_y[-1]
        cur_l2 = sim_y[-2] if len(sim_y) >= 2 else sim_y[-1]
        cur_l12 = sim_y[-12] if len(sim_y) >= 12 else sim_y[-1]

        step_features = [s_m, c_m, cur_l1, cur_l2, cur_l12]
        if has_weather:
            step_features.extend(w_fut[h])

        X_h = np.array(step_features).reshape(1, -1)
        X_h_sc = scaler.transform(X_h)

        # Stage 1 probability
        if clf is not None:
            prob_occ = float(clf.predict_proba(X_h_sc)[0, 1])
        else:
            prob_occ = 1.0
        prob_forecasts.append(prob_occ)

        # Stage 2 intensity
        if reg is not None:
            # Predict log-intensity and exponentiate with half-variance correction
            log_int = float(reg.predict(X_h_sc)[0])
            intensity = float(np.exp(log_int))
        else:
            intensity = pos_mean

        intensity_forecasts.append(intensity)

        # Expected value
        expected_count = max(0.0, prob_occ * intensity)
        point_forecasts.append(expected_count)
        sim_y.append(expected_count)

    preds_arr = np.array(point_forecasts)
    prob_arr = np.array(prob_forecasts)
    int_arr = np.array(intensity_forecasts)

    # Prediction intervals: compute historical residual dispersion
    if clf is not None:
        fitted_prob = clf.predict_proba(X_tr_sc)[:, 1]
    else:
        fitted_prob = np.ones(n)
    if reg is not None:
        fitted_int = np.exp(reg.predict(X_tr_sc))
    else:
        fitted_int = np.full(n, pos_mean)
    fitted_in_sample = fitted_prob * fitted_int
    in_sample_resid_std = float(np.std(y_tr - fitted_in_sample))
    if in_sample_resid_std < 0.1:
        in_sample_resid_std = 1.0

    lower_bound = np.maximum(0.0, preds_arr - 1.96 * in_sample_resid_std)
    upper_bound = preds_arr + 1.96 * in_sample_resid_std

    summary_str = f"Two-Stage Hurdle (Stage 1: Logistic P(Y>0), Stage 2: {'Ridge Intensity' if reg else 'Empirical Positive Mean'})"

    fit_obj = HurdleFitResult(
        classifier=clf,
        regressor=reg,
        scaler=scaler,
        feature_names=feature_cols,
        pos_mean=pos_mean,
        param_summary=summary_str,
        prob_forecasts=prob_arr,
        intensity_forecasts=int_arr,
    )

    return preds_arr, lower_bound, upper_bound, fit_obj
