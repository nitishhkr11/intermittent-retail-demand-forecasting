"""Croston's method for intermittent demand (manual implementation).

Classic Croston (1972), non-bias-corrected (the Syntetos-Boylan
correction is a documented, deliberately separate variant and is not
applied here). Croston separately exponentially smooths two things,
both with the same fixed smoothing parameter `alpha`:

  - the SIZE of a demand observation, updated only on periods where
    demand actually occurs (y_t > 0);
  - the INTERVAL, in periods, since the previous occurrence, also
    updated only on occurrence periods (using the just-elapsed gap).

The forecast is smoothed_size / smoothed_interval, held constant across
the entire requested horizon -- Croston is a flat-line forecast by
construction, not a day-by-day shape. That is a structural property of
the method itself, not a simplification made here.

Leading zeros before the series' first non-zero observation are never
special-cased: `fit()` simply counts elapsed periods from the start of
`history`, so the first interval estimate naturally includes however
many leading zeros preceded the first occurrence.

An all-zero training series gives Croston no observed occurrence at
all, so there is no size/interval ratio to compute -- `fit()` raises
ValueError rather than inventing a 0/0 forecast. This is a genuine
mathematical difference from `src.models.tsb.TSBForecaster`, whose
update rule is instead defined to degrade gracefully to a zero
forecast in that same situation (see that module's docstring) -- not
an inconsistency between the two wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster

#: Fixed smoothing parameter for this first Croston/TSB comparison --
#: the literature-standard default (Croston 1972 and most textbook
#: expositions use alpha in the 0.1-0.2 range), not tuned on any fold's
#: forecast window.
DEFAULT_ALPHA = 0.1


class CrostonForecaster(BaseForecaster):
    """Classic Croston's method: forecast = smoothed_size / smoothed_interval, held flat.

    Fit under a single fixed `alpha` for both the size and interval
    smoothing (the standard classic-Croston formulation). Fitting is
    one deterministic pass over `history` -- there is no iterative
    optimizer here, so there is no convergence/non-convergence to
    report the way there is for ARIMA/SARIMA/Holt-Winters.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        if not np.isfinite(alpha) or not (0 < alpha <= 1):
            raise ValueError(f"alpha must be a finite value in (0, 1], got {alpha!r}.")
        self.alpha = alpha
        self.name = f"Croston(alpha={alpha})"
        self._forecast_value: Optional[float] = None
        self._z_hat: Optional[float] = None  # smoothed demand size
        self._q_hat: Optional[float] = None  # smoothed inter-demand interval
        self._n_obs_fit: Optional[int] = None
        self._n_occurrences: Optional[int] = None

    def fit(self, history: pd.Series) -> "CrostonForecaster":
        if len(history) == 0:
            raise ValueError(f"{self.__class__.__name__}.fit() needs a non-empty history.")
        values = history.to_numpy(dtype=float)
        if np.any(np.isnan(values)):
            raise ValueError(f"{self.__class__.__name__}.fit() received NaN value(s) in history.")
        if np.any(values < 0):
            raise ValueError(
                f"{self.__class__.__name__}.fit() received negative value(s) in history "
                "-- sales cannot be negative."
            )

        alpha = self.alpha
        z_hat: Optional[float] = None
        q_hat: Optional[float] = None
        periods_since_last = 0
        n_occurrences = 0

        for y_t in values:
            periods_since_last += 1
            if y_t > 0:
                n_occurrences += 1
                if z_hat is None:
                    # First-ever occurrence: seed both smoothed values
                    # directly rather than smoothing against an
                    # arbitrary prior. `periods_since_last` here already
                    # equals the 1-based index of this observation, so
                    # any leading zeros are naturally included.
                    z_hat = float(y_t)
                    q_hat = float(periods_since_last)
                else:
                    z_hat = alpha * float(y_t) + (1 - alpha) * z_hat
                    q_hat = alpha * float(periods_since_last) + (1 - alpha) * q_hat
                periods_since_last = 0

        if z_hat is None:
            raise ValueError(
                f"{self.__class__.__name__}.fit() received an all-zero training series "
                "(no demand occurrence at all) -- Croston's method is undefined here: "
                "there is no observed demand size or interval to form a ratio from. "
                "Use TSBForecaster if a defined (zero) forecast is needed in this case."
            )

        self._z_hat = z_hat
        self._q_hat = q_hat
        self._forecast_value = z_hat / q_hat
        self._n_obs_fit = int(len(values))
        self._n_occurrences = n_occurrences
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._forecast_value is None:
            raise RuntimeError(f"{self.__class__.__name__}.predict() called before fit().")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}.")
        return np.full(horizon, self._forecast_value, dtype=float)

    def fit_diagnostics(self) -> Dict[str, Any]:
        """The fitted smoothed states and the forecast they produce.

        `n_occurrences` / `n_obs_fit` together give the empirical
        occurrence rate Croston's own interval estimate is implicitly
        built from -- useful context when comparing a Smooth-category
        item (many occurrences) against a Lumpy one (few) on the same
        fixed `alpha`.
        """
        return {
            "alpha": self.alpha,
            "n_obs_fit": self._n_obs_fit,
            "n_occurrences": self._n_occurrences,
            "demand_size_smoothed": self._z_hat,
            "interval_smoothed": self._q_hat,
            "forecast_value": self._forecast_value,
        }
