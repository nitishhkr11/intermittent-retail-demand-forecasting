"""TSB method for intermittent demand (manual implementation).

Teunter, Syntetos & Babai (2011), "Intermittent demand: Linking
forecasting to inventory obsolescence". TSB replaces Croston's
interval-based update (see `src.models.croston.CrostonForecaster`) with
a smoothed OCCURRENCE PROBABILITY, updated every period including
zero-demand ones -- this is TSB's defining difference from Croston,
which only ever updates on occurrence periods and is consequently blind
to a demand pattern fading out over a long run of zeros.

Two fixed smoothing parameters, `alpha` (probability) and `beta` (demand
size):

  - occurrence probability p_hat is updated EVERY period t, using the
    0/1 occurrence indicator d_t = 1{y_t > 0};
  - demand size z_hat is updated only on occurrence periods (y_t > 0),
    exactly as in Croston.

Forecast = p_hat * z_hat, held constant across the whole requested
horizon, for the same reason Croston's forecast is flat: neither method
models any shape within the horizon.

Both smoothed states are seeded at exactly 0.0 before the first
observation (no empirical warm-up window) -- a plain, documented
initial condition, not fitted or tuned. One direct consequence of that
choice: on an all-zero training series, d_t = 0 and no occurrence ever
happens, so p_hat stays at 0.0 for every period (each update is
`alpha*0 + (1-alpha)*0`) and z_hat is never touched, staying at its 0.0
seed. The forecast is therefore a well-defined 0.0 -- TSB degrades
gracefully here, by construction, unlike
`src.models.croston.CrostonForecaster.fit()`, which raises on the same
input because Croston's ratio has no basis to compute at all. This is a
genuine mathematical difference between the two methods' update rules,
not an inconsistency between these two wrappers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster

#: Fixed smoothing parameters for this first Croston/TSB comparison --
#: literature-standard defaults (Teunter, Syntetos & Babai 2011 evaluate
#: a similar low-alpha/low-beta range), not tuned on any fold's forecast
#: window.
DEFAULT_ALPHA = 0.1  # occurrence-probability smoothing weight
DEFAULT_BETA = 0.1    # demand-size smoothing weight


class TSBForecaster(BaseForecaster):
    """TSB: forecast = smoothed_occurrence_probability * smoothed_demand_size, held flat."""

    def __init__(self, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA):
        if not np.isfinite(alpha) or not (0 < alpha <= 1):
            raise ValueError(f"alpha must be a finite value in (0, 1], got {alpha!r}.")
        if not np.isfinite(beta) or not (0 < beta <= 1):
            raise ValueError(f"beta must be a finite value in (0, 1], got {beta!r}.")
        self.alpha = alpha
        self.beta = beta
        self.name = f"TSB(alpha={alpha},beta={beta})"
        self._forecast_value: Optional[float] = None
        self._p_hat: Optional[float] = None  # smoothed occurrence probability
        self._z_hat: Optional[float] = None  # smoothed demand size
        self._n_obs_fit: Optional[int] = None
        self._n_occurrences: Optional[int] = None

    def fit(self, history: pd.Series) -> "TSBForecaster":
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

        alpha, beta = self.alpha, self.beta
        p_hat = 0.0  # documented 0.0 seed -- see module docstring
        z_hat = 0.0
        n_occurrences = 0

        for y_t in values:
            if y_t > 0:
                n_occurrences += 1
                z_hat = beta * float(y_t) + (1 - beta) * z_hat
                d_t = 1.0
            else:
                d_t = 0.0
            p_hat = alpha * d_t + (1 - alpha) * p_hat  # every period, including zeros

        self._p_hat = p_hat
        self._z_hat = z_hat
        self._forecast_value = p_hat * z_hat
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

        `probability_smoothed` is TSB's own estimate of per-period
        occurrence probability (not a 0/1 classification) -- on a
        genuinely all-zero training window it is exactly 0.0, which is
        what makes `forecast_value` a defined 0.0 rather than NaN (see
        module docstring).
        """
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "n_obs_fit": self._n_obs_fit,
            "n_occurrences": self._n_occurrences,
            "probability_smoothed": self._p_hat,
            "demand_size_smoothed": self._z_hat,
            "forecast_value": self._forecast_value,
        }
