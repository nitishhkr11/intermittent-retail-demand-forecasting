"""Holt-Winters / Exponential Smoothing forecaster wrapper (statsmodels).

Additive seasonal component only (`seasonal="add"`) -- never
multiplicative. The CA_1/FOODS_3 aggregate series has exact zero-sales
days (Walmart store closures each Christmas), and
`statsmodels.tsa.holtwinters.ExponentialSmoothing`'s multiplicative
trend/seasonal components require strictly positive data throughout the
fitted history; multiplicative is mathematically incompatible with a
series that contains true zeros, not merely a stylistic preference.

Whether to add a trend term (`trend="add"`) or omit one (`trend=None`)
is treated as a model-development choice, per the Step 5 brief: both are
compared empirically using mean MAE/RMSE on the same 5 development
backtesting folds every other model in this step is evaluated on --
never the final holdout -- and the loser is dropped. That comparison
itself lives in notebooks/04_classical_models.ipynb (Holt-Winters
section), not in this module; `DEFAULT_TREND` below just records its
outcome, so later code that wants "the Holt-Winters model" without
specifying `trend` gets the winner.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Weekly seasonality for daily retail data, same period as SeasonalNaive/SARIMA.
SEASONAL_PERIODS = 7

#: Winner of the trend='add' vs trend=None comparison in
#: notebooks/04_classical_models.ipynb (Holt-Winters section, development
#: folds only). Used whenever `trend` isn't passed explicitly.
DEFAULT_TREND: Optional[str] = "add"

#: Sentinel so `HoltWintersForecaster(trend=None)` (explicitly no trend)
#: is distinguishable from "trend not specified, use DEFAULT_TREND".
_UNSET = object()


class HoltWintersForecaster(BaseForecaster):
    """Additive-seasonal Holt-Winters exponential smoothing, weekly seasonality.

    `trend`: `"add"` for an additive trend component, `None` for none.
    Defaults to `DEFAULT_TREND` (the development-fold comparison winner)
    when not given explicitly -- pass `trend="add"` or `trend=None`
    directly to force one side of that comparison, as the notebook does
    when running both.
    """

    def __init__(self, trend: Optional[str] = _UNSET, seasonal_periods: int = SEASONAL_PERIODS):  # type: ignore[assignment]
        if trend is _UNSET:
            trend = DEFAULT_TREND
        if trend not in (None, "add"):
            raise ValueError(f"trend must be 'add' or None, got {trend!r}.")
        self.trend = trend
        self.seasonal_periods = seasonal_periods
        trend_tag = self.trend if self.trend is not None else "none"
        self.name = f"HoltWinters(trend={trend_tag})"
        self._results = None
        self.aic: Optional[float] = None
        self.n_obs_fit: Optional[int] = None
        self.convergence_warnings: List[str] = []
        # Phase 2 Step 6 -- residual/DOF bookkeeping, populated by fit().
        self._resid: Optional[np.ndarray] = None
        self.n_params: Optional[int] = None
        self.nobs: Optional[int] = None
        self.loglikelihood_burn: int = 0
        self.converged: Optional[bool] = None
        self.n_iterations: Optional[int] = None
        self.optimizer_message: Optional[str] = None

    def fit(self, history: pd.Series) -> "HoltWintersForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self.convergence_warnings = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ExponentialSmoothing(
                history,
                trend=self.trend,
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                damped_trend=False,
                initialization_method="estimated",
            )
            self._results = model.fit(optimized=True)
            self.convergence_warnings = [
                f"{w.category.__name__}: {w.message}" for w in caught
            ]

        self.aic = float(self._results.aic)
        self.n_obs_fit = int(len(history))

        # Phase 2 Step 6: training (in-sample) residuals and DOF
        # bookkeeping. Unlike SARIMAX, statsmodels' ETS fit here uses
        # `initialization_method="estimated"`: all initial states are
        # free parameters solved for directly, not a diffuse Kalman
        # prior, so there is no equivalent of SARIMAX's
        # `loglikelihood_burn` artifact -- confirmed empirically (the
        # first several residuals on a real development fold are the
        # same order of magnitude as the rest, unlike SARIMAX's). It is
        # kept at 0 here, rather than omitted, so the same
        # burn-in-trimming code path in src.evaluation.residuals works
        # unchanged across all three model families.
        #
        # `n_params` uses `HoltWintersResults.k`, statsmodels' own
        # parameter count (source: statsmodels/tsa/holtwinters/model.py,
        # `k = m*has_seasonal + 2*has_trend + 2 + damped`). This is NOT
        # the same as the number of rows in `.params_formatted` (12 rows
        # for trend="add" here, vs. k=11): `params_formatted` lists all
        # `seasonal_periods` initial-seasonal values, but only
        # `seasonal_periods - 1` of them are independently identified --
        # the standard constraint for an additive seasonal decomposition,
        # under which the seasonal initial states and the initial level
        # are jointly identified only up to one shared degree of freedom.
        # This discrepancy (11 vs 12) is exactly the Holt-Winters
        # residual-DOF limitation Step 6 asks to be documented, not
        # silently resolved: `n_params=k` is statsmodels' own convention
        # for AIC/BIC, used here for consistency, but it is a convention,
        # not a value every textbook would agree is "the" parameter count.
        self._resid = np.asarray(self._results.resid, dtype=float)
        self.n_params = int(self._results.k)
        self.nobs = self.n_obs_fit
        mle_retvals = getattr(self._results, "mle_retvals", None)
        if mle_retvals is not None and hasattr(mle_retvals, "success"):
            self.converged = bool(mle_retvals.success)
        else:
            self.converged = None
        self.n_iterations = getattr(mle_retvals, "nit", None) if mle_retvals is not None else None
        self.optimizer_message = getattr(mle_retvals, "message", None) if mle_retvals is not None else None

        if self.convergence_warnings:
            logger.warning(
                "[%s] fit on %d obs raised %d warning(s): %s",
                self.name,
                self.n_obs_fit,
                len(self.convergence_warnings),
                "; ".join(self.convergence_warnings),
            )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self._results is None:
            raise RuntimeError(f"{self.name}.predict() called before fit().")
        return np.asarray(self._results.forecast(horizon), dtype=float)

    def fit_diagnostics(self) -> Dict[str, Any]:
        """AIC, training-window size, and any warnings from the most recent fit.

        As with ARIMA/SARIMA, `aic` is only meaningful alongside
        `n_obs_fit` and is not comparable across folds of different
        training-window sizes.

        Phase 2 Step 6 additions:
        - `resid`/`resid_diagnostic`: identical in this case (training
          residuals, with no burn-in trim -- see `fit()` for why
          Holt-Winters has no SARIMAX-style burn-in artifact to trim).
        - `n_params`: `HoltWintersResults.k`, statsmodels' own parameter
          count. Documented limitation (see `fit()`): this treats only
          `seasonal_periods - 1` of the `seasonal_periods` initial
          seasonal states as independently free, under the standard
          additive-seasonal identifiability constraint, so it is one
          less than the number of rows in `.params_formatted`. It is a
          convention, not a universally-agreed figure -- used here only
          for consistency with statsmodels' own AIC/BIC computation.
        - `nobs`/`df_resid`: `nobs` is the training-window length
          (HoltWintersResults has no native `.nobs`); `df_resid =
          nobs - n_params`.
        - `converged`/`n_iterations`/`optimizer_message`: from the
          underlying scipy optimizer's own `OptimizeResult`
          (`mle_retvals.success`/`.nit`/`.message`), added for parity
          with ARIMA/SARIMA's `converged` field.
        """
        resid_diagnostic = (
            self._resid[self.loglikelihood_burn :] if self._resid is not None else None
        )
        df_resid = (
            int(len(resid_diagnostic) - self.n_params)
            if resid_diagnostic is not None and self.n_params is not None
            else None
        )
        return {
            "trend": self.trend,
            "seasonal": "add",
            "seasonal_periods": self.seasonal_periods,
            "damped_trend": False,
            "aic": self.aic,
            "n_obs_fit": self.n_obs_fit,
            "convergence_warnings": list(self.convergence_warnings),
            "resid": self._resid,
            "resid_diagnostic": resid_diagnostic,
            "loglikelihood_burn": self.loglikelihood_burn,
            "n_params": self.n_params,
            "nobs": self.nobs,
            "df_resid": df_resid,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "optimizer_message": self.optimizer_message,
            "params": dict(self._results.params) if self._results is not None else None,
        }
