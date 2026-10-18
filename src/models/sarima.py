"""SARIMA forecaster wrapper (statsmodels).

Seasonal ARIMA(p, d, q)(P, D, Q)_m, fit via the same
`statsmodels.tsa.statespace.sarimax.SARIMAX` class as
`src/models/arima.py`'s non-seasonal ARIMA -- deliberately the same
underlying implementation/optimizer/AIC computation, so the two are
directly comparable to each other, while each remains its own separate
`.fit()` call on its own separate model specification.

This separateness is the concrete answer to "fit non-seasonal ARIMA and
seasonal SARIMA separately, don't represent them as two independently
selected models from the same fitted object": `SARIMAForecaster` here
and `src.models.arima.ARIMAForecaster` are two distinct classes, each
producing its own distinct fitted `SARIMAXResults` object. Neither
reuses `pmdarima.auto_arima`'s combined fitted object from Phase 2 Step
4 (which returns `order` and `seasonal_order` together from one search).

`DEFAULT_ORDER = (6, 1, 1)` with `DEFAULT_SEASONAL_ORDER = (0, 0, 2, 7)`
is the Step 4 auto_arima candidate, fixed here per the Step 5 brief --
this step fits and evaluates that candidate, it does not re-run an order
search.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: (p, d, q) -- same non-seasonal order as src.models.arima.DEFAULT_ORDER.
DEFAULT_ORDER: Tuple[int, int, int] = (6, 1, 1)
#: (P, D, Q, m) -- Phase 2 Step 4's auto_arima seasonal candidate, m=7 (weekly).
DEFAULT_SEASONAL_ORDER: Tuple[int, int, int, int] = (0, 0, 2, 7)


class SARIMAForecaster(BaseForecaster):
    """Seasonal ARIMA, fit fresh on each backtest fold's training window.

    Fit under statsmodels' default `enforce_stationarity=True,
    enforce_invertibility=True`, same reasoning as `ARIMAForecaster`: the
    fitted process must actually satisfy the stationary/invertible
    definition of SARIMA rather than relaxing those constraints to avoid
    convergence trouble. `maxiter=500` (statsmodels' default optimizer,
    L-BFGS, defaults to 50) is used for the same empirical reason as
    `ARIMAForecaster`: at the default of 50, this series' largest fold
    fails to converge (`mle_retvals["converged"] is False`); at 500 it
    does. Raising the cap gives the optimizer a fair chance to actually
    reach the optimum, it does not redefine convergence. Warnings raised
    while fitting are captured, not suppressed, and
    `fit_diagnostics()["converged"]` records the optimizer's own verdict
    per fold -- some folds may still not converge even at 500, and that
    is reported rather than hidden. A fold on which fitting fails
    outright is left to raise, so
    `src.evaluation.backtesting.run_backtest` can catch and record it as
    a failed fold.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = DEFAULT_ORDER,
        seasonal_order: Tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.name = f"SARIMA{tuple(order)}x{tuple(seasonal_order)}"
        self._results = None
        self.aic: Optional[float] = None
        self.n_obs_fit: Optional[int] = None
        self.converged: Optional[bool] = None
        self.convergence_warnings: List[str] = []
        # Phase 2 Step 6 -- residual/DOF bookkeeping, populated by fit().
        self._resid: Optional[np.ndarray] = None
        self.n_params: Optional[int] = None
        self.nobs: Optional[int] = None
        self.loglikelihood_burn: Optional[int] = None
        self.n_iterations: Optional[int] = None
        self.warnflag: Optional[int] = None
        self.optimizer: Optional[str] = None

    def fit(self, history: pd.Series) -> "SARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.convergence_warnings = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = SARIMAX(
                history,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            self._results = model.fit(disp=False, maxiter=500)
            self.convergence_warnings = [
                f"{w.category.__name__}: {w.message}" for w in caught
            ]

        self.aic = float(self._results.aic)
        self.n_obs_fit = int(len(history))
        self.converged = self._results.mle_retvals.get("converged")

        # Phase 2 Step 6: see ARIMAForecaster.fit() for the full rationale
        # (same SARIMAX machinery) -- training residuals, DOF bookkeeping
        # (df_resid computed manually since df_resid is inf on this model
        # class), and the optimizer's own convergence evidence beyond the
        # boolean `converged` flag. This is the exact mechanism behind
        # the fold-3 non-convergence investigation (Step 6, section 3):
        # `n_iterations`/`warnflag`/`optimizer` below are what the SARIMA
        # convergence table in the notebook reads.
        self._resid = np.asarray(self._results.resid, dtype=float)
        self.n_params = int(len(self._results.params))
        self.nobs = int(self._results.nobs)
        self.loglikelihood_burn = int(self._results.loglikelihood_burn)
        self.n_iterations = self._results.mle_retvals.get("iterations")
        self.warnflag = self._results.mle_retvals.get("warnflag")
        self.optimizer = self._results.mle_settings.get("optimizer")

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
        forecast = self._results.get_forecast(steps=horizon)
        return np.asarray(forecast.predicted_mean, dtype=float)

    def fit_diagnostics(self) -> Dict[str, Any]:
        """AIC, training-window size, and any warnings from the most recent fit.

        As with ARIMA, `aic` is only meaningful alongside `n_obs_fit`:
        values from different folds' training-window sizes are not
        directly comparable, and a lower AIC does not by itself imply
        better out-of-sample forecast accuracy.

        Phase 2 Step 6 additions -- see ARIMAForecaster.fit_diagnostics()
        for the full rationale (identical SARIMAX machinery, same DOF/
        burn-in reasoning): `resid` (full training residuals),
        `resid_diagnostic` (burn-in-trimmed, for ACF/Ljung-Box/
        histogram), `loglikelihood_burn`, `n_params`, `nobs`, `df_resid`
        (computed manually, `SARIMAXResults.df_resid` is `inf` here),
        `n_iterations`/`warnflag`/`optimizer` (the optimizer's own
        convergence evidence -- the primary data this project's fold-3
        non-convergence investigation reads), and `params` (the fitted
        coefficient vector, for a direct finiteness check).
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
            "order": tuple(self.order),
            "seasonal_order": tuple(self.seasonal_order),
            "enforce_stationarity": True,
            "enforce_invertibility": True,
            "maxiter": 500,
            "aic": self.aic,
            "n_obs_fit": self.n_obs_fit,
            "converged": self.converged,
            "convergence_warnings": list(self.convergence_warnings),
            "resid": self._resid,
            "resid_diagnostic": resid_diagnostic,
            "loglikelihood_burn": self.loglikelihood_burn,
            "n_params": self.n_params,
            "nobs": self.nobs,
            "df_resid": df_resid,
            "n_iterations": self.n_iterations,
            "warnflag": self.warnflag,
            "optimizer": self.optimizer,
            "params": dict(self._results.params) if self._results is not None else None,
        }
