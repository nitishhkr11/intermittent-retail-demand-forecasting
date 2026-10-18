"""Unit tests for src/models/sarima.py (SARIMAForecaster).

Small synthetic series, same rationale as tests/models/test_arima.py.
A couple of tests here specifically check that SARIMA is NOT the same
object/fit as ARIMA -- i.e. that the two are genuinely independent
`.fit()` calls, per the project's explicit "fit them separately" rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.arima import ARIMAForecaster
from src.models.sarima import DEFAULT_ORDER, DEFAULT_SEASONAL_ORDER, SARIMAForecaster


def _history(n: int = 150, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
    values = 100 + weekly + rng.normal(scale=3.0, size=n)
    return pd.Series(values, index=dates)


def test_default_orders_match_step4_candidate():
    assert DEFAULT_ORDER == (6, 1, 1)
    assert DEFAULT_SEASONAL_ORDER == (0, 0, 2, 7)
    model = SARIMAForecaster()
    assert model.order == (6, 1, 1)
    assert model.seasonal_order == (0, 0, 2, 7)


def test_fit_predict_returns_correct_shape_and_finite_values():
    model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 1, 7)).fit(_history())
    forecast = model.predict(28)
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (28,)
    assert np.all(np.isfinite(forecast))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SARIMAForecaster().predict(28)


def test_name_reflects_both_orders():
    model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 1, 7))
    assert model.name == f"SARIMA{(1, 1, 1)}x{(0, 0, 1, 7)}"


def test_fit_diagnostics_after_fit_reports_both_orders():
    history = _history(n=150)
    model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 1, 7)).fit(history)
    diag = model.fit_diagnostics()
    assert isinstance(diag["aic"], float)
    assert diag["n_obs_fit"] == len(history)
    assert diag["order"] == (1, 1, 1)
    assert diag["seasonal_order"] == (0, 0, 1, 7)
    assert diag["maxiter"] == 500
    assert diag["converged"] in (True, False, None)


def test_fit_diagnostics_residual_and_dof_fields_phase2_step6():
    # Phase 2 Step 6: same residual/DOF/convergence-evidence fields as
    # ARIMAForecaster (identical SARIMAX machinery) -- see
    # tests/models/test_arima.py's analogous test for the full rationale.
    history = _history(n=150)
    model = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 1, 7)).fit(history)
    diag = model.fit_diagnostics()

    assert isinstance(diag["resid"], np.ndarray)
    assert len(diag["resid"]) == len(history)
    assert np.all(np.isfinite(diag["resid"]))

    assert diag["loglikelihood_burn"] == 1
    assert len(diag["resid_diagnostic"]) == len(diag["resid"]) - diag["loglikelihood_burn"]

    assert diag["n_params"] == 4  # ar.L1, ma.L1, ma.S.L7, sigma2
    assert diag["nobs"] == len(history)
    assert diag["df_resid"] == len(diag["resid_diagnostic"]) - diag["n_params"]

    assert diag["optimizer"] == "lbfgs"
    assert isinstance(diag["params"], dict) and len(diag["params"]) == diag["n_params"]
    assert all(np.isfinite(v) for v in diag["params"].values())


def test_sarima_and_arima_are_independent_fits_not_shared_state():
    # Fit an ARIMA and a SARIMA model on the SAME history, with the SAME
    # non-seasonal (p, d, q) order, and confirm their fitted results
    # (AIC, forecasts) are each their own -- changing one must never
    # change the other. This is the concrete behavioral check behind
    # "fit non-seasonal ARIMA and seasonal SARIMA separately, never
    # treat them as two reads of one combined fitted object."
    history = _history(n=150)
    arima = ARIMAForecaster(order=(1, 1, 1)).fit(history)
    sarima = SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 1, 7)).fit(history)

    assert arima._results is not sarima._results
    # Seasonal structure means the two AICs need not match (and
    # generally won't) even with the same non-seasonal order.
    assert arima.fit_diagnostics()["seasonal_order"] == (0, 0, 0, 0)
    assert sarima.fit_diagnostics()["seasonal_order"] == (0, 0, 1, 7)


def test_seasonal_order_period_matches_weekly_seasonality():
    # m=7 is weekly seasonality for daily data -- a regression guard
    # against accidentally swapping in a different period.
    assert SARIMAForecaster().seasonal_order[-1] == 7
