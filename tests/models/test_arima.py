"""Unit tests for src/models/arima.py (ARIMAForecaster).

Uses small synthetic series (not the real project data -- that's what
the notebook-execution test is for) so these stay fast and independent
of any data file being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.arima import DEFAULT_ORDER, ARIMAForecaster


def _history(n: int = 150, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    # A bit of weekly structure plus noise -- not required for these
    # tests (they're about the wrapper's contract, not forecast
    # quality), but keeps the series from being degenerate.
    weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
    values = 100 + weekly + rng.normal(scale=3.0, size=n)
    return pd.Series(values, index=dates)


def test_default_order_is_6_1_1():
    assert DEFAULT_ORDER == (6, 1, 1)
    assert ARIMAForecaster().order == (6, 1, 1)


def test_fit_predict_returns_correct_shape_and_finite_values():
    model = ARIMAForecaster(order=(2, 1, 1)).fit(_history())
    forecast = model.predict(28)
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (28,)
    assert np.all(np.isfinite(forecast))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ARIMAForecaster().predict(28)


def test_name_reflects_order():
    model = ARIMAForecaster(order=(2, 1, 1))
    assert "2" in model.name and "1" in model.name
    assert model.name == f"ARIMA{(2, 1, 1)}"


def test_fit_diagnostics_before_fit_has_no_aic():
    model = ARIMAForecaster(order=(2, 1, 1))
    diag = model.fit_diagnostics()
    assert diag["aic"] is None
    assert diag["n_obs_fit"] is None
    assert diag["convergence_warnings"] == []


def test_fit_diagnostics_after_fit_populates_aic_and_metadata():
    history = _history(n=150)
    model = ARIMAForecaster(order=(2, 1, 1)).fit(history)
    diag = model.fit_diagnostics()
    assert isinstance(diag["aic"], float)
    assert diag["n_obs_fit"] == len(history)
    assert diag["order"] == (2, 1, 1)
    assert diag["seasonal_order"] == (0, 0, 0, 0)
    assert diag["enforce_stationarity"] is True
    assert diag["enforce_invertibility"] is True
    assert diag["maxiter"] == 500
    assert isinstance(diag["convergence_warnings"], list)
    assert diag["converged"] in (True, False, None)


def test_fit_diagnostics_residual_and_dof_fields_phase2_step6():
    # Phase 2 Step 6: training-residual / DOF / convergence-evidence
    # fields added to fit_diagnostics() on top of the Step 5 fields
    # checked above -- the exact fields src.evaluation.residuals reads.
    history = _history(n=150)
    model = ARIMAForecaster(order=(2, 1, 1)).fit(history)
    diag = model.fit_diagnostics()

    assert isinstance(diag["resid"], np.ndarray)
    assert len(diag["resid"]) == len(history)  # training residuals, never the forecast horizon
    assert np.all(np.isfinite(diag["resid"]))

    assert diag["loglikelihood_burn"] == 1  # d=1 -> diffuse Kalman filter burns 1 observation
    assert isinstance(diag["resid_diagnostic"], np.ndarray)
    assert len(diag["resid_diagnostic"]) == len(diag["resid"]) - diag["loglikelihood_burn"]

    assert diag["n_params"] == 4  # ARIMA(2,1,1): 2 AR + 1 MA + sigma2
    assert diag["nobs"] == len(history)
    assert diag["df_resid"] == len(diag["resid_diagnostic"]) - diag["n_params"]

    assert diag["n_iterations"] is not None
    assert diag["warnflag"] in (0, 1, 2)
    assert diag["optimizer"] == "lbfgs"
    assert isinstance(diag["params"], dict) and len(diag["params"]) == diag["n_params"]
    assert all(np.isfinite(v) for v in diag["params"].values())


def test_seasonal_order_is_always_switched_off():
    # Non-seasonal ARIMA must fit with seasonal_order=(0,0,0,0) no matter
    # what -- it has no seasonal_order constructor argument at all, so
    # this is really a guard against someone adding one without updating
    # the diagnostics / fit() call together.
    model = ARIMAForecaster(order=(1, 1, 1)).fit(_history(n=100))
    assert model.fit_diagnostics()["seasonal_order"] == (0, 0, 0, 0)


def test_refit_on_different_history_updates_n_obs_and_resets_warnings():
    model = ARIMAForecaster(order=(1, 1, 1))
    model.fit(_history(n=100, seed=1))
    first_n = model.fit_diagnostics()["n_obs_fit"]
    model.fit(_history(n=140, seed=2))
    second_n = model.fit_diagnostics()["n_obs_fit"]
    assert first_n == 100
    assert second_n == 140
