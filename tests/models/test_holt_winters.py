"""Unit tests for src/models/holt_winters.py (HoltWintersForecaster).

One test in particular (`test_fits_cleanly_on_series_with_exact_zeros`)
is the direct check behind this module's whole "additive-only, never
multiplicative" design decision: the real CA_1/FOODS_3 series has exact
zero-sales days (Christmas closures), and multiplicative Holt-Winters
cannot fit through a zero at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.holt_winters import DEFAULT_TREND, SEASONAL_PERIODS, HoltWintersForecaster


def _history(n: int = 150, seed: int = 0, with_zeros: bool = False) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)
    values = 20 + weekly + rng.normal(scale=2.0, size=n)
    values = np.clip(values, 0, None)
    if with_zeros:
        # Simulate a handful of true zero-sales days (store closures).
        values[::37] = 0.0
    return pd.Series(values, index=dates)


def test_default_trend_is_add():
    assert DEFAULT_TREND == "add"
    assert HoltWintersForecaster().trend == "add"


def test_seasonal_periods_defaults_to_weekly():
    assert SEASONAL_PERIODS == 7
    assert HoltWintersForecaster().seasonal_periods == 7


def test_explicit_trend_none_is_not_overridden_by_default():
    # The whole point of the _UNSET sentinel: trend=None must mean "no
    # trend component", not "trend not specified, use DEFAULT_TREND".
    model = HoltWintersForecaster(trend=None)
    assert model.trend is None
    assert "none" in model.name


def test_explicit_trend_add():
    model = HoltWintersForecaster(trend="add")
    assert model.trend == "add"
    assert "add" in model.name


def test_invalid_trend_raises():
    with pytest.raises(ValueError):
        HoltWintersForecaster(trend="multiplicative")


def test_fit_predict_returns_correct_shape_and_finite_values():
    for trend in ("add", None):
        model = HoltWintersForecaster(trend=trend).fit(_history())
        forecast = model.predict(28)
        assert forecast.shape == (28,)
        assert np.all(np.isfinite(forecast))


def test_fits_cleanly_on_series_with_exact_zeros():
    # This is the regression guard for the additive-only design choice:
    # a multiplicative seasonal/trend component cannot fit through a
    # true zero. Additive must handle it without raising.
    history = _history(n=150, with_zeros=True)
    assert (history == 0).any()
    model = HoltWintersForecaster(trend="add").fit(history)
    forecast = model.predict(28)
    assert np.all(np.isfinite(forecast))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        HoltWintersForecaster().predict(28)


def test_fit_diagnostics_after_fit():
    history = _history()
    model = HoltWintersForecaster(trend="add").fit(history)
    diag = model.fit_diagnostics()
    assert isinstance(diag["aic"], float)
    assert diag["n_obs_fit"] == len(history)
    assert diag["trend"] == "add"
    assert diag["seasonal"] == "add"
    assert diag["seasonal_periods"] == 7
    assert diag["damped_trend"] is False


def test_fit_diagnostics_residual_and_dof_fields_phase2_step6():
    # Phase 2 Step 6: training-residual / DOF / convergence fields.
    # Holt-Winters has no SARIMAX-style burn-in artifact (see fit()),
    # so loglikelihood_burn is always 0 and resid_diagnostic == resid.
    history = _history()
    model = HoltWintersForecaster(trend="add").fit(history)
    diag = model.fit_diagnostics()

    assert isinstance(diag["resid"], np.ndarray)
    assert len(diag["resid"]) == len(history)
    assert np.all(np.isfinite(diag["resid"]))

    assert diag["loglikelihood_burn"] == 0
    assert np.array_equal(diag["resid_diagnostic"], diag["resid"])

    # k = m*has_seasonal + 2*has_trend + 2 + damped = 7*1 + 2*1 + 2 + 0 = 11
    # (statsmodels' own AIC parameter count -- see fit()'s docstring for
    # why this is one less than len(results.params_formatted)).
    assert diag["n_params"] == 11
    assert diag["nobs"] == len(history)
    assert diag["df_resid"] == len(diag["resid"]) - diag["n_params"]

    assert diag["converged"] in (True, False, None)
    assert isinstance(diag["params"], dict)


def test_fit_diagnostics_n_params_reflects_trend_none_too():
    # trend=None drops both the trend smoothing param (beta) and the
    # initial_trend state relative to trend="add": k = 7*1 + 2*0 + 2 + 0 = 9.
    history = _history()
    model = HoltWintersForecaster(trend=None).fit(history)
    diag = model.fit_diagnostics()
    assert diag["n_params"] == 9
    assert diag["loglikelihood_burn"] == 0


def test_trend_add_and_trend_none_are_independent_fits():
    history = _history()
    add_model = HoltWintersForecaster(trend="add").fit(history)
    none_model = HoltWintersForecaster(trend=None).fit(history)
    assert add_model._results is not none_model._results
    # Not required to differ numerically, but must be independently
    # fitted objects -- same assertion style as the ARIMA/SARIMA
    # independence test.
    assert add_model.fit_diagnostics()["trend"] == "add"
    assert none_model.fit_diagnostics()["trend"] is None
