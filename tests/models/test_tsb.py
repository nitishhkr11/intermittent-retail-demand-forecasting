"""Unit tests for src/models/tsb.py (TSBForecaster).

Uses small synthetic series (not the real project data -- that's what
the notebook-execution test is for). The all-zero-series tests are this
module's key regression guard: TSB must degrade gracefully to a defined
0.0 forecast in exactly the situation where
src.models.croston.CrostonForecaster is required to raise -- see
tests/models/test_croston.py::test_all_zero_training_series_raises_value_error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tsb import DEFAULT_ALPHA, DEFAULT_BETA, TSBForecaster


def _series(values, start="2020-01-01") -> pd.Series:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=dates)


def test_default_alpha_and_beta_are_0_1():
    assert DEFAULT_ALPHA == 0.1
    assert DEFAULT_BETA == 0.1
    model = TSBForecaster()
    assert model.alpha == 0.1
    assert model.beta == 0.1


# ---- correct prediction length / non-negative output ------------------------------

def test_predict_returns_correct_length_and_finite_values():
    model = TSBForecaster().fit(_series([0, 3, 0, 0, 5, 0, 2, 0, 0, 4]))
    forecast = model.predict(28)
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (28,)
    assert np.all(np.isfinite(forecast))


def test_forecast_is_a_flat_line_across_the_horizon():
    model = TSBForecaster().fit(_series([0, 3, 0, 0, 5, 0, 2, 0, 0, 4]))
    forecast = model.predict(14)
    assert np.all(forecast == forecast[0])


def test_forecast_is_non_negative_for_non_negative_input():
    # probability p_hat in [0, 1] (a convex combination of 0/1
    # indicators starting from a 0.0 seed) times a demand size z_hat >= 0
    # (a convex combination of non-negative sizes starting from a 0.0
    # seed) can never go negative.
    rng = np.random.default_rng(0)
    values = rng.integers(0, 6, size=200).astype(float)
    model = TSBForecaster().fit(_series(values))
    forecast = model.predict(28)
    assert np.all(forecast >= 0)


# ---- hand-computed correctness -----------------------------------------------------

def test_two_occurrences_match_hand_computed_value():
    # history = [0, 0, 4, 0, 2], alpha=beta=0.1, both states seeded at 0.0:
    #   t1 (y=0): d=0, p=0.1*0+0.9*0=0.0
    #   t2 (y=0): d=0, p=0.0
    #   t3 (y=4): z=0.1*4+0.9*0=0.4,  d=1, p=0.1*1+0.9*0=0.1
    #   t4 (y=0): d=0, p=0.1*0+0.9*0.1=0.09
    #   t5 (y=2): z=0.1*2+0.9*0.4=0.56, d=1, p=0.1*1+0.9*0.09=0.181
    #   forecast = p * z = 0.181 * 0.56
    model = TSBForecaster(alpha=0.1, beta=0.1).fit(_series([0, 0, 4, 0, 2]))
    diag = model.fit_diagnostics()
    assert diag["probability_smoothed"] == pytest.approx(0.181)
    assert diag["demand_size_smoothed"] == pytest.approx(0.56)
    assert model.predict(1)[0] == pytest.approx(0.181 * 0.56)


def test_leading_zeros_still_update_probability_every_period():
    # TSB's defining difference from Croston: probability is smoothed
    # EVERY period, including zero-demand ones, not just at occurrences.
    # A long, all-zero prefix should decay p_hat toward 0 even before
    # the first occurrence (there's nothing to decay from here since it
    # starts at 0.0, but this checks p_hat stays exactly 0.0 through a
    # zero-only prefix rather than drifting).
    values = [0.0] * 20 + [5.0]
    model = TSBForecaster(alpha=0.2, beta=0.2).fit(_series(values))
    diag = model.fit_diagnostics()
    # 20 zero periods: p = 0.2*0 + 0.8*0 = 0 every time (stays 0), z
    # untouched (stays 0). Then the one occurrence: p = 0.2*1 + 0.8*0 =
    # 0.2. TSB never special-cases a first occurrence the way Croston
    # seeds directly at the raw value -- z is smoothed from its 0.0 seed
    # even here: z = 0.2*5 + 0.8*0 = 1.0, not 5.0.
    assert diag["probability_smoothed"] == pytest.approx(0.2)
    assert diag["demand_size_smoothed"] == pytest.approx(1.0)


# ---- all-zero training series: TSB degrades gracefully to zero --------------------

def test_all_zero_training_series_forecasts_a_defined_zero():
    model = TSBForecaster().fit(_series([0.0] * 50))
    forecast = model.predict(28)
    assert np.all(np.isfinite(forecast))
    assert np.all(forecast == 0.0)
    diag = model.fit_diagnostics()
    assert diag["probability_smoothed"] == 0.0
    assert diag["demand_size_smoothed"] == 0.0
    assert diag["forecast_value"] == 0.0


def test_all_zero_training_series_does_not_raise():
    # The deliberate asymmetry with Croston: this must NOT raise.
    try:
        TSBForecaster().fit(_series([0.0] * 50))
    except ValueError:
        pytest.fail("TSBForecaster.fit() must not raise on an all-zero series")


# ---- invalid-input handling --------------------------------------------------------

def test_negative_values_raise_value_error():
    with pytest.raises(ValueError, match="[Nn]egative"):
        TSBForecaster().fit(_series([0, 3, -1, 0, 5]))


def test_nan_values_raise_value_error():
    with pytest.raises(ValueError, match="NaN"):
        TSBForecaster().fit(_series([0, 3, np.nan, 0, 5]))


def test_empty_history_raises_value_error():
    with pytest.raises(ValueError, match="non-empty"):
        TSBForecaster().fit(pd.Series([], dtype=float))


def test_horizon_zero_or_negative_raises_value_error():
    model = TSBForecaster().fit(_series([0, 3, 0, 5, 2]))
    with pytest.raises(ValueError):
        model.predict(0)
    with pytest.raises(ValueError):
        model.predict(-1)


def test_predict_before_fit_raises_runtime_error():
    with pytest.raises(RuntimeError):
        TSBForecaster().predict(28)


@pytest.mark.parametrize("bad_value", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_alpha_raises_at_construction(bad_value):
    with pytest.raises(ValueError):
        TSBForecaster(alpha=bad_value)


@pytest.mark.parametrize("bad_value", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_beta_raises_at_construction(bad_value):
    with pytest.raises(ValueError):
        TSBForecaster(beta=bad_value)


# ---- fit_diagnostics ----------------------------------------------------------------

def test_fit_diagnostics_before_fit_is_all_none():
    diag = TSBForecaster().fit_diagnostics()
    assert diag["forecast_value"] is None
    assert diag["probability_smoothed"] is None
    assert diag["demand_size_smoothed"] is None
    assert diag["n_obs_fit"] is None


def test_fit_diagnostics_after_fit_reports_n_obs_and_occurrences():
    model = TSBForecaster().fit(_series([0, 3, 0, 0, 5, 0, 2, 0, 0, 4]))
    diag = model.fit_diagnostics()
    assert diag["n_obs_fit"] == 10
    assert diag["n_occurrences"] == 4
    assert diag["alpha"] == 0.1
    assert diag["beta"] == 0.1
