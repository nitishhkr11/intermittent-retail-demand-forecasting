"""Unit tests for src/models/croston.py (CrostonForecaster).

Uses small synthetic series (not the real project data -- that's what
the notebook-execution test is for) so these stay fast and independent
of any data file being present. Several tests hand-compute the exact
expected smoothed state / forecast value, not just "did it run" --
Croston's recursion is simple enough that an independent-by-hand
calculation is a meaningful regression guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.croston import DEFAULT_ALPHA, CrostonForecaster


def _series(values, start="2020-01-01") -> pd.Series:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=dates)


def test_default_alpha_is_0_1():
    assert DEFAULT_ALPHA == 0.1
    assert CrostonForecaster().alpha == 0.1


# ---- correct prediction length / non-negative output ------------------------------

def test_predict_returns_correct_length_and_finite_values():
    model = CrostonForecaster().fit(_series([0, 3, 0, 0, 5, 0, 2, 0, 0, 4]))
    forecast = model.predict(28)
    assert isinstance(forecast, np.ndarray)
    assert forecast.shape == (28,)
    assert np.all(np.isfinite(forecast))


def test_forecast_is_a_flat_line_across_the_horizon():
    # Croston has no within-horizon shape -- every forecasted day is the
    # same smoothed_size / smoothed_interval ratio, by construction.
    model = CrostonForecaster().fit(_series([0, 3, 0, 0, 5, 0, 2, 0, 0, 4]))
    forecast = model.predict(14)
    assert np.all(forecast == forecast[0])


def test_forecast_is_non_negative_for_non_negative_input():
    # z_hat is a convex combination of non-negative demand sizes (always
    # >= 0) and q_hat is always a positive count of elapsed periods
    # (always > 0) once any occurrence has been seen -- so the ratio can
    # never go negative, for any non-negative input series.
    rng = np.random.default_rng(0)
    values = rng.integers(0, 6, size=200).astype(float)  # zero-heavy, never negative
    model = CrostonForecaster().fit(_series(values))
    forecast = model.predict(28)
    assert np.all(forecast >= 0)


# ---- hand-computed correctness (leading zeros naturally absorbed) -----------------

def test_leading_zeros_and_two_occurrences_match_hand_computed_value():
    # history = [0, 0, 4, 0, 2], alpha=0.1:
    #   occurrence 1 at period 3 (2 leading zeros absorbed into the count):
    #     z_hat=4.0 (seed), q_hat=3.0 (seed)
    #   occurrence 2 at period 5 (2 periods after occurrence 1):
    #     z_hat = 0.1*2 + 0.9*4.0 = 3.8
    #     q_hat = 0.1*2 + 0.9*3.0 = 2.9
    #   forecast = 3.8 / 2.9
    model = CrostonForecaster(alpha=0.1).fit(_series([0, 0, 4, 0, 2]))
    diag = model.fit_diagnostics()
    assert diag["demand_size_smoothed"] == pytest.approx(3.8)
    assert diag["interval_smoothed"] == pytest.approx(2.9)
    assert model.predict(1)[0] == pytest.approx(3.8 / 2.9)


def test_single_occurrence_at_the_very_end_absorbs_all_leading_zeros():
    # A single occurrence at the last of 10 periods: the "interval" is
    # seeded directly at 10 (the 1-based position), with no special
    # casing of the 9 leading zeros that preceded it.
    values = [0.0] * 9 + [7.0]
    model = CrostonForecaster().fit(_series(values))
    diag = model.fit_diagnostics()
    assert diag["demand_size_smoothed"] == pytest.approx(7.0)
    assert diag["interval_smoothed"] == pytest.approx(10.0)
    assert diag["n_occurrences"] == 1


# ---- all-zero training series: Croston is undefined, must raise -------------------

def test_all_zero_training_series_raises_value_error():
    with pytest.raises(ValueError, match="all-zero"):
        CrostonForecaster().fit(_series([0.0] * 50))


# ---- invalid-input handling --------------------------------------------------------

def test_negative_values_raise_value_error():
    with pytest.raises(ValueError, match="[Nn]egative"):
        CrostonForecaster().fit(_series([0, 3, -1, 0, 5]))


def test_nan_values_raise_value_error():
    with pytest.raises(ValueError, match="NaN"):
        CrostonForecaster().fit(_series([0, 3, np.nan, 0, 5]))


def test_empty_history_raises_value_error():
    with pytest.raises(ValueError, match="non-empty"):
        CrostonForecaster().fit(pd.Series([], dtype=float))


def test_horizon_zero_or_negative_raises_value_error():
    model = CrostonForecaster().fit(_series([0, 3, 0, 5, 2]))
    with pytest.raises(ValueError):
        model.predict(0)
    with pytest.raises(ValueError):
        model.predict(-1)


def test_predict_before_fit_raises_runtime_error():
    with pytest.raises(RuntimeError):
        CrostonForecaster().predict(28)


@pytest.mark.parametrize("bad_alpha", [0.0, -0.1, 1.1, float("nan")])
def test_invalid_alpha_raises_at_construction(bad_alpha):
    with pytest.raises(ValueError):
        CrostonForecaster(alpha=bad_alpha)


def test_alpha_of_exactly_1_is_allowed():
    # (0, 1] -- 1.0 is a valid (if extreme: no smoothing memory) choice.
    model = CrostonForecaster(alpha=1.0).fit(_series([0, 3, 0, 5, 2]))
    assert np.isfinite(model.predict(1)[0])


# ---- fit_diagnostics ----------------------------------------------------------------

def test_fit_diagnostics_before_fit_is_all_none():
    diag = CrostonForecaster().fit_diagnostics()
    assert diag["forecast_value"] is None
    assert diag["demand_size_smoothed"] is None
    assert diag["interval_smoothed"] is None
    assert diag["n_obs_fit"] is None
    assert diag["n_occurrences"] is None


def test_fit_diagnostics_after_fit_reports_n_obs_and_occurrences():
    model = CrostonForecaster().fit(_series([0, 3, 0, 0, 5, 0, 2, 0, 0, 4]))
    diag = model.fit_diagnostics()
    assert diag["n_obs_fit"] == 10
    assert diag["n_occurrences"] == 4
    assert diag["alpha"] == 0.1


def test_refit_on_different_history_updates_diagnostics():
    model = CrostonForecaster()
    model.fit(_series([0, 3, 0, 5, 2]))
    first_n = model.fit_diagnostics()["n_obs_fit"]
    model.fit(_series([0, 3, 0, 0, 5, 0, 2, 0, 0, 4]))
    second_n = model.fit_diagnostics()["n_obs_fit"]
    assert first_n == 5
    assert second_n == 10
