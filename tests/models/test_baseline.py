"""Unit tests for src/models/baseline.py (SeasonalNaive)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import SeasonalNaive


def _history(values) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates)


def test_predict_within_one_season_repeats_last_week_exactly():
    # 14 days of history: second week is [7..13], should be forecast verbatim.
    history = _history(list(range(14)))
    model = SeasonalNaive(season_length=7).fit(history)
    forecast = model.predict(7)
    np.testing.assert_array_equal(forecast, np.array([7, 8, 9, 10, 11, 12, 13], dtype=float))


def test_predict_28_day_horizon_tiles_last_week_four_times():
    history = _history(list(range(7)))  # last (and only) week: 0..6
    model = SeasonalNaive(season_length=7).fit(history)
    forecast = model.predict(28)
    assert len(forecast) == 28
    expected = np.tile(np.array([0, 1, 2, 3, 4, 5, 6], dtype=float), 4)
    np.testing.assert_array_equal(forecast, expected)


def test_predict_horizon_not_a_multiple_of_season_length_truncates_cleanly():
    history = _history(list(range(7)))
    model = SeasonalNaive(season_length=7).fit(history)
    forecast = model.predict(10)
    assert len(forecast) == 10
    np.testing.assert_array_equal(forecast, np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2], dtype=float))


def test_fit_raises_when_history_shorter_than_season_length():
    history = _history([1, 2, 3])
    with pytest.raises(ValueError):
        SeasonalNaive(season_length=7).fit(history)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SeasonalNaive().predict(7)


def test_name_is_stable():
    assert SeasonalNaive().name == "SeasonalNaive"


def test_only_uses_most_recent_season_not_earlier_history():
    # If it accidentally averaged/used the whole history instead of just
    # the last 7 observations, this would NOT reproduce the last week exactly.
    history = _history([100, 100, 100, 100, 100, 100, 100] + list(range(7)))
    model = SeasonalNaive(season_length=7).fit(history)
    forecast = model.predict(7)
    np.testing.assert_array_equal(forecast, np.array([0, 1, 2, 3, 4, 5, 6], dtype=float))
