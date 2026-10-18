"""Unit tests for src/evaluation/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, count_zero_actuals, mae, mape, rmse


def test_mae_zero_for_perfect_forecast():
    actual = np.array([1.0, 2.0, 3.0])
    assert mae(actual, actual) == 0.0


def test_mae_matches_hand_computed_value():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    # |2| + |2| + |3| = 7, / 3 = 2.333...
    assert mae(actual, predicted) == pytest.approx(7 / 3)


def test_rmse_zero_for_perfect_forecast():
    actual = np.array([1.0, 2.0, 3.0])
    assert rmse(actual, actual) == 0.0


def test_rmse_matches_hand_computed_value():
    actual = np.array([0.0, 0.0])
    predicted = np.array([3.0, 4.0])
    # sqrt(mean(9, 16)) = sqrt(12.5)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt(12.5))


def test_mae_rmse_raise_on_shape_mismatch():
    with pytest.raises(ValueError):
        mae(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        rmse(np.array([1.0, 2.0]), np.array([1.0]))


def test_mape_matches_hand_computed_value_with_no_zeros():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])
    # |10|/100=0.10, |20|/200=0.10 -> mean 0.10 -> 10%
    assert mape(actual, predicted) == pytest.approx(10.0)


def test_mape_excludes_zero_actual_points():
    # Christmas-style day: actual=0 must not blow up or dominate the average.
    actual = np.array([100.0, 0.0, 100.0])
    predicted = np.array([110.0, 5.0, 90.0])
    # only indices 0 and 2 count: |10|/100=0.10, |10|/100=0.10 -> 10%
    assert mape(actual, predicted) == pytest.approx(10.0)


def test_mape_returns_none_when_all_actuals_zero():
    actual = np.array([0.0, 0.0])
    predicted = np.array([1.0, 2.0])
    assert mape(actual, predicted) is None


def test_count_zero_actuals():
    assert count_zero_actuals(np.array([0.0, 1.0, 0.0, 3.0])) == 2
    assert count_zero_actuals(np.array([1.0, 2.0])) == 0


def test_compute_metrics_bundles_all_three():
    actual = np.array([100.0, 0.0, 100.0])
    predicted = np.array([110.0, 5.0, 90.0])
    result = compute_metrics(actual, predicted)
    assert result.mae == pytest.approx(mae(actual, predicted))
    assert result.rmse == pytest.approx(rmse(actual, predicted))
    assert result.mape == pytest.approx(10.0)
    assert result.mape_n_excluded == 1
    assert result.n == 3
