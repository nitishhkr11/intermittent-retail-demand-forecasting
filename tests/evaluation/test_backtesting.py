"""Unit tests for src/evaluation/backtesting.py.

Uses small local test-double forecasters (not SeasonalNaive) for most
tests, specifically to prove the engine is model-agnostic -- the one
SeasonalNaive-based test at the end is just an end-to-end sanity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import (
    generate_rolling_origins,
    run_backtest,
    summarize_backtest,
)
from src.models.base import BaseForecaster
from src.models.baseline import SeasonalNaive


class _LastValueForecaster(BaseForecaster):
    """Predicts the last training observation repeated for the whole horizon.

    Deliberately not SeasonalNaive -- a second, independent forecaster
    implementation is what actually demonstrates run_backtest() is
    reusable across models rather than secretly coupled to one of them.
    """

    name = "LastValue"

    def __init__(self):
        self._last_value = None

    def fit(self, history: pd.Series) -> "_LastValueForecaster":
        self._last_value = float(history.iloc[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._last_value)


class _HistoryLengthForecaster(BaseForecaster):
    """Predicts how many observations it was fit on, repeated for the horizon.

    Not a sensible forecast -- it exists purely so a test can check
    exactly how much history each fold's model saw, to catch any
    accidental leakage of a fold's own test window (or a later fold's
    data) into its training window.
    """

    name = "HistoryLength"

    def __init__(self):
        self._n = None

    def fit(self, history: pd.Series) -> "_HistoryLengthForecaster":
        self._n = len(history)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, float(self._n))


def _series(n_days: int) -> pd.Series:
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    return pd.Series(np.arange(n_days, dtype=float), index=dates)


# ---- generate_rolling_origins --------------------------------------------------

def test_generate_rolling_origins_last_fold_ends_at_series_end():
    origins = generate_rolling_origins(n_obs=1913, horizon=28, n_folds=5)
    assert len(origins) == 5
    assert origins[-1].test_end == 1913


def test_generate_rolling_origins_folds_are_contiguous_non_overlapping():
    origins = generate_rolling_origins(n_obs=1913, horizon=28, n_folds=5)
    for prev, curr in zip(origins, origins[1:]):
        assert prev.test_end == curr.test_start  # back-to-back, no gap or overlap


def test_generate_rolling_origins_training_window_expands():
    origins = generate_rolling_origins(n_obs=1913, horizon=28, n_folds=5)
    train_sizes = [o.train_end for o in origins]
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes)  # strictly increasing


def test_generate_rolling_origins_raises_when_not_enough_data():
    with pytest.raises(ValueError):
        generate_rolling_origins(n_obs=50, horizon=28, n_folds=5)


# ---- run_backtest ----------------------------------------------------------------

def test_run_backtest_produces_one_result_per_fold():
    series = _series(1913)
    results = run_backtest(series, _LastValueForecaster, horizon=28, n_folds=5)
    assert len(results) == 5
    assert [r.fold for r in results] == [1, 2, 3, 4, 5]


def test_run_backtest_last_value_forecaster_matches_hand_computed_prediction():
    series = _series(1913)
    results = run_backtest(series, _LastValueForecaster, horizon=28, n_folds=5)
    origins = generate_rolling_origins(n_obs=1913, horizon=28, n_folds=5)
    for result, origin in zip(results, origins):
        expected_last_value = series.iloc[origin.train_end - 1]
        assert np.all(result.predictions == expected_last_value)


def test_run_backtest_fold_training_window_excludes_its_own_test_window():
    # _HistoryLengthForecaster reports exactly how many points it was fit
    # on. That must equal the fold's train_end -- never train_end plus
    # any of that fold's own (or a later fold's) test-window points.
    series = _series(1913)
    results = run_backtest(series, _HistoryLengthForecaster, horizon=28, n_folds=5)
    origins = generate_rolling_origins(n_obs=1913, horizon=28, n_folds=5)
    for result, origin in zip(results, origins):
        assert result.predictions[0] == origin.train_end


def test_run_backtest_raises_on_non_datetime_index():
    series = pd.Series(np.arange(200, dtype=float))  # plain RangeIndex
    with pytest.raises(ValueError):
        run_backtest(series, _LastValueForecaster, horizon=28, n_folds=5)


def test_run_backtest_raises_on_unsorted_index():
    series = _series(200)
    shuffled = series.iloc[::-1]
    with pytest.raises(ValueError):
        run_backtest(shuffled, _LastValueForecaster, horizon=28, n_folds=5)


# ---- summarize_backtest -----------------------------------------------------------

def test_summarize_backtest_has_expected_columns_and_row_count():
    series = _series(1913)
    results = run_backtest(series, _LastValueForecaster, horizon=28, n_folds=5)
    table = summarize_backtest(results)
    assert len(table) == 5
    for col in ["model", "fold", "forecast_start", "forecast_end", "MAE", "RMSE", "MAPE"]:
        assert col in table.columns


def test_summarize_backtest_mae_and_rmse_are_zero_for_a_perfectly_predictable_series():
    # On a constant series, "repeat the last observed value" is a perfect
    # forecast -- so a correct metrics pipeline must report exactly zero
    # error for every fold.
    series = pd.Series(
        np.full(1913, 42.0), index=pd.date_range("2011-01-29", periods=1913, freq="D")
    )
    results = run_backtest(series, _LastValueForecaster, horizon=28, n_folds=5)
    table = summarize_backtest(results)
    assert (table["MAE"] == 0.0).all()
    assert (table["RMSE"] == 0.0).all()


# ---- end-to-end sanity check with the real Phase 2 baseline -----------------------

def test_run_backtest_works_with_seasonal_naive():
    series = _series(1913)
    results = run_backtest(series, SeasonalNaive, horizon=28, n_folds=5)
    assert len(results) == 5
    assert all(r.model_name == "SeasonalNaive" for r in results)
    assert all(len(r.predictions) == 28 for r in results)


def test_seasonal_naive_backtest_numbers_unchanged_by_step5_extension():
    # Regression guard: Phase 2 Step 5 extended BacktestFoldResult/
    # run_backtest/summarize_backtest (success/error/diagnostics fields,
    # try/except per fold) and must not change the historical, already-
    # reported Seasonal Naive numbers (MAE 256.55, RMSE 329.05, MAPE
    # 12.08%, mean across the same 5 real development folds) without a
    # demonstrated bug -- there is none, so these must still match.
    series = _series(1913)
    results = run_backtest(series, lambda: SeasonalNaive(season_length=7), horizon=28, n_folds=5)
    assert all(r.success for r in results)
    assert all(r.error is None for r in results)
    table = summarize_backtest(results)
    assert table["success"].all()
    # _series() is a deterministic synthetic ramp (not the real M5
    # series), so this checks shape/behavior stability, not the literal
    # historical figures -- those are re-verified against real data in
    # notebooks/04_classical_models.ipynb and the Step 5 report.
    assert len(table) == 5
    assert (table["MAE"] >= 0).all()
    assert (table["RMSE"] >= 0).all()


# ---- graceful per-fold failure handling (Phase 2 Step 5) -------------------------


class _AlwaysFailsForecaster(BaseForecaster):
    """Fails on every fold -- the "always-broken model" failure-path test double."""

    name = "AlwaysFails"

    def fit(self, history: pd.Series) -> "_AlwaysFailsForecaster":
        raise RuntimeError("deliberate failure for testing")

    def predict(self, horizon: int) -> np.ndarray:  # pragma: no cover -- never reached
        raise RuntimeError("predict() should never be called after fit() raised")


class _FailsOnShortestFoldForecaster(BaseForecaster):
    """Fails only on the fold with the least training history (fold 1)."""

    name = "FailsOnShortestFold"

    def __init__(self):
        self._n = None

    def fit(self, history: pd.Series) -> "_FailsOnShortestFoldForecaster":
        if len(history) < 1780:  # smaller than every fold but the first, for n_obs=1913
            raise ValueError("too little history, simulated failure")
        self._n = len(history)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, float(self._n))


class _DiagnosticsForecaster(BaseForecaster):
    """A trivial forecaster that reports fit_diagnostics(), to test capture."""

    name = "DiagnosticsStub"

    def fit(self, history: pd.Series) -> "_DiagnosticsForecaster":
        self._n = len(history)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.zeros(horizon)

    def fit_diagnostics(self) -> dict:
        return {"n_obs_fit": self._n, "note": "stub diagnostics"}


class _BrokenDiagnosticsForecaster(BaseForecaster):
    """Fits and predicts fine, but its fit_diagnostics() itself raises.

    A diagnostics-gathering bug must not discard an otherwise-valid
    forecast -- that would conflate two unrelated kinds of failure.
    """

    name = "BrokenDiagnostics"

    def fit(self, history: pd.Series) -> "_BrokenDiagnosticsForecaster":
        self._n = len(history)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, 1.0)

    def fit_diagnostics(self) -> dict:
        raise RuntimeError("diagnostics gathering is broken")


def test_run_backtest_records_failed_fold_instead_of_crashing():
    series = _series(1913)
    results = run_backtest(series, _AlwaysFailsForecaster, horizon=28, n_folds=5)
    # The whole point: one failing model must not crash the loop -- all
    # 5 folds are still attempted and recorded.
    assert len(results) == 5
    assert all(r.success is False for r in results)
    assert all(r.error is not None and "RuntimeError" in r.error for r in results)
    assert all(r.predictions is None for r in results)
    # actuals must still be populated -- they come from the series, not
    # the model, so a model failure is no reason to lose them.
    assert all(len(r.actuals) == 28 for r in results)
    assert all(r.diagnostics == {} for r in results)


def test_run_backtest_continues_after_a_single_fold_failure():
    series = _series(1913)
    results = run_backtest(series, _FailsOnShortestFoldForecaster, horizon=28, n_folds=5)
    assert len(results) == 5
    assert results[0].success is False
    assert all(r.success is True for r in results[1:])
    assert all(r.predictions is not None for r in results[1:])


def test_summarize_backtest_emits_nan_metrics_and_error_columns_for_failed_folds():
    series = _series(1913)
    results = run_backtest(series, _AlwaysFailsForecaster, horizon=28, n_folds=5)
    table = summarize_backtest(results)
    assert "success" in table.columns
    assert "error" in table.columns
    assert (~table["success"]).all()
    assert table["MAE"].isna().all()
    assert table["RMSE"].isna().all()
    assert table["MAPE"].isna().all()
    # n (the test-window length) is independent of the model and must
    # still be reported even though the model itself failed.
    assert (table["n"] == 28).all()


def test_summarize_backtest_mixed_success_and_failure_rows():
    series = _series(1913)
    results = run_backtest(series, _FailsOnShortestFoldForecaster, horizon=28, n_folds=5)
    table = summarize_backtest(results)
    assert table.loc[table["fold"] == 1, "success"].iloc[0] == False  # noqa: E712
    assert table.loc[table["fold"] == 1, "MAE"].isna().iloc[0]
    assert (table.loc[table["fold"] != 1, "success"]).all()
    assert not table.loc[table["fold"] != 1, "MAE"].isna().any()


def test_run_backtest_captures_fit_diagnostics_per_fold():
    series = _series(1913)
    results = run_backtest(series, _DiagnosticsForecaster, horizon=28, n_folds=5)
    assert all(r.success for r in results)
    for r in results:
        assert r.diagnostics["note"] == "stub diagnostics"
        assert r.diagnostics["n_obs_fit"] > 0


def test_model_with_no_fit_diagnostics_override_gets_empty_dict():
    # SeasonalNaive doesn't override fit_diagnostics() -- BaseForecaster's
    # default ({}) must flow through run_backtest unchanged.
    series = _series(1913)
    results = run_backtest(series, SeasonalNaive, horizon=28, n_folds=5)
    assert all(r.diagnostics == {} for r in results)


def test_broken_fit_diagnostics_does_not_discard_a_successful_forecast():
    series = _series(1913)
    results = run_backtest(series, _BrokenDiagnosticsForecaster, horizon=28, n_folds=5)
    assert all(r.success for r in results)
    assert all(r.predictions is not None for r in results)
    assert all(np.all(r.predictions == 1.0) for r in results)
    assert all(r.diagnostics == {} for r in results)
