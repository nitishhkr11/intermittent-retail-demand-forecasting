"""Unit tests for src/evaluation/residuals.py (Phase 2 Step 6).

Covers the five categories the Step 6 brief asks for explicitly:
residual calculation using training data only (never the held-out
28-day forecast error); Ljung-Box output structure; residual
diagnostic plotting; correct handling of failed or non-converged fits;
and final-holdout exclusion. Real small ARIMA/SARIMA/Holt-Winters fits
are used wherever the diagnostic *values* matter (never hard-coded),
with low orders and short synthetic series purely for test speed --
the real, full-size development data is exercised separately by
`tests/notebooks/test_classical_models_notebook.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splitter import split_final_holdout
from src.evaluation.backtesting import BacktestFoldResult, run_backtest
from src.evaluation.residuals import (
    FoldResidualDiagnostics,
    LjungBoxResult,
    build_fold_diagnostics,
    compute_fold_residual_diagnostics,
    fold_summary_table,
    ljung_box_test,
    plot_residual_diagnostics,
    plot_residual_diagnostics_grid,
    residual_diagnostics_table,
    select_ljung_box_lag,
)
from src.models.arima import ARIMAForecaster
from src.models.base import BaseForecaster
from src.models.holt_winters import HoltWintersForecaster
from src.models.sarima import SARIMAForecaster


def _series(n: int = 220, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    weekly = 6 * np.sin(2 * np.pi * np.arange(n) / 7)
    values = 100 + 0.05 * np.arange(n) + weekly + rng.normal(scale=3.0, size=n)
    return pd.Series(values, index=dates)


def _fold_result(forecaster: BaseForecaster, history: pd.Series, horizon: int = 28, fold: int = 1) -> BacktestFoldResult:
    """Build one real BacktestFoldResult by hand: fit `forecaster` on
    `history`, exactly as src.evaluation.backtesting.run_backtest does
    internally, so tests can inspect a single fold without paying for a
    full 5-fold backtest.
    """
    forecaster.fit(history)
    predictions = np.asarray(forecaster.predict(horizon), dtype=float)
    diagnostics = forecaster.fit_diagnostics()
    future_dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    return BacktestFoldResult(
        model_name=forecaster.name,
        fold=fold,
        origin_date=history.index[-1],
        forecast_start=future_dates[0],
        forecast_end=future_dates[-1],
        dates=future_dates,
        actuals=np.zeros(horizon),  # unused by residuals.py -- never read as "residuals"
        predictions=predictions,
        success=True,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# 1. Residual calculation using training data only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forecaster_factory",
    [
        lambda: ARIMAForecaster(order=(1, 1, 1)),
        lambda: SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 1, 7)),
        lambda: HoltWintersForecaster(trend=None),
    ],
)
def test_residuals_are_training_window_length_not_forecast_horizon(forecaster_factory):
    history = _series(200)
    fr = _fold_result(forecaster_factory(), history, horizon=28)
    diag = compute_fold_residual_diagnostics(fr)
    assert diag is not None
    # The full residual series is exactly as long as the TRAINING window
    # (200), never the 28-day forecast horizon -- the concrete guard
    # against "in-sample residuals" secretly being forecast errors.
    assert len(diag.resid) == len(history)
    assert len(diag.resid) != 28


def test_training_residuals_are_not_the_oos_forecast_error():
    # Build a fold where actuals/predictions on the TEST window are
    # deliberately very different from the fitted model's TRAINING
    # residuals, then confirm compute_fold_residual_diagnostics reads
    # only the training residuals -- never recomputed from actual-minus-
    # predicted on the held-out window.
    history = _series(200)
    model = ARIMAForecaster(order=(1, 1, 1)).fit(history)
    diag_dict = model.fit_diagnostics()

    fr = BacktestFoldResult(
        model_name=model.name,
        fold=1,
        origin_date=history.index[-1],
        forecast_start=history.index[-1] + pd.Timedelta(days=1),
        forecast_end=history.index[-1] + pd.Timedelta(days=28),
        dates=pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=28, freq="D"),
        actuals=np.full(28, 1_000_000.0),  # absurd, deliberately far from any residual
        predictions=np.zeros(28),  # forecast error would be ~1,000,000 at every point
        success=True,
        diagnostics=diag_dict,
    )
    diag = compute_fold_residual_diagnostics(fr)
    assert diag is not None
    # If this module ever started computing "residuals" as
    # actuals - predictions on the test window, every value here would
    # be ~1,000,000. It must not be anywhere close.
    assert np.max(np.abs(diag.resid)) < 1000


def test_residual_values_are_not_hard_coded_across_different_series():
    # Two different synthetic series must produce two different
    # residual series and two different Ljung-Box statistics -- a cheap
    # guard against the diagnostics ever being a stubbed constant.
    fr_a = _fold_result(ARIMAForecaster(order=(1, 1, 1)), _series(200, seed=1), fold=1)
    fr_b = _fold_result(ARIMAForecaster(order=(1, 1, 1)), _series(200, seed=2), fold=1)
    diag_a = compute_fold_residual_diagnostics(fr_a)
    diag_b = compute_fold_residual_diagnostics(fr_b)
    assert not np.allclose(diag_a.resid, diag_b.resid)
    assert diag_a.ljung_box.table["lb_stat"].to_numpy() != pytest.approx(
        diag_b.ljung_box.table["lb_stat"].to_numpy()
    )


def test_burn_in_trim_matches_loglikelihood_burn_for_sarimax_models():
    history = _series(200)
    fr = _fold_result(SARIMAForecaster(order=(1, 1, 1), seasonal_order=(0, 0, 1, 7)), history, fold=1)
    diag = compute_fold_residual_diagnostics(fr)
    burn = fr.diagnostics["loglikelihood_burn"]
    assert diag.loglikelihood_burn == burn
    assert len(diag.resid_diagnostic) == len(diag.resid) - burn
    assert len(diag.dates_diagnostic) == len(diag.dates) - burn


def test_holt_winters_has_no_burn_in():
    history = _series(200)
    fr = _fold_result(HoltWintersForecaster(trend="add"), history, fold=1)
    diag = compute_fold_residual_diagnostics(fr)
    assert diag.loglikelihood_burn == 0
    assert len(diag.resid_diagnostic) == len(diag.resid)


# ---------------------------------------------------------------------------
# 2. Ljung-Box output structure
# ---------------------------------------------------------------------------


def test_select_ljung_box_lag_uses_hyndman_athanasopoulos_rule():
    # min(2*m, T/5): for m=7, 2*m=14 binds whenever T/5 >= 14 (T >= 70).
    assert select_ljung_box_lag(n_obs=1000, seasonal_period=7) == 14
    # For a short series, T/5 binds instead.
    assert select_ljung_box_lag(n_obs=50, seasonal_period=7) == 10
    with pytest.raises(ValueError):
        select_ljung_box_lag(n_obs=0)


def test_ljung_box_test_structure_on_white_noise():
    rng = np.random.default_rng(0)
    resid = rng.normal(size=500)
    result = ljung_box_test(resid, lags=[7, 14, 21], model_df=0)
    assert isinstance(result, LjungBoxResult)
    assert list(result.table.index) == [7, 14, 21]
    assert set(result.table.columns) >= {"lb_stat", "lb_pvalue", "reject_h0"}
    assert result.n_obs == 500
    # White noise should (almost always) fail to reject H0 at these lags.
    assert result.table.loc[14, "lb_pvalue"] > 0.01
    assert result.untestable_lags() == []
    assert "H0" in result.summary()


def test_ljung_box_test_detects_real_autocorrelation():
    # An AR(1) series with a strong coefficient must reject H0 at a lag
    # that actually captures the dependency -- the test must have real
    # power to detect what's genuinely there, not just run without error.
    rng = np.random.default_rng(0)
    n = 500
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = 0.8 * resid[t - 1] + rng.normal(scale=1.0)
    result = ljung_box_test(resid, lags=[5], model_df=0, alpha=0.05)
    assert result.table.loc[5, "lb_pvalue"] < 0.01
    assert bool(result.table.loc[5, "reject_h0"]) is True


def test_ljung_box_model_df_marks_small_lags_untestable_not_silently_wrong():
    rng = np.random.default_rng(0)
    resid = rng.normal(size=300)
    # model_df=10 (as for this project's SARIMA(6,1,1)(0,0,2,7)) makes
    # lag 7 untestable (7 <= model_df) -- statsmodels itself returns NaN
    # for lb_pvalue there, and reject_h0 must be None, not a misleading
    # False ("fails to reject" would imply a valid test was run).
    result = ljung_box_test(resid, lags=[7, 14], model_df=10)
    assert result.untestable_lags() == [7]
    assert pd.isna(result.table.loc[7, "lb_pvalue"])
    assert result.table.loc[7, "reject_h0"] is None
    assert "not computable" in result.summary()


def test_ljung_box_rejects_lag_not_smaller_than_sample_size():
    resid = np.random.default_rng(0).normal(size=10)
    with pytest.raises(ValueError):
        ljung_box_test(resid, lags=[10], model_df=0)


def test_ljung_box_drops_nans_before_testing():
    rng = np.random.default_rng(0)
    resid = rng.normal(size=200)
    resid_with_nan = np.concatenate([[np.nan, np.nan], resid])
    result = ljung_box_test(resid_with_nan, lags=[5], model_df=0)
    assert result.n_obs == 200


# ---------------------------------------------------------------------------
# 3. Residual diagnostic plotting
# ---------------------------------------------------------------------------


def test_plot_residual_diagnostics_runs_and_returns_three_axes():
    history = _series(200)
    fr = _fold_result(ARIMAForecaster(order=(1, 1, 1)), history, fold=1)
    diag = compute_fold_residual_diagnostics(fr)
    fig = plot_residual_diagnostics(diag)
    assert len(fig.axes) == 3
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_residual_diagnostics_grid_one_row_per_fold():
    fr1 = _fold_result(ARIMAForecaster(order=(1, 1, 1)), _series(150, seed=1), fold=1)
    fr2 = _fold_result(ARIMAForecaster(order=(1, 1, 1)), _series(150, seed=2), fold=2)
    diagnostics = build_fold_diagnostics([fr1, fr2])
    fig = plot_residual_diagnostics_grid(diagnostics)
    assert len(fig.axes) == 2 * 3  # 2 folds x 3 panels
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_residual_diagnostics_grid_raises_on_empty_input():
    with pytest.raises(ValueError):
        plot_residual_diagnostics_grid([])


def test_plot_residual_diagnostics_flags_non_convergence_in_title():
    history = _series(200)
    fr = _fold_result(ARIMAForecaster(order=(1, 1, 1)), history, fold=1)
    diag = compute_fold_residual_diagnostics(fr)
    # Force a non-converged state the same way a real fold reports it,
    # without needing an actual non-convergent fit in this fast test.
    diag.converged = False
    fig = plot_residual_diagnostics(diag)
    title = fig.axes[0].get_title()
    assert "NOT converge" in title
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Correct handling of failed or non-converged fits
# ---------------------------------------------------------------------------


class _NoDiagnosticsForecaster(BaseForecaster):
    """Mirrors SeasonalNaive: no fit_diagnostics() override, default {}."""

    name = "NoDiagnosticsStub"

    def fit(self, history):
        self._n = len(history)
        return self

    def predict(self, horizon):
        return np.zeros(horizon)


def test_failed_fold_is_skipped_not_raised():
    fr = BacktestFoldResult(
        model_name="SomeModel",
        fold=1,
        origin_date=pd.Timestamp("2020-01-01"),
        forecast_start=pd.Timestamp("2020-01-02"),
        forecast_end=pd.Timestamp("2020-01-29"),
        dates=pd.date_range("2020-01-02", periods=28, freq="D"),
        actuals=np.zeros(28),
        predictions=None,
        success=False,
        error="RuntimeError: deliberate failure",
        diagnostics={},
    )
    assert compute_fold_residual_diagnostics(fr) is None


def test_model_with_no_residual_diagnostics_is_skipped_not_raised():
    history = _series(100)
    fr = _fold_result(_NoDiagnosticsForecaster(), history, fold=1)
    assert fr.diagnostics == {}
    assert compute_fold_residual_diagnostics(fr) is None


def test_build_fold_diagnostics_skips_mixed_failed_and_successful_folds():
    good = _fold_result(ARIMAForecaster(order=(1, 1, 1)), _series(150), fold=1)
    bad = BacktestFoldResult(
        model_name="ARIMA(1, 1, 1)",
        fold=2,
        origin_date=pd.Timestamp("2020-06-01"),
        forecast_start=pd.Timestamp("2020-06-02"),
        forecast_end=pd.Timestamp("2020-06-29"),
        dates=pd.date_range("2020-06-02", periods=28, freq="D"),
        actuals=np.zeros(28),
        predictions=None,
        success=False,
        error="RuntimeError: deliberate failure",
        diagnostics={},
    )
    diagnostics = build_fold_diagnostics([good, bad])
    assert len(diagnostics) == 1
    assert diagnostics[0].fold == 1


def test_non_converged_fit_is_not_skipped_but_is_flagged():
    # A fold that fit successfully but whose optimizer did not converge
    # (success=True, converged=False -- exactly SARIMA fold 3's real
    # situation) must still produce full diagnostics, with a note.
    history = _series(200)
    fr = _fold_result(ARIMAForecaster(order=(1, 1, 1)), history, fold=1)
    fr.diagnostics["converged"] = False  # simulate without a slow real non-convergent fit
    diag = compute_fold_residual_diagnostics(fr)
    assert diag is not None
    assert diag.converged is False
    assert any("did NOT converge" in note for note in diag.notes)


def test_fold_summary_table_and_long_table_shapes():
    history = _series(200)
    fr1 = _fold_result(ARIMAForecaster(order=(1, 1, 1)), history, fold=1)
    fr2 = _fold_result(ARIMAForecaster(order=(1, 1, 1)), _series(200, seed=3), fold=2)
    diagnostics = build_fold_diagnostics([fr1, fr2])
    summary = fold_summary_table(diagnostics)
    assert len(summary) == 2
    assert {"model", "fold", "n_obs_fit", "n_params", "df_resid"} <= set(summary.columns)

    long_table = residual_diagnostics_table(diagnostics)
    # One row per (fold, lag): 2 folds * len(lags tested for n=200 ARIMA).
    assert set(long_table["fold"]) == {1, 2}
    assert {"lag", "lb_stat", "lb_pvalue", "reject_h0"} <= set(long_table.columns)


# ---------------------------------------------------------------------------
# 5. Final-holdout exclusion
# ---------------------------------------------------------------------------


def test_reconstructed_dates_never_exceed_origin_date():
    history = _series(200)
    fr = _fold_result(ARIMAForecaster(order=(1, 1, 1)), history, fold=1)
    diag = compute_fold_residual_diagnostics(fr)
    assert diag.dates.max() == fr.origin_date
    assert diag.dates_diagnostic.max() == fr.origin_date


def test_full_backtest_never_produces_diagnostics_reaching_the_holdout():
    # The real, end-to-end guard: split a synthetic series into
    # development/holdout via the project's own splitter, run a real
    # backtest on the development side only, and confirm every fold's
    # reconstructed diagnostic dates stay strictly before the holdout
    # start -- mirroring exactly how notebooks/04_classical_models.ipynb
    # section 11 is built.
    full_series = _series(400).to_frame("sales").reset_index(names="date")
    split = split_final_holdout(full_series, holdout_days=28, date_col="date")
    dev = split.development.set_index("date")["sales"].asfreq("D")

    results = run_backtest(dev, lambda: ARIMAForecaster(order=(1, 1, 1)), horizon=28, n_folds=5)
    diagnostics = build_fold_diagnostics(results)
    assert len(diagnostics) == 5  # all 5 folds fit cleanly on this well-behaved synthetic series

    for d in diagnostics:
        assert d.dates.max() < split.test_start
        assert d.dates_diagnostic.max() < split.test_start
        assert d.forecast_end < split.test_start
