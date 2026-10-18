"""Unit tests for src/analysis/diagnostics.py (ADF test + first differencing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.diagnostics import (
    LAGS_OF_INTEREST,
    MIN_OBS_FOR_ADF,
    AutoARIMAResult,
    ACFPACFResult,
    adf_test,
    compute_acf,
    compute_acf_pacf,
    compute_pacf,
    first_difference,
    run_auto_arima_cross_check,
)
from src.data.splitter import split_final_holdout


# ---- ADF result structure ----------------------------------------------------

def test_adf_test_returns_expected_structure():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(size=200))
    result = adf_test(series)
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.n_lags, int)
    assert isinstance(result.n_obs, int)
    assert set(result.critical_values.keys()) == {"1%", "5%", "10%"}
    assert result.alpha == 0.05


def test_adf_test_n_obs_consistent_with_input_length_and_lags():
    # statsmodels' own accounting: nobs = len(input) - 1 - n_lags (the
    # test regresses the differenced series on its own lags). Tiny noise
    # keeps the internal design matrix non-singular -- a perfectly
    # noise-free trend is a degenerate case this test isn't about.
    rng = np.random.default_rng(3)
    series = pd.Series(np.arange(300, dtype=float) + rng.normal(scale=0.01, size=300))
    result = adf_test(series)
    assert result.n_obs == len(series) - 1 - result.n_lags


def test_adf_test_drops_nulls_before_testing():
    rng = np.random.default_rng(1)
    values = rng.normal(size=200)
    with_nulls = pd.Series(np.concatenate([[np.nan], values, [np.nan]]))
    clean = pd.Series(values)
    result_with_nulls = adf_test(with_nulls)
    result_clean = adf_test(clean)
    assert result_with_nulls.n_obs == result_clean.n_obs


# ---- stationary vs non-stationary verdicts ------------------------------------

def test_adf_test_rejects_h0_for_stationary_white_noise():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(size=300))
    result = adf_test(series)
    assert result.is_stationary
    assert result.p_value < 0.05


def test_adf_test_fails_to_reject_h0_for_random_walk():
    rng = np.random.default_rng(42)
    steps = rng.normal(size=500)
    series = pd.Series(np.cumsum(steps))  # classic unit-root / non-stationary case
    result = adf_test(series)
    assert not result.is_stationary
    assert result.p_value >= 0.05


def test_summary_mentions_correct_verdict_language():
    rng = np.random.default_rng(42)
    stationary = adf_test(pd.Series(rng.normal(size=300)))
    assert "reject H0 -> evidence of stationarity" in stationary.summary()

    walk = pd.Series(np.cumsum(rng.normal(size=500)))
    nonstationary = adf_test(walk)
    assert "fail to reject H0" in nonstationary.summary()


# ---- insufficient observations -------------------------------------------------

def test_adf_test_raises_on_insufficient_observations():
    series = pd.Series([1.0, 2.0, 3.0])
    assert len(series) < MIN_OBS_FOR_ADF
    with pytest.raises(ValueError):
        adf_test(series)


def test_adf_test_boundary_just_below_minimum_raises():
    series = pd.Series(np.arange(MIN_OBS_FOR_ADF - 1, dtype=float))
    with pytest.raises(ValueError):
        adf_test(series)


# ---- differencing behavior ------------------------------------------------------

def test_first_difference_length_and_values():
    series = pd.Series([10.0, 12.0, 9.0, 15.0])
    diff = first_difference(series)
    assert len(diff) == len(series) - 1
    np.testing.assert_allclose(diff.to_numpy(), [2.0, -3.0, 6.0])


def test_first_difference_removes_a_pure_linear_trend():
    series = pd.Series(np.arange(1, 51, dtype=float))  # pure linear trend
    diff = first_difference(series)
    assert np.allclose(diff.to_numpy(), 1.0)


def test_first_difference_is_not_applied_twice_automatically():
    series = pd.Series(np.arange(1, 51, dtype=float))
    diff = first_difference(series)
    # Calling it again is a separate, explicit decision by the caller --
    # the function itself holds no state and doesn't chain automatically.
    diff_again = first_difference(series)
    pd.testing.assert_series_equal(diff.reset_index(drop=True), diff_again.reset_index(drop=True))


def test_differencing_a_random_walk_tends_toward_stationarity():
    # A random walk is (by construction) non-stationary; its first
    # difference is exactly the i.i.d. step series, which is stationary.
    rng = np.random.default_rng(7)
    steps = rng.normal(size=500)
    walk = pd.Series(np.cumsum(steps))
    walk_result = adf_test(walk)
    diff_result = adf_test(first_difference(walk))
    assert not walk_result.is_stationary
    assert diff_result.is_stationary


# ---- no accidental use of the final holdout -------------------------------------

def test_adf_test_combined_with_split_never_sees_holdout_values():
    # Development: a smooth, bounded sine pattern. Holdout: a deliberately
    # enormous, obviously-distinct spike. If the holdout leaked into what
    # gets tested, n_obs would be 28 higher than development's own length
    # and the huge values would appear in the tested series at all.
    rng = np.random.default_rng(11)
    dates = pd.date_range("2011-01-29", periods=300, freq="D")
    dev_values = np.sin(np.linspace(0, 30, 272)) * 5 + 100 + rng.normal(scale=0.01, size=272)
    holdout_values = np.full(28, 1_000_000.0)
    df = pd.DataFrame(
        {"date": dates, "sales": np.concatenate([dev_values, holdout_values])}
    )

    split = split_final_holdout(df, holdout_days=28, date_col="date")
    dev_series = split.development.set_index("date")["sales"]

    assert 1_000_000.0 not in dev_series.to_numpy()
    assert len(dev_series) == 272

    result = adf_test(dev_series)
    assert result.n_obs == len(dev_series) - 1 - result.n_lags


# =============================================================================
# Phase 2 Step 4 -- ACF / PACF output structure
# =============================================================================

def test_compute_acf_pacf_returns_expected_structure():
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(size=300))
    result = compute_acf_pacf(series, nlags=40)

    assert isinstance(result, ACFPACFResult)
    assert result.nlags == 40
    assert len(result.lags) == 41
    assert result.acf_values.shape == (41,)
    assert result.pacf_values.shape == (41,)
    assert result.acf_confint.shape == (41, 2)
    assert result.pacf_confint.shape == (41, 2)
    # Lag 0 is the trivial self-correlation, by both statsmodels' and this
    # module's convention.
    assert result.acf_values[0] == pytest.approx(1.0)
    assert result.pacf_values[0] == pytest.approx(1.0)


def test_notable_lags_table_has_expected_columns_and_real_values():
    # A strong AR(1) process (phi=0.9) has a sharp, unmistakable spike at
    # lag 1 in both ACF and PACF -- a known, checkable structure rather
    # than an arbitrary hard-coded number.
    rng = np.random.default_rng(9)
    n = 500
    phi = 0.9
    x = np.zeros(n)
    noise = rng.normal(scale=1.0, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    series = pd.Series(x)

    result = compute_acf_pacf(series, nlags=20)
    table = result.notable_lags_table(lags_of_interest=(1, 2, 5, 10))

    assert list(table.columns) == ["lag", "acf", "acf_notable", "pacf", "pacf_notable"]
    assert len(table) == 4
    row_lag1 = table.loc[table["lag"] == 1].iloc[0]
    assert row_lag1["acf"] > 0.5
    assert bool(row_lag1["acf_notable"]) is True
    assert bool(row_lag1["pacf_notable"]) is True


def test_notable_acf_pacf_lags_return_dicts_keyed_by_lag():
    rng = np.random.default_rng(13)
    series = pd.Series(rng.normal(size=300))
    result = compute_acf_pacf(series, nlags=60)

    notable_acf = result.notable_acf_lags()
    notable_pacf = result.notable_pacf_lags()

    assert set(notable_acf.keys()) == set(LAGS_OF_INTEREST)
    assert set(notable_pacf.keys()) == set(LAGS_OF_INTEREST)
    assert all(isinstance(v, bool) for v in notable_acf.values())
    assert all(isinstance(v, bool) for v in notable_pacf.values())


# =============================================================================
# Phase 2 Step 4 -- valid lag handling
# =============================================================================

def test_compute_acf_raises_on_nlags_too_large_for_series():
    series = pd.Series(np.arange(30, dtype=float))
    with pytest.raises(ValueError):
        compute_acf(series, nlags=30)  # nlags must be < len(series)


def test_compute_pacf_raises_on_nlags_too_large_for_series():
    series = pd.Series(np.arange(30, dtype=float))
    with pytest.raises(ValueError):
        compute_pacf(series, nlags=50)


def test_compute_acf_raises_on_nonpositive_nlags():
    series = pd.Series(np.arange(100, dtype=float))
    with pytest.raises(ValueError):
        compute_acf(series, nlags=0)


def test_acf_pacf_result_raises_on_out_of_range_lag_request():
    rng = np.random.default_rng(21)
    series = pd.Series(rng.normal(size=200))
    result = compute_acf_pacf(series, nlags=10)
    with pytest.raises(ValueError):
        result.notable_acf_lags(lags_of_interest=(1, 2, 56))  # 56 > nlags=10


def test_compute_acf_pacf_valid_lag_range_produces_full_length_arrays():
    rng = np.random.default_rng(33)
    series = pd.Series(rng.normal(size=400))
    for nlags in (1, 7, 60):
        result = compute_acf_pacf(series, nlags=nlags)
        assert len(result.acf_values) == nlags + 1
        assert len(result.pacf_values) == nlags + 1


# =============================================================================
# Phase 2 Step 4 -- correct differenced input is used
# =============================================================================

def test_acf_on_raw_trend_vs_first_differenced_trend_differ_as_expected():
    # A pure linear trend has ACF that stays high at every lag (it barely
    # decays) -- first-differencing it removes the trend entirely, so the
    # differenced series' ACF should collapse toward the significance
    # band at the same lag. If a caller accidentally fed the raw series
    # into the ACF/PACF step instead of first_difference(series), this
    # is exactly the discrepancy that would go undetected.
    rng = np.random.default_rng(17)
    n = 300
    trend = pd.Series(np.arange(n, dtype=float) + rng.normal(scale=0.5, size=n))

    raw_result = compute_acf_pacf(trend, nlags=20)
    diff_result = compute_acf_pacf(first_difference(trend), nlags=20)

    assert raw_result.acf_values[10] > 0.5
    assert abs(diff_result.acf_values[10]) < 0.3


def test_first_difference_output_is_what_feeds_compute_acf_pacf():
    # Exercises the exact composition the notebook uses: first_difference()
    # then compute_acf_pacf() on its result.
    rng = np.random.default_rng(23)
    series = pd.Series(np.cumsum(rng.normal(size=250)))  # random walk
    diff = first_difference(series)
    assert len(diff) == len(series) - 1

    result = compute_acf_pacf(diff, nlags=15)
    assert result.nlags == 15
    # The differenced random walk is (by construction) close to i.i.d.
    # noise, so it shouldn't look wildly non-stationary at lag 1.
    assert abs(result.acf_values[1]) < 0.5


# =============================================================================
# Phase 2 Step 4 -- no final-holdout leakage (ACF/PACF)
# =============================================================================

def test_compute_acf_pacf_combined_with_split_never_sees_holdout_values():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2011-01-29", periods=300, freq="D")
    dev_values = np.sin(np.linspace(0, 30, 272)) * 5 + 100 + rng.normal(scale=0.01, size=272)
    holdout_values = np.full(28, 1_000_000.0)
    df = pd.DataFrame({"date": dates, "sales": np.concatenate([dev_values, holdout_values])})

    split = split_final_holdout(df, holdout_days=28, date_col="date")
    dev_series = split.development.set_index("date")["sales"]

    # Ground truth: compute_acf_pacf on the development values alone,
    # constructed independently of the split.
    standalone = pd.Series(dev_values)
    expected = compute_acf_pacf(standalone, nlags=30)

    # What the pipeline actually does: feed split.development through.
    actual = compute_acf_pacf(dev_series, nlags=30)

    np.testing.assert_allclose(actual.acf_values, expected.acf_values)
    np.testing.assert_allclose(actual.pacf_values, expected.pacf_values)
    assert len(dev_series) == 272
    assert 1_000_000.0 not in dev_series.to_numpy()


# =============================================================================
# Phase 2 Step 4 -- auto_arima cross-check
# =============================================================================

def test_run_auto_arima_cross_check_returns_expected_structure():
    rng = np.random.default_rng(29)
    n = 200
    series = pd.Series(np.cumsum(rng.normal(size=n)))  # random walk -> d=1 sensible

    result = run_auto_arima_cross_check(
        series, d=1, seasonal_m=7, max_p=2, max_q=2, max_P=1, max_Q=1,
    )

    assert isinstance(result, AutoARIMAResult)
    assert len(result.order) == 3
    assert len(result.seasonal_order) == 4
    assert result.order[1] == 1  # d was fixed at 1
    assert result.seasonal_order[3] == 7  # seasonal period m=7
    assert isinstance(result.aic, float)
    assert result.n_obs == n
    assert "AIC" in result.summary()


def test_run_auto_arima_cross_check_no_holdout_leakage():
    # Same holdout-isolation pattern as the ADF test above: an obviously
    # distinct holdout block, confirmed excluded by observation count.
    rng = np.random.default_rng(41)
    dates = pd.date_range("2011-01-29", periods=228, freq="D")
    dev_values = np.cumsum(rng.normal(size=200))
    holdout_values = np.full(28, 1_000_000.0)
    df = pd.DataFrame({"date": dates, "sales": np.concatenate([dev_values, holdout_values])})

    split = split_final_holdout(df, holdout_days=28, date_col="date")
    dev_series = split.development.set_index("date")["sales"]
    assert len(dev_series) == 200
    assert 1_000_000.0 not in dev_series.to_numpy()

    result = run_auto_arima_cross_check(
        dev_series, d=1, seasonal_m=7, max_p=2, max_q=2, max_P=1, max_Q=1,
    )
    # If the holdout had leaked in, n_obs would be 228, not 200.
    assert result.n_obs == 200
