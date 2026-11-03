"""Tests for src/models/prophet_model.py (leakage-safe Prophet forecasters).

Every series and calendar here is SYNTHETIC (named `synthetic_*`); nothing in this file is an M5 result.
Pure helpers (dates, regressor frames, configuration) are tested without prophet; tests that fit a model skip
themselves when prophet is not installed.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import run_backtest
from src.evaluation.forecast_artifacts import audit_forecast_table, backtest_results_to_forecast_table
from src.models.prophet_model import (
    ProphetBackendError,
    ProphetForecaster,
    ProphetSettings,
    ProphetVariant,
    build_future_frame,
    build_training_frame,
    check_history,
    future_dates,
    primary_variant_name,
    regressor_frame,
    variants_from_config,
)
from src.models.xgboost_model import EVENT_FLAG_COLUMNS
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

HORIZON = 14
ADDITIVE = ProphetVariant("Prophet(additive)", "additive", True)
MULTIPLICATIVE = ProphetVariant("Prophet(multiplicative)", "multiplicative", True)
HISTORY_ONLY = ProphetVariant("Prophet(additive,no-regressors)", "additive", False)


def synthetic_series(n=300, start="2020-01-01", seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 200 + 0.1 * t + 30 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, n)
    return pd.Series(np.clip(values, 0, None), index=pd.date_range(start, periods=n, freq="D"), name="synthetic_sales")


def synthetic_calendar(start="2019-06-01", n=800) -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="D", name="date")
    flags = pd.DataFrame(0, index=index, columns=list(EVENT_FLAG_COLUMNS))
    flags.loc[index.dayofweek == 5, "snap_ca"] = 1
    flags.loc[(index.month == 12) & (index.day == 25), "is_christmas"] = 1
    flags.loc[index.day == 10, "event_cultural"] = 1
    return flags


def settings(**overrides) -> ProphetSettings:
    return ProphetSettings(**{**dict(horizon=HORIZON), **overrides})


@pytest.fixture(scope="module")
def prophet_module():
    return pytest.importorskip("prophet")


# --------------------------------------------------------------------------- configuration

def test_real_config_is_valid_and_matches_the_documented_settings():
    config = load_config()
    s = ProphetSettings.from_config(config)
    assert (s.yearly_seasonality, s.weekly_seasonality, s.daily_seasonality) == (True, True, False)
    assert (s.changepoint_prior_scale, s.n_changepoints, s.changepoint_range, s.interval_width) == (0.05, 25, 0.8, 0.8)
    assert s.mcmc_samples == 0 and s.horizon == 28 and s.regressors == tuple(EVENT_FLAG_COLUMNS)
    names = [v.name for v in variants_from_config(config)]
    assert names == ["Prophet(additive)", "Prophet(multiplicative)", "Prophet(additive,no-regressors)"]
    assert primary_variant_name(config) == "Prophet(additive)"
    assert [v.use_regressors for v in variants_from_config(config)] == [True, True, False]


@pytest.mark.parametrize(
    "key, bad",
    [
        ("yearly_seasonality", "yes"), ("weekly_seasonality", 1), ("changepoint_prior_scale", 0.0), ("seasonality_prior_scale", -1.0),
        ("n_changepoints", -1), ("n_changepoints", 2.5), ("changepoint_range", 0.0), ("changepoint_range", 1.5),
        ("interval_width", 1.0), ("interval_width", 0.0), ("mcmc_samples", 200), ("uncertainty_samples", 0), ("random_seed", -1),
        ("horizon", 0), ("regressors", ("snap_ca", "snap_ca")), ("regressors", ("not_a_flag",)),
    ],
)
def test_invalid_settings_raise_a_message_naming_the_key(key, bad):
    with pytest.raises(ValueError, match=key):
        ProphetSettings(**{key: bad})


def test_invalid_config_file_values_and_missing_sections_are_rejected():
    config = load_config()
    config["prophet"]["interval_width"] = 2
    with pytest.raises(ValueError, match="prophet.interval_width"):
        ProphetSettings.from_config(config)
    with pytest.raises(ValueError, match="no 'prophet' section"):
        ProphetSettings.from_config({"backtest": {"horizon": 28}})
    with pytest.raises(ValueError, match="backtest.horizon"):
        ProphetSettings.from_config({"prophet": {}, "backtest": {}})
    assert ProphetSettings.from_config({"prophet": {}, "backtest": {"horizon": 7}}).horizon == 7      # documented defaults apply


def test_invalid_variants_are_rejected():
    config = load_config()
    config["prophet"]["variants"] = [{"name": "x", "seasonality_mode": "exponential", "use_regressors": True}]
    with pytest.raises(ValueError, match="seasonality_mode"):
        variants_from_config(config)
    config["prophet"]["variants"] = [{"name": "x", "seasonality_mode": "additive"}]
    with pytest.raises(ValueError, match="exactly the keys"):
        variants_from_config(config)
    config["prophet"]["variants"] = [{"name": "x", "seasonality_mode": "additive", "use_regressors": True}] * 2
    with pytest.raises(ValueError, match="twice"):
        variants_from_config(config)
    config["prophet"]["variants"] = []
    with pytest.raises(ValueError, match="variants"):
        variants_from_config(config)
    config = load_config()
    config["prophet"]["primary_variant"] = "Prophet(unknown)"
    with pytest.raises(ValueError, match="primary_variant"):
        primary_variant_name(config)
    with pytest.raises(ValueError, match="use_regressors"):
        ProphetVariant("x", "additive", "yes")


# --------------------------------------------------------------------------- dates and regressor frames

def test_check_history_accepts_a_clean_daily_series_and_rejects_everything_else():
    series = synthetic_series(40)
    assert check_history(series) is series
    with pytest.raises(TypeError):
        check_history(series.to_numpy())
    with pytest.raises(ValueError, match="indexed by date"):
        check_history(pd.Series(series.to_numpy()))
    with pytest.raises(ValueError, match="empty"):
        check_history(series.iloc[:0])
    with pytest.raises(ValueError, match="continuous"):
        check_history(series.drop(series.index[5]))
    with pytest.raises(ValueError, match="sorted"):
        check_history(series.iloc[::-1])
    with pytest.raises(ValueError, match="non-finite"):
        check_history(series.where(series.index != series.index[3]))
    with pytest.raises(ValueError, match="negative"):
        check_history(series.where(series.index != series.index[3], -1.0))


def test_future_dates_start_the_day_after_the_history_and_are_daily():
    days = future_dates(pd.Timestamp("2016-03-27"), 28)
    assert days[0] == pd.Timestamp("2016-03-28") and days[-1] == pd.Timestamp("2016-04-24") and len(days) == 28
    assert days.freqstr == "D" and days.is_monotonic_increasing
    for bad in (0, -3, 2.5, True, "7"):
        with pytest.raises(ValueError, match="positive integer"):
            future_dates(pd.Timestamp("2016-03-27"), bad)


def test_regressor_frame_reads_calendar_facts_aligned_with_the_requested_days():
    calendar = synthetic_calendar()
    index = pd.date_range("2020-12-20", periods=10, freq="D")
    frame = regressor_frame(index, calendar, ["snap_ca", "is_christmas"])
    assert list(frame.columns) == ["snap_ca", "is_christmas"] and len(frame) == 10 and frame.index.equals(pd.RangeIndex(10))
    np.testing.assert_array_equal(frame["snap_ca"].to_numpy(), (index.dayofweek == 5).astype(float))
    assert frame["is_christmas"].to_numpy().tolist() == [0.0] * 5 + [1.0] + [0.0] * 4          # 2020-12-25 is the 6th day


def test_regressor_frame_never_fills_missing_dates_or_accepts_bad_flags():
    calendar = synthetic_calendar(start="2020-01-01", n=10)
    with pytest.raises(ValueError, match="does not cover 5"):
        regressor_frame(pd.date_range("2020-01-06", periods=10, freq="D"), calendar, ["snap_ca"])
    with pytest.raises(ValueError, match="missing regressor"):
        regressor_frame(calendar.index, calendar.drop(columns=["snap_ca"]), ["snap_ca"])
    bad = calendar.copy()
    bad.iloc[2, 0] = 3
    with pytest.raises(ValueError, match="0/1"):
        regressor_frame(calendar.index, bad, ["snap_ca"])
    with pytest.raises(ValueError, match="indexed by date"):
        regressor_frame(calendar.index, calendar.reset_index(drop=True), ["snap_ca"])


def test_training_and_future_frames_have_the_columns_prophet_expects():
    history, calendar = synthetic_series(60), synthetic_calendar()
    regressors = ("snap_ca", "is_christmas")
    training = build_training_frame(history, calendar, regressors)
    assert list(training.columns) == ["ds", "y", "snap_ca", "is_christmas"] and len(training) == 60
    np.testing.assert_allclose(training["y"].to_numpy(), history.to_numpy())
    future = build_future_frame(history.index[-1], 7, calendar, regressors)
    assert list(future.columns) == ["ds", "snap_ca", "is_christmas"] and len(future) == 7
    assert pd.Timestamp(future["ds"].iloc[0]) == history.index[-1] + pd.Timedelta(days=1)
    assert "y" not in future.columns                                             # the future frame cannot carry sales
    assert list(build_training_frame(history, None, ()).columns) == ["ds", "y"]
    assert list(build_future_frame(history.index[-1], 7, None, ()).columns) == ["ds"]
    with pytest.raises(ValueError, match="calendar is required"):
        build_training_frame(history, None, regressors)
    with pytest.raises(ValueError, match="calendar is required"):
        build_future_frame(history.index[-1], 7, None, regressors)


# --------------------------------------------------------------------------- missing prophet / backend failure

def test_using_the_model_without_prophet_raises_an_informative_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "prophet", None)                      # makes `from prophet import Prophet` raise ImportError
    forecaster = ProphetForecaster(HISTORY_ONLY, settings())
    with pytest.raises(ImportError) as excinfo:
        forecaster.fit(synthetic_series())
    message = str(excinfo.value)
    assert "python -m pip install prophet" in message and "requirements-phase3.txt" in message
    assert "No other method is substituted" in message


def test_the_module_imports_without_prophet():
    code = "import sys; sys.modules['prophet'] = None; import src.models.prophet_model as m; print(len(m.variants_from_config()))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3"


def test_a_failing_stan_backend_gives_an_explanation_and_no_substitute_forecast(prophet_module, monkeypatch):
    def broken_fit(self, df, **kwargs):
        raise RuntimeError("Error during optimization! Command 'prophet_model.exe' failed")

    monkeypatch.setattr(prophet_module.Prophet, "fit", broken_fit)
    forecaster = ProphetForecaster(HISTORY_ONLY, settings())
    with pytest.raises(ProphetBackendError) as excinfo:
        forecaster.fit(synthetic_series())
    message = str(excinfo.value)
    assert "WITHOUT spaces" in message and "force-reinstall prophet" in message and "No other method is substituted" in message
    assert not forecaster.is_fitted


# --------------------------------------------------------------------------- the forecaster (needs prophet)

def test_constructor_validates_its_arguments():
    with pytest.raises(TypeError, match="ProphetVariant"):
        ProphetForecaster("Prophet(additive)", settings())
    with pytest.raises(TypeError, match="ProphetSettings"):
        ProphetForecaster(ADDITIVE, {"horizon": 7})
    with pytest.raises(ValueError, match="needs the event calendar"):
        ProphetForecaster(ADDITIVE, settings())
    assert ProphetForecaster(HISTORY_ONLY, settings()).regressors == ()
    assert ProphetForecaster(ADDITIVE, settings(), calendar=synthetic_calendar()).regressors == tuple(EVENT_FLAG_COLUMNS)
    with pytest.raises(ValueError, match="Unknown Prophet variant"):
        ProphetForecaster.from_config("Prophet(garch)", load_config())
    assert ProphetForecaster.from_config("Prophet(multiplicative)", load_config(), calendar=synthetic_calendar()).name == "Prophet(multiplicative)"


def test_predict_before_fit_raises_a_clear_error():
    forecaster = ProphetForecaster(HISTORY_ONLY, settings())
    assert not forecaster.is_fitted and forecaster.fit_diagnostics() == {}
    for call in (lambda: forecaster.predict(3), lambda: forecaster.predict_interval(3), forecaster.forecast_index,
                 lambda: forecaster.predict_frame(pd.date_range("2020-01-01", periods=3))):
        with pytest.raises(RuntimeError, match="not been fitted"):
            call()


@pytest.mark.parametrize("variant", [ADDITIVE, MULTIPLICATIVE, HISTORY_ONLY], ids=lambda v: v.name)
def test_every_variant_forecasts_the_right_days_non_negative_with_ordered_intervals(prophet_module, variant):
    history = synthetic_series(300)
    forecaster = ProphetForecaster(variant, settings(), calendar=synthetic_calendar())
    assert forecaster.fit(history) is forecaster
    forecast = forecaster.predict(HORIZON)
    lower, upper = forecaster.predict_interval(HORIZON)
    assert forecast.shape == lower.shape == upper.shape == (HORIZON,)
    assert np.all(np.isfinite(forecast)) and np.all(forecast >= 0) and np.all(lower >= 0)
    assert np.all(lower <= forecast + 1e-9) and np.all(forecast <= upper + 1e-9)
    dates = forecaster.forecast_index(HORIZON)
    assert dates[0] == history.index[-1] + pd.Timedelta(days=1) and len(dates) == HORIZON and dates.freqstr == "D"
    np.testing.assert_allclose(forecaster.predict(5), forecast[:5])                      # a shorter horizon is a prefix
    np.testing.assert_allclose(forecaster.predict_interval(5)[1], upper[:5])


def test_a_longer_horizon_than_the_configured_one_is_extrapolated_only_where_the_calendar_reaches(prophet_module):
    history = synthetic_series(300)                                                       # ends 2020-10-26
    forecaster = ProphetForecaster(HISTORY_ONLY, settings(), calendar=None).fit(history)
    assert forecaster.predict(40).shape == (40,)
    with_regressors = ProphetForecaster(ADDITIVE, settings(), calendar=synthetic_calendar(start="2019-06-01", n=560)).fit(history)
    assert with_regressors.predict(15).shape == (15,)
    with pytest.raises(ValueError, match="does not cover"):
        with_regressors.predict(200)
    with pytest.raises(ValueError, match="positive integer"):
        with_regressors.predict(0)


def test_forecasts_and_intervals_are_deterministic_and_do_not_disturb_the_callers_random_state(prophet_module):
    history, calendar = synthetic_series(300), synthetic_calendar()
    np.random.seed(2024)
    expected_next = np.random.rand(3)
    np.random.seed(2024)
    first = ProphetForecaster(ADDITIVE, settings(), calendar=calendar).fit(history)
    second = ProphetForecaster(ADDITIVE, settings(), calendar=calendar).fit(history)
    np.testing.assert_array_equal(first.predict(HORIZON), second.predict(HORIZON))
    np.testing.assert_array_equal(first.predict_interval(HORIZON)[0], second.predict_interval(HORIZON)[0])
    np.testing.assert_array_equal(first.predict_interval(HORIZON)[1], second.predict_interval(HORIZON)[1])
    np.testing.assert_array_equal(np.random.rand(3), expected_next)                       # global RNG state restored after both fits


def test_the_forecast_depends_only_on_the_history_it_was_given(prophet_module):
    full = synthetic_series(340)
    history, calendar = full.iloc[:300], synthetic_calendar()
    reference = ProphetForecaster(ADDITIVE, settings(), calendar=calendar).fit(history).predict(HORIZON)
    future_changed = full.copy()
    future_changed.iloc[300:] = 99999.0                                                    # sales after the origin exist but can never be read
    again = ProphetForecaster(ADDITIVE, settings(), calendar=calendar).fit(future_changed.iloc[:300]).predict(HORIZON)
    np.testing.assert_array_equal(reference, again)


def test_future_regressor_values_come_from_the_calendar_and_actually_matter(prophet_module):
    history = synthetic_series(300)
    calendar = synthetic_calendar()
    target_days = future_dates(history.index[-1], HORIZON)
    changed = calendar.copy()
    changed.loc[target_days, "snap_ca"] = 1 - changed.loc[target_days, "snap_ca"]          # flip the SNAP flag on the forecast days only
    base = ProphetForecaster(ADDITIVE, settings(), calendar=calendar).fit(history).predict(HORIZON)
    flipped = ProphetForecaster(ADDITIVE, settings(), calendar=changed).fit(history).predict(HORIZON)
    assert not np.allclose(base, flipped)                                                  # the regressor variant uses the future calendar
    history_only_a = ProphetForecaster(HISTORY_ONLY, settings(), calendar=calendar).fit(history).predict(HORIZON)
    history_only_b = ProphetForecaster(HISTORY_ONLY, settings(), calendar=changed).fit(history).predict(HORIZON)
    np.testing.assert_array_equal(history_only_a, history_only_b)                          # the history-only variant ignores the calendar entirely
    beyond = calendar.copy()
    beyond.loc[beyond.index > target_days[-1], :] = 1                                      # flags AFTER the forecast window can never matter
    np.testing.assert_array_equal(ProphetForecaster(ADDITIVE, settings(), calendar=beyond).fit(history).predict(HORIZON), base)


def test_a_calendar_that_does_not_cover_the_forecast_days_fails_fast(prophet_module):
    history = synthetic_series(300)                                                        # ends 2020-10-26
    short = synthetic_calendar(start="2019-06-01", n=(history.index[-1] - pd.Timestamp("2019-06-01")).days + 4)
    forecaster = ProphetForecaster(ADDITIVE, settings(), calendar=short)
    with pytest.raises(ValueError, match="does not cover"):
        forecaster.fit(history)
    assert not forecaster.is_fitted
    late = synthetic_calendar(start="2020-03-01", n=400)                                    # does not cover the training days
    with pytest.raises(ValueError, match="does not cover"):
        ProphetForecaster(ADDITIVE, settings(), calendar=late).fit(history)


def test_a_history_that_is_too_short_or_invalid_is_refused(prophet_module):
    with pytest.raises(ValueError, match="too short"):
        ProphetForecaster(HISTORY_ONLY, settings()).fit(synthetic_series(20))
    with pytest.raises(ValueError, match="negative"):
        ProphetForecaster(HISTORY_ONLY, settings()).fit(synthetic_series(60).where(lambda s: s.index != s.index[3], -5.0))


def test_a_falling_series_is_clipped_at_zero_and_the_clipping_is_reported(prophet_module):
    t = np.arange(200)
    history = pd.Series(np.clip(400 - 2.4 * t, 0, None), index=pd.date_range("2020-01-01", periods=200, freq="D"))
    forecaster = ProphetForecaster(HISTORY_ONLY, settings(weekly_seasonality=False, yearly_seasonality=False)).fit(history)
    forecast = forecaster.predict(HORIZON)
    assert np.all(forecast >= 0) and np.all(forecaster.predict_interval(HORIZON)[0] >= 0)
    assert forecaster.fit_diagnostics()["n_negative_values_clipped"] > 0


def test_diagnostics_describe_the_fit_and_are_json_friendly(prophet_module):
    history = synthetic_series(300)
    forecaster = ProphetForecaster(MULTIPLICATIVE, settings(), calendar=synthetic_calendar()).fit(history)
    d = forecaster.fit_diagnostics()
    assert d["model"] == "Prophet(multiplicative)" and d["seasonality_mode"] == "multiplicative" and d["uses_regressors"] is True
    assert d["n_obs"] == 300 and d["history_end"] == str(history.index[-1].date()) and d["regressors"] == list(EVENT_FLAG_COLUMNS)
    assert d["n_changepoints_fitted"] == 25 and d["estimation"].startswith("MAP")
    assert d["optimizer_converged"] is True and d["fit_seconds"] > 0 and d["prophet_version"]
    json.dumps(d)
    assert ProphetForecaster(HISTORY_ONLY, settings()).fit(history).fit_diagnostics()["regressors"] == []


def test_predict_frame_exposes_components_for_history_days_without_clipping(prophet_module):
    history = synthetic_series(300)
    forecaster = ProphetForecaster(ADDITIVE, settings(), calendar=synthetic_calendar()).fit(history)
    frame = forecaster.predict_frame(history.index[-60:])
    assert len(frame) == 60 and {"trend", "weekly", "yearly", "snap_ca", "yhat"} <= set(frame.columns)
    assert pd.DatetimeIndex(frame["ds"]).equals(history.index[-60:])
    weekly = forecaster.predict_frame(pd.date_range("2020-03-02", periods=7, freq="D"))["weekly"]
    assert abs(float(weekly.sum())) < 1e-6 * 7 + 5.0                                       # a weekly pattern sums to about zero over a week


def test_run_backtest_records_a_failed_fold_and_scores_the_others(prophet_module):
    series = synthetic_series(60)                          # fold 1 trains on 60 - 3*14 = 18 days: below the 28-day minimum
    results = run_backtest(series, lambda: ProphetForecaster(HISTORY_ONLY, settings()), horizon=HORIZON, n_folds=3)
    assert [r.success for r in results] == [False, True, True]
    assert "too short" in results[0].error and results[0].predictions is None
    table = backtest_results_to_forecast_table(results)
    assert table.loc[table["fold"] == 1, "predicted"].isna().all()                          # no substitute forecast


def test_run_backtest_integration_produces_audited_non_negative_forecasts(prophet_module):
    series = synthetic_series(300)
    results = run_backtest(series, lambda: ProphetForecaster(ADDITIVE, settings(), calendar=synthetic_calendar()), horizon=HORIZON, n_folds=3)
    assert all(r.success for r in results) and [r.fold for r in results] == [1, 2, 3]
    for result in results:
        assert len(result.predictions) == HORIZON and np.all(result.predictions >= 0)
        assert result.dates[0] == result.origin_date + pd.Timedelta(days=1)
        assert result.diagnostics["model"] == "Prophet(additive)" and result.diagnostics["optimizer_converged"] is True
    checks = audit_forecast_table(backtest_results_to_forecast_table(results), kind="aggregate", dev_series=series, horizon=HORIZON, n_folds=3)
    assert [c.name for c in checks if not c.passed] == []
