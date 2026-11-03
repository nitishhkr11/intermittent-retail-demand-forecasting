"""Tests for src/models/lstm_gru.py (direct multi-horizon LSTM / GRU forecasters).

Every series here is SYNTHETIC (named `synthetic_*`); nothing in this file is an M5 result.
Pure helpers (windows, scaler, calendar features, configuration) are tested without torch;
tests that train a network skip themselves when torch is not installed.
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
from src.models.lstm_gru import (
    CALENDAR_FEATURE_NAMES,
    NeuralConfig,
    NeuralSequenceForecaster,
    Scaler,
    build_windows,
    calendar_feature_matrix,
    check_history,
    make_forecast_input,
    split_origins,
    variant_name,
    variants_from_config,
    window_indices,
)
from src.models.xgboost_model import EVENT_FLAG_COLUMNS
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

HORIZON = 7
TINY = dict(
    input_window=7, hidden_size=4, num_layers=1, dropout=0.0, learning_rate=0.01, weight_decay=0.0,
    batch_size=16, max_epochs=3, patience=2, validation_days=14, loss="huber", huber_delta=1.0,
    grad_clip=1.0, seeds=(0, 1), torch_num_threads=1, horizon=HORIZON,
)


def tiny_config(**overrides) -> NeuralConfig:
    return NeuralConfig(**{**TINY, **overrides})


def synthetic_series(n=120, start="2020-01-01", seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 50 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n)
    return pd.Series(np.clip(values, 0, None), index=pd.date_range(start, periods=n, freq="D"), name="synthetic_sales")


def synthetic_calendar(start="2019-12-01", n=400) -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="D", name="date")
    flags = pd.DataFrame(0, index=index, columns=list(EVENT_FLAG_COLUMNS))
    flags.loc[index.dayofweek == 5, "snap_ca"] = 1
    flags.loc[index.day == 25, "event_religious"] = 1
    return flags


@pytest.fixture(scope="module")
def torch_module():
    return pytest.importorskip("torch")


# --------------------------------------------------------------------------- configuration

def test_real_config_is_valid_and_matches_the_documented_settings():
    cfg = NeuralConfig.from_config(load_config())
    assert (cfg.input_window, cfg.hidden_size, cfg.num_layers, cfg.horizon) == (56, 32, 1, 28)
    assert cfg.seeds == (0, 1, 2, 3, 4) and cfg.loss == "huber" and cfg.validation_days == 56
    assert cfg.required_history == 56 + 28 + 56 + 10 - 1
    variants = variants_from_config(load_config())
    assert [v.name for v in variants] == ["LSTM(calendar)", "GRU(calendar)", "LSTM(univariate)", "GRU(univariate)"]


@pytest.mark.parametrize(
    "key, bad",
    [
        ("input_window", 1), ("hidden_size", 0), ("num_layers", 0), ("batch_size", 0), ("max_epochs", 0), ("patience", 0),
        ("dropout", 1.0), ("dropout", -0.1), ("learning_rate", 0.0), ("learning_rate", float("nan")), ("weight_decay", -1.0),
        ("loss", "l1"), ("validation_days", HORIZON - 1), ("seeds", ()), ("seeds", (1, 1)), ("seeds", (0.5,)),
        ("torch_num_threads", 0), ("grad_clip", 0.0), ("huber_delta", -1.0), ("horizon", 0),
    ],
)
def test_invalid_settings_raise_a_message_naming_the_key(key, bad):
    with pytest.raises(ValueError, match=key):
        tiny_config(**{key: bad})


@pytest.mark.parametrize("key, bad", [("hidden_size", "big"), ("max_epochs", 2.5), ("learning_rate", None), ("seeds", None), ("loss", "")])
def test_invalid_config_file_values_are_rejected_with_the_key_name(key, bad):
    config = load_config()
    config["neural"][key] = bad
    with pytest.raises(ValueError, match=f"neural.{key}"):
        NeuralConfig.from_config(config)


def test_missing_sections_and_unknown_variants_are_rejected():
    with pytest.raises(ValueError, match="no 'neural' section"):
        NeuralConfig.from_config({"backtest": {"horizon": 28}})
    with pytest.raises(ValueError, match="backtest.horizon"):
        NeuralConfig.from_config({"neural": load_config()["neural"], "backtest": {}})
    config = load_config()
    config["neural"]["variants"] = [{"cell": "rnn", "features": "calendar"}]
    with pytest.raises(ValueError, match="variants"):
        variants_from_config(config)
    config["neural"]["variants"] = [{"cell": "lstm", "features": "calendar"}, {"cell": "lstm", "features": "calendar"}]
    with pytest.raises(ValueError, match="twice"):
        variants_from_config(config)
    config["neural"]["variants"] = []
    with pytest.raises(ValueError, match="variants"):
        variants_from_config(config)


def test_variant_names_are_the_stable_artifact_keys():
    assert variant_name("lstm", "calendar") == "LSTM(calendar)"
    assert variant_name("gru", "univariate") == "GRU(univariate)"
    with pytest.raises(ValueError):
        variant_name("transformer", "calendar")
    with pytest.raises(ValueError):
        variant_name("lstm", "multivariate")


# --------------------------------------------------------------------------- history validation

def test_check_history_accepts_a_clean_daily_series_and_rejects_everything_else():
    series = synthetic_series(30)
    np.testing.assert_allclose(check_history(series), series.to_numpy())
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


# --------------------------------------------------------------------------- scaler

def test_scaler_round_trips_and_handles_a_constant_series():
    values = np.array([2.0, 4.0, 6.0, 8.0])
    scaler = Scaler.fit(values)
    assert scaler.mean == 5.0 and scaler.scale == pytest.approx(np.sqrt(5.0))
    np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values)
    constant = Scaler.fit([3.0, 3.0, 3.0])
    assert constant.scale == 1.0 and np.all(constant.transform([3.0, 3.0]) == 0.0)
    with pytest.raises(ValueError):
        Scaler.fit([1.0])
    with pytest.raises(ValueError):
        Scaler.fit([1.0, np.nan])


# --------------------------------------------------------------------------- windows and the chronological split

def test_window_indices_read_the_last_input_window_days_and_the_next_horizon_days():
    past, future = window_indices([10, 11], input_window=4, horizon=3)
    np.testing.assert_array_equal(past, [[7, 8, 9, 10], [8, 9, 10, 11]])
    np.testing.assert_array_equal(future, [[11, 12, 13], [12, 13, 14]])


def test_build_windows_cuts_inputs_and_targets_by_position():
    values = np.arange(30, dtype=float)
    covariates = np.column_stack([np.arange(30) * 10, np.arange(30) * 100]).astype(float)
    origins = split_origins(30, input_window=5, horizon=3, validation_days=6)[0][:2]
    past, future, target = build_windows(values, covariates, origins, input_window=5, horizon=3)
    t = origins[0]
    np.testing.assert_array_equal(past[0, :, 0], np.arange(t - 4, t + 1))
    np.testing.assert_array_equal(past[0, :, 1], covariates[t - 4:t + 1, 0])
    np.testing.assert_array_equal(target[0], np.arange(t + 1, t + 4))
    np.testing.assert_array_equal(future[0, :, 1], covariates[t + 1:t + 4, 1])
    assert past.shape == (2, 5, 3) and future.shape == (2, 3, 2) and target.shape == (2, 3)
    assert past.dtype == np.float32 and target.dtype == np.float32
    univariate_past, univariate_future, _ = build_windows(values, None, origins, 5, 3)
    assert univariate_past.shape == (2, 5, 1) and univariate_future.shape == (2, 3, 0)


def test_build_windows_refuses_origins_whose_windows_leave_the_history():
    values = np.arange(30, dtype=float)
    with pytest.raises(ValueError, match="origin positions"):
        build_windows(values, None, [27], input_window=5, horizon=3)        # target would run past the end
    with pytest.raises(ValueError, match="origin positions"):
        build_windows(values, None, [3], input_window=5, horizon=3)         # input window would start before day 0
    with pytest.raises(ValueError, match="covariates has"):
        build_windows(values, np.zeros((29, 2)), [10], 5, 3)


def test_training_targets_end_before_the_validation_tail_and_validation_targets_lie_inside_it():
    n_obs, window, horizon, tail = 100, 10, 7, 14
    train, validation = split_origins(n_obs, window, horizon, tail)
    tail_start = n_obs - tail
    assert train.min() == window - 1
    assert train.max() + horizon == tail_start - 1                   # last training target is the day BEFORE the tail
    assert validation.min() + 1 == tail_start                        # first validation target is the first tail day
    assert validation.max() + horizon == n_obs - 1                   # last validation target is the last history day
    assert len(validation) == tail - horizon + 1
    assert set(train).isdisjoint(set(validation))


def test_training_arrays_do_not_change_when_the_validation_tail_values_change():
    values = synthetic_series(100).to_numpy()
    perturbed = values.copy()
    perturbed[100 - 14:] += 1000.0
    train, _ = split_origins(100, 10, 7, 14)
    a = build_windows(values, None, train, 10, 7)
    b = build_windows(perturbed, None, train, 10, 7)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)


def test_split_origins_rejects_too_short_histories_and_short_validation_tails():
    with pytest.raises(ValueError, match="too short"):
        split_origins(30, 10, 7, 14)
    with pytest.raises(ValueError, match="at least the horizon"):
        split_origins(100, 10, 7, 6)


def test_forecast_input_is_the_last_window_plus_the_future_calendar():
    values = np.arange(20, dtype=float)
    past_cov = np.column_stack([np.arange(20), np.arange(20) + 0.5]).astype(float)
    future_cov = np.full((4, 2), 7.0)
    past, future = make_forecast_input(values, past_cov, future_cov, input_window=5, horizon=4)
    np.testing.assert_array_equal(past[0, :, 0], np.arange(15, 20))
    np.testing.assert_array_equal(past[0, :, 1], np.arange(15, 20))
    assert future.shape == (1, 4, 2) and np.all(future == 7.0)
    univariate_past, univariate_future = make_forecast_input(values, None, None, 5, 4)
    assert univariate_past.shape == (1, 5, 1) and univariate_future.shape == (1, 4, 0)
    with pytest.raises(ValueError, match="both"):
        make_forecast_input(values, past_cov, None, 5, 4)
    with pytest.raises(ValueError, match="cover 4 days"):
        make_forecast_input(values, past_cov, future_cov[:3], 5, 4)
    with pytest.raises(ValueError, match="at least 5"):
        make_forecast_input(values[:3], None, None, 5, 4)


# --------------------------------------------------------------------------- calendar features

def test_calendar_matrix_has_one_hot_weekdays_and_the_event_flags():
    calendar = synthetic_calendar()
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    matrix = calendar_feature_matrix(index, calendar)
    assert matrix.shape == (10, len(CALENDAR_FEATURE_NAMES)) == (10, 13) and matrix.dtype == np.float32
    assert np.all(matrix[:, :7].sum(axis=1) == 1)
    assert matrix[0, 2] == 1                                           # 2020-01-01 was a Wednesday
    snap = CALENDAR_FEATURE_NAMES.index("snap_ca")
    np.testing.assert_array_equal(matrix[:, snap], (index.dayofweek == 5).astype(float))


def test_calendar_matrix_never_fills_missing_dates_or_accepts_non_binary_flags():
    calendar = synthetic_calendar(start="2020-01-01", n=10)
    with pytest.raises(ValueError, match="does not cover 5"):
        calendar_feature_matrix(pd.date_range("2020-01-06", periods=10, freq="D"), calendar)
    with pytest.raises(ValueError, match="missing column"):
        calendar_feature_matrix(calendar.index, calendar.drop(columns=["snap_ca"]))
    bad = calendar.copy()
    bad.iloc[2, 0] = 2
    with pytest.raises(ValueError, match="0/1"):
        calendar_feature_matrix(calendar.index, bad)
    with pytest.raises(ValueError, match="indexed by date"):
        calendar_feature_matrix(calendar.index, calendar.reset_index(drop=True))


# --------------------------------------------------------------------------- missing torch

def test_using_the_model_without_torch_raises_an_informative_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)                     # makes `import torch` raise ImportError
    forecaster = NeuralSequenceForecaster("lstm", "univariate", tiny_config())
    with pytest.raises(ImportError) as excinfo:
        forecaster.fit(synthetic_series())
    message = str(excinfo.value)
    assert "python -m pip install torch" in message and "requirements-phase3.txt" in message
    assert "No other method is substituted" in message


def test_the_module_imports_without_torch():
    code = "import sys; sys.modules['torch'] = None; import src.models.lstm_gru as m; print(m.variant_name('gru', 'calendar'))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "GRU(calendar)"


# --------------------------------------------------------------------------- the forecaster (needs torch)

def test_constructor_validates_variant_and_calendar():
    with pytest.raises(ValueError, match="cell"):
        NeuralSequenceForecaster("rnn", "univariate", tiny_config())
    with pytest.raises(ValueError, match="features"):
        NeuralSequenceForecaster("lstm", "full", tiny_config())
    with pytest.raises(ValueError, match="need the event calendar"):
        NeuralSequenceForecaster("lstm", "calendar", tiny_config())
    with pytest.raises(TypeError, match="NeuralConfig"):
        NeuralSequenceForecaster("lstm", "univariate", {"hidden_size": 4})
    assert NeuralSequenceForecaster("gru", "calendar", tiny_config(), calendar=synthetic_calendar()).name == "GRU(calendar)"


def test_predict_before_fit_raises_a_clear_error():
    forecaster = NeuralSequenceForecaster("lstm", "univariate", tiny_config())
    assert not forecaster.is_fitted
    for call in (lambda: forecaster.predict(3), forecaster.seed_predictions, forecaster.training_history, forecaster.forecast_index):
        with pytest.raises(RuntimeError, match="not been fitted"):
            call()


@pytest.mark.parametrize("cell", ["lstm", "gru"])
@pytest.mark.parametrize("features", ["calendar", "univariate"])
def test_every_variant_fits_and_forecasts_the_right_number_of_non_negative_days(torch_module, cell, features):
    history = synthetic_series(120)
    forecaster = NeuralSequenceForecaster(cell, features, tiny_config(), calendar=synthetic_calendar())
    assert forecaster.fit(history) is forecaster
    forecast = forecaster.predict(HORIZON)
    assert forecast.shape == (HORIZON,) and forecast.dtype == float
    assert np.all(np.isfinite(forecast)) and np.all(forecast >= 0)
    assert forecaster.predict(3).shape == (3,)
    np.testing.assert_allclose(forecaster.predict(3), forecast[:3])        # a shorter horizon is a prefix of the full forecast
    dates = forecaster.forecast_index(HORIZON)
    assert dates[0] == history.index[-1] + pd.Timedelta(days=1) and len(dates) == HORIZON
    assert dates.freqstr == "D"


def test_the_horizon_is_checked(torch_module):
    forecaster = NeuralSequenceForecaster("gru", "univariate", tiny_config()).fit(synthetic_series())
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            forecaster.predict(bad)
    with pytest.raises(ValueError, match="at most 7 days"):
        forecaster.predict(HORIZON + 1)
    with pytest.raises(ValueError, match="at most 7 days"):
        forecaster.seed_predictions(HORIZON + 1)


def test_the_reported_forecast_is_the_mean_of_the_seed_forecasts(torch_module):
    forecaster = NeuralSequenceForecaster("lstm", "univariate", tiny_config(seeds=(0, 1, 2))).fit(synthetic_series())
    members = forecaster.seed_predictions()
    assert members.shape == (3, HORIZON) and np.all(members >= 0)
    np.testing.assert_allclose(forecaster.predict(HORIZON), members.mean(axis=0))
    assert not np.allclose(members[0], members[1])                         # different seeds give different members


def test_same_seed_gives_bit_identical_forecasts_and_different_seeds_do_not(torch_module):
    history = synthetic_series()
    first = NeuralSequenceForecaster("gru", "calendar", tiny_config(), calendar=synthetic_calendar()).fit(history).predict(HORIZON)
    second = NeuralSequenceForecaster("gru", "calendar", tiny_config(), calendar=synthetic_calendar()).fit(history).predict(HORIZON)
    np.testing.assert_array_equal(first, second)
    other = NeuralSequenceForecaster("gru", "calendar", tiny_config(seeds=(5, 6)), calendar=synthetic_calendar()).fit(history).predict(HORIZON)
    assert not np.array_equal(first, other)


def test_fitting_does_not_change_the_callers_global_random_state(torch_module):
    import torch

    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    NeuralSequenceForecaster("lstm", "univariate", tiny_config(seeds=(0,))).fit(synthetic_series())
    np.testing.assert_array_equal(torch.rand(3).numpy(), expected.numpy())


def test_scaler_statistics_come_from_the_inner_training_part_only(torch_module):
    history = synthetic_series(120)
    forecaster = NeuralSequenceForecaster("lstm", "univariate", tiny_config()).fit(history)
    diagnostics = forecaster.fit_diagnostics()
    inner_train = history.to_numpy()[: 120 - 14]
    assert diagnostics["scaler_mean"] == pytest.approx(inner_train.mean())
    assert diagnostics["scaler_scale"] == pytest.approx(inner_train.std(ddof=0))
    assert diagnostics["scaler_fit_end_date"] == str(history.index[120 - 14 - 1].date())
    # changing only the validation tail must not move the scaler
    perturbed = history.copy()
    perturbed.iloc[-14:] += 500.0
    other = NeuralSequenceForecaster("lstm", "univariate", tiny_config()).fit(perturbed).fit_diagnostics()
    assert other["scaler_mean"] == diagnostics["scaler_mean"] and other["scaler_scale"] == diagnostics["scaler_scale"]


def test_the_forecast_depends_only_on_the_history_it_was_given(torch_module):
    full = synthetic_series(150)
    history = full.iloc[:120]
    reference = NeuralSequenceForecaster("lstm", "calendar", tiny_config(), calendar=synthetic_calendar()).fit(history).predict(HORIZON)
    # values AFTER the history end exist in `full` but can never be read: rebuilding the history from a perturbed future changes nothing
    future_changed = full.copy()
    future_changed.iloc[120:] = 9999.0
    again = NeuralSequenceForecaster("lstm", "calendar", tiny_config(), calendar=synthetic_calendar()).fit(future_changed.iloc[:120]).predict(HORIZON)
    np.testing.assert_array_equal(reference, again)


def test_a_calendar_that_does_not_cover_the_forecast_days_raises(torch_module):
    history = synthetic_series(120)                                        # ends 2020-04-29
    short_calendar = synthetic_calendar(start="2019-12-01", n=(history.index[-1] - pd.Timestamp("2019-12-01")).days + 3)
    forecaster = NeuralSequenceForecaster("lstm", "calendar", tiny_config(), calendar=short_calendar)
    with pytest.raises(ValueError, match="does not cover"):
        forecaster.fit(history)
    assert not forecaster.is_fitted
    late = synthetic_calendar(start="2020-02-01", n=300)                  # does not cover the training days
    with pytest.raises(ValueError, match="does not cover"):
        NeuralSequenceForecaster("gru", "calendar", tiny_config(), calendar=late).fit(history)


def test_a_history_that_is_too_short_is_refused_not_fitted(torch_module):
    forecaster = NeuralSequenceForecaster("lstm", "univariate", tiny_config())
    with pytest.raises(ValueError, match="too short"):
        forecaster.fit(synthetic_series(30))
    assert not forecaster.is_fitted


def test_a_zero_series_still_gives_zero_or_tiny_non_negative_forecasts(torch_module):
    history = pd.Series(0.0, index=pd.date_range("2020-01-01", periods=100, freq="D"))
    forecaster = NeuralSequenceForecaster("gru", "univariate", tiny_config()).fit(history)
    forecast = forecaster.predict(HORIZON)
    assert np.all(forecast >= 0) and np.all(np.isfinite(forecast))
    assert isinstance(forecaster.fit_diagnostics()["n_negative_values_clipped"], int)


def test_negative_raw_outputs_are_clipped_and_counted(torch_module, monkeypatch):
    forecaster = NeuralSequenceForecaster("lstm", "univariate", tiny_config(seeds=(0,)))
    original_inverse = Scaler.inverse
    monkeypatch.setattr(Scaler, "inverse", lambda self, values: original_inverse(self, values) - 1e6)
    forecaster.fit(synthetic_series())
    assert np.all(forecaster.predict(HORIZON) == 0.0)
    assert forecaster.fit_diagnostics()["n_negative_values_clipped"] == HORIZON


def test_training_history_and_diagnostics_describe_the_fit(torch_module):
    history = synthetic_series()
    forecaster = NeuralSequenceForecaster("gru", "calendar", tiny_config(max_epochs=4, patience=3), calendar=synthetic_calendar()).fit(history)
    log = forecaster.training_history()
    assert list(log.columns) == ["seed", "epoch", "train_loss", "val_loss", "is_best_epoch"]
    assert set(log["seed"]) == {0, 1} and np.isfinite(log[["train_loss", "val_loss"]].to_numpy()).all()
    assert log.groupby("seed")["is_best_epoch"].sum().eq(1).all()
    best = log[log["is_best_epoch"]].set_index("seed")
    assert (best["val_loss"] == log.groupby("seed")["val_loss"].min()).all()      # the restored weights are the lowest-validation-loss epoch
    diagnostics = forecaster.fit_diagnostics()
    assert diagnostics["model"] == "GRU(calendar)" and diagnostics["n_obs"] == 120 and diagnostics["seeds"] == [0, 1]
    assert diagnostics["n_train_windows"] == 120 - 14 - HORIZON - (7 - 1) and diagnostics["n_validation_windows"] == 14 - HORIZON + 1
    assert diagnostics["n_past_features"] == 14 and diagnostics["n_future_features"] == 13
    assert all(1 <= e <= 4 for e in diagnostics["epochs_run"]) and diagnostics["history_end"] == str(history.index[-1].date())
    json.dumps(diagnostics)                                                            # JSON-friendly for the saved diagnostics file
    univariate = NeuralSequenceForecaster("lstm", "univariate", tiny_config()).fit(history).fit_diagnostics()
    assert univariate["n_past_features"] == 1 and univariate["n_future_features"] == 0


def test_early_stopping_stops_after_patience_epochs_without_improvement(torch_module):
    forecaster = NeuralSequenceForecaster("lstm", "univariate", tiny_config(max_epochs=60, patience=2, learning_rate=0.05)).fit(synthetic_series())
    diagnostics = forecaster.fit_diagnostics()
    for ran, best, stopped in zip(diagnostics["epochs_run"], diagnostics["best_epoch"], diagnostics["stopped_early"]):
        assert ran <= 60
        if stopped:
            assert ran - best == 2                                                     # patience epochs after the best one


def test_run_backtest_records_a_failed_fold_instead_of_hiding_it(torch_module):
    series = synthetic_series(60)                                  # fold 1 trains on 60 - 5*7 = 25 days: too short for this config
    results = run_backtest(series, lambda: NeuralSequenceForecaster("lstm", "univariate", tiny_config()), horizon=HORIZON, n_folds=5)
    assert [r.success for r in results] == [False, False, True, True, True]
    failed = [r for r in results if not r.success]
    assert all("too short" in r.error and r.predictions is None for r in failed)
    table = backtest_results_to_forecast_table(results)
    assert table.loc[table["fold"] == failed[0].fold, "predicted"].isna().all()    # no substitute forecast


def test_run_backtest_integration_produces_audited_non_negative_forecasts(torch_module):
    series = synthetic_series(140)
    results = run_backtest(series, lambda: NeuralSequenceForecaster("gru", "calendar", tiny_config(), calendar=synthetic_calendar()), horizon=HORIZON, n_folds=2)
    assert all(r.success for r in results) and [r.fold for r in results] == [1, 2]
    for result in results:
        assert len(result.predictions) == HORIZON and np.all(result.predictions >= 0)
        assert result.dates[0] == result.origin_date + pd.Timedelta(days=1)
        assert result.diagnostics["model"] == "GRU(calendar)"
    table = backtest_results_to_forecast_table(results)
    checks = audit_forecast_table(table, kind="aggregate", dev_series=series, horizon=HORIZON, n_folds=2)
    assert [c.name for c in checks if not c.passed] == []
