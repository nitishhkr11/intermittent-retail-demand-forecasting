"""Tests for src/data/aggregate_series.py: the development-only aggregate series loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.aggregate_series import (
    DevelopmentWindow,
    aggregate_series_path,
    load_aggregate_development,
    load_aggregate_development_series,
)
from src.utils.config import load_config
from src.utils.paths import resolve_path

CONFIG = {"backtest": {"holdout_days": 28}}


def _write(tmp_path, n_days=100, start="2020-01-01", **overrides):
    dates = pd.date_range(start, periods=n_days, freq="D")
    table = pd.DataFrame({"date": dates, "sales": np.arange(n_days, dtype=float) + 10.0})
    for column, value in overrides.items():
        table[column] = value
    path = tmp_path / "aggregate.parquet"
    table.to_parquet(path)
    return path, table


def test_series_ends_on_the_last_development_date_and_window_holds_dates_only(tmp_path):
    path, table = _write(tmp_path, n_days=100)
    series, window = load_aggregate_development(CONFIG, path=path)
    assert isinstance(window, DevelopmentWindow)
    assert len(series) == 72 and window.n_development_days == 72 and window.n_holdout_days == 28
    assert window.n_total_days == 100
    assert series.index[0] == pd.Timestamp("2020-01-01") == window.dev_start
    assert series.index[-1] == window.dev_end == pd.Timestamp("2020-03-12")
    assert window.holdout_start == pd.Timestamp("2020-03-13") and window.holdout_end == pd.Timestamp("2020-04-09")
    assert series.index.max() < window.holdout_start
    assert series.dtype == float and series.index.is_monotonic_increasing and series.index.freqstr in (None, "D")


def test_holdout_values_never_reach_the_returned_series(tmp_path):
    path, table = _write(tmp_path, n_days=100)
    baseline, _ = load_aggregate_development(CONFIG, path=path)
    perturbed = table.copy()
    perturbed.loc[perturbed.index[-28:], "sales"] = 1e9
    perturbed.to_parquet(path)
    changed, _ = load_aggregate_development(CONFIG, path=path)
    pd.testing.assert_series_equal(baseline, changed)
    assert changed.max() < 1e6


def test_series_shortcut_returns_only_the_series(tmp_path):
    path, _ = _write(tmp_path, n_days=60)
    series = load_aggregate_development_series(CONFIG, path=path)
    assert isinstance(series, pd.Series) and len(series) == 32


def test_missing_file_message_names_the_file_and_the_command(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_aggregate_development(CONFIG, path=tmp_path / "nope.parquet")
    message = str(excinfo.value)
    assert "nope.parquet" in message and "python -m src.analysis.demand_classification" in message
    assert str(tmp_path) not in message          # no machine-specific folder in the message


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda t: t.drop(columns=["sales"]), "missing column"),
        (lambda t: pd.concat([t, t.iloc[[3]]]).sort_values("date"), "duplicated dates"),
        (lambda t: t.drop(index=[10, 11]), "not a continuous daily series"),
        (lambda t: t.assign(sales=t["sales"].where(t.index != 5)), "missing sales"),
        (lambda t: t.assign(sales=t["sales"].where(t.index != 5, -1.0)), "negative sales"),
    ],
)
def test_invalid_tables_are_rejected_with_a_specific_message(tmp_path, mutate, expected):
    _, table = _write(tmp_path, n_days=80)
    path = tmp_path / "bad.parquet"
    mutate(table).to_parquet(path)
    with pytest.raises(ValueError, match=expected):
        load_aggregate_development(CONFIG, path=path)


def test_unsorted_input_is_sorted_before_splitting(tmp_path):
    _, table = _write(tmp_path, n_days=80)
    path = tmp_path / "shuffled.parquet"
    table.sample(frac=1.0, random_state=1).to_parquet(path)
    series, window = load_aggregate_development(CONFIG, path=path)
    assert series.index.is_monotonic_increasing and window.dev_end == pd.Timestamp("2020-02-21")


def test_real_project_series_has_the_documented_calendar():
    config = load_config()
    if not aggregate_series_path(config).exists():
        pytest.skip("the processed aggregate parquet file is not present")
    series, window = load_aggregate_development(config)
    assert len(series) == 1913 and window.n_holdout_days == 28
    assert series.index[0] == pd.Timestamp("2011-01-29")
    assert series.index[-1] == pd.Timestamp("2016-04-24")
    assert window.holdout_start == pd.Timestamp("2016-04-25") and window.holdout_end == pd.Timestamp("2016-05-22")
    assert aggregate_series_path(config) == resolve_path(
        config["data"]["processed_dir"] + "/" + config["output"]["aggregate_filename"]
    )
