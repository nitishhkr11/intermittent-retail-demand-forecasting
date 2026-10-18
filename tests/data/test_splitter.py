"""Unit tests for src/data/splitter.py (split_final_holdout)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splitter import split_final_holdout


def _series_df(n_days: int) -> pd.DataFrame:
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "sales": range(n_days)})


def test_split_lengths():
    df = _series_df(100)
    split = split_final_holdout(df, holdout_days=28)
    assert split.dev_length == 72
    assert split.test_length == 28
    assert split.dev_length + split.test_length == len(df)


def test_split_is_contiguous_and_chronological_with_no_overlap():
    df = _series_df(100)
    split = split_final_holdout(df, holdout_days=28)
    assert split.dev_end == split.test_start - pd.Timedelta(days=1)
    assert split.development["date"].max() < split.test["date"].min()


def test_split_boundaries_match_data():
    df = _series_df(100)
    split = split_final_holdout(df, holdout_days=28)
    assert split.dev_start == df["date"].iloc[0]
    assert split.test_end == df["date"].iloc[-1]


def test_raises_if_not_sorted():
    df = _series_df(100).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        split_final_holdout(df, holdout_days=28)


def test_raises_if_date_col_missing():
    df = _series_df(100).rename(columns={"date": "ds"})
    with pytest.raises(ValueError):
        split_final_holdout(df, holdout_days=28)


def test_raises_if_too_few_rows_for_holdout():
    df = _series_df(10)
    with pytest.raises(ValueError):
        split_final_holdout(df, holdout_days=28)


def test_development_never_touches_test_rows():
    df = _series_df(100)
    split = split_final_holdout(df, holdout_days=28)
    assert set(split.development["date"]).isdisjoint(set(split.test["date"]))
