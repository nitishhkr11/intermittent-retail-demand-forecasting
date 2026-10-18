"""Unit tests for src/data/series_validator.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validator import DataValidationError
from src.data.series_validator import (
    check_continuous_range,
    check_daily_frequency,
    check_no_duplicate_dates,
    check_no_missing_dates,
    check_no_negative_sales,
    run_all_series_validations,
)


def _clean_series(n_days: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "sales": range(n_days)})


# ---- check_daily_frequency ---------------------------------------------------

def test_check_daily_frequency_passes_on_clean_series():
    assert check_daily_frequency(_clean_series()).passed


def test_check_daily_frequency_fails_on_two_day_gap():
    df = _clean_series()
    df.loc[5, "date"] = df.loc[5, "date"] + pd.Timedelta(days=1)
    assert not check_daily_frequency(df).passed


# ---- check_no_duplicate_dates -------------------------------------------------

def test_check_no_duplicate_dates_passes_on_clean_series():
    assert check_no_duplicate_dates(_clean_series()).passed


def test_check_no_duplicate_dates_catches_duplicate():
    df = _clean_series()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not check_no_duplicate_dates(df).passed


# ---- check_no_missing_dates ----------------------------------------------------

def test_check_no_missing_dates_passes_on_clean_series():
    assert check_no_missing_dates(_clean_series()).passed


def test_check_no_missing_dates_catches_gap():
    df = _clean_series().drop(index=5).reset_index(drop=True)
    assert not check_no_missing_dates(df).passed


# ---- check_continuous_range ----------------------------------------------------

def test_check_continuous_range_passes_on_clean_series():
    assert check_continuous_range(_clean_series()).passed


def test_check_continuous_range_catches_short_series():
    df = _clean_series().drop(index=5).reset_index(drop=True)
    assert not check_continuous_range(df).passed


# ---- check_no_negative_sales ---------------------------------------------------

def test_check_no_negative_sales_passes_on_clean_series():
    assert check_no_negative_sales(_clean_series()).passed


def test_check_no_negative_sales_catches_negative_value():
    df = _clean_series()
    df.loc[0, "sales"] = -5
    assert not check_no_negative_sales(df).passed


def test_check_no_negative_sales_allows_zero():
    # Zero-sales days (e.g. Christmas closures) are legitimate, not an error.
    df = _clean_series()
    df.loc[0, "sales"] = 0
    assert check_no_negative_sales(df).passed


# ---- run_all_series_validations -------------------------------------------------

def test_run_all_series_validations_passes_on_clean_series():
    results = run_all_series_validations(_clean_series())
    assert all(r.passed for r in results)


def test_run_all_series_validations_raises_on_duplicate_dates():
    df = _clean_series()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError):
        run_all_series_validations(df)


def test_run_all_series_validations_raises_on_negative_sales():
    df = _clean_series()
    df.loc[0, "sales"] = -1
    with pytest.raises(DataValidationError):
        run_all_series_validations(df)
