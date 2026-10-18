"""Unit tests for src/data/validator.py.

Each check gets one test on a clean panel (should pass) and one test
on a deliberately broken panel (should fail), plus a couple of tests
for the aggregate run_all_validations() runner.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validator import (
    DataValidationError,
    check_data_types,
    check_date_continuity,
    check_duplicate_keys,
    check_expected_columns,
    check_join_duplication,
    check_missing_dates,
    check_valid_sales,
    run_all_validations,
)


def _clean_panel() -> pd.DataFrame:
    """A small, structurally-valid panel: 2 items x 3 continuous days each."""
    dates = pd.date_range("2011-01-29", periods=3, freq="D")
    rows = []
    for item in ["FOODS_3_001", "FOODS_3_002"]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "item_id": item,
                    "dept_id": "FOODS_3",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "d": f"d_{i + 1}",
                    "sales": i,
                    "date": d,
                    "wm_yr_wk": 11101,
                    "sell_price": 2.5,
                }
            )
    df = pd.DataFrame(rows)
    df["item_id"] = df["item_id"].astype("category")
    df["sales"] = df["sales"].astype("int64")
    return df


# ---- check_expected_columns -------------------------------------------------

def test_check_expected_columns_passes_on_clean_panel():
    assert check_expected_columns(_clean_panel()).passed


def test_check_expected_columns_fails_when_column_missing():
    df = _clean_panel().drop(columns=["sell_price"])
    result = check_expected_columns(df)
    assert not result.passed
    assert "sell_price" in result.message


# ---- check_data_types --------------------------------------------------------

def test_check_data_types_passes_on_clean_panel():
    assert check_data_types(_clean_panel()).passed


def test_check_data_types_fails_on_wrong_sales_dtype():
    df = _clean_panel()
    df["sales"] = df["sales"].astype("float64")
    assert not check_data_types(df).passed


# ---- check_duplicate_keys ----------------------------------------------------

def test_check_duplicate_keys_passes_on_clean_panel():
    assert check_duplicate_keys(_clean_panel()).passed


def test_check_duplicate_keys_catches_duplicate_row():
    df = _clean_panel()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not check_duplicate_keys(df).passed


# ---- check_missing_dates -----------------------------------------------------

def test_check_missing_dates_passes_on_clean_panel():
    assert check_missing_dates(_clean_panel()).passed


def test_check_missing_dates_catches_null_date():
    df = _clean_panel()
    df.loc[0, "date"] = pd.NaT
    assert not check_missing_dates(df).passed


# ---- check_date_continuity ---------------------------------------------------

def test_check_date_continuity_passes_on_clean_panel():
    assert check_date_continuity(_clean_panel()).passed


def test_check_date_continuity_catches_gap():
    df = _clean_panel()
    df = df.drop(df.index[1])  # remove the middle day for the first item
    assert not check_date_continuity(df).passed


# ---- check_valid_sales --------------------------------------------------------

def test_check_valid_sales_passes_on_clean_panel():
    assert check_valid_sales(_clean_panel()).passed


def test_check_valid_sales_flags_negative_but_is_not_critical():
    df = _clean_panel()
    df.loc[0, "sales"] = -5
    result = check_valid_sales(df)
    assert not result.passed
    assert result.critical is False


# ---- check_join_duplication ---------------------------------------------------

def test_check_join_duplication_passes_when_row_count_unchanged():
    assert check_join_duplication(100, 100).passed


def test_check_join_duplication_fails_when_row_count_grows():
    assert not check_join_duplication(100, 150).passed


# ---- run_all_validations -------------------------------------------------------

def test_run_all_validations_passes_on_clean_panel():
    results = run_all_validations(_clean_panel())
    assert all(r.passed for r in results)


def test_run_all_validations_raises_on_critical_failure():
    df = _clean_panel()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate key -> critical
    with pytest.raises(DataValidationError):
        run_all_validations(df)


def test_run_all_validations_does_not_raise_on_noncritical_failure():
    df = _clean_panel()
    df.loc[0, "sales"] = -5  # non-critical
    results = run_all_validations(df)  # should not raise
    assert any(r.name == "valid_sales" and not r.passed for r in results)
