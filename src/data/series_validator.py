"""Series-level data-quality checks for Phase 2's daily aggregate demand series.

src/data/validator.py checks the item-level (item_id, date) panel built
in Phase 1 -- a different grain with a different expected-columns list,
so its checks don't apply here, and this module deliberately doesn't
touch that file. This module checks the single two-column (date,
sales) aggregate series Phase 2 backtesting runs on, reusing
validator.py's ValidationResult/DataValidationError so both report
through the same shape.

These are the Requirement 1 checks: daily frequency, no duplicate
dates, no missing dates, a continuous date range, and no negative
sales. Forward-filling is deliberately never done anywhere in this
module or its caller -- a zero-sales day (e.g. Christmas, when stores
are closed) is a legitimate observation, not a gap to patch over.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from src.data.validator import DataValidationError, ValidationResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def check_daily_frequency(df: pd.DataFrame, date_col: str = "date") -> ValidationResult:
    """Every consecutive pair of rows must be exactly one day apart."""
    diffs = df[date_col].diff().dropna()
    passed = bool((diffs == pd.Timedelta(days=1)).all())
    message = (
        "Daily frequency confirmed (every row exactly 1 day apart)."
        if passed
        else f"{int((diffs != pd.Timedelta(days=1)).sum())} row-to-row gap(s) are not exactly 1 day."
    )
    return ValidationResult("daily_frequency", passed, message)


def check_no_duplicate_dates(df: pd.DataFrame, date_col: str = "date") -> ValidationResult:
    """No date should appear more than once."""
    n_dupes = int(df[date_col].duplicated().sum())
    passed = n_dupes == 0
    message = "No duplicate dates." if passed else f"{n_dupes} duplicate date(s) found."
    return ValidationResult("no_duplicate_dates", passed, message)


def check_no_missing_dates(df: pd.DataFrame, date_col: str = "date") -> ValidationResult:
    """Every calendar day between the min and max date must have a row."""
    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    missing = full_range.difference(df[date_col])
    passed = len(missing) == 0
    message = (
        "No missing dates in range."
        if passed
        else f"{len(missing)} date(s) missing from the range, e.g. {list(missing[:3].date)}."
    )
    return ValidationResult("no_missing_dates", passed, message)


def check_continuous_range(df: pd.DataFrame, date_col: str = "date") -> ValidationResult:
    """Row count must equal the number of calendar days the min/max dates span.

    A cheap cross-check of the two checks above: if both pass, this one
    is implied, so a failure here with the others passing would itself
    indicate a bug in this validation code rather than the data.
    """
    expected = (df[date_col].max() - df[date_col].min()).days + 1
    passed = len(df) == expected
    message = (
        f"Continuous: {len(df)} rows across a {expected}-day range."
        if passed
        else f"{len(df)} rows but a {expected}-day range -- {expected - len(df)} day(s) unaccounted for."
    )
    return ValidationResult("continuous_range", passed, message)


def check_no_negative_sales(df: pd.DataFrame, sales_col: str = "sales") -> ValidationResult:
    """Sales should be non-negative -- M5 documents demand as unit counts >= 0."""
    n_negative = int((df[sales_col] < 0).sum())
    passed = n_negative == 0
    message = "No negative sales." if passed else f"{n_negative} row(s) have negative sales."
    return ValidationResult("no_negative_sales", passed, message)


def run_all_series_validations(
    df: pd.DataFrame, date_col: str = "date", sales_col: str = "sales"
) -> List[ValidationResult]:
    """Run every Requirement-1 check, log each result, raise if any fails.

    All five checks are treated as critical: Phase 2 modeling assumes a
    clean, gap-free, duplicate-free daily series, and silently modeling
    over a violation of that would produce results that look valid but
    aren't.
    """
    checks = [
        check_daily_frequency(df, date_col),
        check_no_duplicate_dates(df, date_col),
        check_no_missing_dates(df, date_col),
        check_continuous_range(df, date_col),
        check_no_negative_sales(df, sales_col),
    ]

    for result in checks:
        if result.passed:
            logger.info("[%s] %s", result.name, result.message)
        else:
            logger.error("[%s] %s", result.name, result.message)

    failed = [r for r in checks if not r.passed]
    if failed:
        names = ", ".join(r.name for r in failed)
        raise DataValidationError(f"Series validation check(s) failed: {names}")

    return checks
