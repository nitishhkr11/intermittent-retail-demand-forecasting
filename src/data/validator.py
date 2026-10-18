"""Data-quality checks for the CA_1 / FOODS_3 panel.

Each `check_*` function inspects one property of the data and returns
a `ValidationResult` — it never raises on its own, so every check can
be unit tested in isolation. `run_all_validations` is what actually
enforces them: it runs every check, logs each result, and raises
`DataValidationError` if any *critical* check fails.

Two checks are reused rather than duplicated for related items in the
Phase 1 checklist:
  - "missing date mapping" (a 'd' value with no matching calendar row)
    shows up as a null 'date' column, so it's checked with
    `check_missing_dates`, called right after the calendar join.
  - "join duplication" is checked with `check_join_duplication`,
    called with before/after row counts right after each merge
    (calendar join and price join both use it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns the rest of the pipeline (and later phases) rely on existing.
EXPECTED_COLUMNS = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "d", "sales", "date", "wm_yr_wk", "sell_price",
]

class DataValidationError(Exception):
    """Raised when a critical validation check fails."""


@dataclass
class ValidationResult:
    """Outcome of one validation check."""

    name: str
    passed: bool
    message: str
    critical: bool = field(default=True)


def check_expected_columns(df: pd.DataFrame) -> ValidationResult:
    """All columns later code assumes exist must be present."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    passed = not missing
    message = (
        "All expected columns present."
        if passed
        else f"Missing columns: {missing}"
    )
    return ValidationResult("expected_columns", passed, message)


def check_data_types(df: pd.DataFrame) -> ValidationResult:
    """Key columns must have the dtype later code assumes they have.

    'date' is checked as "some datetime64 resolution", not an exact
    'datetime64[ns]' string — pandas 2.x can legitimately produce
    datetime64[us] or [ns] depending on construction path, and neither
    is wrong for day-level dates. 'sales' is checked exactly, since we
    explicitly cast it to int64 ourselves in finalize_dtypes().
    """
    problems = []
    if "sales" in df.columns and str(df["sales"].dtype) != "int64":
        problems.append(f"sales: expected int64, got {df['sales'].dtype}")
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        problems.append(f"date: expected a datetime64 dtype, got {df['date'].dtype}")
    passed = not problems
    message = "All dtypes as expected." if passed else "; ".join(problems)
    return ValidationResult("data_types", passed, message)


def check_duplicate_keys(df: pd.DataFrame) -> ValidationResult:
    """No (item_id, date) pair should appear more than once."""
    n_dupes = int(df.duplicated(subset=["item_id", "date"]).sum())
    passed = n_dupes == 0
    message = (
        "No duplicate (item_id, date) rows."
        if passed
        else f"{n_dupes} duplicate (item_id, date) rows found."
    )
    return ValidationResult("duplicate_keys", passed, message)


def check_missing_dates(df: pd.DataFrame) -> ValidationResult:
    """The 'date' column must never be null.

    A null here after the calendar join means some 'd' identifier in
    the sales data had no matching row in calendar.csv.
    """
    n_missing = int(df["date"].isna().sum())
    passed = n_missing == 0
    message = (
        "No missing dates." if passed else f"{n_missing} rows have a null date."
    )
    return ValidationResult("missing_dates", passed, message)


def check_date_continuity(df: pd.DataFrame) -> ValidationResult:
    """Every item should have exactly one row per day in its date range."""
    problems = []
    for item_id, group in df.groupby("item_id", observed=True):
        expected_days = (group["date"].max() - group["date"].min()).days + 1
        actual_days = group["date"].nunique()
        if actual_days != expected_days:
            problems.append(str(item_id))
    passed = not problems
    message = (
        "Every item has a continuous daily date range."
        if passed
        else f"{len(problems)} item(s) have date gaps, e.g. {problems[:3]}."
    )
    return ValidationResult("date_continuity", passed, message)


def check_valid_sales(df: pd.DataFrame) -> ValidationResult:
    """Sales should be non-negative — M5 documents demand as unit counts >= 0.

    Not treated as critical: a negative value is worth investigating,
    but Phase 1 shouldn't halt the whole pipeline over a data question
    it hasn't looked into yet.
    """
    n_negative = int((df["sales"] < 0).sum())
    passed = n_negative == 0
    message = (
        "No negative sales values."
        if passed
        else f"{n_negative} rows have negative sales."
    )
    return ValidationResult("valid_sales", passed, message, critical=False)


def check_join_duplication(
    pre_join_rows: int, post_join_rows: int, join_name: str = "join"
) -> ValidationResult:
    """Row count must be identical before and after a many-to-one join."""
    passed = pre_join_rows == post_join_rows
    message = (
        f"{join_name}: row count unchanged ({pre_join_rows})."
        if passed
        else (
            f"{join_name}: row count changed {pre_join_rows} -> "
            f"{post_join_rows}, join likely duplicated rows."
        )
    )
    return ValidationResult(f"join_duplication_{join_name}", passed, message)


def run_all_validations(df: pd.DataFrame) -> List[ValidationResult]:
    """Run every whole-panel check, log each result, raise on critical failure.

    Called once at the end of build_panel(), after the join-specific
    checks have already run inline. This is the final gate before the
    panel is saved.
    """
    checks = [
        check_expected_columns(df),
        check_data_types(df),
        check_duplicate_keys(df),
        check_missing_dates(df),
        check_date_continuity(df),
        check_valid_sales(df),
    ]

    for result in checks:
        if result.passed:
            logger.info("[%s] %s", result.name, result.message)
        elif result.critical:
            logger.error("[%s] %s", result.name, result.message)
        else:
            logger.warning("[%s] %s", result.name, result.message)

    failed_critical = [r for r in checks if not r.passed and r.critical]
    if failed_critical:
        names = ", ".join(r.name for r in failed_critical)
        raise DataValidationError(f"Critical validation check(s) failed: {names}")

    return checks
