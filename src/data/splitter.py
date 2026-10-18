"""Chronological train/validation/test splitting. No random splitting, per the locked validation rules.

Phase 2 Step 2 needs exactly one kind of split here: carving a final,
untouched holdout off the tail of the daily aggregate series. Rolling-
origin folds for backtesting live in src/evaluation/backtesting.py
instead, since they're tied to the evaluation loop rather than being a
one-off split.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HoldoutSplit:
    """Result of carving a final holdout off the tail of a chronological series."""

    development: pd.DataFrame
    test: pd.DataFrame
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def dev_length(self) -> int:
        return len(self.development)

    @property
    def test_length(self) -> int:
        return len(self.test)

    def summary(self) -> str:
        return (
            f"development: {self.dev_start.date()} -> {self.dev_end.date()} "
            f"({self.dev_length} days)\n"
            f"final test:  {self.test_start.date()} -> {self.test_end.date()} "
            f"({self.test_length} days)"
        )


def split_final_holdout(
    df: pd.DataFrame, holdout_days: int = 28, date_col: str = "date"
) -> HoldoutSplit:
    """Reserve the final `holdout_days` rows of a chronological series as an untouched test set.

    `df` must already be sorted ascending by `date_col` with one row per
    day -- the Requirement-1 checks (src.data.series_validator) are what
    should have verified that before this is called; this function
    trusts its input rather than re-deriving continuity itself.

    The final holdout must never be used for model selection or tuning
    -- it exists only to report one honest number at the very end.
    Model/backtest comparison during development uses `development` only.
    """
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' column not found in df.")
    if not df[date_col].is_monotonic_increasing:
        raise ValueError(f"df must be sorted ascending by '{date_col}' before splitting.")
    if len(df) <= holdout_days:
        raise ValueError(
            f"Not enough rows ({len(df)}) to reserve a {holdout_days}-day holdout."
        )

    development = df.iloc[:-holdout_days].reset_index(drop=True)
    test = df.iloc[-holdout_days:].reset_index(drop=True)

    split = HoldoutSplit(
        development=development,
        test=test,
        dev_start=development[date_col].iloc[0],
        dev_end=development[date_col].iloc[-1],
        test_start=test[date_col].iloc[0],
        test_end=test[date_col].iloc[-1],
    )
    logger.info("Final holdout split:\n%s", split.summary())
    return split
