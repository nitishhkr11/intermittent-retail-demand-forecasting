"""Scope, reshape, and enrich raw M5 data into one clean panel.

This module turns the three raw M5 files into a single long-format
DataFrame — one row per (item, date) — restricted to one store and
department, with calendar and price information attached. That output
(data/interim/panel_ca1_foods3.parquet) is what every later phase
(classification, modeling, dashboard) reads. Nothing downstream should
ever need to touch the raw CSVs directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data.loader import load_calendar, load_prices, load_sales
from src.data.validator import (
    DataValidationError,
    check_join_duplication,
    check_missing_dates,
    run_all_validations,
)
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import resolve_path

logger = get_logger(__name__)

ID_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

CALENDAR_COLUMNS = [
    "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]


def scope_and_melt_sales(
    sales_df: pd.DataFrame, store_id: str, dept_id: str
) -> pd.DataFrame:
    """Filter sales to one store/department, then melt from wide to long.

    Filtering before melting matters: sales_train_evaluation.csv has
    ~30,000 rows (one per item-store) but ~1,900+ day columns. Melting
    first would explode memory use for stores/departments we don't
    need; filtering first keeps this cheap.
    """
    scoped = sales_df[
        (sales_df["store_id"] == store_id) & (sales_df["dept_id"] == dept_id)
    ].copy()
    if scoped.empty:
        raise ValueError(
            f"No rows found for store_id='{store_id}', dept_id='{dept_id}'. "
            "Check these values match the raw data exactly (case-sensitive)."
        )
    logger.info("Scoped to store=%s, dept=%s: %d items", store_id, dept_id, len(scoped))

    day_columns = [c for c in scoped.columns if c.startswith("d_")]
    long_df = scoped.melt(
        id_vars=ID_COLUMNS, value_vars=day_columns, var_name="d", value_name="sales"
    )
    logger.info("Melted to long format: %d rows", len(long_df))
    return long_df


def attach_calendar(long_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Join calendar info onto the long sales panel using the 'd' identifier."""
    calendar_subset = calendar_df[[c for c in CALENDAR_COLUMNS if c in calendar_df.columns]]

    before = len(long_df)
    merged = long_df.merge(calendar_subset, on="d", how="left", validate="many_to_one")

    dup_check = check_join_duplication(before, len(merged), join_name="calendar")
    if not dup_check.passed:
        raise DataValidationError(dup_check.message)

    date_check = check_missing_dates(merged)
    if not date_check.passed:
        raise DataValidationError(date_check.message)

    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def attach_prices(long_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Join sell price onto the panel using (store_id, item_id, wm_yr_wk).

    Prices in M5 vary weekly, not daily, so this is a many-to-one join:
    many daily sales rows share one weekly price row. `validate=` makes
    pandas raise immediately if that assumption is ever violated — that
    guarantee, plus the row-count check below, is what actually
    prevents duplicate item-date rows here, not just a comment
    promising it won't happen.
    """
    before = len(long_df)
    merged = long_df.merge(
        prices_df[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    dup_check = check_join_duplication(before, len(merged), join_name="prices")
    if not dup_check.passed:
        raise DataValidationError(dup_check.message)

    n_missing_price = int(merged["sell_price"].isna().sum())
    if n_missing_price:
        logger.warning(
            "%d rows have no matching price (expected for weeks before an "
            "item was listed for sale) — kept as NaN, not dropped.",
            n_missing_price,
        )
    return merged


def finalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to sensible, memory-efficient dtypes."""
    df = df.copy()
    for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df["sales"] = df["sales"].astype("int64")
    df["date"] = df["date"].astype("datetime64[ns]")
    return df


def build_panel(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Run the full raw-to-interim pipeline and return the cleaned panel.

    Ties loader.py, this module, and validator.py together. Also
    runnable directly: `python -m src.data.preprocessor`.
    """
    config = config or load_config()
    raw_dir = resolve_path(config["data"]["raw_dir"])
    store_id = config["scope"]["store_id"]
    dept_id = config["scope"]["dept_id"]

    sales_df = load_sales(raw_dir / config["data"]["sales_file"])
    calendar_df = load_calendar(raw_dir / config["data"]["calendar_file"])
    prices_df = load_prices(raw_dir / config["data"]["prices_file"])

    panel = scope_and_melt_sales(sales_df, store_id, dept_id)
    panel = attach_calendar(panel, calendar_df)
    panel = attach_prices(panel, prices_df)
    panel = finalize_dtypes(panel)

    run_all_validations(panel)  # final gate; raises DataValidationError if broken

    logger.info("Final panel: %d rows, %d columns", len(panel), panel.shape[1])
    return panel


def save_panel(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Path:
    """Save the panel to data/interim/ as Parquet, per the approved architecture."""
    config = config or load_config()
    interim_dir = resolve_path(config["data"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / config["output"]["panel_filename"]
    df.to_parquet(out_path, index=False)
    logger.info("Saved panel to %s", out_path)
    return out_path


if __name__ == "__main__":
    cfg = load_config()
    result_df = build_panel(cfg)
    save_panel(result_df, cfg)
