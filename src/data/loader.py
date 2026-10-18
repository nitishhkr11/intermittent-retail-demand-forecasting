"""Raw M5 file loading.

Each function here does exactly one thing: read one raw CSV into a
DataFrame and return it, unmodified. No filtering, no reshaping, no
merging happens in this module — that's preprocessor.py's job. Keeping
loading this simple means these functions stay reusable regardless of
which store or department gets analyzed later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _read_csv(path: Union[str, Path], name: str) -> pd.DataFrame:
    """Shared CSV-reading helper with a clear error if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {name} at '{path}'. Download the M5 "
            "Forecasting - Accuracy dataset from Kaggle and place its "
            "CSVs in data/raw/ before running the pipeline."
        )
    logger.info("Loading %s from %s", name, path)
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows, %d columns", name, len(df), df.shape[1])
    return df


def load_sales(path: Union[str, Path]) -> pd.DataFrame:
    """Load sales_train_evaluation.csv.

    Wide format: one row per (item_id, store_id), with day columns
    d_1 ... d_n holding daily unit sales.
    """
    return _read_csv(path, "sales_train_evaluation.csv")


def load_calendar(path: Union[str, Path]) -> pd.DataFrame:
    """Load calendar.csv.

    One row per day. Maps each 'd_n' identifier used in the sales file
    to a real calendar date, plus weekday/month/year, event flags, and
    SNAP (food-assistance) flags.
    """
    return _read_csv(path, "calendar.csv")


def load_prices(path: Union[str, Path]) -> pd.DataFrame:
    """Load sell_prices.csv.

    One row per (store_id, item_id, wm_yr_wk) — prices in M5 vary by
    week, not by day.
    """
    return _read_csv(path, "sell_prices.csv")
