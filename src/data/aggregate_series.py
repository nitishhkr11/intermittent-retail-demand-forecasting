"""Loader for the CA_1/FOODS_3 AGGREGATE daily series, development period only.

Every aggregate-series experiment in Phase 2 and Phase 3 (Seasonal Naive, ARIMA,
SARIMA, Holt-Winters, LSTM/GRU, Prophet) is evaluated on the same series: the daily
total of unit sales across all 823 products. This module is the one place that
series is turned into a development-period `pd.Series`, so every notebook and test
cuts the final holdout off in exactly the same way -- through the existing
`src.data.splitter.split_final_holdout`, the mechanism the Phase 2 notebook used.

The parquet file itself still contains the reserved final days. They are removed
immediately: the Series returned here ends on the last DEVELOPMENT date, and the
reserved window is exposed only as two DATES (`DevelopmentWindow`), never as values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.data.splitter import split_final_holdout
from src.utils.config import load_config
from src.utils.paths import resolve_path


@dataclass(frozen=True)
class DevelopmentWindow:
    """Calendar facts about the development / reserved-holdout split (dates only)."""

    dev_start: pd.Timestamp
    dev_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    n_development_days: int
    n_holdout_days: int

    @property
    def n_total_days(self) -> int:
        return self.n_development_days + self.n_holdout_days


def aggregate_series_path(config: Optional[Dict[str, Any]] = None) -> Path:
    """Absolute path of the saved aggregate-series parquet file named in config.yaml."""
    config = load_config() if config is None else config
    return resolve_path(Path(config["data"]["processed_dir"]) / config["output"]["aggregate_filename"])


def _read_aggregate_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Aggregate series file '{path.name}' not found. Create it with: "
            "python -m src.analysis.demand_classification"
        )
    table = pd.read_parquet(path)
    missing = {"date", "sales"} - set(table.columns)
    if missing:
        raise ValueError(f"'{path.name}' is missing column(s): {sorted(missing)}.")
    table = table[["date", "sales"]].copy()
    table["date"] = pd.to_datetime(table["date"]).astype("datetime64[ns]")
    table = table.sort_values("date").reset_index(drop=True)
    if table["date"].duplicated().any():
        raise ValueError("The aggregate series has duplicated dates.")
    expected = pd.date_range(table["date"].iloc[0], table["date"].iloc[-1], freq="D")
    if len(expected) != len(table):
        raise ValueError(
            f"The aggregate series is not a continuous daily series: {len(table)} rows for "
            f"{len(expected)} calendar days."
        )
    if table["sales"].isna().any():
        raise ValueError("The aggregate series contains missing sales values.")
    if (table["sales"] < 0).any():
        raise ValueError("The aggregate series contains negative sales values.")
    return table


def load_aggregate_development(
    config: Optional[Dict[str, Any]] = None, path: Optional[Union[str, Path]] = None
) -> "tuple[pd.Series, DevelopmentWindow]":
    """The development-period aggregate Series plus the development/holdout calendar.

    `path` overrides the config's parquet location (used by tests with synthetic data).
    The returned Series is date-indexed (daily, ascending), float-typed, and ends on
    the last development date; nothing after it is returned.
    """
    config = load_config() if config is None else config
    holdout_days = int(config["backtest"]["holdout_days"])
    table = _read_aggregate_table(Path(path) if path is not None else aggregate_series_path(config))
    split = split_final_holdout(table, holdout_days=holdout_days, date_col="date")

    development = split.development
    series = pd.Series(
        development["sales"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(development["date"], name="date"),
        name="aggregate_sales",
    )
    window = DevelopmentWindow(
        dev_start=split.dev_start,
        dev_end=split.dev_end,
        holdout_start=split.test_start,
        holdout_end=split.test_end,
        n_development_days=split.dev_length,
        n_holdout_days=split.test_length,
    )
    return series, window


def load_aggregate_development_series(
    config: Optional[Dict[str, Any]] = None, path: Optional[Union[str, Path]] = None
) -> pd.Series:
    """Shortcut for `load_aggregate_development(...)[0]`: just the development Series."""
    return load_aggregate_development(config=config, path=path)[0]
