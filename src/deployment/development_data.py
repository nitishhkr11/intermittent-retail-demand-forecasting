"""Create dashboard data files containing development sales only.

Date columns may be scanned to determine the cutoff. Sales columns are read with
the cutoff as a Parquet row filter, so reserved-period targets are never loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.provenance import write_json

OUTPUT_DIR = Path("artifacts/dashboard_data")
AGGREGATE_NAME = "aggregate_development.parquet"
ALL_ITEMS_NAME = "all_items_development.parquet"
SELECTED_ITEMS_NAME = "selected_items_development.parquet"
MANIFEST_NAME = "development_data_manifest.json"


def _hash_table(table: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes()).hexdigest()


def _verified_existing_development_data(output: Path) -> Optional[Dict[str, Any]]:
    """Reuse a verified development-only package without touching source history."""
    manifest_path = output / MANIFEST_NAME
    expected_names = (AGGREGATE_NAME, ALL_ITEMS_NAME, SELECTED_ITEMS_NAME)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("scope") != "development only"
                or manifest.get("holdout_dates_included") is not False
                or manifest.get("holdout_target_values_read") is not False
                or manifest.get("files") != sorted(expected_names)):
            return None
        start = pd.Timestamp(manifest["development_start"])
        end = pd.Timestamp(manifest["development_end"])
        dates = pd.date_range(start, end, freq="D")
        if len(dates) != int(manifest["development_days"]):
            return None
        aggregate = pd.read_parquet(output / AGGREGATE_NAME)
        all_items = pd.read_parquet(output / ALL_ITEMS_NAME)
        selected = pd.read_parquet(output / SELECTED_ITEMS_NAME)
        tables = {AGGREGATE_NAME: aggregate, ALL_ITEMS_NAME: all_items, SELECTED_ITEMS_NAME: selected}
        if not {"date", "sales"}.issubset(aggregate.columns):
            return None
        if not {"date", "item_id", "sales"}.issubset(all_items.columns):
            return None
        if not {"date", "item_id", "sales"}.issubset(selected.columns):
            return None
        for name, table in tables.items():
            table_dates = pd.DatetimeIndex(pd.to_datetime(table["date"]).sort_values().unique())
            if len(table_dates) == 0 or table_dates[0] != start or table_dates[-1] != end:
                return None
            if name != AGGREGATE_NAME and len(table) != table["item_id"].nunique() * len(dates):
                return None
            if _hash_table(table) != manifest.get({
                AGGREGATE_NAME: "aggregate_sha256",
                ALL_ITEMS_NAME: "all_items_sha256",
                SELECTED_ITEMS_NAME: "selected_items_sha256",
            }[name]):
                return None
        if len(aggregate) != len(dates) or aggregate["sales"].isna().any() or (aggregate["sales"] < 0).any():
            return None
        if len(all_items) != int(manifest["all_item_days"]) or len(selected) != int(manifest["selected_item_days"]):
            return None
        if not pd.DatetimeIndex(pd.to_datetime(aggregate["date"]).sort_values()).equals(dates):
            return None
        for table in (all_items, selected):
            if table.duplicated(["item_id", "date"]).any():
                return None
            counts = table.groupby("item_id", observed=True).size()
            if not counts.eq(len(dates)).all():
                return None
        return manifest
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def build_development_data(config: Optional[Mapping[str, Any]] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Build aggregate, all-item and selected-item parquet files ending at dev_end."""
    config = load_config() if config is None else config
    output = resolve_path(output_dir or OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    reusable = _verified_existing_development_data(output)
    if reusable is not None:
        return reusable
    processed = Path(config["data"]["processed_dir"])
    interim = Path(config["data"]["interim_dir"])
    aggregate_path = resolve_path(processed / config["output"]["aggregate_filename"])
    panel_path = resolve_path(interim / config["output"]["panel_filename"])
    selected_path = resolve_path(processed / config["output"]["selected_items_daily_sales_filename"])
    holdout_days = int(config["backtest"]["holdout_days"])

    # Read date labels only to establish the cutoff; this does not load any target values.
    date_column = pd.read_parquet(aggregate_path, columns=["date"])["date"]
    dates = pd.DatetimeIndex(pd.to_datetime(date_column)).sort_values().unique()
    if len(dates) <= holdout_days:
        raise ValueError("aggregate date column is too short to establish the development cutoff")
    dev_end = pd.Timestamp(dates[-holdout_days - 1])
    dev_start = pd.Timestamp(dates[0])
    cutoff = pd.Timestamp(dev_end)
    filters = [("date", "<=", cutoff)]

    # Parquet predicate pushdown ensures only development-period sales are read.
    aggregate = pd.read_parquet(aggregate_path, columns=["date", "sales"], filters=filters).sort_values("date")
    items = pd.read_parquet(panel_path, columns=["date", "item_id", "sales"], filters=filters).sort_values(["date", "item_id"])
    selected = pd.read_parquet(selected_path, columns=["date", "item_id", "sales"], filters=filters).sort_values(["date", "item_id"])

    expected_dates = pd.date_range(dev_start, dev_end, freq="D")
    if not pd.DatetimeIndex(pd.to_datetime(aggregate["date"])).equals(expected_dates):
        raise ValueError("filtered aggregate does not match the complete development date range")
    if aggregate["sales"].isna().any() or (aggregate["sales"] < 0).any():
        raise ValueError("development aggregate contains invalid sales")
    if pd.to_datetime(items["date"]).max() > dev_end or pd.to_datetime(selected["date"]).max() > dev_end:
        raise AssertionError("development parquet filter returned reserved-period dates")
    expected_items = len(items["item_id"].unique())
    expected_rows = expected_items * len(expected_dates)
    if len(items) != expected_rows or items.duplicated(["item_id", "date"]).any():
        raise ValueError("development item panel is incomplete or has duplicate item/date rows")

    files = {
        AGGREGATE_NAME: aggregate[["date", "sales"]].reset_index(drop=True),
        ALL_ITEMS_NAME: items[["date", "item_id", "sales"]].reset_index(drop=True),
        SELECTED_ITEMS_NAME: selected[["date", "item_id", "sales"]].reset_index(drop=True),
    }
    for name, table in files.items():
        destination = output / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        table.to_parquet(temporary, index=False)
        temporary.replace(destination)
    manifest = {
        "schema_version": 1,
        "scope": "development only",
        "development_start": dev_start.date().isoformat(),
        "development_end": dev_end.date().isoformat(),
        "development_days": len(expected_dates),
        "holdout_dates_included": False,
        "holdout_target_values_read": False,
        "all_item_days": len(items),
        "selected_item_days": len(selected),
        "aggregate_sha256": _hash_table(files[AGGREGATE_NAME]),
        "all_items_sha256": _hash_table(files[ALL_ITEMS_NAME]),
        "selected_items_sha256": _hash_table(files[SELECTED_ITEMS_NAME]),
        "files": sorted(files),
    }
    write_json(output / MANIFEST_NAME, manifest)
    return manifest
