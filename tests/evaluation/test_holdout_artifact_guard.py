"""Guard: no saved development artifact may contain a date inside the reserved final holdout.

The reserved window is 2016-04-25 .. 2016-05-22 (config `backtest.holdout_days`).
Every CSV the project saves under artifacts/forecasts, artifacts/metrics and
artifacts/inventory is a DEVELOPMENT artifact, so none of its date-like columns may
extend past the last development day. The (separate) artifacts/final folder is the only
place holdout dates may ever appear, and only through src/evaluation/final_holdout.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.aggregate_series import aggregate_series_path, load_aggregate_development
from src.utils.config import load_config
from src.utils.paths import resolve_path

DATE_COLUMNS = ("date", "forecast_start", "forecast_end", "origin_date", "ds", "cutoff", "training_cutoff")
DEVELOPMENT_FOLDERS = ("artifacts/forecasts", "artifacts/metrics", "artifacts/inventory", "artifacts/residuals")


@pytest.fixture(scope="module")
def development_end():
    config = load_config()
    if not aggregate_series_path(config).exists():
        pytest.skip("the processed aggregate parquet file is not present")
    _, window = load_aggregate_development(config)
    return window.dev_end


def _date_columns_of(path):
    header = pd.read_csv(path, nrows=0).columns
    return [c for c in header if c in DATE_COLUMNS]


def test_no_development_artifact_has_a_date_in_the_reserved_holdout(development_end):
    offenders = []
    checked = 0
    for folder in DEVELOPMENT_FOLDERS:
        directory = resolve_path(folder)
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            columns = _date_columns_of(path)
            if not columns:
                continue
            table = pd.read_csv(path, usecols=columns)
            checked += 1
            for column in columns:
                latest = pd.to_datetime(table[column], errors="coerce").max()
                if pd.notna(latest) and latest > development_end:
                    offenders.append(f"{folder}/{path.name}: {column} reaches {latest.date()}")
    assert checked > 0, "no development artifact with a date column was found to check"
    assert offenders == [], "artifacts contain dates after the development end: " + "; ".join(offenders)


def test_the_final_folder_is_not_scanned_as_a_development_folder():
    assert all("final" not in folder for folder in DEVELOPMENT_FOLDERS)
