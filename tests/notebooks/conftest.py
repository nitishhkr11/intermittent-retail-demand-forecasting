"""Read-only development references used by saved forecast artifact audits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT


@pytest.fixture(scope="module")
def aggregate_development_reference() -> pd.Series:
    """Use the bundled 1,913-day development-only reference when available.

    The deployment/CI bundle has already removed the final holdout. Do not
    apply the configured 28-day split a second time to that reference.
    """
    path = PROJECT_ROOT / "artifacts" / "dashboard_data" / "aggregate_development.parquet"
    if not path.is_file():
        pytest.skip("the packaged development-only aggregate reference is unavailable; the full-history source is protected")
    table = pd.read_parquet(path, columns=["date", "sales"])
    table["date"] = pd.to_datetime(table["date"])
    table = table.sort_values("date").reset_index(drop=True)
    assert len(table) == 1913
    assert table["date"].max() == pd.Timestamp("2016-04-24")
    return pd.Series(table["sales"].to_numpy(float), index=pd.DatetimeIndex(table["date"], name="date"), name="aggregate_sales")


@pytest.fixture(scope="module")
def selected_items_development_reference() -> dict[str, pd.Series]:
    """Use exact bundled development-only item actuals for artifact audits."""
    path = PROJECT_ROOT / "artifacts" / "dashboard_data" / "selected_items_development.parquet"
    if not path.is_file():
        pytest.skip("the packaged development-only item reference is unavailable; the full-history source is protected")
    daily = pd.read_parquet(path, columns=["date", "item_id", "sales"])
    daily["date"] = pd.to_datetime(daily["date"])
    assert daily["item_id"].nunique() == 20
    assert daily.groupby("item_id").size().eq(1913).all()
    assert daily["date"].max() == pd.Timestamp("2016-04-24")
    result = {}
    for item, group in daily.groupby("item_id", sort=True):
        group = group.sort_values("date")
        result[str(item)] = pd.Series(
            group["sales"].to_numpy(float),
            index=pd.DatetimeIndex(group["date"], name="date"),
            name=str(item),
        )
    return result
