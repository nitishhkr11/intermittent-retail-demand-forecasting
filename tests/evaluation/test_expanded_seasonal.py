"""Tests for the additional seasonal-window plan and its leakage boundaries."""

import pandas as pd
import pytest

from src.evaluation.expanded_seasonal import seasonal_origins


@pytest.fixture
def development_dates():
    return pd.date_range("2011-01-29", "2016-04-24", freq="D")


def test_quarterly_plan_covers_four_quarters_over_four_years_without_overlap(development_dates):
    folds = seasonal_origins(development_dates)

    assert len(folds) == 16
    assert {(row["year"], row["quarter"]) for row in folds} == {
        (year, quarter) for year in range(2012, 2016) for quarter in range(1, 5)
    }
    assert all(row["training_days"] >= 365 and row["horizon"] == 28 for row in folds)
    assert all(
        pd.Timestamp(row["forecast_end"]) - pd.Timestamp(row["forecast_start"]) == pd.Timedelta(days=27)
        for row in folds
    )
    assert all(
        pd.Timestamp(left["forecast_end"]) < pd.Timestamp(right["forecast_start"])
        for left, right in zip(folds, folds[1:])
    )
    assert folds[0]["forecast_start"] == "2012-01-29"
    assert folds[-1]["forecast_end"] <= "2016-04-24"


def test_seasonal_plan_refuses_insufficient_valid_windows(development_dates):
    with pytest.raises(ValueError, match="at least 12"):
        seasonal_origins(development_dates, years=(2014, 2015))


def test_seasonal_plan_rejects_non_daily_or_duplicate_indices():
    with pytest.raises(ValueError, match="unique and ascending"):
        seasonal_origins(pd.DatetimeIndex(["2011-01-01", "2011-01-03", "2011-01-03"]))
