"""Unit tests for src/analysis/eda.py. All synthetic data -- no real M5 data
is used or required by any test in this file.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend for tests

import pandas as pd
import pytest

from src.analysis.eda import (
    build_aggregate_series,
    build_selected_items_daily_sales,
    compute_day_of_week_pattern,
    compute_rolling_stats,
    dataset_overview,
    plot_adi_cv2_quadrants,
    plot_representative_series,
    summarize_demand_classes,
    summarize_missing_values,
)


# ---- build_aggregate_series (Stage 2 gap -- untested until now) -----------------

def test_build_aggregate_series_sums_across_all_items():
    panel = pd.DataFrame({
        "item_id": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
        "sales": [1, 2, 3, 4],
    })
    result = build_aggregate_series(panel)
    assert list(result["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    assert list(result["sales"]) == [4, 6]  # (1+3), (2+4)


def test_build_aggregate_series_is_sorted_by_date():
    panel = pd.DataFrame({
        "item_id": ["A", "A"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-01"]),  # out of order on purpose
        "sales": [2, 1],
    })
    result = build_aggregate_series(panel)
    assert list(result["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-02"]))


# ---- build_selected_items_daily_sales (Phase 2 Step 7 data pipeline) --------------

def _panel_two_items_two_days():
    return pd.DataFrame({
        "item_id": ["A", "A", "B", "B", "C", "C"],  # C is NOT selected
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 3),
        "sales": [1, 2, 3, 4, 5, 6],
    })


def _selected_two_items():
    return pd.DataFrame({
        "item_id": ["A", "B"],
        "demand_class": ["Smooth", "Lumpy"],
        "ADI": [1.1, 2.2],
        "CV2": [0.3, 0.6],
    })


def test_build_selected_items_daily_sales_filters_to_selected_items_only():
    result = build_selected_items_daily_sales(_panel_two_items_two_days(), _selected_two_items())
    assert set(result["item_id"]) == {"A", "B"}  # item C excluded
    assert len(result) == 4  # 2 items x 2 days


def test_build_selected_items_daily_sales_keeps_real_sales_values_unchanged():
    result = build_selected_items_daily_sales(_panel_two_items_two_days(), _selected_two_items())
    a_sales = result.loc[result["item_id"] == "A"].sort_values("date")["sales"].tolist()
    b_sales = result.loc[result["item_id"] == "B"].sort_values("date")["sales"].tolist()
    assert a_sales == [1, 2]   # straight from the panel, nothing fabricated or altered
    assert b_sales == [3, 4]


def test_build_selected_items_daily_sales_joins_demand_class_correctly():
    result = build_selected_items_daily_sales(_panel_two_items_two_days(), _selected_two_items())
    classes = result.drop_duplicates("item_id").set_index("item_id")["demand_class"]
    assert classes["A"] == "Smooth"
    assert classes["B"] == "Lumpy"


def test_build_selected_items_daily_sales_has_expected_columns():
    result = build_selected_items_daily_sales(_panel_two_items_two_days(), _selected_two_items())
    assert list(result.columns) == ["item_id", "date", "sales", "demand_class"]


def test_build_selected_items_daily_sales_is_sorted_by_item_then_date():
    panel = pd.DataFrame({
        "item_id": ["B", "A", "B", "A"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-01"]),
        "sales": [4, 2, 3, 1],
    })
    result = build_selected_items_daily_sales(panel, _selected_two_items())
    assert list(result["item_id"]) == ["A", "A", "B", "B"]
    assert list(result["date"]) == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]))


def test_build_selected_items_daily_sales_raises_if_a_selected_item_is_missing_from_panel():
    panel = pd.DataFrame({
        "item_id": ["A", "A"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "sales": [1, 2],
    })
    # selected_items names B too, but B never appears in panel.
    with pytest.raises(ValueError, match="not found in panel"):
        build_selected_items_daily_sales(panel, _selected_two_items())


def test_build_selected_items_daily_sales_does_not_fabricate_missing_dates():
    # Item A has 2 days, item B only 1 -- a genuine gap in the source
    # panel. This function must NOT invent a row to fill it; it only
    # returns what's really in `panel`, uneven row counts and all.
    panel = pd.DataFrame({
        "item_id": ["A", "A", "B"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
        "sales": [1, 2, 3],
    })
    result = build_selected_items_daily_sales(panel, _selected_two_items())
    assert len(result) == 3  # not padded up to 4
    assert (result.loc[result["item_id"] == "B", "date"] == pd.Timestamp("2024-01-01")).all()


# ---- dataset_overview -------------------------------------------------------------

def _tiny_panel(n_days=10) -> pd.DataFrame:
    return pd.DataFrame({
        "item_id": ["A"] * n_days,
        "date": pd.date_range("2024-01-01", periods=n_days, freq="D"),
        "sales": range(n_days),
        "store_id": ["CA_1"] * n_days,
        "dept_id": ["FOODS_3"] * n_days,
    })


def test_dataset_overview_basic_counts():
    overview = dataset_overview(_tiny_panel(10))
    assert overview["n_rows"] == 10
    assert overview["n_items"] == 1
    assert overview["n_days"] == 10
    assert overview["stores"] == ["CA_1"]
    assert overview["departments"] == ["FOODS_3"]


def test_dataset_overview_infers_daily_frequency():
    overview = dataset_overview(_tiny_panel(10))
    assert overview["inferred_freq"] == "D"


# ---- summarize_missing_values -------------------------------------------------------

def test_summarize_missing_values_flags_expected_vs_unexpected():
    df = pd.DataFrame({
        "sell_price": [1.0, None, None],       # expected to be missing sometimes
        "event_name_1": [None, None, None],    # expected to be missing sometimes
        "item_id": ["A", None, "C"],           # NOT expected -- should be flagged
    })
    result = summarize_missing_values(df)
    by_col = result.set_index("column")
    assert bool(by_col.loc["sell_price", "is_expected"]) is True
    assert bool(by_col.loc["event_name_1", "is_expected"]) is True
    assert bool(by_col.loc["item_id", "is_expected"]) is False


def test_summarize_missing_values_excludes_complete_columns():
    df = pd.DataFrame({"complete": [1, 2, 3], "sell_price": [1.0, None, 3.0]})
    result = summarize_missing_values(df)
    assert "complete" not in result["column"].values
    assert "sell_price" in result["column"].values


# ---- compute_rolling_stats --------------------------------------------------------

def test_compute_rolling_stats_known_values():
    # window=3 over [1,2,3,4]: means are 1, 1.5, 2, 3
    series = pd.Series([1, 2, 3, 4])
    result = compute_rolling_stats(series, window=3)
    assert result["rolling_mean"].tolist() == pytest.approx([1.0, 1.5, 2.0, 3.0])


# ---- compute_day_of_week_pattern ----------------------------------------------------

def test_compute_day_of_week_pattern_known_values_and_order():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=7, freq="D"),  # Mon..Sun
        "sales": [10, 20, 30, 40, 50, 60, 70],
    })
    result = compute_day_of_week_pattern(df)
    assert result["Monday"] == 10.0
    assert result["Sunday"] == 70.0
    assert list(result.index) == [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ]


# ---- summarize_demand_classes -----------------------------------------------------

def test_summarize_demand_classes_counts_and_percentages():
    classification = pd.DataFrame({
        "item_id": ["a", "b", "c", "d"],
        "demand_class": ["Smooth", "Smooth", "Lumpy", "NoDemand"],
    })
    result = summarize_demand_classes(classification).set_index("demand_class")
    assert result.loc["Smooth", "count"] == 2
    assert result.loc["Smooth", "pct_of_catalog"] == 50.0
    assert result.loc["NoDemand", "count"] == 1


# ---- plotting smoke tests (no image comparison -- just "does it crash / right type") --

def test_plot_adi_cv2_quadrants_smoke():
    classification = pd.DataFrame({
        "item_id": ["a", "b", "c"],
        "ADI": [1.0, 2.0, None],
        "CV2": [0.1, 0.6, None],
        "demand_class": ["Smooth", "Lumpy", "NoDemand"],
    })
    ax = plot_adi_cv2_quadrants(classification, adi_threshold=1.32, cv2_threshold=0.49)
    assert ax is not None
    assert ax.get_xlabel().startswith("ADI")


def test_plot_representative_series_smoke():
    panel = pd.DataFrame({
        "item_id": ["a", "a", "b", "b"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
        "sales": [1, 2, 3, 4],
    })
    selected = pd.DataFrame({
        "item_id": ["a", "b"],
        "demand_class": ["Smooth", "Lumpy"],
    })
    fig = plot_representative_series(panel, selected, items_per_category=1)
    assert fig is not None
    assert len(fig.axes) == 2  # one subplot per category present
