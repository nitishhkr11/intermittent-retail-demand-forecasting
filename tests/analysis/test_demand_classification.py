"""Unit tests for src/analysis/demand_classification.py.

Every numeric assertion here was verified against the actual functions
before being written (not hand-derived and assumed correct) — see the
docstring on each test for what it's checking and why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.demand_classification import (
    NO_DEMAND_LABEL,
    INSUFFICIENT_DATA_LABEL,
    classify_catalog,
    classify_from_stats,
    classify_series,
    compute_adi,
    compute_cv_squared,
    compute_nonzero_stats,
    select_representative_items,
)


# ---- compute_nonzero_stats --------------------------------------------------

def test_compute_nonzero_stats_basic():
    stats = compute_nonzero_stats(pd.Series([0, 2, 0, 4, 0]))
    assert stats["n_periods"] == 5
    assert stats["n_nonzero"] == 2
    assert stats["mean_nonzero"] == pytest.approx(3.0)


def test_compute_nonzero_stats_all_zero():
    stats = compute_nonzero_stats(pd.Series([0, 0, 0]))
    assert stats["n_nonzero"] == 0
    assert np.isnan(stats["mean_nonzero"])
    assert np.isnan(stats["std_nonzero"])


def test_compute_nonzero_stats_single_nonzero_std_is_nan():
    stats = compute_nonzero_stats(pd.Series([0, 0, 5, 0]))
    assert stats["n_nonzero"] == 1
    assert stats["mean_nonzero"] == 5.0
    assert np.isnan(stats["std_nonzero"])  # std needs >= 2 points


# ---- compute_adi --------------------------------------------------------------

def test_compute_adi_known_value():
    # 10 periods, 5 non-zero -> ADI = 10 / 5 = 2.0
    series = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    assert compute_adi(series) == pytest.approx(2.0)


def test_compute_adi_every_period_has_demand():
    # ADI = 1.0 when every single period has non-zero demand
    series = pd.Series([1, 2, 3, 1, 2])
    assert compute_adi(series) == pytest.approx(1.0)


def test_compute_adi_all_zero_is_nan():
    assert np.isnan(compute_adi(pd.Series([0, 0, 0, 0])))


# ---- compute_cv_squared --------------------------------------------------------

def test_compute_cv_squared_known_value():
    # non-zero values [2,4,6,8] (zeros interleaved don't affect this calc):
    # mean=5, sample std=sqrt(20/3)=2.5820, CV=0.5164, CV^2=0.2667
    series = pd.Series([2, 0, 4, 0, 6, 0, 8])
    assert compute_cv_squared(series) == pytest.approx(0.26667, abs=1e-4)


def test_compute_cv_squared_constant_nonzero_is_zero():
    # No variability in demand size when it occurs -> CV^2 = 0
    series = pd.Series([0, 3, 0, 3, 0, 3])
    assert compute_cv_squared(series) == pytest.approx(0.0)


def test_compute_cv_squared_all_zero_is_nan():
    assert np.isnan(compute_cv_squared(pd.Series([0, 0, 0])))


def test_compute_cv_squared_single_nonzero_is_nan():
    # Can't compute a standard deviation from one point
    assert np.isnan(compute_cv_squared(pd.Series([0, 0, 5, 0])))


# ---- classify_from_stats: boundary conditions -----------------------------------

def test_boundary_adi_exactly_at_threshold_is_intermittent_side():
    # ADI == threshold uses the ">=" branch, per the spec
    assert classify_from_stats(1.32, 0.3) == "Intermittent"


def test_boundary_adi_just_below_threshold_is_smooth_side():
    assert classify_from_stats(1.319999, 0.3) == "Smooth"


def test_boundary_cv2_exactly_at_threshold_is_erratic_side():
    assert classify_from_stats(0.5, 0.49) == "Erratic"


def test_boundary_cv2_just_below_threshold_is_smooth_side():
    assert classify_from_stats(0.5, 0.489999) == "Smooth"


def test_boundary_both_at_threshold_is_lumpy():
    assert classify_from_stats(1.32, 0.49) == "Lumpy"


def test_classify_from_stats_nan_adi_is_no_demand():
    assert classify_from_stats(np.nan, 0.3) == NO_DEMAND_LABEL


def test_classify_from_stats_nan_cv2_is_insufficient_data():
    # NaN ADI is checked first, so this only shows InsufficientData
    # when ADI itself is a real number and only CV2 is NaN.
    assert classify_from_stats(2.0, np.nan) == INSUFFICIENT_DATA_LABEL


# ---- classify_series: known synthetic examples, one per category -----------------

def test_known_example_smooth():
    # Demand every day, low size variability (2,3 alternating)
    series = pd.Series([2, 3, 2, 3, 2, 3, 2, 3, 2, 3])
    result = classify_series(series)
    assert result["demand_class"] == "Smooth"
    assert result["ADI"] < 1.32
    assert result["CV2"] < 0.49


def test_known_example_intermittent():
    # Sparse (3 of 13 periods), but constant size (always 2) when it occurs
    series = pd.Series([0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0])
    result = classify_series(series)
    assert result["demand_class"] == "Intermittent"
    assert result["ADI"] >= 1.32
    assert result["CV2"] < 0.49


def test_known_example_erratic():
    # Demand every period, but highly variable size
    series = pd.Series([1, 10, 2, 15, 1, 8, 3, 20, 2, 12])
    result = classify_series(series)
    assert result["demand_class"] == "Erratic"
    assert result["ADI"] < 1.32
    assert result["CV2"] >= 0.49


def test_known_example_lumpy():
    # Sparse (2 of 13 periods) AND highly variable size (5 vs 25)
    series = pd.Series([0, 0, 0, 5, 0, 0, 0, 0, 0, 25, 0, 0, 0])
    result = classify_series(series)
    assert result["demand_class"] == "Lumpy"
    assert result["ADI"] >= 1.32
    assert result["CV2"] >= 0.49


def test_known_example_no_demand():
    result = classify_series(pd.Series([0, 0, 0, 0, 0]))
    assert result["demand_class"] == NO_DEMAND_LABEL


def test_known_example_insufficient_data():
    result = classify_series(pd.Series([0, 0, 5, 0, 0]))
    assert result["demand_class"] == INSUFFICIENT_DATA_LABEL


# ---- classify_catalog: multi-item panel, nothing dropped ------------------------

def _synthetic_catalog_panel() -> pd.DataFrame:
    """4 items covering smooth/intermittent/no-demand/insufficient-data."""
    rows = []
    series_by_item = {
        "ITEM_SMOOTH": [2, 3, 2, 3, 2, 3, 2, 3, 2, 3],
        "ITEM_INTERMITTENT": [0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
        "ITEM_NODEMAND": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "ITEM_INSUFFICIENT": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
    }
    for item_id, values in series_by_item.items():
        for v in values:
            rows.append({"item_id": item_id, "sales": v})
    df = pd.DataFrame(rows)
    df["item_id"] = df["item_id"].astype("category")
    return df


def test_classify_catalog_covers_every_item():
    panel = _synthetic_catalog_panel()
    result = classify_catalog(panel)
    assert set(result["item_id"]) == {
        "ITEM_SMOOTH", "ITEM_INTERMITTENT", "ITEM_NODEMAND", "ITEM_INSUFFICIENT",
    }
    assert len(result) == 4  # nothing dropped, including the edge cases


def test_classify_catalog_assigns_expected_classes():
    panel = _synthetic_catalog_panel()
    result = classify_catalog(panel).set_index("item_id")
    assert result.loc["ITEM_SMOOTH", "demand_class"] == "Smooth"
    assert result.loc["ITEM_INTERMITTENT", "demand_class"] == "Intermittent"
    assert result.loc["ITEM_NODEMAND", "demand_class"] == NO_DEMAND_LABEL
    assert result.loc["ITEM_INSUFFICIENT", "demand_class"] == INSUFFICIENT_DATA_LABEL


# ---- select_representative_items ------------------------------------------------

def _classification_with_counts(n_smooth, n_intermittent, n_erratic, n_lumpy) -> pd.DataFrame:
    rows = []
    for cat, n in [
        ("Smooth", n_smooth), ("Intermittent", n_intermittent),
        ("Erratic", n_erratic), ("Lumpy", n_lumpy),
    ]:
        for i in range(n):
            rows.append({"item_id": f"{cat}_{i}", "ADI": 1.0, "CV2": 0.1, "demand_class": cat})
    return pd.DataFrame(rows)


def test_select_representative_items_takes_n_per_category_when_available():
    classification = _classification_with_counts(10, 10, 10, 10)
    selected = select_representative_items(classification, n_per_category=5, random_state=1)
    counts = selected["demand_class"].value_counts()
    assert counts["Smooth"] == 5
    assert counts["Intermittent"] == 5
    assert counts["Erratic"] == 5
    assert counts["Lumpy"] == 5
    assert len(selected) == 20


def test_select_representative_items_does_not_fabricate_when_short():
    # Only 2 Lumpy items exist -- must return exactly 2, never padded to 5.
    # The other categories have 10 available but are still capped at
    # n_per_category=5 each.
    classification = _classification_with_counts(10, 10, 10, 2)
    selected = select_representative_items(classification, n_per_category=5, random_state=1)
    counts = selected["demand_class"].value_counts()
    assert counts["Lumpy"] == 2
    assert len(selected) == 5 + 5 + 5 + 2


def test_select_representative_items_handles_empty_category():
    # Zero Erratic items -- category contributes nothing, no crash
    classification = _classification_with_counts(10, 10, 0, 10)
    selected = select_representative_items(classification, n_per_category=5, random_state=1)
    assert "Erratic" not in selected["demand_class"].values
    assert len(selected) == 5 + 5 + 0 + 5


def test_select_representative_items_excludes_no_demand_and_insufficient_data():
    classification = _classification_with_counts(10, 10, 10, 10)
    extra = pd.DataFrame([
        {"item_id": "X1", "ADI": np.nan, "CV2": np.nan, "demand_class": NO_DEMAND_LABEL},
        {"item_id": "X2", "ADI": 5.0, "CV2": np.nan, "demand_class": INSUFFICIENT_DATA_LABEL},
    ])
    classification = pd.concat([classification, extra], ignore_index=True)
    selected = select_representative_items(classification, n_per_category=5, random_state=1)
    assert NO_DEMAND_LABEL not in selected["demand_class"].values
    assert INSUFFICIENT_DATA_LABEL not in selected["demand_class"].values


def test_select_representative_items_is_reproducible_with_same_seed():
    classification = _classification_with_counts(10, 10, 10, 10)
    first = select_representative_items(classification, n_per_category=5, random_state=7)
    second = select_representative_items(classification, n_per_category=5, random_state=7)
    assert sorted(first["item_id"]) == sorted(second["item_id"])
