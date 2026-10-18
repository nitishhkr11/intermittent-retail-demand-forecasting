"""Reusable EDA helpers.

Two build stages live here, both part of the faculty's Phase 1 (data
understanding / synopsis preparation, not modeling):

  Stage 2 (demand classification support): build_aggregate_series,
  save_aggregate_series, build_selected_items_daily_sales,
  save_selected_items_daily_sales.

  Stage 3 (EDA): everything else -- dataset overview stats, missing-value
  summary, rolling statistics, day-of-week pattern, demand-class summary,
  and the two plotting functions shared with notebooks/03 (the ADI/CV2
  quadrant scatter) so that plot isn't duplicated between notebooks.

Nothing in this module fits, forecasts, or evaluates a model. Plotting
functions take an optional `ax` and draw onto it (creating one if not
given), returning the Axes -- the standard matplotlib composition
pattern, and easy to call from a notebook cell that then does its own
`plt.show()` / `fig.savefig()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import resolve_path

logger = get_logger(__name__)

# Columns where missing values are an expected, documented data
# characteristic (see src/data/preprocessor.py: attach_calendar /
# attach_prices) rather than a data-quality problem.
DEFAULT_EXPECTED_MISSING_COLUMNS = [
    "event_name_1", "event_type_1", "event_name_2", "event_type_2", "sell_price",
]

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Muted, non-neon palette kept consistent with the monochrome dashboard
# planned for Phase 3 (see app/theme.py).
CATEGORY_COLORS = {
    "Smooth": "#4C4C4C",
    "Intermittent": "#8C8C8C",
    "Erratic": "#B0862B",
    "Lumpy": "#8B3A3A",
    "NoDemand": "#C9C9C9",
    "InsufficientData": "#C9C9C9",
}


# ---- Stage 2: aggregate series -------------------------------------------------

def build_aggregate_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Sum demand across the ENTIRE CA_1/FOODS_3 catalog into one daily series.

    Uses every item in `panel`, not just a classified or selected
    subset -- aggregating the full catalog, rather than only ~20-25
    items, is what makes this series dense enough to be useful for
    aggregate-level modeling later.

    Returns a DataFrame with columns `date` and `sales`, one row per day.
    """
    aggregate = (
        panel.groupby("date", observed=True)["sales"]
        .sum()
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    logger.info(
        "Built aggregate series: %d days, %s to %s",
        len(aggregate), aggregate["date"].min().date(), aggregate["date"].max().date(),
    )
    return aggregate


def save_aggregate_series(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Path:
    """Save the aggregate series to data/processed/."""
    config = config or load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / config["output"]["aggregate_filename"]
    df.to_parquet(out_path, index=False)
    logger.info("Saved aggregate series to %s", out_path)
    return out_path


# ---- Phase 2 Step 7: item-level daily sales for the 20 selected items -----------

def build_selected_items_daily_sales(
    panel: pd.DataFrame, selected_items: pd.DataFrame
) -> pd.DataFrame:
    """Daily sales for just the 20 representative items chosen in Stage 2,
    with each item's demand_class attached -- the Phase 2 Step 7 item-level
    modeling input (Croston / TSB / per-item Seasonal Naive).

    Filters `panel` down to the item_ids present in `selected_items` (no
    re-aggregation needed: `panel` already has one row per item per day)
    and keeps only `item_id`, `date`, `sales`, then left-joins
    `demand_class` from `selected_items` by `item_id`.

    This never fills, interpolates, or otherwise invents a sales value --
    every row returned is a real `panel` row. It only checks that nothing
    is missing: every selected item_id must actually be present in
    `panel` (raises otherwise) and must receive a demand_class from the
    join (raises on any row that doesn't -- that would mean an id in
    `selected_items` wasn't matched, which should never happen since
    `selected_items` is itself derived from this same panel). An uneven
    row count per item (a genuine gap in the source panel, not something
    this function would ever paper over) is logged as a warning, not
    silently fixed.
    """
    selected = selected_items[["item_id", "demand_class"]].copy()
    selected["item_id"] = selected["item_id"].astype(str)
    selected_ids = selected["item_id"].unique()

    subset = panel.loc[
        panel["item_id"].astype(str).isin(selected_ids), ["item_id", "date", "sales"]
    ].copy()
    subset["item_id"] = subset["item_id"].astype(str)

    found_ids = set(subset["item_id"].unique())
    missing_ids = set(selected_ids) - found_ids
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} selected item_id(s) not found in panel: "
            f"{sorted(missing_ids)}"
        )

    merged = subset.merge(selected, on="item_id", how="left")
    n_missing_class = int(merged["demand_class"].isna().sum())
    if n_missing_class:
        raise ValueError(
            f"{n_missing_class} row(s) have no demand_class after the join -- "
            "every item in selected_items should match by item_id."
        )

    merged = merged.sort_values(["item_id", "date"]).reset_index(drop=True)

    counts_per_item = merged.groupby("item_id", observed=True)["date"].count()
    if counts_per_item.nunique() > 1:
        logger.warning(
            "Uneven row counts per item in build_selected_items_daily_sales "
            "(min=%d, max=%d) -- the source panel may have missing item-day "
            "rows for some items; none were filled in.",
            counts_per_item.min(), counts_per_item.max(),
        )

    logger.info(
        "Built selected-items daily sales: %d items, %d rows, %s to %s",
        merged["item_id"].nunique(), len(merged),
        merged["date"].min().date(), merged["date"].max().date(),
    )
    return merged


def save_selected_items_daily_sales(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Path:
    """Save the 20-item daily sales table to data/processed/."""
    config = config or load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / config["output"]["selected_items_daily_sales_filename"]
    df.to_parquet(out_path, index=False)
    logger.info("Saved selected-items daily sales to %s", out_path)
    return out_path


# ---- Stage 3: dataset overview and continuity -----------------------------------

def dataset_overview(panel: pd.DataFrame) -> Dict[str, Any]:
    """Summary stats for the "1. Dataset overview" EDA section.

    Returns row count, unique item count, date range, inferred
    frequency, the stores/departments present, and basic sales stats --
    everything the notebook needs to report without recomputing it
    inline.
    """
    dates = pd.Series(sorted(panel["date"].unique()))
    inferred_freq = pd.infer_freq(dates) if len(dates) > 2 else None
    return {
        "n_rows": int(len(panel)),
        "n_items": int(panel["item_id"].nunique()),
        "date_min": dates.min(),
        "date_max": dates.max(),
        "n_days": int(len(dates)),
        "inferred_freq": inferred_freq,
        "stores": sorted(panel["store_id"].unique().astype(str).tolist()),
        "departments": sorted(panel["dept_id"].unique().astype(str).tolist()),
        "sales_summary": panel["sales"].describe().to_dict(),
    }


# ---- Stage 3: missing values -----------------------------------------------------

def summarize_missing_values(
    df: pd.DataFrame,
    expected_missing_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Missing-value count/percentage per column, flagging the unexpected ones.

    `expected_missing_columns` defaults to the calendar-event and price
    columns that are legitimately missing sometimes (see
    DEFAULT_EXPECTED_MISSING_COLUMNS). Any OTHER column with missing
    values is flagged `is_expected=False` -- that's the signal to look
    closer, not every NaN in the table.
    """
    expected = set(expected_missing_columns or DEFAULT_EXPECTED_MISSING_COLUMNS)
    n_missing = df.isnull().sum()
    n_missing = n_missing[n_missing > 0].sort_values(ascending=False)
    result = pd.DataFrame({
        "column": n_missing.index,
        "n_missing": n_missing.values,
        "pct_missing": (n_missing.values / len(df) * 100).round(2),
    })
    result["is_expected"] = result["column"].isin(expected)
    unexpected = result[~result["is_expected"]]
    if not unexpected.empty:
        logger.warning(
            "Unexpected missing values in columns not on the expected list: %s",
            unexpected["column"].tolist(),
        )
    return result.reset_index(drop=True)


# ---- Stage 3: aggregate demand analysis -------------------------------------------

def compute_rolling_stats(series: pd.Series, window: int) -> pd.DataFrame:
    """Rolling mean and standard deviation of `series` over `window` periods.

    A purely descriptive smoothing view -- not a model fit, and not
    used to produce any forecast.
    """
    return pd.DataFrame({
        "rolling_mean": series.rolling(window=window, min_periods=1).mean(),
        "rolling_std": series.rolling(window=window, min_periods=1).std(),
    })


def compute_day_of_week_pattern(
    df: pd.DataFrame, date_col: str = "date", value_col: str = "sales"
) -> pd.Series:
    """Mean of `value_col` by day of week, ordered Monday through Sunday."""
    weekday_names = pd.to_datetime(df[date_col]).dt.day_name()
    weekday_names = pd.Categorical(weekday_names, categories=DAY_ORDER, ordered=True)
    return df.groupby(weekday_names, observed=True)[value_col].mean()


# ---- Stage 3: demand pattern summary -----------------------------------------------

def summarize_demand_classes(classification: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of items in each demand_class, including the
    non-standard NoDemand / InsufficientData labels if present.
    """
    counts = classification["demand_class"].value_counts()
    pct = (counts / len(classification) * 100).round(1)
    return pd.DataFrame({"count": counts, "pct_of_catalog": pct}).reset_index(
        names="demand_class"
    )


def plot_adi_cv2_quadrants(
    classification: pd.DataFrame,
    adi_threshold: float,
    cv2_threshold: float,
    ax=None,
):
    """ADI (x) vs CV2 (y) scatter with the quadrant boundaries drawn and
    labelled. Shared by notebooks/03_demand_classification.ipynb and
    notebooks/02_eda.ipynb so this plot exists in exactly one place.

    Only items with both ADI and CV2 defined can be plotted; the number
    excluded (NoDemand / InsufficientData) is logged, not hidden.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    plot_df = classification.dropna(subset=["ADI", "CV2"])
    n_excluded = len(classification) - len(plot_df)
    if n_excluded:
        logger.info(
            "plot_adi_cv2_quadrants: excluding %d item(s) with undefined ADI/CV2",
            n_excluded,
        )

    for demand_class, group in plot_df.groupby("demand_class", observed=True):
        ax.scatter(
            group["ADI"], group["CV2"], label=demand_class, s=50, alpha=0.85,
            color=CATEGORY_COLORS.get(demand_class, "#4C4C4C"),
        )

    ax.axvline(adi_threshold, linestyle="--", color="gray", linewidth=1)
    ax.axhline(cv2_threshold, linestyle="--", color="gray", linewidth=1)

    if not plot_df.empty:
        xmax = max(plot_df["ADI"].max(), adi_threshold) * 1.15
        ymax = max(plot_df["CV2"].max(), cv2_threshold) * 1.15
        ax.text(adi_threshold * 0.5, cv2_threshold * 0.5, "Smooth", ha="center", color="gray")
        ax.text((adi_threshold + xmax) / 2, cv2_threshold * 0.5, "Intermittent", ha="center", color="gray")
        ax.text(adi_threshold * 0.5, (cv2_threshold + ymax) / 2, "Erratic", ha="center", color="gray")
        ax.text((adi_threshold + xmax) / 2, (cv2_threshold + ymax) / 2, "Lumpy", ha="center", color="gray")

    ax.set_xlabel("ADI (Average Demand Interval)")
    ax.set_ylabel("CV² (squared coefficient of variation)")
    ax.set_title("Demand classification quadrants")
    ax.legend(title="Assigned class")
    return ax


def plot_representative_series(
    panel: pd.DataFrame,
    selected_items: pd.DataFrame,
    items_per_category: int = 2,
):
    """Small-multiples plot of daily demand for a few items per category.

    Picks the first `items_per_category` selected items per category
    (deterministic -- not re-sampled here, since the sampling already
    happened in select_representative_items) and plots each item's raw
    daily series so the shape differences between categories are
    visible directly, without any smoothing or model fitting.

    Returns the created Figure.
    """
    import matplotlib.pyplot as plt

    categories = [c for c in ["Smooth", "Intermittent", "Erratic", "Lumpy"]
                  if c in selected_items["demand_class"].values]
    fig, axes = plt.subplots(len(categories), 1, figsize=(9, 2.5 * len(categories)), sharex=True)
    if len(categories) == 1:
        axes = [axes]

    for ax, category in zip(axes, categories):
        item_ids = (
            selected_items.loc[selected_items["demand_class"] == category, "item_id"]
            .head(items_per_category)
        )
        for item_id in item_ids:
            item_series = panel.loc[panel["item_id"] == item_id].sort_values("date")
            ax.plot(item_series["date"], item_series["sales"], label=str(item_id), linewidth=1)
        ax.set_ylabel("Units sold")
        ax.set_title(category, loc="left", fontsize=10, color=CATEGORY_COLORS.get(category, "#4C4C4C"))
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Representative daily demand by category")
    fig.tight_layout()
    return fig
