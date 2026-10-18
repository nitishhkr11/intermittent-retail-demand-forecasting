"""Demand-pattern classification (Smooth / Intermittent / Erratic / Lumpy).

Implements the Syntetos-Boylan-Croston (SBC) classification scheme:
every item is scored on two axes computed independently of each other,
which is the whole point of using two numbers instead of one --

  ADI (Average Demand Interval)
      How SPARSE demand is over time: periods per demand-day.
      ADI = n_periods / n_nonzero_periods.
      Low ADI = demand happens most days. High ADI = long gaps between
      demand events.

  CV^2 (squared coefficient of variation of non-zero demand SIZE)
      How VARIABLE the size of demand is, given that it occurred, with
      the zero-days excluded on purpose -- mixing zeros into a "how
      variable is demand" measure would conflate sparsity with size
      variability, which is exactly what having two separate axes
      avoids.
      CV^2 = (std_nonzero / mean_nonzero) ** 2.

The two axes combine into four quadrants at the thresholds given by
the project methodology (ADI=1.32, CV2=0.49):

    ADI < thr, CV2 < thr  -> Smooth        (frequent, regular size)
    ADI >= thr, CV2 < thr -> Intermittent  (sparse, regular size)
    ADI < thr, CV2 >= thr -> Erratic       (frequent, variable size)
    ADI >= thr, CV2 >= thr-> Lumpy         (sparse AND variable -- hardest)

Two non-standard labels handle series the quadrant rule can't score:

    NoDemand         -- zero non-zero periods. ADI is undefined
                         (division by zero) and so is CV2.
    InsufficientData -- exactly one non-zero period. ADI is defined,
                         but CV2 needs at least two non-zero
                         observations to compute a standard deviation.

Neither case is dropped from the classification table -- they're
labelled explicitly so nothing goes missing silently. They're excluded
only from representative *selection*, since they don't fit the
four-quadrant scheme being sampled from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import resolve_path

logger = get_logger(__name__)

STANDARD_CATEGORIES = ["Smooth", "Intermittent", "Erratic", "Lumpy"]
NO_DEMAND_LABEL = "NoDemand"
INSUFFICIENT_DATA_LABEL = "InsufficientData"


def compute_nonzero_stats(sales: pd.Series) -> Dict[str, float]:
    """Basic stats of the non-zero (positive) values in a demand series.

    "Non-zero" is defined as sales > 0, not sales != 0, since demand
    counts shouldn't be negative (Stage 1's validator already flags
    negative values as a data-quality warning).
    """
    n_periods = int(len(sales))
    nonzero = sales[sales > 0]
    n_nonzero = int(len(nonzero))
    return {
        "n_periods": n_periods,
        "n_nonzero": n_nonzero,
        "mean_nonzero": float(nonzero.mean()) if n_nonzero > 0 else np.nan,
        # std needs >= 2 points; pandas returns NaN for n=1 with ddof=1
        # anyway, but being explicit here documents the reason.
        "std_nonzero": float(nonzero.std()) if n_nonzero > 1 else np.nan,
    }


def compute_adi(sales: pd.Series) -> float:
    """Average Demand Interval = total periods / periods with non-zero demand.

    Returns NaN for an all-zero series (no demand to space out).
    """
    stats = compute_nonzero_stats(sales)
    if stats["n_nonzero"] == 0:
        return np.nan
    return stats["n_periods"] / stats["n_nonzero"]


def compute_cv_squared(sales: pd.Series) -> float:
    """Squared coefficient of variation of non-zero demand size.

    Returns NaN if there are fewer than 2 non-zero observations (a
    standard deviation needs at least 2 points) or if the non-zero
    mean is 0 (shouldn't happen given "non-zero" means > 0, but
    guarded rather than assumed).
    """
    stats = compute_nonzero_stats(sales)
    if stats["n_nonzero"] < 2 or not stats["mean_nonzero"]:
        return np.nan
    return (stats["std_nonzero"] / stats["mean_nonzero"]) ** 2


def classify_from_stats(
    adi: float,
    cv2: float,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
) -> str:
    """Apply the SBC quadrant rule to already-computed ADI/CV2.

    Kept separate from classify_series() so the threshold logic --
    including exact boundary behavior at the thresholds themselves --
    is trivial to unit test without needing to construct a real series
    that happens to land on a precise ADI or CV2 value.
    """
    if np.isnan(adi):
        return NO_DEMAND_LABEL
    if np.isnan(cv2):
        return INSUFFICIENT_DATA_LABEL
    if adi < adi_threshold and cv2 < cv2_threshold:
        return "Smooth"
    if adi >= adi_threshold and cv2 < cv2_threshold:
        return "Intermittent"
    if adi < adi_threshold and cv2 >= cv2_threshold:
        return "Erratic"
    return "Lumpy"


def classify_series(
    sales: pd.Series,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
) -> Dict[str, Any]:
    """Compute ADI, CV2, and the resulting demand class for one item's series."""
    adi = compute_adi(sales)
    cv2 = compute_cv_squared(sales)
    demand_class = classify_from_stats(adi, cv2, adi_threshold, cv2_threshold)
    return {"ADI": adi, "CV2": cv2, "demand_class": demand_class}


def classify_catalog(
    panel: pd.DataFrame,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
) -> pd.DataFrame:
    """Classify every item_id in `panel`. Never drops an item.

    `panel` must have `item_id` and `sales` columns -- the full
    CA_1/FOODS_3 catalog, not a pre-filtered subset. Classification
    has to run on everyone before anyone gets selected, or "representative"
    stops meaning anything.
    """
    rows = []
    for item_id, group in panel.groupby("item_id", observed=True):
        result = classify_series(group["sales"], adi_threshold, cv2_threshold)
        result["item_id"] = item_id
        rows.append(result)

    result_df = pd.DataFrame(rows)[["item_id", "ADI", "CV2", "demand_class"]]
    logger.info("Classified %d items", len(result_df))
    for label, count in result_df["demand_class"].value_counts().items():
        logger.info("  %s: %d", label, count)
    return result_df


def select_representative_items(
    classification: pd.DataFrame,
    n_per_category: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample up to n_per_category items from each of the 4 standard categories.

    Never fabricates items: a category with fewer than n_per_category
    contributes all of it has, not padded to look like a full set.
    NoDemand / InsufficientData items are never selected -- they don't
    fit the four-quadrant scheme this selection samples from.
    """
    selected_parts = []
    for category in STANDARD_CATEGORIES:
        pool = classification[classification["demand_class"] == category]
        n_take = min(n_per_category, len(pool))
        if len(pool) < n_per_category:
            logger.warning(
                "Category '%s' has only %d item(s), fewer than the requested %d "
                "-- taking all of them, not fabricating more.",
                category, len(pool), n_per_category,
            )
        if n_take > 0:
            selected_parts.append(pool.sample(n=n_take, random_state=random_state))

    if not selected_parts:
        return classification.iloc[0:0].copy()
    return pd.concat(selected_parts, ignore_index=True)


def save_classification(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Path:
    """Save the full classification table to data/processed/."""
    config = config or load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / config["output"]["classification_filename"]
    df.to_parquet(out_path, index=False)
    logger.info("Saved classification table to %s", out_path)
    return out_path


def save_selected_items(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Path:
    """Save the representative selection to data/processed/."""
    config = config or load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / config["output"]["selected_items_filename"]
    df.to_parquet(out_path, index=False)
    logger.info("Saved selected items to %s", out_path)
    return out_path


if __name__ == "__main__":
    from src.analysis.eda import build_aggregate_series, save_aggregate_series

    cfg = load_config()
    interim_path = resolve_path(cfg["data"]["interim_dir"]) / cfg["output"]["panel_filename"]
    if not interim_path.exists():
        raise FileNotFoundError(
            f"'{interim_path}' not found. Run `python -m src.data.preprocessor` first."
        )
    panel_df = pd.read_parquet(interim_path)

    classification_df = classify_catalog(
        panel_df, cfg["classification"]["adi_threshold"], cfg["classification"]["cv2_threshold"]
    )
    save_classification(classification_df, cfg)

    selected_df = select_representative_items(
        classification_df, cfg["classification"]["n_per_category"], cfg["classification"]["random_seed"]
    )
    save_selected_items(selected_df, cfg)

    aggregate_df = build_aggregate_series(panel_df)
    save_aggregate_series(aggregate_df, cfg)

    logger.info("Stage 2 complete: classification, selection, and aggregate series saved.")
