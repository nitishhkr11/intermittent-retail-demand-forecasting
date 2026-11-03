"""Engine behind notebook 05: Seasonal Naive vs. Croston vs. TSB on intermittent demand.

Everything the notebook shows is computed here, by small functions that can be unit
tested on synthetic series:

* describing demand patterns (zero share, zero runs, demand size / interval /
  occurrence probability);
* running the three baselines through the SAME five walk-forward folds as every other
  Phase 2 / Phase 3 experiment, using the existing, unmodified
  `run_backtest` / `summarize_backtest` / `fold_results_to_long_table`;
* scoring them with MAE, RMSE, zero-protected MAPE, WAPE, bias and MASE
  (`forecast_artifacts.metrics_by_group`);
* a deliberately naive "forecast zero everywhere" reference, which shows what MAE
  rewards on a series that is mostly zeros;
* the internal smoothed states of Croston and TSB next to the empirical size /
  interval / probability they are trying to track;
* a baselines-only robustness run on ALL items, with demand classes recomputed from
  each fold's own training window ("fold-local" classes).

Nothing here reads the final holdout: every loader cuts the data at the development
window (`src.data.aggregate_series.load_aggregate_development`) before any item series
is built, so the 28 reserved days never reach a model, a metric or a plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.demand_classification import classify_series, compute_adi, compute_cv_squared
from src.data.aggregate_series import DevelopmentWindow
from src.data.splitter import split_final_holdout
from src.evaluation.backtesting import RollingOrigin, generate_rolling_origins, run_backtest, summarize_backtest
from src.evaluation.forecast_artifacts import metrics_by_group, seasonal_naive_scale
from src.evaluation.item_level import fold_results_to_long_table
from src.models.base import BaseForecaster
from src.models.baseline import SeasonalNaive
from src.models.croston import CrostonForecaster
from src.models.tsb import TSBForecaster
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.paths import resolve_path

logger = get_logger(__name__)

CLASS_ORDER = ["Smooth", "Intermittent", "Erratic", "Lumpy"]
ZERO_MODEL_NAME = "ZeroForecast(reference)"
#: Plot style per model: colour AND marker/hatch, so nothing depends on colour alone.
MODEL_STYLE = {
    "SeasonalNaive": {"color": "#6B6B6B", "hatch": "//", "marker": "o", "linestyle": "-"},
    "Croston": {"color": "#B0862B", "hatch": "\\\\", "marker": "s", "linestyle": "--"},
    "TSB": {"color": "#2F6F8F", "hatch": "xx", "marker": "^", "linestyle": "-."},
    "XGBoost": {"color": "#8B3A3A", "hatch": "oo", "marker": "d", "linestyle": "-"},
    ZERO_MODEL_NAME: {"color": "#FFFFFF", "hatch": "..", "marker": "x", "linestyle": ":"},
}


def model_style(model_name: str) -> Dict[str, str]:
    """Style entry for a model name (prefix match, so `Croston(alpha=0.1)` finds `Croston`)."""
    for key, style in MODEL_STYLE.items():
        if model_name.startswith(key):
            return style
    return {"color": "#444444", "hatch": "", "marker": "d", "linestyle": "-"}


def _class_rank(label: str) -> int:
    return CLASS_ORDER.index(label) if label in CLASS_ORDER else len(CLASS_ORDER)


# ----------------------------------------------------------------------------------------
# Demand patterns
# ----------------------------------------------------------------------------------------
def _validated_values(series: pd.Series, what: str = "series") -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.size == 0:
        raise ValueError(f"{what} is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{what} contains NaN or infinite value(s).")
    if np.any(values < 0):
        raise ValueError(f"{what} contains negative value(s); unit sales cannot be negative.")
    return values


def zero_run_lengths(values: Sequence[float]) -> np.ndarray:
    """Lengths of the maximal runs of consecutive zero-sales days (empty array if none)."""
    arr = _validated_values(pd.Series(values), "values")
    is_zero = np.concatenate(([False], arr == 0, [False])).astype(np.int8)
    edges = np.diff(is_zero)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return (ends - starts).astype(int)


def demand_pattern_summary(series: pd.Series) -> Dict[str, Any]:
    """One item's size / interval / probability description.

    * `occurrence_probability` = share of days with sales > 0.
    * `ADI` (average demand interval) = days / demand days (the Phase 1 definition).
    * `mean_interval` = mean gap, in days, between consecutive demand days, counting the
      first demand day's distance from the start of the series (the quantity Croston's
      method smooths).
    * `mean_size` / `median_size` / `max_size` = demand size on demand days only.
    * `CV2` = squared coefficient of variation of the demand size (Phase 1 definition).
    * zero-run statistics = runs of consecutive zero-sales days.
    * `leading_zero_days` = days before the first sale (the item was probably not yet on sale);
      `trailing_zero_days` = days since the last sale at the end of the series.
    """
    values = _validated_values(series, "series")
    series = pd.Series(values)
    n = int(values.size)
    positive = values > 0
    n_demand = int(positive.sum())
    sizes = values[positive]
    runs = zero_run_lengths(values)
    if n_demand:
        positions = np.flatnonzero(positive) + 1
        gaps = np.diff(positions, prepend=0)
        mean_interval = float(gaps.mean())
    else:
        mean_interval = float("nan")
    return {
        "n_days": n,
        "n_demand_days": n_demand,
        "zero_share": float(1.0 - n_demand / n),
        "occurrence_probability": float(n_demand / n),
        "ADI": float(compute_adi(series)),
        "mean_interval": mean_interval,
        "mean_size": float(sizes.mean()) if n_demand else float("nan"),
        "median_size": float(np.median(sizes)) if n_demand else float("nan"),
        "max_size": float(sizes.max()) if n_demand else float("nan"),
        "CV2": float(compute_cv_squared(series)),
        "mean_daily_demand": float(values.mean()),
        "leading_zero_days": int(np.argmax(positive)) if n_demand else n,
        "trailing_zero_days": int(np.argmax(positive[::-1])) if n_demand else n,
        "n_zero_runs": int(runs.size),
        "longest_zero_run": int(runs.max()) if runs.size else 0,
        "mean_zero_run": float(runs.mean()) if runs.size else 0.0,
    }


def demand_pattern_table(series_by_item: Mapping[str, pd.Series], classes: Mapping[str, str]) -> pd.DataFrame:
    """`demand_pattern_summary` for every item, ordered by demand class then item id."""
    rows = []
    for item_id, series in series_by_item.items():
        record = {"item_id": item_id, "demand_class": classes[item_id]}
        record.update(demand_pattern_summary(series))
        rows.append(record)
    table = pd.DataFrame(rows)
    table["_rank"] = table["demand_class"].map(_class_rank)
    return table.sort_values(["_rank", "item_id"]).drop(columns="_rank").reset_index(drop=True)


def zero_run_table(series_by_item: Mapping[str, pd.Series], classes: Mapping[str, str]) -> pd.DataFrame:
    """Every zero run of every item: one row per run (`item_id`, `demand_class`, `run_length`)."""
    rows = []
    for item_id, series in series_by_item.items():
        for length in zero_run_lengths(series.to_numpy(dtype=float)):
            rows.append({"item_id": item_id, "demand_class": classes[item_id], "run_length": int(length)})
    return pd.DataFrame(rows, columns=["item_id", "demand_class", "run_length"])


# ----------------------------------------------------------------------------------------
# Models, folds, scoring
# ----------------------------------------------------------------------------------------
def baseline_factories(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Callable[[], BaseForecaster]]:
    """The three Phase 2 item-level models, with exactly the Phase 2 settings from config.yaml."""
    config = load_config() if config is None else config
    season = int(config["backtest"]["season_length"])
    models = config["models"]
    croston_alpha = float(models["croston_alpha"])
    tsb_alpha, tsb_beta = float(models["tsb_alpha"]), float(models["tsb_beta"])
    return {
        SeasonalNaive(season_length=season).name: lambda: SeasonalNaive(season_length=season),
        CrostonForecaster(alpha=croston_alpha).name: lambda: CrostonForecaster(alpha=croston_alpha),
        TSBForecaster(alpha=tsb_alpha, beta=tsb_beta).name: lambda: TSBForecaster(alpha=tsb_alpha, beta=tsb_beta),
    }


def fold_origins(n_obs: int, horizon: int, n_folds: int) -> List[RollingOrigin]:
    """The shared walk-forward fold layout (the existing generator, unchanged)."""
    return generate_rolling_origins(n_obs=n_obs, horizon=horizon, n_folds=n_folds)


def item_mase_scales(
    series_by_item: Mapping[str, pd.Series], horizon: int, n_folds: int, lag: int = 7
) -> pd.DataFrame:
    """MASE denominator per (item, fold): mean |y_t - y_(t-lag)| on that fold's TRAINING window."""
    rows = []
    for item_id, series in series_by_item.items():
        values = series.to_numpy(dtype=float)
        for origin in fold_origins(len(values), horizon, n_folds):
            rows.append(
                {
                    "item_id": item_id,
                    "fold": origin.fold,
                    "mase_scale": seasonal_naive_scale(values[: origin.train_end], lag),
                }
            )
    return pd.DataFrame(rows, columns=["item_id", "fold", "mase_scale"])


def run_item_baselines(
    series_by_item: Mapping[str, pd.Series],
    classes: Mapping[str, str],
    factories: Mapping[str, Callable[[], BaseForecaster]],
    horizon: int,
    n_folds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run every model on every item through the existing walk-forward backtest.

    Returns `(long_table, fold_summary)`: one row per (item, model, fold, date) and one per
    (item, model, fold). Failed folds stay in both tables (NaN prediction, `success=False`,
    the error text), so a failure is visible rather than silently missing.
    """
    long_tables, summaries = [], []
    for item_id, series in series_by_item.items():
        for model_name, factory in factories.items():
            results = run_backtest(series, factory, horizon=horizon, n_folds=n_folds)
            long_tables.append(fold_results_to_long_table(results, item_id, classes[item_id]))
            summary = summarize_backtest(results)
            summary.insert(0, "item_id", item_id)
            summary.insert(1, "demand_class", classes[item_id])
            summaries.append(summary)
    return pd.concat(long_tables, ignore_index=True), pd.concat(summaries, ignore_index=True)


def zero_forecast_long_table(long_table: pd.DataFrame) -> pd.DataFrame:
    """The "always forecast zero" reference, on exactly the evaluation cells of `long_table`.

    It needs no fitting, so it is defined even where a model failed. It exists to show what
    the error metrics reward on a mostly-zero series; it is never presented as a candidate.
    """
    cells = long_table.drop_duplicates(["item_id", "fold", "date"]).copy()
    cells["model"] = ZERO_MODEL_NAME
    cells["predicted"] = 0.0
    cells["success"] = True
    cells["error"] = ""
    return cells.reset_index(drop=True)


def score_long_table(long_table: pd.DataFrame, mase_scales: pd.DataFrame) -> pd.DataFrame:
    """Fold-level MAE, RMSE, zero-protected MAPE, WAPE, bias and MASE per (item, model, fold)."""
    keys = ["item_id", "demand_class", "model", "fold"]
    metrics = metrics_by_group(long_table, keys, mase_scales)
    spans = (
        long_table.groupby(keys, sort=False)["date"]
        .agg(forecast_start="min", forecast_end="max")
        .reset_index()
    )
    return metrics.merge(spans, on=keys, how="left")


_METRIC_COLUMNS = ["MAE", "RMSE", "MAPE", "WAPE", "MASE", "bias"]


def _mean_and_count(values: pd.Series) -> Tuple[float, int]:
    valid = values.dropna()
    return (float(valid.mean()) if len(valid) else float("nan")), int(len(valid))


def item_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric over an item's SUCCESSFUL folds (NaN if it has none), plus fold spread.

    MAPE / WAPE / MASE are averaged only over folds where they are defined, and the number of
    such folds is reported next to them. `MAE_std` is the fold-to-fold standard deviation.
    """
    need = {"item_id", "demand_class", "model", "fold", "success", *_METRIC_COLUMNS}
    missing = need - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold_metrics is missing column(s): {sorted(missing)}")
    rows = []
    for (item_id, demand_class, model), group in fold_metrics.groupby(
        ["item_id", "demand_class", "model"], sort=False, observed=True
    ):
        ok = group[group["success"].astype(bool)]
        record: Dict[str, Any] = {
            "item_id": item_id,
            "demand_class": demand_class,
            "model": model,
            "n_folds": int(len(group)),
            "n_success": int(len(ok)),
        }
        for metric in _METRIC_COLUMNS:
            mean, count = _mean_and_count(ok[metric]) if len(ok) else (float("nan"), 0)
            record[f"{metric}_mean"] = mean
            if metric in ("MAPE", "WAPE", "MASE"):
                record[f"{metric}_n_folds_valid"] = count
        record["MAE_std"] = float(ok["MAE"].std(ddof=1)) if len(ok) > 1 else float("nan")
        rows.append(record)
    return pd.DataFrame(rows)


def class_summary(item_table: pd.DataFrame, class_column: str = "demand_class") -> pd.DataFrame:
    """Macro average (every item weighted equally) of the per-item means, per class and model.

    Volume-weighted averages are deliberately not computed: they would let the highest-volume
    items decide each class's number.
    """
    need = {"item_id", class_column, "model", *(f"{m}_mean" for m in _METRIC_COLUMNS)}
    missing = need - set(item_table.columns)
    if missing:
        raise ValueError(f"item_table is missing column(s): {sorted(missing)}")
    rows = []
    for (label, model), group in item_table.groupby([class_column, "model"], sort=False, observed=True):
        record: Dict[str, Any] = {
            class_column: label,
            "model": model,
            "n_items": int(group["item_id"].nunique()),
        }
        for metric in _METRIC_COLUMNS:
            mean, count = _mean_and_count(group[f"{metric}_mean"])
            record[f"{metric}_macro_mean"] = mean
            if metric in ("MAPE", "WAPE", "MASE"):
                record[f"{metric}_n_items_valid"] = count
        mae = group["MAE_mean"].dropna()
        record["MAE_item_min"] = float(mae.min()) if len(mae) else float("nan")
        record["MAE_item_max"] = float(mae.max()) if len(mae) else float("nan")
        rows.append(record)
    table = pd.DataFrame(rows)
    table["_rank"] = table[class_column].map(_class_rank)
    return table.sort_values(["_rank", "model"], kind="stable").drop(columns="_rank").reset_index(drop=True)


def reproduction_check(
    new_fold_summary: pd.DataFrame, saved_fold_summary: pd.DataFrame, atol: float = 1e-9
) -> Dict[str, Any]:
    """Does a fresh run reproduce the saved Phase 2 item-level fold metrics?

    Rows are matched on (item_id, model, fold). MAE, RMSE and MAPE are compared (NaN equals NaN).
    Nothing is rewritten: this only reports.
    """
    keys = ["item_id", "model", "fold"]
    left = new_fold_summary[keys + ["MAE", "RMSE", "MAPE"]]
    right = saved_fold_summary[keys + ["MAE", "RMSE", "MAPE"]]
    merged = left.merge(right, on=keys, how="outer", suffixes=("_new", "_saved"), indicator=True)
    only_new = int((merged["_merge"] == "left_only").sum())
    only_saved = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    diffs: Dict[str, float] = {}
    for metric in ("MAE", "RMSE", "MAPE"):
        a = both[f"{metric}_new"].to_numpy(dtype=float)
        b = both[f"{metric}_saved"].to_numpy(dtype=float)
        nan_mismatch = bool(np.any(np.isnan(a) != np.isnan(b)))
        both_valid = ~np.isnan(a) & ~np.isnan(b)
        diff = float(np.max(np.abs(a[both_valid] - b[both_valid]))) if both_valid.any() else 0.0
        diffs[metric] = float("inf") if nan_mismatch else diff
    ok = only_new == 0 and only_saved == 0 and all(v <= atol for v in diffs.values())
    return {
        "reproduced": bool(ok),
        "n_compared": int(len(both)),
        "n_only_in_new_run": only_new,
        "n_only_in_saved_file": only_saved,
        "max_abs_diff_MAE": diffs["MAE"],
        "max_abs_diff_RMSE": diffs["RMSE"],
        "max_abs_diff_MAPE": diffs["MAPE"],
        "tolerance": atol,
    }


# ----------------------------------------------------------------------------------------
# What Croston and TSB have learned
# ----------------------------------------------------------------------------------------
def smoothing_state_table(
    series_by_item: Mapping[str, pd.Series],
    classes: Mapping[str, str],
    horizon: int,
    n_folds: int,
    croston_alpha: float,
    tsb_alpha: float,
    tsb_beta: float,
) -> pd.DataFrame:
    """Fitted Croston / TSB states at every fold origin next to the empirical quantities.

    Columns starting `train_` describe the TRAINING window only. `window_mean_actual` is the
    realised mean demand of the fold's forecast window; it is an evaluation-side reference for
    reading the forecast level and is never fed back into any fit.
    """
    rows = []
    for item_id, series in series_by_item.items():
        values = series.to_numpy(dtype=float)
        for origin in fold_origins(len(values), horizon, n_folds):
            train = values[: origin.train_end]
            window = values[origin.train_end : origin.test_end]
            n_demand = int((train > 0).sum())
            row: Dict[str, Any] = {
                "item_id": item_id,
                "demand_class": classes[item_id],
                "fold": origin.fold,
                "n_train": int(train.size),
                "train_demand_days": n_demand,
                "train_probability": n_demand / train.size,
                "train_mean_size": float(train[train > 0].mean()) if n_demand else float("nan"),
                "train_interval": train.size / n_demand if n_demand else float("nan"),
                "train_mean_demand": float(train.mean()),
                "train_recent_mean_28": float(train[-28:].mean()),
                "window_mean_actual": float(window.mean()),
            }
            history = pd.Series(train)
            try:
                croston = CrostonForecaster(alpha=croston_alpha).fit(history).fit_diagnostics()
                row.update(
                    croston_size=croston["demand_size_smoothed"],
                    croston_interval=croston["interval_smoothed"],
                    croston_forecast=croston["forecast_value"],
                    croston_status="ok",
                )
            except ValueError as exc:
                row.update(
                    croston_size=np.nan, croston_interval=np.nan, croston_forecast=np.nan,
                    croston_status=f"failed: {exc}"[:160],
                )
            tsb = TSBForecaster(alpha=tsb_alpha, beta=tsb_beta).fit(history).fit_diagnostics()
            row.update(
                tsb_probability=tsb["probability_smoothed"],
                tsb_size=tsb["demand_size_smoothed"],
                tsb_forecast=tsb["forecast_value"],
            )
            row["croston_level_error"] = row["croston_forecast"] - row["window_mean_actual"]
            row["tsb_level_error"] = row["tsb_forecast"] - row["window_mean_actual"]
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# All-item robustness run with fold-local demand classes
# ----------------------------------------------------------------------------------------
def panel_path(config: Optional[Mapping[str, Any]] = None) -> Path:
    config = load_config() if config is None else config
    return resolve_path(Path(config["data"]["interim_dir"]) / config["output"]["panel_filename"])


def load_all_item_development_sales(
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[pd.DataFrame, DevelopmentWindow]:
    """Daily sales of EVERY item as a date x item table, cut at the development window.

    The cut is applied right after reading, before anything is pivoted, so the reserved
    final-holdout days never enter the returned table. Raises if the panel has gaps or
    duplicate (item, date) rows.
    """
    config = load_config() if config is None else config
    path = panel_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing '{path.name}'. Create it from the raw M5 files with: python -m src.data.preprocessor"
        )
    panel = pd.read_parquet(path, columns=["item_id", "date", "sales"])
    # Same splitter the aggregate loader uses, applied to the panel's own calendar.
    calendar = pd.DataFrame({"date": pd.to_datetime(panel["date"].drop_duplicates()).sort_values()})
    expected_days = (calendar["date"].iloc[-1] - calendar["date"].iloc[0]).days + 1
    if len(calendar) != expected_days:
        raise ValueError(
            f"The panel's calendar has {len(calendar)} distinct dates for {expected_days} calendar days; "
            "a daily series with holes would shift every weekly and lag feature."
        )
    split = split_final_holdout(calendar, holdout_days=int(config["backtest"]["holdout_days"]), date_col="date")
    window = DevelopmentWindow(
        dev_start=split.dev_start,
        dev_end=split.dev_end,
        holdout_start=split.test_start,
        holdout_end=split.test_end,
        n_development_days=split.dev_length,
        n_holdout_days=split.test_length,
    )
    panel = panel[panel["date"] <= window.dev_end].copy()
    panel["item_id"] = panel["item_id"].astype(str)
    wide = panel.pivot(index="date", columns="item_id", values="sales").sort_index().astype(float)
    if wide.shape[0] != window.n_development_days:
        raise ValueError(
            f"Panel has {wide.shape[0]} development days, expected {window.n_development_days}."
        )
    if wide.isna().any().any():
        raise ValueError("Panel has missing (item, date) cells; it must be a complete calendar.")
    if wide.index.max() > window.dev_end:  # defensive; the filter above already guarantees it
        raise AssertionError("Development table extends into the reserved holdout.")
    return wide, window


def fold_local_class_table(
    wide: pd.DataFrame, horizon: int, n_folds: int, phase1_classes: Optional[Mapping[str, str]] = None
) -> pd.DataFrame:
    """ADI / CV2 / class of every item recomputed from each fold's own TRAINING prefix only.

    Phase 1 labelled items using the whole history (including days that later became forecast
    windows). Model SELECTION must not use a label built from the days being forecast, so the
    comparisons below group items by these fold-local labels instead.
    """
    rows = []
    values = wide.to_numpy(dtype=float)
    for origin in fold_origins(values.shape[0], horizon, n_folds):
        for j, item_id in enumerate(wide.columns):
            result = classify_series(pd.Series(values[: origin.train_end, j]))
            rows.append(
                {
                    "fold": origin.fold,
                    "item_id": item_id,
                    "n_train_obs": origin.train_end,
                    "ADI": result["ADI"],
                    "CV2": result["CV2"],
                    "fold_local_class": result["demand_class"],
                    "phase1_class": (phase1_classes or {}).get(item_id, ""),
                }
            )
    return pd.DataFrame(rows)


def class_agreement(class_table: pd.DataFrame) -> pd.DataFrame:
    """Per fold: how many items keep their Phase 1 label when classified on the training prefix."""
    if "phase1_class" not in class_table.columns:
        raise ValueError("class_table needs a 'phase1_class' column.")
    known = class_table[class_table["phase1_class"] != ""]
    rows = []
    for fold, group in known.groupby("fold"):
        same = group["fold_local_class"] == group["phase1_class"]
        rows.append(
            {
                "fold": int(fold),
                "n_items": int(len(group)),
                "n_same_label": int(same.sum()),
                "share_same_label": float(same.mean()),
                "n_changed_label": int((~same).sum()),
            }
        )
    return pd.DataFrame(rows)


def attach_fold_local_class(fold_metrics: pd.DataFrame, class_table: pd.DataFrame) -> pd.DataFrame:
    """Add the fold-local class (and its ADI / CV2) to per-(item, model, fold) metrics."""
    columns = ["item_id", "fold", "fold_local_class", "ADI", "CV2"]
    return fold_metrics.merge(class_table[columns], on=["item_id", "fold"], how="left", validate="many_to_one")


def fold_class_summary(fold_metrics_with_class: pd.DataFrame) -> pd.DataFrame:
    """Per (fold, fold-local class, model): number of items and macro-mean metrics over those items."""
    rows = []
    usable = fold_metrics_with_class[fold_metrics_with_class["success"].astype(bool)]
    for (fold, label, model), group in usable.groupby(["fold", "fold_local_class", "model"], sort=False):
        record: Dict[str, Any] = {
            "fold": int(fold),
            "fold_local_class": label,
            "model": model,
            "n_items": int(group["item_id"].nunique()),
        }
        for metric in _METRIC_COLUMNS:
            mean, count = _mean_and_count(group[metric])
            record[f"{metric}_macro_mean"] = mean
            if metric in ("MAPE", "WAPE", "MASE"):
                record[f"{metric}_n_items_valid"] = count
        rows.append(record)
    table = pd.DataFrame(rows)
    table["_rank"] = table["fold_local_class"].map(_class_rank)
    return table.sort_values(["fold", "_rank", "model"], kind="stable").drop(columns="_rank").reset_index(drop=True)


def all_items_class_summary(fold_class_table: pd.DataFrame) -> pd.DataFrame:
    """Per (fold-local class, model): the five fold-level macro means averaged, with their spread.

    `MAE_across_folds_std` is the standard deviation of the five fold-level class means, i.e. how
    much the class number moves from one forecast window to the next.
    """
    rows = []
    for (label, model), group in fold_class_table.groupby(["fold_local_class", "model"], sort=False):
        record: Dict[str, Any] = {
            "fold_local_class": label,
            "model": model,
            "n_folds": int(group["fold"].nunique()),
            "mean_items_per_fold": float(group["n_items"].mean()),
        }
        for metric in _METRIC_COLUMNS:
            record[f"{metric}_mean_over_folds"] = float(group[f"{metric}_macro_mean"].mean())
        record["MAE_across_folds_std"] = (
            float(group["MAE_macro_mean"].std(ddof=1)) if len(group) > 1 else float("nan")
        )
        rows.append(record)
    table = pd.DataFrame(rows)
    table["_rank"] = table["fold_local_class"].map(_class_rank)
    return table.sort_values(["_rank", "model"], kind="stable").drop(columns="_rank").reset_index(drop=True)


def run_all_items_baselines(
    wide: pd.DataFrame,
    factories: Mapping[str, Callable[[], BaseForecaster]],
    horizon: int,
    n_folds: int,
    phase1_classes: Optional[Mapping[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Baselines on every item. Returns `(fold_metrics_with_fold_local_class, class_table)`."""
    series_by_item = {
        str(item_id): pd.Series(wide[item_id].to_numpy(dtype=float), index=wide.index) for item_id in wide.columns
    }
    labels = {item_id: (phase1_classes or {}).get(item_id, "Unlabelled") for item_id in series_by_item}
    long_table, _ = run_item_baselines(series_by_item, labels, factories, horizon, n_folds)
    long_table = pd.concat([long_table, zero_forecast_long_table(long_table)], ignore_index=True)
    scales = item_mase_scales(series_by_item, horizon, n_folds)
    fold_metrics = score_long_table(long_table, scales)
    class_table = fold_local_class_table(wide, horizon, n_folds, phase1_classes)
    return attach_fold_local_class(fold_metrics, class_table), class_table


# ----------------------------------------------------------------------------------------
# Plots (matplotlib imported lazily; every distinction also uses a marker or hatch)
# ----------------------------------------------------------------------------------------
def plot_demand_patterns(
    series_by_item: Mapping[str, pd.Series], pattern_table: pd.DataFrame, days: int = 120
):
    """Three views of the 20 items: a daily-sales example per class, zero shares, size vs probability."""
    import matplotlib.pyplot as plt

    markers = {"Smooth": "o", "Intermittent": "s", "Erratic": "^", "Lumpy": "D"}
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    present = [c for c in CLASS_ORDER if c in set(pattern_table["demand_class"])]
    for k, label in enumerate(present):
        ax = fig.add_subplot(grid[0, k])
        item_id = pattern_table.loc[pattern_table["demand_class"] == label, "item_id"].iloc[0]
        tail = series_by_item[item_id].iloc[-days:]
        ax.vlines(tail.index, 0, tail.values, color="#333333", linewidth=1.2)
        ax.plot(tail.index, tail.values, linestyle="none", marker=markers[label], markersize=3.5, color="#333333")
        ax.set_title(f"{label}: {item_id}\nlast {len(tail)} development days", fontsize=9)
        ax.set_ylabel("units sold per day")
        ax.tick_params(axis="x", rotation=40, labelsize=7)
    ax = fig.add_subplot(grid[1, 0:2])
    ordered = pattern_table.reset_index(drop=True)
    for label in present:
        part = ordered[ordered["demand_class"] == label]
        ax.plot(part.index, part["zero_share"] * 100, linestyle="none", marker=markers[label], markersize=8,
                markerfacecolor="white" if label in ("Intermittent", "Lumpy") else "#333333",
                markeredgecolor="#333333", label=label)
    ax.set_xticks(ordered.index)
    ax.set_xticklabels(ordered["item_id"].str.replace("FOODS_3_", "", regex=False), rotation=90, fontsize=7)
    ax.set_xlabel("item (FOODS_3_ number)")
    ax.set_ylabel("days with zero sales (%)")
    ax.set_title("Share of zero-sales days, per item", fontsize=10)
    ax.legend(title="Phase 1 class", fontsize=8)
    ax.grid(alpha=0.3)
    ax = fig.add_subplot(grid[1, 2:4])
    for label in present:
        part = ordered[ordered["demand_class"] == label]
        ax.plot(part["occurrence_probability"], part["mean_size"], linestyle="none", marker=markers[label],
                markersize=8, markerfacecolor="white" if label in ("Intermittent", "Lumpy") else "#333333",
                markeredgecolor="#333333", label=label)
    ax.set_xlabel("probability of a demand day")
    ax.set_ylabel("mean units on a demand day")
    ax.set_title("Occurrence probability vs. demand size", fontsize=10)
    ax.legend(title="Phase 1 class", fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle("Demand patterns of the 20 selected items (development window only)", fontsize=12)
    return fig


def plot_zero_runs(run_table: pd.DataFrame, max_length: int = 60):
    """Histogram of zero-run lengths per class (log counts)."""
    import matplotlib.pyplot as plt

    present = [c for c in CLASS_ORDER if c in set(run_table["demand_class"])]
    fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 3.8), sharey=True, squeeze=False,
                             constrained_layout=True)
    bins = np.arange(0.5, max_length + 1.5, 1.0)
    hatches = {"Smooth": "//", "Intermittent": "\\\\", "Erratic": "xx", "Lumpy": ".."}
    for ax, label in zip(axes[0], present):
        lengths = run_table.loc[run_table["demand_class"] == label, "run_length"].clip(upper=max_length)
        ax.hist(lengths, bins=bins, color="#DDDDDD", edgecolor="#333333", hatch=hatches[label])
        ax.set_yscale("log")
        ax.set_title(f"{label} (n={len(lengths)} runs)", fontsize=10)
        ax.set_xlabel(f"consecutive zero-sales days (>{max_length} shown as {max_length})")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("number of zero runs (log scale)")
    fig.suptitle("How long do zero-sales runs last?", fontsize=12)
    return fig


def plot_class_metric(summary: pd.DataFrame, metric: str, class_column: str = "demand_class", title: str = ""):
    """Grouped bars (colour + hatch) of a macro-mean metric per class and model, with item range marks."""
    import matplotlib.pyplot as plt

    column = f"{metric}_macro_mean"
    if column not in summary.columns:
        raise ValueError(f"summary has no column {column!r}.")
    classes = [c for c in CLASS_ORDER if c in set(summary[class_column])]
    models = list(dict.fromkeys(summary["model"]))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    for m, model in enumerate(models):
        style = model_style(model)
        values = [
            summary.loc[(summary[class_column] == c) & (summary["model"] == model), column].squeeze()
            if ((summary[class_column] == c) & (summary["model"] == model)).any() else np.nan
            for c in classes
        ]
        positions = np.arange(len(classes)) + (m - (len(models) - 1) / 2) * width
        ax.bar(positions, values, width=width * 0.92, color=style["color"], edgecolor="#222222",
               hatch=style["hatch"], label=model)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel(f"{metric} (macro mean over items)")
    ax.set_title(title or f"{metric} by demand class")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return fig
