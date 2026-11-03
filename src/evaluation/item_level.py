"""Item-level backtesting orchestration (Phase 2 Step 7).

Steps 2-6 modeled a single series: the CA_1/FOODS_3 daily aggregate.
Step 7 moves to 20 individual item-level series (5 per demand class:
Smooth, Intermittent, Erratic, Lumpy) and compares per-item Seasonal
Naive, Croston, and TSB on each one.

Nothing here changes `src.evaluation.backtesting`
(`run_backtest`/`summarize_backtest`) or `src.evaluation.metrics` --
both are already-approved, already-tested Phase 2 Step 2/5 deliverables
and are reused completely unmodified, once per (item, model) pair. This
module only adds the per-item loop around them, a long-format
fold-level results table (actual/predicted per date, not just the
per-fold summary `summarize_backtest` already gives), and the
unweighted-across-items ("macro") category summary the Step 7 brief
asks for.

Item-level error magnitudes (MAE/RMSE) are NOT comparable to the
aggregate-series error magnitudes reported in Steps 2-6: one item's
daily sales are a small fraction of the full CA_1/FOODS_3 aggregate, so
item-level errors are structurally far smaller just from scale, not
from being "more accurate" models. Nothing in this module mixes the two.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from src.data.splitter import split_final_holdout
from src.evaluation.backtesting import (
    BacktestFoldResult,
    generate_rolling_origins,
    run_backtest,
    summarize_backtest,
)
from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)


def item_development_series(
    daily_sales: pd.DataFrame, item_id: str, holdout_days: int = 28
) -> pd.Series:
    """Date-indexed, holdout-excluded sales Series for one item.

    `daily_sales` is the `selected_items_daily_sales` table (columns
    `item_id, date, sales, demand_class`). This calls the existing,
    unmodified `src.data.splitter.split_final_holdout` on that one
    item's rows and returns only its `development` half -- the exact
    same holdout mechanism every prior Phase 2 step used on the
    aggregate series, applied per item here rather than a separately
    reasoned-about date literal. The final `holdout_days` rows
    (2016-04-25 to 2016-05-22, for this project's 28-day holdout) are
    never returned by this function at all.
    """
    item_df = (
        daily_sales.loc[daily_sales["item_id"] == item_id, ["date", "sales"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if item_df.empty:
        raise ValueError(f"item_id {item_id!r} not found in daily_sales.")

    split = split_final_holdout(item_df, holdout_days=holdout_days, date_col="date")
    development = split.development
    return pd.Series(
        development["sales"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(development["date"]),
        name=item_id,
    )


def fold_results_to_long_table(
    results: List[BacktestFoldResult], item_id: str, demand_class: str
) -> pd.DataFrame:
    """Expand one (item, model) backtest's fold results into one row per (fold, date).

    This is the granular "fold-level actuals, predictions, dates, item
    IDs, demand classes, model names" table the Step 7 brief asks to
    save -- distinct from `summarize_backtest()`'s one-row-per-fold
    metric summary (built separately, from the same `results`, by
    `run_item_level_backtests` below).

    A failed fold (`predictions is None`) still contributes its
    actual-only rows, with `predicted=NaN`, so a failed fold is visible
    in the saved table rather than silently missing from it.
    """
    rows = []
    for r in results:
        preds = r.predictions if r.predictions is not None else [np.nan] * len(r.dates)
        for date, actual, pred in zip(r.dates, r.actuals, preds):
            rows.append(
                {
                    "item_id": item_id,
                    "demand_class": demand_class,
                    "model": r.model_name,
                    "fold": r.fold,
                    "date": pd.Timestamp(date).date(),
                    "actual": float(actual),
                    "predicted": float(pred) if pred is not None and not pd.isna(pred) else np.nan,
                    "success": r.success,
                    "error": r.error,
                }
            )
    return pd.DataFrame(rows)


def run_item_level_backtests(
    daily_sales: pd.DataFrame,
    item_ids: List[str],
    model_factories: Dict[str, Callable[[], BaseForecaster]],
    horizon: int = 28,
    n_folds: int = 5,
    holdout_days: int = 28,
) -> Dict[str, pd.DataFrame]:
    """Run every model in `model_factories` through every item in `item_ids`.

    For each item, `item_development_series` carves off that item's
    (holdout-excluded) development series once; every model in
    `model_factories` is then run through the existing
    `src.evaluation.backtesting.run_backtest` on that same series,
    unmodified -- identical fold boundaries for every model on a given
    item, exactly as Steps 2-6 required for the single aggregate series.

    Returns a dict with two long tables:
      - "long": one row per (item, model, fold, date) -- actual/predicted.
      - "fold_summary": one row per (item, model, fold) -- MAE/RMSE/MAPE,
        from `summarize_backtest` with `item_id`/`demand_class` attached.
    """
    class_lookup = (
        daily_sales[["item_id", "demand_class"]]
        .drop_duplicates()
        .set_index("item_id")["demand_class"]
    )

    long_tables = []
    fold_summaries = []
    for item_id in item_ids:
        demand_class = class_lookup.loc[item_id]
        series = item_development_series(daily_sales, item_id, holdout_days=holdout_days)
        logger.info(
            "Item %s (%s): development series %d obs, %s to %s",
            item_id, demand_class, len(series), series.index.min().date(), series.index.max().date(),
        )
        for model_name, factory in model_factories.items():
            results = run_backtest(series, factory, horizon=horizon, n_folds=n_folds)
            long_tables.append(fold_results_to_long_table(results, item_id, demand_class))

            fold_summary = summarize_backtest(results)
            fold_summary.insert(0, "item_id", item_id)
            fold_summary.insert(1, "demand_class", demand_class)
            fold_summaries.append(fold_summary)

    return {
        "long": pd.concat(long_tables, ignore_index=True),
        "fold_summary": pd.concat(fold_summaries, ignore_index=True),
    }


def per_item_summary(fold_summary: pd.DataFrame) -> pd.DataFrame:
    """Mean MAE/RMSE/MAPE per (item_id, demand_class, model), averaged across that item's folds.

    Averaged only over that item-model's successful folds (`success`);
    a model that failed every fold for an item gets NaN means here
    rather than a misleadingly-partial average. MAPE is averaged only
    over folds where it was additionally defined (not all-zero actuals
    in that fold's test window) -- `MAPE_n_folds_valid` reports how many
    that was, same convention as `src.evaluation.comparison.summarize_model_means`.
    """
    required = {"item_id", "demand_class", "model", "MAE", "RMSE", "MAPE", "success"}
    missing = required - set(fold_summary.columns)
    if missing:
        raise ValueError(f"fold_summary is missing column(s): {sorted(missing)}")

    rows = []
    for (item_id, demand_class, model), grp in fold_summary.groupby(
        ["item_id", "demand_class", "model"], sort=False, observed=True
    ):
        ok = grp[grp["success"].astype(bool)]
        mape_valid = ok["MAPE"].dropna()
        rows.append(
            {
                "item_id": item_id,
                "demand_class": demand_class,
                "model": model,
                "n_folds": len(grp),
                "n_success": len(ok),
                "MAE_mean": ok["MAE"].mean() if len(ok) else np.nan,
                "RMSE_mean": ok["RMSE"].mean() if len(ok) else np.nan,
                "MAPE_mean": mape_valid.mean() if len(mape_valid) else np.nan,
                "MAPE_n_folds_valid": int(len(mape_valid)),
            }
        )
    return pd.DataFrame(rows)


def category_macro_average_summary(item_summary: pd.DataFrame) -> pd.DataFrame:
    """Macro-averaged (unweighted mean ACROSS ITEMS) MAE/RMSE/MAPE per (demand_class, model).

    Each input row is already one item's own mean-over-its-folds error
    (`per_item_summary`'s output); this averages those item-level means
    again, with every item weighted equally regardless of its sales
    volume -- exactly the Step 7 brief's "use macro-averages ... so
    products with larger sales volumes do not automatically dominate".
    A pooled/volume-weighted average is deliberately not computed here.
    """
    required = {"item_id", "demand_class", "model", "MAE_mean", "RMSE_mean", "MAPE_mean"}
    missing = required - set(item_summary.columns)
    if missing:
        raise ValueError(f"item_summary is missing column(s): {sorted(missing)}")

    rows = []
    for (demand_class, model), grp in item_summary.groupby(
        ["demand_class", "model"], sort=False, observed=True
    ):
        mape_valid = grp["MAPE_mean"].dropna()
        mae_valid = grp["MAE_mean"].dropna()
        rmse_valid = grp["RMSE_mean"].dropna()
        rows.append(
            {
                "demand_class": demand_class,
                "model": model,
                "n_items": grp["item_id"].nunique(),
                "MAE_macro_mean": mae_valid.mean() if len(mae_valid) else np.nan,
                "RMSE_macro_mean": rmse_valid.mean() if len(rmse_valid) else np.nan,
                "MAPE_macro_mean": mape_valid.mean() if len(mape_valid) else np.nan,
                "MAPE_n_items_valid": int(len(mape_valid)),
            }
        )
    return pd.DataFrame(rows)


CATEGORY_ORDER = ["Smooth", "Intermittent", "Erratic", "Lumpy"]
_CATEGORY_COLORS = {
    "Smooth": "#4C4C4C",
    "Intermittent": "#8C8C8C",
    "Erratic": "#B0862B",
    "Lumpy": "#8B3A3A",
}


def plot_category_comparison(category_summary: pd.DataFrame, metric: str = "MAE_macro_mean", ax=None):
    """Grouped bar chart: one group per demand category, one bar per model.

    Bar height is the macro-averaged `metric` (e.g. "MAE_macro_mean",
    "RMSE_macro_mean"). Monochrome-leaning palette, consistent with this
    project's established plot style (see `src.evaluation.comparison`
    and `src.analysis.eda.CATEGORY_COLORS`) -- color marks the model,
    position marks the category.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    categories = [c for c in CATEGORY_ORDER if c in category_summary["demand_class"].unique()]
    models = list(dict.fromkeys(category_summary["model"]))  # first-seen order, de-duplicated
    n_models = max(len(models), 1)
    x = np.arange(len(categories))
    width = 0.8 / n_models
    palette = ["#4C4C4C", "#B0862B", "#8B3A3A", "#8C8C8C", "#3B6E8F"]

    for i, model in enumerate(models):
        heights = []
        for c in categories:
            match = category_summary.loc[
                (category_summary["demand_class"] == c) & (category_summary["model"] == model), metric
            ]
            heights.append(match.iloc[0] if len(match) else np.nan)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, heights, width, label=model, color=palette[i % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Item-level {metric.replace('_', ' ')} by demand category (macro-average across items)")
    ax.legend(title="Model")
    ax.figure.tight_layout()
    return ax


_MODEL_PLOT_COLORS = {
    "SeasonalNaive": "#8C8C8C",
    "Croston": "#B0862B",
    "TSB": "#3B6E8F",
    "XGBoost": "#8B3A3A",  # Phase 3 Step 1; prefix-matched, so every XGBoost variant shares it
}


def plot_example_item_forecasts(long_table: pd.DataFrame, item_ids: List[str], fold: int, ax_cols: int = 2):
    """Small-multiples actual-vs-predicted plot for a handful of example items, one fold.

    One subplot per item in `item_ids`, each showing the shared black
    "Actual" line plus one dashed line per model's `fold`-th-fold
    forecast -- a visual complement (not a replacement) to the
    category-level macro-averaged MAE/RMSE bar charts above, built
    straight from the saved long-format results table (`long_table`:
    `item_id, demand_class, model, fold, date, actual, predicted`), the
    same monochrome-leaning, black-actual-line convention as
    `src.evaluation.backtesting.plot_fold_forecast`.
    """
    import matplotlib.pyplot as plt

    items = [i for i in item_ids if i in long_table["item_id"].unique()]
    if not items:
        raise ValueError("None of the given item_ids were found in long_table.")

    ncols = min(ax_cols, len(items))
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, item_id in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        item_fold = long_table[(long_table["item_id"] == item_id) & (long_table["fold"] == fold)]
        if item_fold.empty:
            ax.set_title(f"{item_id} -- no data for fold {fold}")
            ax.axis("off")
            continue

        demand_class = item_fold["demand_class"].iloc[0]
        actual_plotted = False
        for model_name, grp in item_fold.groupby("model", sort=False):
            grp = grp.sort_values("date")
            if not actual_plotted:
                ax.plot(grp["date"], grp["actual"], label="Actual", color="black", linewidth=1.5)
                actual_plotted = True
            color = next(
                (c for key, c in _MODEL_PLOT_COLORS.items() if model_name.startswith(key)), "#4C4C4C"
            )
            ax.plot(grp["date"], grp["predicted"], label=model_name, linestyle="--", linewidth=1.2, color=color)

        ax.set_title(f"{item_id} ({demand_class}) -- fold {fold}", fontsize=10)
        ax.set_ylabel("Sales (units)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7)

    for idx in range(len(items), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"Example item-level forecasts vs. actual -- fold {fold}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Phase 3 Step 1 additions -- pooled (global) models.
#
# Everything above is the completed Phase 2 Step 7 code and is unchanged.
# The functions below only ADD to it: a walk-forward runner for a model that
# is fit ONCE per fold on all items together, and a guard that two result
# tables describe exactly the same evaluation cells. The per-fold metric
# tables are still built by the existing `summarize_backtest` and
# `fold_results_to_long_table`, and `per_item_summary` /
# `category_macro_average_summary` above are reused unmodified on the result.
# ---------------------------------------------------------------------------


def run_pooled_item_level_backtest(
    daily_sales: pd.DataFrame,
    item_ids: List[str],
    pooled_factory: Callable[[], Any],
    horizon: int = 28,
    n_folds: int = 5,
    holdout_days: int = 28,
) -> Dict[str, pd.DataFrame]:
    """Walk-forward evaluation of ONE pooled model that is fit on all `item_ids` together, per fold.

    Fold boundaries come from the existing `generate_rolling_origins` on the
    development series (the same function `run_backtest` uses), so every
    item has exactly the folds, training windows and forecast windows it has
    in `run_item_level_backtests`. Each item's series is carved off by the
    existing `item_development_series`, which never returns the final
    holdout days; nothing here reads a date after the development end.

    `pooled_factory()` must return an object with
      * `fit({item_id: date-indexed Series})` -- given training prefixes only,
      * `predict(horizon) -> {item_id: array of length horizon}` -- called
        with no data argument, so no forecast-window actuals can reach it,
      * `forecast_dates(horizon)` -- checked here against the fold's test dates,
      * `fit_diagnostics() -> dict`,
      * optionally `item_class_table()` and `feature_importance()`.
    A new instance is built for every fold.

    A fit or predict exception is recorded on that fold (every item gets
    `success=False` and the error text, exactly as `run_backtest` records a
    failed fold) instead of aborting the run. A misaligned forecast calendar
    or a wrong-shaped prediction is a protocol error, not a model failure,
    and raises.

    Returns a dict of DataFrames:
      "long"              one row per (item, fold, date); same schema as
                          `run_item_level_backtests`'s "long"
      "fold_summary"      one row per (item, fold); same schema as that
                          function's "fold_summary" (`summarize_backtest` plus
                          item_id / demand_class)
      "fold_diagnostics"  one row per fold: success/error plus whatever the
                          model's `fit_diagnostics()` reported
      "fold_item_classes" per fold and item, the demand class computed from
                          that fold's training window next to the Phase 1 label
                          (empty if the model has no `item_class_table`)
      "feature_importance" per fold and feature, the model's gain importance
                          (empty if the model has no `feature_importance`)
    `demand_class` in "long"/"fold_summary" is the Phase 1 label, used only to
    group results -- the same role it has in `run_item_level_backtests`.
    """
    if not item_ids:
        raise ValueError("item_ids must not be empty.")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("item_ids contains duplicates.")

    class_lookup = (
        daily_sales[["item_id", "demand_class"]].drop_duplicates().set_index("item_id")["demand_class"]
    )
    series_by_item = {
        item_id: item_development_series(daily_sales, item_id, holdout_days=holdout_days)
        for item_id in item_ids
    }
    reference_index = series_by_item[item_ids[0]].index
    for item_id, series in series_by_item.items():
        if not series.index.equals(reference_index):
            raise ValueError(
                f"Item {item_id!r} has a different development calendar from {item_ids[0]!r}; "
                "a pooled model needs every item on one shared set of dates."
            )
    origins = generate_rolling_origins(len(reference_index), horizon=horizon, n_folds=n_folds)
    logger.info(
        "Pooled backtest: %d items, development %s to %s (%d obs), %d folds, horizon %d",
        len(item_ids), reference_index[0].date(), reference_index[-1].date(),
        len(reference_index), n_folds, horizon,
    )

    results_by_item: Dict[str, List[BacktestFoldResult]] = {item_id: [] for item_id in item_ids}
    diagnostic_rows: List[Dict[str, Any]] = []
    class_tables: List[pd.DataFrame] = []
    importance_tables: List[pd.DataFrame] = []

    for origin in origins:
        training = {item_id: series.iloc[: origin.train_end] for item_id, series in series_by_item.items()}
        test_index = reference_index[origin.test_start : origin.test_end]

        forecaster = None
        predictions: Dict[str, Any] = {}
        diagnostics: Dict[str, Any] = {}
        success = True
        error = None
        try:
            forecaster = pooled_factory()
            forecaster.fit(training)
            predictions = forecaster.predict(horizon)
        except Exception as exc:  # noqa: BLE001 -- a model failure is a reportable fold, not a crash
            success = False
            error = f"{type(exc).__name__}: {exc}"
            logger.error("[pooled] fold %d FAILED during fit/predict: %s", origin.fold, error)
        else:
            try:
                diagnostics = forecaster.fit_diagnostics() or {}
            except Exception as diag_exc:  # noqa: BLE001
                logger.warning("[pooled] fold %d: fit_diagnostics() raised %s -- continuing", origin.fold, diag_exc)

        if success:
            # Protocol integrity: these are bugs if they ever fail, so they raise.
            if not forecaster.forecast_dates(horizon).equals(test_index):
                raise ValueError(
                    f"Fold {origin.fold}: the model's forecast dates differ from the fold's test dates."
                )
            if set(predictions) != set(item_ids):
                raise ValueError(f"Fold {origin.fold}: predictions cover a different set of items than were fit.")
            for item_id in item_ids:
                if np.asarray(predictions[item_id], dtype=float).shape != (horizon,):
                    raise ValueError(f"Fold {origin.fold}: forecast for {item_id!r} does not have length {horizon}.")

        model_name = getattr(forecaster, "name", None) or getattr(pooled_factory, "__name__", "UnknownModel")

        for item_id in item_ids:
            series = series_by_item[item_id]
            results_by_item[item_id].append(
                BacktestFoldResult(
                    model_name=model_name,
                    fold=origin.fold,
                    origin_date=training[item_id].index[-1],
                    forecast_start=test_index[0],
                    forecast_end=test_index[-1],
                    dates=test_index,
                    actuals=series.iloc[origin.test_start : origin.test_end].to_numpy(dtype=float),
                    predictions=np.asarray(predictions[item_id], dtype=float) if success else None,
                    success=success,
                    error=error,
                    diagnostics=dict(diagnostics),
                )
            )

        row: Dict[str, Any] = {
            "model": model_name,
            "fold": origin.fold,
            "origin_date": training[item_ids[0]].index[-1].date(),
            "forecast_start": test_index[0].date(),
            "forecast_end": test_index[-1].date(),
            "success": success,
            "error": error,
        }
        for key, value in diagnostics.items():
            row.setdefault(key, value)
        diagnostic_rows.append(row)

        if success and hasattr(forecaster, "item_class_table"):
            table = forecaster.item_class_table().rename(columns={"demand_class": "fold_local_class"})
            table.insert(0, "fold", origin.fold)
            table["phase1_class"] = table["item_id"].map(class_lookup)
            class_tables.append(table)
        if success and hasattr(forecaster, "feature_importance"):
            try:
                importance = forecaster.feature_importance("gain")
            except Exception as imp_exc:  # noqa: BLE001
                logger.warning("[pooled] fold %d: feature_importance() raised %s -- skipped", origin.fold, imp_exc)
            else:
                importance_tables.append(
                    pd.DataFrame({"fold": origin.fold, "feature": importance.index, "gain": importance.to_numpy()})
                )

    long_tables = []
    fold_summaries = []
    for item_id in item_ids:
        demand_class = class_lookup.loc[item_id]
        results = results_by_item[item_id]
        long_tables.append(fold_results_to_long_table(results, item_id, demand_class))
        fold_summary = summarize_backtest(results)
        fold_summary.insert(0, "item_id", item_id)
        fold_summary.insert(1, "demand_class", demand_class)
        fold_summaries.append(fold_summary)

    return {
        "long": pd.concat(long_tables, ignore_index=True),
        "fold_summary": pd.concat(fold_summaries, ignore_index=True),
        "fold_diagnostics": pd.DataFrame(diagnostic_rows),
        "fold_item_classes": pd.concat(class_tables, ignore_index=True) if class_tables else pd.DataFrame(),
        "feature_importance": pd.concat(importance_tables, ignore_index=True) if importance_tables else pd.DataFrame(),
    }


def check_same_evaluation_grid(
    reference_long: pd.DataFrame, other_long: pd.DataFrame, atol: float = 0.0
) -> Dict[str, Any]:
    """Raise unless two long result tables evaluate exactly the same (item, fold, date) cells.

    "Same" means the same set of cells and, in every cell, the same actual
    value (within `atol`). Prediction and model columns are ignored, and either
    table may hold several models. `date` may be a date, a Timestamp or a
    string (a CSV read back from disk), and is compared as a calendar day.

    Used to guarantee that a new model is scored on the same items, folds and
    days as the Phase 2 baselines before their errors are put side by side.
    Returns a small dict describing the grid that matched.
    """
    key = ["item_id", "fold", "date"]
    grids = {}
    for label, table in (("reference", reference_long), ("other", other_long)):
        missing = set(key + ["actual"]) - set(table.columns)
        if missing:
            raise ValueError(f"{label} table is missing column(s): {sorted(missing)}")
        grid = table[key + ["actual"]].copy()
        grid["date"] = pd.to_datetime(grid["date"]).dt.normalize()
        grid = grid.drop_duplicates()
        if grid.duplicated(key).any():
            raise ValueError(f"{label} table has more than one actual value for the same (item_id, fold, date) cell.")
        grids[label] = grid

    merged = grids["reference"].merge(
        grids["other"], on=key, how="outer", suffixes=("_reference", "_other"), indicator=True
    )
    n_only_reference = int((merged["_merge"] == "left_only").sum())
    n_only_other = int((merged["_merge"] == "right_only").sum())
    if n_only_reference or n_only_other:
        raise ValueError(
            f"Evaluation grids differ: {n_only_reference} cell(s) only in the reference table, "
            f"{n_only_other} only in the other table."
        )
    max_diff = float((merged["actual_reference"] - merged["actual_other"]).abs().max()) if len(merged) else 0.0
    if max_diff > atol:
        raise ValueError(f"Actual values differ between the tables (max absolute difference {max_diff:g}).")
    return {
        "n_cells": int(len(merged)),
        "n_items": int(merged["item_id"].nunique()),
        "n_folds": int(merged["fold"].nunique()),
        "max_abs_actual_diff": max_diff,
    }


def plot_category_comparison_by_model_color(
    category_summary: pd.DataFrame, metric: str = "MAE_macro_mean", ax=None
):
    """Grouped bars by demand category, one bar per model, coloured by MODEL NAME.

    Same layout and input as `plot_category_comparison` (which is left unchanged
    for the Phase 2 figures). The difference: a model's colour comes from
    `_MODEL_PLOT_COLORS` (prefix-matched, the same mapping
    `plot_example_item_forecasts` uses) instead of from its position in the
    legend, so a model keeps one colour across every Phase 3 figure however many
    models a chart shows. Models that share a prefix (the XGBoost variants)
    share a colour -- show one variant per family in a single chart.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    categories = [c for c in CATEGORY_ORDER if c in set(category_summary["demand_class"])]
    models = list(dict.fromkeys(category_summary["model"]))
    n_models = max(len(models), 1)
    x = np.arange(len(categories))
    width = 0.8 / n_models

    for i, model in enumerate(models):
        heights = []
        for c in categories:
            match = category_summary.loc[
                (category_summary["demand_class"] == c) & (category_summary["model"] == model), metric
            ]
            heights.append(match.iloc[0] if len(match) else np.nan)
        color = next((col for key, col in _MODEL_PLOT_COLORS.items() if str(model).startswith(key)), "#4C4C4C")
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, heights, width, label=model, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Item-level {metric.replace('_', ' ')} by demand category (macro-average across items)")
    ax.legend(title="Model")
    ax.figure.tight_layout()
    return ax
