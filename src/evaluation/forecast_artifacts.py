"""Reading, auditing and scoring SAVED forecast tables (Phase 3).

Notebook 09 (model comparison), the inventory layer and the Streamlit dashboard all
work from forecast tables that earlier notebooks saved under `artifacts/forecasts/`.
This module is the single place those files are read and checked, so every consumer
applies the same rules:

  * A missing file raises `MissingArtifactError` naming the file AND the exact
    command that creates it -- never an empty chart and never made-up numbers.
  * A file that exists but is unusable (wrong columns, wrong dates, actuals that do
    not match the development series, negative unit forecasts, any date inside the
    reserved final holdout) raises `ArtifactError` and says which check failed.
  * Metrics are always RECOMPUTED from the saved (actual, predicted) pairs with the
    project's own `src.evaluation.metrics` functions. A score typed into a notebook
    is never trusted.

Two table kinds exist and are never mixed:

  "aggregate"   columns model, fold, date, actual, predicted
                -- the single CA_1/FOODS_3 daily-total series.
  "item"        columns item_id, demand_class, model, fold, date, actual, predicted
                (+ success, error) -- one row per item, model, fold and day.

Aggregate and item-level errors sit on different scales (one item is a tiny
fraction of the department total), so nothing here ranks one against the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.evaluation.backtesting import generate_rolling_origins
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT, resolve_path

AGGREGATE_REQUIRED: Tuple[str, ...] = ("model", "fold", "date", "actual", "predicted")
ITEM_REQUIRED: Tuple[str, ...] = (
    "item_id", "demand_class", "model", "fold", "date", "actual", "predicted",
)
_KEY_COLUMNS = {"aggregate": ("model",), "item": ("item_id", "model")}
_NEGATIVE_TOLERANCE = 1e-9


class ArtifactError(ValueError):
    """A saved artifact exists but cannot be used as a valid forecast table."""


class MissingArtifactError(FileNotFoundError):
    """A required artifact file does not exist; the message says how to create it."""


# --------------------------------------------------------------------------- how artifacts are produced

#: Filename prefix -> the notebook that writes it (used only to build helpful messages).
_PRODUCER_RULES: Tuple[Tuple[str, str], ...] = (
    ("lstm_gru_", "07_lstm_gru.ipynb"),
    ("prophet_", "08_prophet.ipynb"),
    ("xgboost_", "06_xgboost.ipynb"),
    ("intermittent_", "05_intermittent_models.ipynb"),
    ("comparison_", "09_model_comparison.ipynb"),
    ("inventory_", "10_inventory_decision_support.ipynb"),
    ("item_level_", "04_classical_models.ipynb"),
    ("seasonal_naive_", "04_classical_models.ipynb"),
    ("arima_", "04_classical_models.ipynb"),
    ("sarima_", "04_classical_models.ipynb"),
    ("holt_winters_", "04_classical_models.ipynb"),
    ("model_comparison_", "04_classical_models.ipynb"),
    ("residual_diagnostics_", "04_classical_models.ipynb"),
)


def notebook_command(notebook_filename: str) -> str:
    """PowerShell/bash command that executes a project notebook in place."""
    return f"python -m jupyter nbconvert --to notebook --execute --inplace notebooks/{notebook_filename}"


def producer_notebook(filename: str) -> Optional[str]:
    """Name of the notebook that creates `filename`, or None if it is not a known artifact."""
    for prefix, notebook in _PRODUCER_RULES:
        if filename.startswith(prefix):
            return notebook
    return None


def producer_hint(filename: str) -> str:
    """One sentence saying how to create `filename`."""
    notebook = producer_notebook(filename)
    if notebook is None:
        return "Re-run the notebook that writes this file."
    return f"It is written by notebooks/{notebook}. Create it with: {notebook_command(notebook)}"


def _display_path(path: Path) -> str:
    """Repo-relative path for messages (never an absolute, machine-specific path)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def forecasts_dir(config: Optional[Mapping[str, Any]] = None) -> Path:
    """Absolute folder that holds the saved forecast tables."""
    config = load_config() if config is None else config
    return resolve_path(config["phase3"]["forecasts_dir"])


def metrics_dir(config: Optional[Mapping[str, Any]] = None) -> Path:
    """Absolute folder that holds the saved metric tables."""
    config = load_config() if config is None else config
    return resolve_path(config["phase3"]["metrics_dir"])


# --------------------------------------------------------------------------- reading

def read_forecast_table(path: Union[str, Path], kind: str = "aggregate") -> pd.DataFrame:
    """Read one saved forecast CSV into a typed, sorted DataFrame.

    Dates become `datetime64[ns]` (calendar days), `fold` an int, `actual` /
    `predicted` floats. A table without a `success` column is treated as all
    successful (the Phase 2 aggregate files have none); a row may only have a
    missing `predicted` value if `success` is False.
    """
    if kind not in _KEY_COLUMNS:
        raise ValueError(f"kind must be 'aggregate' or 'item', got {kind!r}.")
    path = Path(path)
    if not path.exists():
        raise MissingArtifactError(f"Missing artifact '{_display_path(path)}'. {producer_hint(path.name)}")

    table = pd.read_csv(path)
    required = AGGREGATE_REQUIRED if kind == "aggregate" else ITEM_REQUIRED
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ArtifactError(
            f"'{path.name}' is not a valid {kind} forecast table: missing column(s) {missing}; "
            f"it has {list(table.columns)}."
        )
    if table.empty:
        raise ArtifactError(f"'{path.name}' contains no rows.")

    table = table.copy()
    table["date"] = pd.to_datetime(table["date"]).dt.normalize().astype("datetime64[ns]")
    table["fold"] = table["fold"].astype(int)
    for column in ("actual", "predicted"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
    if "success" in table.columns:
        table["success"] = table["success"].astype(bool)
    else:
        table["success"] = True
    if "error" not in table.columns:
        table["error"] = None

    order = list(_KEY_COLUMNS[kind]) + ["fold", "date"]
    return table.sort_values(order).reset_index(drop=True)


def load_aggregate_forecast_tables(
    filenames: Sequence[str], config: Optional[Mapping[str, Any]] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Read several aggregate forecast files; return (tables_by_filename, problems_by_filename).

    A file that is missing or invalid is reported in `problems`, NOT filled with
    anything: the caller shows the message and carries on with the rest.
    """
    directory = forecasts_dir(config)
    tables: Dict[str, pd.DataFrame] = {}
    problems: Dict[str, str] = {}
    for filename in filenames:
        try:
            tables[filename] = read_forecast_table(directory / filename, kind="aggregate")
        except (MissingArtifactError, ArtifactError) as exc:
            problems[filename] = str(exc)
    return tables, problems


def load_item_forecast_tables(
    filenames: Sequence[str], config: Optional[Mapping[str, Any]] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Item-level counterpart of `load_aggregate_forecast_tables`."""
    directory = forecasts_dir(config)
    tables: Dict[str, pd.DataFrame] = {}
    problems: Dict[str, str] = {}
    for filename in filenames:
        try:
            tables[filename] = read_forecast_table(directory / filename, kind="item")
        except (MissingArtifactError, ArtifactError) as exc:
            problems[filename] = str(exc)
    return tables, problems


# --------------------------------------------------------------------------- writing

FORECAST_TABLE_COLUMNS: Tuple[str, ...] = ("model", "fold", "date", "actual", "predicted", "success", "error")


def backtest_results_to_forecast_table(results: Sequence[Any]) -> pd.DataFrame:
    """Turn `run_backtest` results (BacktestFoldResult objects) into the saved aggregate table.

    One row per (model, fold, forecast day) with the columns of FORECAST_TABLE_COLUMNS.
    A failed fold keeps its rows with `predicted` = NaN, `success` = False and the
    recorded error text: it stays visible in every later table instead of silently
    disappearing, and it is never filled with a substitute forecast.
    """
    rows: List[Tuple[Any, ...]] = []
    for result in results:
        usable = bool(result.success) and result.predictions is not None
        predictions = np.asarray(result.predictions, dtype=float) if usable else np.full(len(result.dates), np.nan)
        if len(predictions) != len(result.dates):
            raise ArtifactError(
                f"{result.model_name} fold {result.fold}: {len(predictions)} predictions for {len(result.dates)} dates."
            )
        for day, actual, predicted in zip(result.dates, result.actuals, predictions):
            rows.append(
                (result.model_name, int(result.fold), pd.Timestamp(day), float(actual), float(predicted),
                 bool(result.success), result.error)
            )
    return pd.DataFrame(rows, columns=list(FORECAST_TABLE_COLUMNS))


def write_forecast_table(table: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Write a forecast table as CSV: ISO calendar dates, no index, parent folders created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, lineterminator="\n")
    return path


# --------------------------------------------------------------------------- the expected evaluation grid

def expected_fold_windows(dev_dates: pd.DatetimeIndex, horizon: int, n_folds: int) -> pd.DataFrame:
    """The fold layout every model must have used, from the development calendar.

    Uses the project's own `generate_rolling_origins`, so this is the grid the
    Phase 2 backtests ran on. Columns: fold, n_train, origin_date (last training
    day), forecast_start, forecast_end.
    """
    dev_dates = pd.DatetimeIndex(dev_dates)
    rows = []
    for origin in generate_rolling_origins(len(dev_dates), horizon=horizon, n_folds=n_folds):
        rows.append(
            {
                "fold": origin.fold,
                "n_train": origin.train_end,
                "origin_date": dev_dates[origin.train_end - 1],
                "forecast_start": dev_dates[origin.test_start],
                "forecast_end": dev_dates[origin.test_end - 1],
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one audit check on one saved table."""

    name: str
    passed: bool
    detail: str = ""


def audit_forecast_table(
    table: pd.DataFrame,
    *,
    kind: str,
    dev_series: Optional[pd.Series] = None,
    item_series: Optional[Mapping[str, pd.Series]] = None,
    dev_dates: Optional[pd.DatetimeIndex] = None,
    horizon: int = 28,
    n_folds: int = 5,
    require_non_negative: bool = True,
    atol: float = 1e-9,
) -> List[CheckResult]:
    """Run every integrity check on a forecast table and return ALL results (never raises).

    Checks: no date after the development end (the reserved holdout); every
    (item,) model has every fold; each fold has exactly `horizon` consecutive days
    equal to the expected forecast window; no duplicate rows; actuals equal the
    development series (aggregate: `dev_series`; item-level: `item_series`, when
    given); predictions finite on successful rows; predictions non-negative.
    """
    if kind not in _KEY_COLUMNS:
        raise ValueError(f"kind must be 'aggregate' or 'item', got {kind!r}.")
    if dev_dates is None:
        if dev_series is None:
            raise ValueError("Pass dev_series or dev_dates so the expected folds can be derived.")
        dev_dates = dev_series.index
    dev_dates = pd.DatetimeIndex(dev_dates)
    keys = list(_KEY_COLUMNS[kind])
    windows = expected_fold_windows(dev_dates, horizon, n_folds).set_index("fold")
    results: List[CheckResult] = []

    # 1. the reserved holdout must be absent
    late = table.loc[table["date"] > dev_dates[-1], "date"]
    results.append(
        CheckResult(
            "no_dates_after_development_end",
            late.empty,
            "ok" if late.empty else f"{len(late)} row(s) dated after {dev_dates[-1].date()} (the reserved holdout)",
        )
    )

    # 2. duplicates
    duplicated = int(table.duplicated(keys + ["fold", "date"]).sum())
    results.append(CheckResult("no_duplicate_rows", duplicated == 0, f"{duplicated} duplicated row(s)"))

    # 3. fold coverage and window dates, per (item,) model
    bad_groups: List[str] = []
    for group_key, group in table.groupby(keys, sort=False):
        label = group_key if isinstance(group_key, str) else "/".join(map(str, group_key))
        folds_present = sorted(group["fold"].unique())
        if folds_present != list(range(1, n_folds + 1)):
            bad_groups.append(f"{label}: folds {folds_present}, expected 1..{n_folds}")
            continue
        for fold, fold_rows in group.groupby("fold"):
            expected_days = pd.date_range(windows.loc[fold, "forecast_start"], periods=horizon, freq="D")
            actual_days = pd.DatetimeIndex(fold_rows["date"].sort_values())
            if not actual_days.equals(expected_days):
                bad_groups.append(
                    f"{label}: fold {fold} covers {actual_days.min().date()}..{actual_days.max().date()} "
                    f"({len(actual_days)} days), expected {expected_days[0].date()}..{expected_days[-1].date()}"
                )
    results.append(
        CheckResult(
            "folds_and_windows_match_protocol",
            not bad_groups,
            "ok" if not bad_groups else "; ".join(bad_groups[:3]) + (" ..." if len(bad_groups) > 3 else ""),
        )
    )

    # 4. actuals equal the development series
    if kind == "aggregate" and dev_series is not None:
        reference = dev_series.reindex(pd.DatetimeIndex(table["date"])).to_numpy(dtype=float)
        diff = np.abs(reference - table["actual"].to_numpy(dtype=float))
        worst = float(np.nanmax(diff)) if len(diff) else 0.0
        unmatched = int(np.isnan(reference).sum())
        passed = unmatched == 0 and worst <= atol
        results.append(
            CheckResult(
                "actuals_equal_development_series",
                passed,
                "ok" if passed else f"max |actual - development series| = {worst:g}; {unmatched} date(s) not in the series",
            )
        )
    elif kind == "item" and item_series is not None:
        worst, unmatched = 0.0, 0
        for item_id, group in table.groupby("item_id", sort=False):
            if item_id not in item_series:
                unmatched += len(group)
                continue
            reference = item_series[item_id].reindex(pd.DatetimeIndex(group["date"])).to_numpy(dtype=float)
            unmatched += int(np.isnan(reference).sum())
            diff = np.abs(reference - group["actual"].to_numpy(dtype=float))
            if len(diff) and not np.all(np.isnan(diff)):
                worst = max(worst, float(np.nanmax(diff)))
        passed = unmatched == 0 and worst <= atol
        results.append(
            CheckResult(
                "actuals_equal_development_series",
                passed,
                "ok" if passed else f"max |actual - item development series| = {worst:g}; {unmatched} row(s) unmatched",
            )
        )
    else:
        results.append(CheckResult("actuals_equal_development_series", True, "skipped: no reference series supplied"))

    # 5. predictions finite where the fit succeeded, and non-negative
    ok_rows = table[table["success"]]
    non_finite = int((~np.isfinite(ok_rows["predicted"].to_numpy(dtype=float))).sum())
    results.append(
        CheckResult("predictions_finite_on_successful_rows", non_finite == 0, f"{non_finite} non-finite prediction(s)")
    )
    if require_non_negative:
        negative = int((ok_rows["predicted"] < -_NEGATIVE_TOLERANCE).sum())
        results.append(CheckResult("predictions_non_negative", negative == 0, f"{negative} negative prediction(s)"))
    return results


def validate_forecast_table(table: pd.DataFrame, **kwargs: Any) -> List[CheckResult]:
    """Like `audit_forecast_table`, but raises `ArtifactError` if any check failed."""
    results = audit_forecast_table(table, **kwargs)
    failed = [r for r in results if not r.passed]
    if failed:
        raise ArtifactError("Forecast table failed integrity check(s): " + "; ".join(f"{r.name} ({r.detail})" for r in failed))
    return results


def check_same_grid(
    reference: pd.DataFrame, other: pd.DataFrame, key_columns: Sequence[str], atol: float = 0.0
) -> Dict[str, Any]:
    """Raise unless two tables score exactly the same cells with the same actual values.

    `key_columns` identifies a cell, e.g. ["fold", "date"] for aggregate tables or
    ["item_id", "fold", "date"] for item-level ones. Prediction and model columns
    are ignored and either table may hold several models.
    """
    keys = list(key_columns)
    grids = {}
    for label, table in (("reference", reference), ("other", other)):
        missing = set(keys + ["actual"]) - set(table.columns)
        if missing:
            raise ArtifactError(f"{label} table is missing column(s): {sorted(missing)}")
        grid = table[keys + ["actual"]].copy()
        grid["date"] = pd.to_datetime(grid["date"]).dt.normalize()
        grid = grid.drop_duplicates()
        if grid.duplicated(keys).any():
            raise ArtifactError(f"{label} table has more than one actual value for the same {keys} cell.")
        grids[label] = grid
    merged = grids["reference"].merge(
        grids["other"], on=keys, how="outer", suffixes=("_reference", "_other"), indicator=True
    )
    only_reference = int((merged["_merge"] == "left_only").sum())
    only_other = int((merged["_merge"] == "right_only").sum())
    if only_reference or only_other:
        raise ArtifactError(
            f"Evaluation grids differ: {only_reference} cell(s) only in the reference table, {only_other} only in the other."
        )
    max_diff = float((merged["actual_reference"] - merged["actual_other"]).abs().max()) if len(merged) else 0.0
    if max_diff > atol:
        raise ArtifactError(f"Actual values differ between the tables (max absolute difference {max_diff:g}).")
    return {"n_cells": int(len(merged)), "max_abs_actual_diff": max_diff}


# --------------------------------------------------------------------------- scoring

def seasonal_naive_scale(train: Sequence[float], lag: int = 7) -> float:
    """MASE denominator: mean |y_t - y_{t-lag}| over a TRAINING window, or NaN if undefined.

    It is NaN when the window is not longer than `lag`, and when every seasonal
    difference is zero (for example a series that never sells): a scale of zero
    would make MASE infinite, so it is reported as undefined instead.
    """
    values = np.asarray(train, dtype=float)
    if lag < 1:
        raise ValueError("lag must be at least 1.")
    if len(values) <= lag or not np.all(np.isfinite(values)):
        return float("nan")
    scale = float(np.mean(np.abs(values[lag:] - values[:-lag])))
    return scale if scale > 0 else float("nan")


def fold_training_scales(series: pd.Series, windows: pd.DataFrame, lag: int = 7) -> Dict[int, float]:
    """MASE scale of each fold's training window (`series.iloc[:n_train]`)."""
    return {
        int(row.fold): seasonal_naive_scale(series.iloc[: int(row.n_train)].to_numpy(dtype=float), lag)
        for row in windows.itertuples(index=False)
    }


def metrics_by_group(
    table: pd.DataFrame,
    group_columns: Sequence[str],
    mase_scales: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """MAE, RMSE, zero-protected MAPE, WAPE, bias and MASE for each group of rows.

    `group_columns` normally ends in "fold" (e.g. ["model", "fold"]). `mase_scales`
    is an optional DataFrame holding a `mase_scale` column plus whichever of the
    group columns identify the scale (e.g. ["fold"], or ["item_id", "fold"]).

    A group with any failed or missing prediction gets NaN metrics and
    `success=False` -- it is never scored on a partial window.

    Definitions (u = actual, f = forecast, over the group's rows):
      MAE   mean |u - f|                         RMSE  sqrt(mean (u - f)^2)
      MAPE  mean |u - f| / u over u > 0, in %    (zero actuals excluded and counted)
      WAPE  sum |u - f| / sum u                  NaN when sum u = 0
      bias  mean (f - u)                         positive = over-forecast
      MASE  MAE / scale                          NaN when the scale is undefined
    """
    group_columns = list(group_columns)
    missing = set(group_columns + ["actual", "predicted"]) - set(table.columns)
    if missing:
        raise ArtifactError(f"table is missing column(s): {sorted(missing)}")
    scale_keys: List[str] = []
    scale_lookup: Dict[Tuple, float] = {}
    if mase_scales is not None:
        if "mase_scale" not in mase_scales.columns:
            raise ValueError("mase_scales needs a 'mase_scale' column.")
        scale_keys = [c for c in mase_scales.columns if c != "mase_scale"]
        absent = [c for c in scale_keys if c not in group_columns]
        if absent:
            raise ValueError(f"mase_scales key column(s) {absent} are not among the group columns.")
        for row in mase_scales.itertuples(index=False):
            record = row._asdict()
            scale_lookup[tuple(record[c] for c in scale_keys)] = float(record["mase_scale"])

    rows = []
    for key, group in table.groupby(group_columns, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key))
        actual = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        success = bool(group["success"].all()) if "success" in group.columns else True
        success = success and bool(np.all(np.isfinite(predicted)))
        record.update(MAE=np.nan, RMSE=np.nan, MAPE=np.nan, mape_n_excluded=np.nan, WAPE=np.nan, bias=np.nan, MASE=np.nan)
        record["n"] = int(len(group))
        record["success"] = success
        if success:
            m = compute_metrics(actual, predicted)
            record["MAE"], record["RMSE"] = m.mae, m.rmse
            record["MAPE"] = np.nan if m.mape is None else m.mape
            record["mape_n_excluded"] = m.mape_n_excluded
            denominator = float(np.sum(actual))
            record["WAPE"] = float(np.sum(np.abs(actual - predicted)) / denominator) if denominator > 0 else np.nan
            record["bias"] = float(np.mean(predicted - actual))
            if scale_keys:
                lookup_key = tuple(record[c] for c in scale_keys)
                scale = scale_lookup.get(lookup_key, np.nan)
                record["MASE"] = m.mae / scale if np.isfinite(scale) and scale > 0 else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def add_horizon_step(table: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Add `horizon_step` (1 = first forecast day) within each (group, fold) window."""
    keys = list(group_columns) + ["fold"]
    ordered = table.sort_values(keys + ["date"]).copy()
    ordered["horizon_step"] = ordered.groupby(keys, sort=False).cumcount() + 1
    return ordered


def truncate_horizon(table: pd.DataFrame, horizon: int, group_columns: Sequence[str]) -> pd.DataFrame:
    """Keep only the first `horizon` forecast days of every (group, fold) window.

    The saved forecasts are 28-day forecasts; the first h days of a 28-day forecast
    ARE the h-day forecast issued at the same origin (no model here re-plans), so
    this is a lawful way to score or display shorter horizons.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    stepped = add_horizon_step(table, group_columns)
    return stepped[stepped["horizon_step"] <= horizon].reset_index(drop=True)


# --------------------------------------------------------------------------- reference series loaders

def load_selected_item_series(config: Optional[Mapping[str, Any]] = None) -> Dict[str, pd.Series]:
    """Development series (holdout removed) of the 20 selected items, keyed by item_id."""
    from src.evaluation.item_level import item_development_series

    config = load_config() if config is None else config
    path = resolve_path(
        Path(config["data"]["processed_dir"]) / config["output"]["selected_items_daily_sales_filename"]
    )
    if not path.exists():
        raise MissingArtifactError(
            f"Missing '{_display_path(path)}'. Create it with: python -m src.analysis.demand_classification"
        )
    daily = pd.read_parquet(path)
    holdout_days = int(config["backtest"]["holdout_days"])
    splitter_log = logging.getLogger("src.data.splitter")
    previous_level = splitter_log.level
    splitter_log.setLevel(logging.WARNING)      # one "Final holdout split" line per item would be noise
    try:
        return {
            item_id: item_development_series(daily, item_id, holdout_days=holdout_days)
            for item_id in sorted(daily["item_id"].unique())
        }
    finally:
        splitter_log.setLevel(previous_level)
