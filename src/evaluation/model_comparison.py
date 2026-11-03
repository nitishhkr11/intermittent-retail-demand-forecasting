"""Unified comparison of every saved forecast artifact (Phase 3, notebook 09).

Everything here is recomputed from SAVED forecast tables, never retyped and never re-fitted:

* `aggregate_sources` / `item_sources` name the artifact each model family writes (from `config.yaml`);
* `load_sources` reads them, audits each one and returns an explicit, per-artifact status. A missing or invalid
  artifact is reported with the exact command that creates it; it is never replaced by a stand-in forecast;
* `combine_tables` checks that all models were scored on exactly the same (fold, date) cells with the same actual
  values and that no model name appears twice with different forecasts;
* `fold_metrics` / `model_summary` give MAE, RMSE, zero-protected MAPE (with the number of zero days excluded),
  WAPE, bias and MASE per fold, then the mean, spread and range across folds;
* `fold_wins` counts, fold by fold, how often a model has a lower error than a reference model.

Aggregate-series errors (thousands of units per day) and item-level errors (units of one product per day) are on
different scales. They live in separate tables, plots and sections here, and nothing in this module ranks one
against the other. No significance test is performed: with five consecutive folds none can be justified.

MASE scale = mean |y_t - y_(t-7)| over the TRAINING window of each fold (item level: of each item and fold); it is NaN
(MASE undefined) when that scale is zero. WAPE = sum |error| / sum actual (NaN if no demand). MAPE excludes days with
zero actual demand and reports how many were excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.forecast_artifacts import (
    ArtifactError,
    MissingArtifactError,
    audit_forecast_table,
    check_same_grid,
    expected_fold_windows,
    fold_training_scales,
    forecasts_dir,
    metrics_by_group,
    producer_hint,
    read_forecast_table,
)
from src.evaluation.intermittent_experiment import class_summary, item_mase_scales, item_summary
from src.utils.config import load_config

AGGREGATE_FAMILIES = (
    ("SeasonalNaive", "classical (Phase 2)"),
    ("ARIMA", "classical (Phase 2)"),
    ("SARIMA", "classical (Phase 2)"),
    ("HoltWinters", "classical (Phase 2)"),
    ("LSTM", "neural (Phase 3)"),
    ("GRU", "neural (Phase 3)"),
    ("Prophet", "Prophet (Phase 3)"),
)
ITEM_FAMILIES = (
    ("SeasonalNaive", "baseline (Phase 2)"),
    ("Croston", "baseline (Phase 2)"),
    ("TSB", "baseline (Phase 2)"),
    ("XGBoost", "XGBoost (Phase 3)"),
)
FAMILY_STYLE = {
    "classical (Phase 2)": {"color": "#6B6B6B", "hatch": "//"},
    "baseline (Phase 2)": {"color": "#6B6B6B", "hatch": "//"},
    "neural (Phase 3)": {"color": "#2F6F8F", "hatch": "xx"},
    "Prophet (Phase 3)": {"color": "#B0862B", "hatch": "\\\\"},
    "XGBoost (Phase 3)": {"color": "#8B3A3A", "hatch": ".."},
    "other": {"color": "#444444", "hatch": ""},
}
#: Phase 3 model names that are the pre-declared primary variants; every other variant is an ablation.
PRIMARY_MODELS = ("LSTM(calendar)", "GRU(calendar)", "Prophet(additive)", "XGBoost(pooled)")


def model_family(model: str, scope: str = "aggregate") -> str:
    """Family label of a model name (prefix match), for grouping and plot styling."""
    for prefix, family in (AGGREGATE_FAMILIES if scope == "aggregate" else ITEM_FAMILIES):
        if model.startswith(prefix):
            return family
    return "other"


# ----------------------------------------------------------------------------------------- the artifact registry
@dataclass(frozen=True)
class ArtifactSource:
    """One saved forecast table that the comparison can read."""

    label: str
    filename: str
    scope: str  # "aggregate" or "item"

    @property
    def hint(self) -> str:
        return producer_hint(self.filename)


def aggregate_sources(config: Optional[Mapping[str, Any]] = None) -> List[ArtifactSource]:
    """The aggregate-series forecast files: four Phase 2 models, LSTM/GRU, Prophet."""
    config = load_config() if config is None else config
    p2, p3 = config["phase2"], config["phase3"]
    return [
        ArtifactSource("Seasonal Naive", p2["seasonal_naive_forecasts_filename"], "aggregate"),
        ArtifactSource("ARIMA", p2["arima_forecasts_filename"], "aggregate"),
        ArtifactSource("SARIMA", p2["sarima_forecasts_filename"], "aggregate"),
        ArtifactSource("Holt-Winters", p2["holt_winters_forecasts_filename"], "aggregate"),
        ArtifactSource("LSTM / GRU", p3["neural_forecasts_filename"], "aggregate"),
        ArtifactSource("Prophet", p3["prophet_forecasts_filename"], "aggregate"),
    ]


def item_sources(config: Optional[Mapping[str, Any]] = None) -> List[ArtifactSource]:
    """The item-level forecast files: Phase 2 baselines, XGBoost, and notebook 05's re-run of the baselines."""
    config = load_config() if config is None else config
    p2, p3 = config["phase2"], config["phase3"]
    return [
        ArtifactSource("Phase 2 item-level baselines", p2["item_level_long_results_filename"], "item"),
        ArtifactSource("XGBoost", p3["xgboost_long_results_filename"], "item"),
        ArtifactSource("Notebook 05 baselines (re-run)", p3["intermittent_long_filename"], "item"),
    ]


AUDIT_COLUMNS = ["artifact", "scope", "status", "rows", "models", "failed_checks", "detail"]


def load_sources(
    sources: Sequence[ArtifactSource],
    *,
    dev_series: Optional[pd.Series] = None,
    item_series: Optional[Mapping[str, pd.Series]] = None,
    dev_dates: Optional[pd.DatetimeIndex] = None,
    config: Optional[Mapping[str, Any]] = None,
    horizon: int = 28,
    n_folds: int = 5,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Read and audit every source. Returns `(tables_by_filename, audit_table)`.

    Statuses: `ok`, `missing` (file absent; the detail holds the command that creates it), `invalid` (unreadable table)
    and `audit failed` (readable but a check failed; such a table is NOT returned, so it cannot be scored).
    """
    directory = forecasts_dir(config)
    tables: Dict[str, pd.DataFrame] = {}
    rows: List[Dict[str, Any]] = []
    for source in sources:
        base = {"artifact": source.filename, "scope": source.scope, "rows": 0, "models": "", "failed_checks": 0}
        try:
            table = read_forecast_table(directory / source.filename, kind=source.scope)
        except MissingArtifactError:
            rows.append({**base, "status": "missing", "detail": f"Not found. {source.hint}"})
            continue
        except ArtifactError as exc:
            rows.append({**base, "status": "invalid", "detail": str(exc)})
            continue
        kwargs: Dict[str, Any] = {"kind": source.scope, "horizon": horizon, "n_folds": n_folds}
        if source.scope == "aggregate":
            kwargs["dev_series"] = dev_series
        else:
            kwargs["item_series"] = item_series
            kwargs["dev_dates"] = dev_dates
        checks = audit_forecast_table(table, **kwargs)
        failed = [c for c in checks if not c.passed]
        record = {**base, "rows": int(len(table)), "models": ", ".join(sorted(map(str, table["model"].unique())))}
        if failed:
            rows.append({**record, "status": "audit failed", "failed_checks": len(failed),
                         "detail": "; ".join(f"{c.name}: {c.detail}" for c in failed)})
            continue
        tables[source.filename] = table
        rows.append({**record, "status": "ok", "detail": f"{len(checks)} audit checks passed"})
    return tables, pd.DataFrame(rows, columns=AUDIT_COLUMNS)


# ----------------------------------------------------------------------------------------- combining tables safely
def combine_tables(
    tables: Mapping[str, pd.DataFrame], scope: str, atol: float = 1e-9
) -> Tuple[pd.DataFrame, List[str]]:
    """Stack several saved tables of ONE scope into a single table, verifying identifiers.

    Checks that (1) every table scores exactly the same cells (`fold`, `date`[, `item_id`]) with identical actual
    values; (2) a model name that appears in more than one table has the same forecasts in each (a re-run), in which
    case the first table's rows are kept; (3) a model name that appears twice with DIFFERENT forecasts is an error.
    Returns `(combined, notes)`; the notes list what was de-duplicated.
    """
    if scope not in ("aggregate", "item"):
        raise ValueError("scope must be 'aggregate' or 'item'.")
    if not tables:
        raise ValueError("No tables to combine.")
    keys = ["fold", "date"] if scope == "aggregate" else ["item_id", "fold", "date"]
    for name, table in tables.items():
        if scope == "aggregate" and "item_id" in table.columns:
            raise ArtifactError(
                f"'{name}' has an item_id column: it is an item-level table. Aggregate and item-level errors are on "
                "different scales and are never combined."
            )
        if scope == "item" and "item_id" not in table.columns:
            raise ArtifactError(f"'{name}' has no item_id column, so it is not an item-level table.")
    names = list(tables)
    reference = tables[names[0]]
    notes: List[str] = []
    kept: List[pd.DataFrame] = []
    seen: Dict[str, Tuple[str, pd.DataFrame]] = {}
    for name in names:
        table = tables[name].copy()
        table["date"] = pd.to_datetime(table["date"]).dt.normalize()
        if name != names[0]:
            check_same_grid(reference, table, keys, atol=atol)
        for model, part in table.groupby("model", sort=False):
            if model in seen:
                first_name, first = seen[model]
                cols = keys
                merged = first.merge(part, on=cols, suffixes=("_first", "_again"), validate="one_to_one")
                a = merged["predicted_first"].to_numpy(dtype=float)
                b = merged["predicted_again"].to_numpy(dtype=float)
                same = len(merged) == len(first) == len(part) and bool(
                    np.all((np.isnan(a) & np.isnan(b)) | (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) <= atol))
                )
                if not same:
                    raise ArtifactError(
                        f"Model name {model!r} appears in {first_name!r} and {name!r} with DIFFERENT forecasts; "
                        "model names must identify one forecast."
                    )
                notes.append(f"{model}: identical in {first_name} and {name}; counted once (kept {first_name}).")
                continue
            seen[model] = (name, part)
            kept.append(part)
    combined = pd.concat(kept, ignore_index=True)
    return combined, notes


# ----------------------------------------------------------------------------------------- metrics
def aggregate_fold_metrics(table: pd.DataFrame, dev_series: pd.Series, horizon: int = 28, n_folds: int = 5) -> pd.DataFrame:
    """Per (model, fold) MAE, RMSE, MAPE, WAPE, bias and MASE of the aggregate series, plus the model family."""
    windows = expected_fold_windows(dev_series.index, horizon, n_folds)
    scales = pd.DataFrame(
        [{"fold": fold, "mase_scale": scale} for fold, scale in fold_training_scales(dev_series, windows).items()]
    )
    metrics = metrics_by_group(table, ["model", "fold"], scales)
    metrics["family"] = metrics["model"].map(lambda m: model_family(m, "aggregate"))
    metrics["primary"] = metrics["model"].isin(PRIMARY_MODELS)
    return metrics


def item_fold_metrics(table: pd.DataFrame, series_by_item: Mapping[str, pd.Series], horizon: int = 28, n_folds: int = 5) -> pd.DataFrame:
    """Per (item, model, fold) metrics of the item-level table, plus the model family."""
    scales = item_mase_scales(series_by_item, horizon, n_folds)
    metrics = metrics_by_group(table, ["item_id", "demand_class", "model", "fold"], scales)
    metrics["family"] = metrics["model"].map(lambda m: model_family(m, "item"))
    return metrics


_SUMMARY_METRICS = ("MAE", "RMSE", "MAPE", "WAPE", "MASE", "bias")


def model_summary(fold_table: pd.DataFrame) -> pd.DataFrame:
    """Per model: mean, standard deviation, min and max over folds of each metric (aggregate series).

    Only successful folds count. `folds_scored` and `folds_failed` show how many; `mape_zero_days_excluded` is the total
    number of zero-actual days left out of MAPE. Standard deviation is the sample (n-1) one across folds; with five
    consecutive, overlapping-history folds it describes spread, not a confidence interval.
    """
    need = {"model", "fold", "success", *_SUMMARY_METRICS, "mape_n_excluded"}
    missing = need - set(fold_table.columns)
    if missing:
        raise ValueError(f"fold_table is missing column(s): {sorted(missing)}")
    rows = []
    for model, group in fold_table.groupby("model", sort=False):
        ok = group[group["success"].astype(bool)]
        record: Dict[str, Any] = {
            "model": model,
            "family": group["family"].iloc[0] if "family" in group.columns else "",
            "folds_scored": int(len(ok)),
            "folds_failed": int(len(group) - len(ok)),
        }
        for metric in _SUMMARY_METRICS:
            values = ok[metric].dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            record[f"{metric}_min"] = float(values.min()) if len(values) else float("nan")
            record[f"{metric}_max"] = float(values.max()) if len(values) else float("nan")
        record["mape_zero_days_excluded"] = int(ok["mape_n_excluded"].sum()) if len(ok) else 0
        rows.append(record)
    return pd.DataFrame(rows)


def fold_wins(fold_table: pd.DataFrame, reference_model: str, metric: str = "MAE") -> pd.DataFrame:
    """For each model: in how many folds is its `metric` strictly lower than `reference_model`'s, and by how much on average.

    Folds where either model failed are not counted. This is a descriptive count of 5 folds, not a test.
    """
    if reference_model not in set(fold_table["model"]):
        raise ValueError(f"reference model {reference_model!r} is not in the table.")
    wide = fold_table[fold_table["success"].astype(bool)].pivot(index="fold", columns="model", values=metric)
    ref = wide[reference_model]
    rows = []
    for model in wide.columns:
        if model == reference_model:
            continue
        both = wide[[model]].join(ref.rename("ref")).dropna()
        diff = both[model] - both["ref"]
        rows.append(
            {
                "model": model,
                "reference": reference_model,
                "folds_compared": int(len(both)),
                "folds_lower": int((diff < 0).sum()),
                "mean_difference": float(diff.mean()) if len(diff) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def lowest_by_metric(summary: pd.DataFrame, columns: Sequence[str] = ("MAE_mean", "RMSE_mean", "WAPE_mean", "MASE_mean")) -> pd.DataFrame:
    """Which model has the lowest mean of each metric, within ONE scope. A description of these folds, not a ranking claim."""
    rows = []
    for column in columns:
        valid = summary.dropna(subset=[column])
        if valid.empty:
            rows.append({"metric": column.replace("_mean", ""), "lowest_model": "n/a", "value": float("nan")})
            continue
        best = valid.loc[valid[column].idxmin()]
        rows.append({"metric": column.replace("_mean", ""), "lowest_model": best["model"], "value": float(best[column])})
    return pd.DataFrame(rows)


def item_model_class_summary(fold_table: pd.DataFrame) -> pd.DataFrame:
    """Macro-average (items weighted equally) per demand class and model for the item-level table."""
    return class_summary(item_summary(fold_table))


# ----------------------------------------------------------------------------------------- plots
def plot_aggregate_comparison(summary: pd.DataFrame):
    """Mean MAE and RMSE per model with the across-fold standard deviation; colour AND hatch encode the family."""
    import matplotlib.pyplot as plt

    ordered = summary.reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 1.1 * len(ordered) + 5), 5), sharey=False, constrained_layout=True)
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for i, row in ordered.iterrows():
            style = FAMILY_STYLE.get(row["family"], FAMILY_STYLE["other"])
            ax.bar(i, row[f"{metric}_mean"], yerr=row[f"{metric}_std"] if np.isfinite(row[f"{metric}_std"]) else None,
                   color=style["color"], edgecolor="#222222", hatch=style["hatch"], capsize=4,
                   alpha=1.0 if row["model"] in PRIMARY_MODELS or not row["family"].endswith("(Phase 3)") else 0.55)
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(ordered["model"], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"{metric} (units/day, aggregate series)")
        ax.set_title(f"Mean {metric} over the five folds (bars: +/- 1 std across folds)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    handles = []
    from matplotlib.patches import Patch

    for family in dict.fromkeys(ordered["family"]):
        style = FAMILY_STYLE.get(family, FAMILY_STYLE["other"])
        handles.append(Patch(facecolor=style["color"], edgecolor="#222222", hatch=style["hatch"], label=family))
    axes[0].legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle("Aggregate series (CA_1 / FOODS_3 daily total): development folds only; lighter bars = pre-declared ablation variants", fontsize=11)
    return fig


def plot_fold_variation(fold_table: pd.DataFrame, metric: str = "MAE"):
    """Per-fold metric of every model (lines with distinct markers and dash styles)."""
    import matplotlib.pyplot as plt

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p"]
    dashes = ["-", "--", "-.", ":"]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, (model, group) in enumerate(fold_table[fold_table["success"].astype(bool)].groupby("model", sort=False)):
        group = group.sort_values("fold")
        style = FAMILY_STYLE.get(group["family"].iloc[0], FAMILY_STYLE["other"])
        ax.plot(group["fold"], group[metric], marker=markers[i % len(markers)], linestyle=dashes[i % len(dashes)],
                color=style["color"] if style["color"] != "#FFFFFF" else "#222222", markersize=6, linewidth=1.4, label=model)
    ax.set_xticks(sorted(fold_table["fold"].unique()))
    ax.set_xlabel("fold (1 = oldest forecast window, 5 = most recent)")
    ax.set_ylabel(f"{metric} (units/day, aggregate series)")
    ax.set_title(f"{metric} of every model in each of the five folds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")
    return fig
