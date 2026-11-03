"""Cached artifact loading -- the only module allowed to touch artifacts/ from the app layer.

The dashboard never trains a model and never forecasts: it reads the forecast and metric tables that the notebooks saved
(`artifacts/forecasts`, `artifacts/metrics`, `artifacts/inventory`) and the development-period data. Nothing here reads a
day of the reserved final holdout. Every loader returns a `Loaded` object, so a missing or invalid artifact becomes an
explicit message (with the command that creates it) instead of an exception or an invented number.

Caching: loaders take a `stamp` (file names, sizes and modification times), so a re-run notebook is picked up on the next
interaction without restarting the app, and unchanged files are not re-read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.data.aggregate_series import DevelopmentWindow, aggregate_series_path, load_aggregate_development
from src.evaluation.forecast_artifacts import (
    forecasts_dir,
    load_selected_item_series,
    metrics_dir,
    notebook_command,
    producer_hint,
    producer_notebook,
)
from src.evaluation.model_comparison import (
    aggregate_fold_metrics,
    aggregate_sources,
    combine_tables,
    item_fold_metrics,
    item_sources,
    load_sources,
    model_family,
    model_summary,
)
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT, resolve_path
from src.utils.provenance import config_section_hash, file_sha256, read_json

METADATA_FILES = {
    "lstm_gru": ("neural_metadata_filename", ("neural", "backtest")),
    "prophet": ("prophet_metadata_filename", ("prophet", "backtest")),
    "intermittent": ("intermittent_metadata_filename", ("models", "backtest")),
    "inventory": ("inventory_metadata_filename", ("inventory", "backtest")),
    "comparison": ("comparison_metadata_filename", ("comparison", "backtest")),
}


@dataclass
class Loaded:
    """A value or the reason there is none. `command` says how to create what is missing."""

    value: Any = None
    error: Optional[str] = None
    command: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Bundle:
    """Everything the forecast-based pages need, loaded and audited once."""

    window: DevelopmentWindow
    dev_series: pd.Series
    item_series: Dict[str, pd.Series]
    aggregate: Optional[pd.DataFrame]            # all aggregate models, long format (model, fold, date, actual, predicted, ...)
    item: Optional[pd.DataFrame]                 # all item-level models, long format (item_id, ... )
    audit: pd.DataFrame                          # one row per expected forecast file
    sources: Dict[str, str] = field(default_factory=dict)   # model name -> forecast file it came from
    notes: List[str] = field(default_factory=list)


# ----------------------------------------------------------------------------------------------------------- basics
#: How the development data is built (needs the raw M5 files in data/raw/; see README.md).
DATA_COMMAND = "python -m src.data.preprocessor\npython -m src.analysis.demand_classification"


def get_config() -> Dict[str, Any]:
    return _config()


@st.cache_resource(show_spinner=False)
def _config() -> Dict[str, Any]:
    return load_config()


def repo_path(path: Path) -> str:
    """Repo-relative POSIX path for display (never an absolute, machine-specific path)."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def artifact_stamp(*paths: Path) -> Tuple:
    """Cache key that changes when any of the files appears, disappears or is rewritten."""
    out = []
    for p in paths:
        p = Path(p)
        if p.exists():
            info = p.stat()
            out.append((p.name, info.st_size, info.st_mtime_ns))
        else:
            out.append((p.name, None, None))
    return tuple(out)


def _forecast_files(config: Dict[str, Any]) -> List[Path]:
    directory = forecasts_dir(config)
    return [directory / s.filename for s in aggregate_sources(config) + item_sources(config)]


def file_record(path: Path) -> Dict[str, Any]:
    """Provenance facts about one artifact file: where it is, when it was written, and its content hash."""
    path = Path(path)
    if not path.exists():
        return {"file": repo_path(path), "exists": False, "modified_utc": "-", "size_kb": None, "sha256": "-"}
    info = path.stat()
    return {
        "file": repo_path(path),
        "exists": True,
        "modified_utc": datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "size_kb": round(info.st_size / 1024, 1),
        "sha256": file_sha256(path)[:12],
    }


def missing_artifact(filename: str, *, folder: str = "forecasts") -> Loaded:
    config = get_config()
    base = {"forecasts": forecasts_dir(config), "metrics": metrics_dir(config)}.get(folder, resolve_path(folder))
    return Loaded(error=f"'{repo_path(base / filename)}' was not found.", command=notebook_command(producer_notebook(filename) or ""))


# ----------------------------------------------------------------------------------------------------------- forecasts
def load_bundle() -> Loaded:
    """Audited forecast tables for every saved model. Cached on the files' stamps."""
    config = get_config()
    stamp = artifact_stamp(aggregate_series_path(config), *_forecast_files(config))
    return _load_bundle(stamp)


@st.cache_data(show_spinner="Reading saved forecasts ...", max_entries=2)
def _load_bundle(stamp: Tuple) -> Loaded:
    config = get_config()
    horizon, n_folds = int(config["backtest"]["horizon"]), int(config["backtest"]["n_folds"])
    try:
        dev_series, window = load_aggregate_development(config)
        item_series = load_selected_item_series(config)
    except FileNotFoundError as exc:
        return Loaded(error=str(exc).split(" Create it")[0], command=DATA_COMMAND)
    except Exception as exc:                       # an unreadable development file is reported, not hidden
        return Loaded(error=f"The development data could not be read: {exc}", command=DATA_COMMAND)
    sources = aggregate_sources(config) + item_sources(config)
    tables, audit = load_sources(
        sources, dev_series=dev_series, item_series=item_series, dev_dates=dev_series.index,
        config=config, horizon=horizon, n_folds=n_folds,
    )
    notes: List[str] = []
    result: Dict[str, Optional[pd.DataFrame]] = {"aggregate": None, "item": None}
    origin: Dict[str, str] = {}
    for scope in ("aggregate", "item"):
        parts = {s.filename: tables[s.filename] for s in sources if s.scope == scope and s.filename in tables}
        if not parts:
            continue
        try:
            combined, scope_notes = combine_tables(parts, scope)
        except Exception as exc:
            notes.append(f"{scope} tables could not be combined safely: {exc}")
            continue
        notes += scope_notes
        result[scope] = combined
        for filename, table in parts.items():
            for model in table["model"].unique():
                origin.setdefault(str(model), filename)
    return Loaded(Bundle(window, dev_series, item_series, result["aggregate"], result["item"], audit, origin, notes))


def ordered_models(table: pd.DataFrame, scope: str) -> List[str]:
    """Model names of a table in a stable, readable order (family first, then name)."""
    models = [str(m) for m in table["model"].unique()]
    return sorted(models, key=lambda m: (model_family(m, scope), m))


def load_metrics(scope: str) -> Loaded:
    """Per-(model, fold) metrics computed from the saved forecasts of `scope` ("aggregate" or "item")."""
    bundle = load_bundle()
    if not bundle.ok:
        return bundle
    stamp = artifact_stamp(*_forecast_files(get_config()))
    return _load_metrics(scope, stamp)


@st.cache_data(show_spinner=False, max_entries=4)
def _load_metrics(scope: str, stamp: Tuple) -> Loaded:
    config = get_config()
    bundle: Bundle = load_bundle().value
    horizon, n_folds = int(config["backtest"]["horizon"]), int(config["backtest"]["n_folds"])
    table = bundle.aggregate if scope == "aggregate" else bundle.item
    if table is None:
        names = [s.filename for s in (aggregate_sources(config) if scope == "aggregate" else item_sources(config))]
        return Loaded(error=f"No {scope}-level forecast file passed its audit (expected: {', '.join(names)}).",
                      command=producer_hint(names[0]).split("Create it with: ")[-1])
    if scope == "aggregate":
        folds = aggregate_fold_metrics(table, bundle.dev_series, horizon, n_folds)
    else:
        folds = item_fold_metrics(table, bundle.item_series, horizon, n_folds)
    return Loaded({"folds": folds, "summary": None})


def aggregate_summary(fold_table: pd.DataFrame) -> pd.DataFrame:
    return model_summary(fold_table)


def macro_item_fold_table(item_folds: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight mean over items, per (model, fold). A fold fails if any item's forecast failed."""
    rows = []
    for (model, fold), group in item_folds.groupby(["model", "fold"], sort=False):
        ok = bool(group["success"].astype(bool).all())
        record: Dict[str, Any] = {"model": model, "fold": int(fold), "success": ok, "family": group["family"].iloc[0]}
        for metric in ("MAE", "RMSE", "MAPE", "WAPE", "MASE", "bias"):
            record[metric] = float(group[metric].mean()) if ok else float("nan")
        record["mape_n_excluded"] = int(group["mape_n_excluded"].fillna(0).sum()) if ok else 0
        record["n_items"] = int(group["item_id"].nunique())
        rows.append(record)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- overview
def load_overview() -> Loaded:
    config = get_config()
    panel = resolve_path(Path(config["data"]["interim_dir"]) / config["output"]["panel_filename"])
    classification = resolve_path(Path(config["data"]["processed_dir"]) / config["output"]["classification_filename"])
    return _load_overview(artifact_stamp(aggregate_series_path(config), panel, classification))


@st.cache_data(show_spinner="Summarising the development data ...", max_entries=2)
def _load_overview(stamp: Tuple) -> Loaded:
    from src.analysis.demand_classification import classify_series
    from src.evaluation.intermittent_experiment import load_all_item_development_sales

    config = get_config()
    try:
        series, window = load_aggregate_development(config)
        wide, _ = load_all_item_development_sales(config)
    except FileNotFoundError as exc:
        return Loaded(error=str(exc).split(" Create it")[0], command=DATA_COMMAND)
    except Exception as exc:
        return Loaded(error=f"The development data could not be read: {exc}", command=DATA_COMMAND)
    classes = pd.DataFrame([{"item_id": c, **classify_series(wide[c])} for c in wide.columns])
    classification_path = resolve_path(Path(config["data"]["processed_dir"]) / config["output"]["classification_filename"])
    phase1 = pd.read_parquet(classification_path) if classification_path.exists() else None
    panel_rows = None
    try:
        import pyarrow.parquet as pq

        panel_rows = int(pq.ParquetFile(resolve_path(Path(config["data"]["interim_dir"]) / config["output"]["panel_filename"])).metadata.num_rows)
    except Exception:
        panel_rows = None
    n_items, n_days = wide.shape[1], wide.shape[0]
    return Loaded({
        "series": series, "window": window,
        "n_items": n_items, "n_days": n_days, "dev_item_days": int(n_items * n_days),
        "panel_item_days": panel_rows,
        "zero_share": float((wide.to_numpy() == 0).mean()),
        "dev_classes": classes["demand_class"].value_counts().rename_axis("demand_class").reset_index(name="items"),
        "phase1_classes": None if phase1 is None else phase1["demand_class"].value_counts().rename_axis("demand_class").reset_index(name="items"),
        "total_units": float(series.sum()),
    })


def holdout_state() -> Dict[str, Any]:
    """Whether the final-holdout files exist. Only file existence is checked; no holdout value is read."""
    from src.evaluation import final_holdout as fh

    config = get_config()
    return {
        "frozen": fh.frozen_path(config).exists(),
        "evaluated": fh.results_path(config).exists() or fh.record_path(config).exists(),
    }


# ----------------------------------------------------------------------------------------------------------- reliability
def read_optional_csv(filename: str, folder: str = "metrics") -> Loaded:
    config = get_config()
    base = metrics_dir(config) if folder == "metrics" else forecasts_dir(config)
    path = base / filename
    if not path.exists():
        return missing_artifact(filename, folder=folder)
    return _read_csv(str(path), artifact_stamp(path))


@st.cache_data(show_spinner=False, max_entries=24)
def _read_csv(path: str, stamp: Tuple) -> Loaded:
    try:
        return Loaded(pd.read_csv(path))
    except Exception as exc:
        return Loaded(error=f"'{repo_path(Path(path))}' could not be read: {exc}")


def load_run_metadata() -> Dict[str, Dict[str, Any]]:
    """The saved `*_run_metadata.json` of each Phase 3 component, plus whether its config hash still matches."""
    config = get_config()
    out: Dict[str, Dict[str, Any]] = {}
    for component, (key, _sections) in METADATA_FILES.items():
        directory = resolve_path(config["phase3"].get("inventory_dir")) if component == "inventory" else metrics_dir(config)
        path = directory / config["phase3"][key]
        record: Dict[str, Any] = {"file": repo_path(path), "exists": path.exists()}
        if path.exists():
            try:
                saved = read_json(path)
                sections = saved.get("config_sections") or []
                record.update(
                    generated=saved.get("generated_at_utc"),
                    versions={k: v for k, v in (saved.get("environment", {}).get("libraries") or {}).items() if v},
                    python=saved.get("environment", {}).get("python"),
                    config_matches=(config_section_hash(config, sections) == saved.get("config_sha256")) if sections and saved.get("config_sha256") else None,
                    details=saved.get("details", {}),
                )
            except Exception as exc:
                record["error"] = str(exc)
        out[component] = record
    return out


def residual_acf(table: pd.DataFrame, model: str, fold: int, max_lag: int = 14) -> Optional[Dict[str, Any]]:
    """Autocorrelation of one fold's forecast errors (actual - forecast), from the saved forecast table."""
    from statsmodels.tsa.stattools import acf

    part = table[(table["model"] == model) & (table["fold"] == fold)].sort_values("date")
    if part.empty or "success" in part.columns and not part["success"].astype(bool).all():
        return None
    errors = (part["actual"] - part["predicted"]).to_numpy(dtype=float)
    n = len(errors)
    lags = min(max_lag, n - 2)
    if lags < 1 or np.allclose(errors, errors[0]):
        return None
    values = acf(errors, nlags=lags, fft=False)
    return {"lags": list(range(1, lags + 1)), "acf": [float(v) for v in values[1:]], "n": n, "band": float(1.96 / np.sqrt(n))}


def data_validation_summary() -> Loaded:
    """Re-run the series checks on the saved development series (cheap) so the page shows what was actually verified."""
    from src.data import series_validator as sv

    config = get_config()
    path = aggregate_series_path(config)
    if not path.exists():
        return Loaded(error=f"'{repo_path(path)}' was not found.", command="python -m src.analysis.demand_classification")
    series, window = load_aggregate_development(config)
    frame = pd.DataFrame({"date": series.index, "sales": series.to_numpy()})
    results = [sv.check_daily_frequency(frame), sv.check_no_duplicate_dates(frame), sv.check_no_missing_dates(frame),
               sv.check_continuous_range(frame), sv.check_no_negative_sales(frame)]
    rows = [{"check": r.name, "passed": bool(r.passed), "detail": r.message} for r in results]
    return Loaded({"checks": pd.DataFrame(rows), "window": window})
