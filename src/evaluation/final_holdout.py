"""Controlled, one-time evaluation of the reserved final holdout (the last 28 days, 2016-04-25 .. 2016-05-22).

The holdout exists to report ONE honest out-of-sample number per model at the very end of the project. Nothing
in the development workflow (notebooks 04-10, the dashboard) reads it. This module is the only code path that is
allowed to, and it only does so after a deliberate three-step procedure:

    python -m src.evaluation.final_holdout status     read-only: what is frozen, what is missing, what to do next
    python -m src.evaluation.final_holdout freeze     record the candidate models and the headline choice, decided
                                                      from DEVELOPMENT folds only (no holdout value is read)
    python -m src.evaluation.final_holdout run --confirm EVALUATE-FINAL-HOLDOUT-ONCE
                                                      fit every frozen candidate on development data, forecast the
                                                      28 holdout days, THEN read the holdout and score once

Safeguards (each is enforced in code and covered by tests/evaluation/test_final_holdout.py):

  * `run` refuses without a frozen-candidates file, without the exact confirmation phrase, when `config.yaml` has
    changed since the freeze (hash of the sections that define the models), and when a result already exists
    (unless `--override-existing` is given, which archives the old files and records the repeat evaluation).
  * Every candidate is fitted and has produced its 28 forecasts BEFORE the holdout actuals are read: the actuals
    are only reachable through `HoldoutInputs.read_actuals`, which is called once, after all forecasts exist.
  * A frozen candidate whose rebuilt model no longer has the frozen name is refused, so settings cannot drift
    silently between freeze and run.
  * A candidate that fails is reported as failed. No other method is ever substituted for it.
  * ALL frozen candidates are reported, not only the headline one. The headline per scope is the candidate with the
    lowest mean development-fold MAE, fixed at freeze time. Aggregate and item-level results are separate scopes
    and are never ranked against each other.
  * Freezing never reads a holdout value: it uses the saved development forecast tables only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.aggregate_series import _read_aggregate_table, aggregate_series_path, load_aggregate_development
from src.evaluation.forecast_artifacts import (
    forecasts_dir,
    load_selected_item_series,
    metrics_by_group,
    seasonal_naive_scale,
)
from src.evaluation.model_comparison import (
    aggregate_fold_metrics,
    aggregate_sources,
    combine_tables,
    item_fold_metrics,
    item_sources,
    load_sources,
    model_summary,
)
from src.models.base import BaseForecaster
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT, resolve_path
from src.utils.provenance import (
    build_run_metadata,
    config_section_hash,
    environment_snapshot,
    file_sha256,
    read_json,
    write_json,
)

SCOPES = ("aggregate", "item_level")
CONFIRMATION_PHRASE = "EVALUATE-FINAL-HOLDOUT-ONCE"
SCHEMA_VERSION = 1

#: The config sections that define the candidate models. Their hash is stored at freeze time and compared at run time.
FROZEN_CONFIG_SECTIONS = ("backtest", "models", "xgboost", "neural", "prophet", "final_holdout")

#: Short candidate names accepted in `final_holdout.candidates` of config.yaml.
KNOWN_CANDIDATES = {
    "aggregate": ("SeasonalNaive", "ARIMA", "SARIMA", "HoltWinters", "LSTM(calendar)", "GRU(calendar)",
                  "LSTM(univariate)", "GRU(univariate)", "Prophet(additive)", "Prophet(multiplicative)",
                  "Prophet(additive,no-regressors)"),
    "item_level": ("SeasonalNaive", "Croston", "TSB", "XGBoost(pooled)"),
}

REPEAT_WARNING = (
    "WARNING: the final holdout has now been evaluated more than once. A result that could have influenced a "
    "decision is no longer an honest single-shot estimate; report the evaluation count with the numbers."
)


class FinalHoldoutError(RuntimeError):
    """A refusal: the requested step would break the final-holdout protocol. The message says what to do."""


class NotReadyToFreeze(FinalHoldoutError):
    """Development evidence for one or more candidates is missing or failed its audit."""


# ----------------------------------------------------------------------------------------------------- paths

def final_dir(config: Optional[Mapping[str, Any]] = None) -> Path:
    config = load_config() if config is None else config
    return resolve_path(config["phase3"]["final_dir"])


def _final_path(config: Mapping[str, Any], key: str) -> Path:
    return final_dir(config) / config["final_holdout"][key]


def frozen_path(config: Optional[Mapping[str, Any]] = None) -> Path:
    config = load_config() if config is None else config
    return _final_path(config, "frozen_filename")


def results_path(config: Optional[Mapping[str, Any]] = None) -> Path:
    config = load_config() if config is None else config
    return _final_path(config, "results_filename")


def forecasts_path(config: Optional[Mapping[str, Any]] = None) -> Path:
    config = load_config() if config is None else config
    return _final_path(config, "forecasts_filename")


def record_path(config: Optional[Mapping[str, Any]] = None) -> Path:
    config = load_config() if config is None else config
    return _final_path(config, "record_filename")


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _utc(now: Optional[datetime]) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------------------------------- candidates

def configured_candidates(config: Mapping[str, Any]) -> Dict[str, List[str]]:
    """The candidate short names from config.yaml, validated."""
    section = config["final_holdout"]["candidates"]
    out: Dict[str, List[str]] = {}
    for scope in SCOPES:
        names = [str(n) for n in (section.get(scope) or [])]
        if not names:
            raise FinalHoldoutError(f"config.yaml final_holdout.candidates.{scope} is empty.")
        if len(set(names)) != len(names):
            raise FinalHoldoutError(f"final_holdout.candidates.{scope} lists a model twice.")
        unknown = [n for n in names if n not in KNOWN_CANDIDATES[scope]]
        if unknown:
            raise FinalHoldoutError(
                f"Unknown {scope} candidate(s) {unknown}. Known candidates: {list(KNOWN_CANDIDATES[scope])}."
            )
        out[scope] = names
    return out


def name_matches(short_name: str, full_name: str) -> bool:
    """`ARIMA` matches `ARIMA(6, 1, 1)`; `XGBoost(pooled)` matches only itself, not `XGBoost(pooled,poisson)`."""
    return full_name == short_name or full_name.startswith(short_name + "(")


def resolve_model_name(short_name: str, available: Sequence[str]) -> Optional[str]:
    """The saved model name that a short candidate name refers to (None if absent; error if ambiguous)."""
    hits = [m for m in available if name_matches(short_name, m)]
    if len(hits) > 1:
        raise FinalHoldoutError(f"Candidate {short_name!r} is ambiguous in the saved results: {hits}.")
    return hits[0] if hits else None


def candidate_settings(scope: str, short_name: str, config: Mapping[str, Any]) -> Dict[str, Any]:
    """The settings a candidate will be built with, copied out of config.yaml for the freeze record."""
    models, neural, prophet = config["models"], config["neural"], config["prophet"]
    season = int(config["backtest"]["season_length"])
    if short_name == "SeasonalNaive":
        return {"season_length": season}
    if short_name == "ARIMA":
        return {"order": list(models["arima_order"])}
    if short_name == "SARIMA":
        return {"order": list(models["sarima_order"]), "seasonal_order": list(models["sarima_seasonal_order"])}
    if short_name == "HoltWinters":
        return {"seasonal_periods": int(models["holt_winters_seasonal_periods"])}
    if short_name in ("LSTM(calendar)", "GRU(calendar)", "LSTM(univariate)", "GRU(univariate)"):
        cell, features = short_name[:-1].split("(")
        shared = {k: v for k, v in neural.items() if k not in ("variants", "primary_features")}
        return {"cell": cell.lower(), "features": features, **shared}
    if short_name.startswith("Prophet("):
        variant = next((v for v in prophet["variants"] if v["name"] == short_name), None)
        if variant is None:
            raise FinalHoldoutError(f"Prophet variant {short_name!r} is not declared in config.yaml prophet.variants.")
        return {"variant": dict(variant), **{k: v for k, v in prophet.items() if k not in ("variants",)}}
    if short_name == "Croston":
        return {"alpha": float(models["croston_alpha"])}
    if short_name == "TSB":
        return {"alpha": float(models["tsb_alpha"]), "beta": float(models["tsb_beta"])}
    if short_name == "XGBoost(pooled)":
        return {"xgboost": dict(config["xgboost"])}
    raise FinalHoldoutError(f"No settings are defined for candidate {short_name!r} ({scope}).")


Factory = Callable[[], Any]


def default_factories(config: Mapping[str, Any], calendar: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Factory]]:
    """Build-on-demand constructors for every known candidate, with exactly the config.yaml settings.

    Heavy libraries (PyTorch, Prophet, XGBoost) are imported only when their candidate is built, so a machine
    without them can still freeze and check status; building such a candidate raises the model module's own
    informative ImportError, and it is never replaced by another method.
    """
    from src.models.arima import ARIMAForecaster
    from src.models.baseline import SeasonalNaive
    from src.models.croston import CrostonForecaster
    from src.models.holt_winters import HoltWintersForecaster
    from src.models.sarima import SARIMAForecaster
    from src.models.tsb import TSBForecaster

    models = config["models"]
    season = int(config["backtest"]["season_length"])
    seasonal_naive: Factory = lambda: SeasonalNaive(season_length=season)

    def neural(cell: str, features: str) -> Factory:
        def build():
            from src.models.lstm_gru import NeuralSequenceForecaster

            return NeuralSequenceForecaster.from_config(cell, features, config, calendar=calendar)

        return build

    def prophet(name: str) -> Factory:
        def build():
            from src.models.prophet_model import ProphetForecaster

            return ProphetForecaster.from_config(name, config, calendar=calendar)

        return build

    def pooled_xgboost():
        from src.models.xgboost_model import PooledXGBoostForecaster, XGBoostConfig

        return PooledXGBoostForecaster(XGBoostConfig.from_config(config), calendar)

    aggregate: Dict[str, Factory] = {
        "SeasonalNaive": seasonal_naive,
        "ARIMA": lambda: ARIMAForecaster(order=tuple(models["arima_order"])),
        "SARIMA": lambda: SARIMAForecaster(
            order=tuple(models["sarima_order"]), seasonal_order=tuple(models["sarima_seasonal_order"])
        ),
        "HoltWinters": lambda: HoltWintersForecaster(seasonal_periods=int(models["holt_winters_seasonal_periods"])),
    }
    for cell in ("lstm", "gru"):
        for features in ("calendar", "univariate"):
            aggregate[f"{cell.upper()}({features})"] = neural(cell, features)
    for variant in config["prophet"]["variants"]:
        aggregate[variant["name"]] = prophet(variant["name"])
    item_level: Dict[str, Factory] = {
        "SeasonalNaive": seasonal_naive,
        "Croston": lambda: CrostonForecaster(alpha=float(models["croston_alpha"])),
        "TSB": lambda: TSBForecaster(alpha=float(models["tsb_alpha"]), beta=float(models["tsb_beta"])),
        "XGBoost(pooled)": pooled_xgboost,
    }
    return {"aggregate": aggregate, "item_level": item_level}


# ----------------------------------------------------------------------------------------------------- development evidence

@dataclass
class DevelopmentEvidence:
    """Mean development-fold metrics per saved model, for each scope, plus the audit of the files they came from."""

    summaries: Dict[str, pd.DataFrame]              # scope -> model, MAE_mean, RMSE_mean, folds_scored, folds_failed
    audit: pd.DataFrame
    window: Any                                     # DevelopmentWindow: calendar dates of the development / holdout split
    artifact_sha256: Dict[str, str] = field(default_factory=dict)


def _macro_fold_table(item_metrics: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight mean over items of each metric, per (model, fold); a fold fails if any item's forecast failed."""
    rows = []
    for (model, fold), group in item_metrics.groupby(["model", "fold"], sort=False):
        ok = bool(group["success"].astype(bool).all())
        record: Dict[str, Any] = {"model": model, "fold": int(fold), "success": ok}
        for metric in ("MAE", "RMSE", "MAPE", "WAPE", "MASE", "bias"):
            record[metric] = float(group[metric].mean()) if ok else float("nan")
        record["mape_n_excluded"] = int(group["mape_n_excluded"].fillna(0).sum()) if ok else 0
        rows.append(record)
    return pd.DataFrame(rows)


def load_development_evidence(config: Optional[Mapping[str, Any]] = None) -> DevelopmentEvidence:
    """Read and audit the saved development forecasts, then summarise each model's folds. No holdout value is read."""
    config = load_config() if config is None else config
    horizon, n_folds = int(config["backtest"]["horizon"]), int(config["backtest"]["n_folds"])
    dev_series, window = load_aggregate_development(config)
    item_series = load_selected_item_series(config)
    sources = aggregate_sources(config) + item_sources(config)
    tables, audit = load_sources(
        sources, dev_series=dev_series, item_series=item_series, dev_dates=dev_series.index,
        config=config, horizon=horizon, n_folds=n_folds,
    )
    summaries: Dict[str, pd.DataFrame] = {}
    hashes: Dict[str, str] = {}
    directory = forecasts_dir(config)
    for filename in tables:
        hashes[filename] = file_sha256(directory / filename)
    aggregate_tables = {s.filename: tables[s.filename] for s in sources if s.scope == "aggregate" and s.filename in tables}
    item_tables = {s.filename: tables[s.filename] for s in sources if s.scope == "item" and s.filename in tables}
    if aggregate_tables:
        combined, _ = combine_tables(aggregate_tables, "aggregate")
        summaries["aggregate"] = model_summary(aggregate_fold_metrics(combined, dev_series, horizon, n_folds))
    if item_tables:
        combined, _ = combine_tables(item_tables, "item")
        summaries["item_level"] = model_summary(_macro_fold_table(item_fold_metrics(combined, item_series, horizon, n_folds)))
    return DevelopmentEvidence(summaries=summaries, audit=audit, window=window, artifact_sha256=hashes)


def choose_headline(summary: pd.DataFrame, candidates: Mapping[str, str]) -> Tuple[str, pd.DataFrame]:
    """Lowest mean development-fold MAE among candidates whose folds all succeeded (ties: alphabetical short name).

    Returns `(short_name, table)`; the table lists every candidate with its development MAE / RMSE and eligibility.
    """
    rows = []
    for short, full in candidates.items():
        match = summary[summary["model"] == full]
        if match.empty:
            raise NotReadyToFreeze(f"No development result for {full!r}.")
        row = match.iloc[0]
        eligible = int(row["folds_failed"]) == 0 and bool(np.isfinite(row["MAE_mean"]))
        rows.append({"candidate": short, "model": full, "dev_MAE_mean": float(row["MAE_mean"]),
                     "dev_RMSE_mean": float(row["RMSE_mean"]), "folds_scored": int(row["folds_scored"]),
                     "folds_failed": int(row["folds_failed"]), "eligible_for_headline": eligible})
    table = pd.DataFrame(rows)
    eligible = table[table["eligible_for_headline"]].sort_values(["dev_MAE_mean", "candidate"])
    if eligible.empty:
        raise NotReadyToFreeze("No candidate has a complete, finite development result, so no headline model can be chosen.")
    return str(eligible.iloc[0]["candidate"]), table


def _not_ready_message(missing: Sequence[str], audit: pd.DataFrame) -> str:
    lines = ["Cannot freeze yet: development evidence is incomplete.", ""]
    lines += [f"  missing: {m}" for m in missing]
    bad = audit[audit["status"] != "ok"] if len(audit) else audit
    if len(bad):
        lines += ["", "Artifacts that are absent or failed their audit:"]
        lines += [f"  {r.artifact} [{r.status}]: {r.detail}" for r in bad.itertuples(index=False)]
    lines += ["", "Re-run the producing notebooks (commands above), then freeze again. Nothing was written."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------- freeze

def freeze(
    config: Optional[Mapping[str, Any]] = None,
    *,
    evidence: Optional[DevelopmentEvidence] = None,
    overwrite: bool = False,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Write the frozen-candidates file. Uses development evidence only; never reads a holdout value."""
    config = load_config() if config is None else config
    if results_path(config).exists() or record_path(config).exists():
        raise FinalHoldoutError("The final holdout has already been evaluated; the candidates can no longer be re-frozen.")
    target = frozen_path(config)
    if target.exists() and not overwrite:
        raise FinalHoldoutError(
            f"'{_display(target)}' already exists. Candidates are frozen once; pass --overwrite only to replace a "
            "freeze that has not been evaluated yet (for example after fixing a development artifact)."
        )
    candidates = configured_candidates(config)
    evidence = load_development_evidence(config) if evidence is None else evidence

    resolved: Dict[str, Dict[str, str]] = {}
    missing: List[str] = []
    for scope in SCOPES:
        summary = evidence.summaries.get(scope)
        available = [] if summary is None else list(summary["model"])
        resolved[scope] = {}
        for short in candidates[scope]:
            full = resolve_model_name(short, available)
            if full is None:
                missing.append(f"{scope} candidate {short}")
            else:
                resolved[scope][short] = full
    if missing:
        raise NotReadyToFreeze(_not_ready_message(missing, evidence.audit))

    scopes_record: Dict[str, Any] = {}
    for scope in SCOPES:
        headline, table = choose_headline(evidence.summaries[scope], resolved[scope])
        scopes_record[scope] = {
            "headline_candidate": headline,
            "headline_model": resolved[scope][headline],
            "candidates": [
                {**row, "settings": candidate_settings(scope, row["candidate"], config)}
                for row in table.to_dict(orient="records")
            ],
        }

    window = evidence.window
    record = {
        "schema_version": SCHEMA_VERSION,
        "frozen_at_utc": _utc(now),
        "holdout_evaluated": False,
        "config_sections": list(FROZEN_CONFIG_SECTIONS),
        "config_sha256": config_section_hash(config, FROZEN_CONFIG_SECTIONS),
        "development_window": {"start": str(window.dev_start.date()), "end": str(window.dev_end.date()),
                               "days": int(window.n_development_days)},
        "reserved_holdout": {"start": str(window.holdout_start.date()), "end": str(window.holdout_end.date()),
                             "days": int(window.n_holdout_days)},
        "selection_rule": config["final_holdout"]["selection_rule"],
        "metrics": list(config["final_holdout"]["metrics"]),
        "scopes": scopes_record,
        "development_artifact_sha256": dict(sorted(evidence.artifact_sha256.items())),
        "environment": environment_snapshot(),
        "note": "Chosen from development folds only. No holdout value was read while freezing.",
    }
    write_json(target, record)
    return record


# ----------------------------------------------------------------------------------------------------- status

@dataclass
class FinalHoldoutStatus:
    frozen: bool
    config_matches: Optional[bool]
    evaluated: bool
    evaluation_count: int
    missing: List[str]
    next_steps: List[str]
    frozen_at: Optional[str] = None

    def to_text(self) -> str:
        lines = ["Final holdout status", "--------------------"]
        lines.append(f"candidates frozen        : {'yes (' + str(self.frozen_at) + ')' if self.frozen else 'no'}")
        if self.frozen:
            lines.append(f"config unchanged         : {'yes' if self.config_matches else 'NO - re-freeze or restore config.yaml'}")
        lines.append(f"holdout evaluated        : {'yes, ' + str(self.evaluation_count) + ' time(s)' if self.evaluated else 'no'}")
        if self.missing:
            lines.append("what is missing:")
            lines += [f"  - {m}" for m in self.missing]
        lines.append("next steps:")
        lines += [f"  {i}. {s}" for i, s in enumerate(self.next_steps, 1)]
        return "\n".join(lines)


def status(config: Optional[Mapping[str, Any]] = None, *, evidence: Optional[DevelopmentEvidence] = None) -> FinalHoldoutStatus:
    """Read-only summary of where the final-holdout procedure stands. Never reads a holdout value."""
    config = load_config() if config is None else config
    frozen_file, record_file = frozen_path(config), record_path(config)
    evaluated = results_path(config).exists() or record_file.exists()
    count = 0
    if record_file.exists():
        try:
            count = int(read_json(record_file).get("evaluation_count", 1))
        except (ValueError, OSError):
            count = 1
    frozen = frozen_file.exists()
    frozen_at = matches = None
    if frozen:
        stored = read_json(frozen_file)
        frozen_at = stored.get("frozen_at_utc")
        matches = stored.get("config_sha256") == config_section_hash(config, FROZEN_CONFIG_SECTIONS)

    missing: List[str] = []
    if not frozen:
        try:
            evidence = load_development_evidence(config) if evidence is None else evidence
            candidates = configured_candidates(config)
            for scope in SCOPES:
                summary = evidence.summaries.get(scope)
                available = [] if summary is None else list(summary["model"])
                for short in candidates[scope]:
                    if resolve_model_name(short, available) is None:
                        missing.append(f"development results for {scope} candidate {short}")
            bad = evidence.audit[evidence.audit["status"] != "ok"] if len(evidence.audit) else evidence.audit
            missing += [f"{r.artifact} [{r.status}] - {r.detail}" for r in bad.itertuples(index=False)]
        except (FileNotFoundError, FinalHoldoutError, ValueError) as exc:
            missing.append(f"development evidence could not be loaded: {exc}")

    steps: List[str] = []
    if evaluated:
        steps.append("The holdout is evaluated. Report the numbers together with the evaluation count in the record file.")
    elif not frozen:
        if missing:
            steps.append("Create every missing development artifact (commands above), then review notebook 09 before freezing.")
        else:
            steps.append("All development artifacts are present. Review notebook 09, decide that no model will change, then freeze.")
        steps.append("python -m src.evaluation.final_holdout freeze")
        steps.append(f"python -m src.evaluation.final_holdout run --confirm {CONFIRMATION_PHRASE}")
    elif not matches:
        steps.append("config.yaml changed after the freeze. Restore it, or re-freeze with --overwrite and review the new freeze.")
    else:
        steps.append(f"python -m src.evaluation.final_holdout run --confirm {CONFIRMATION_PHRASE}")
    return FinalHoldoutStatus(frozen, matches, evaluated, count, missing, steps, frozen_at)


# ----------------------------------------------------------------------------------------------------- inputs

@dataclass
class HoldoutActuals:
    aggregate: pd.Series                      # date-indexed holdout sales of the aggregate series
    items: Dict[str, pd.Series]               # item_id -> date-indexed holdout sales


@dataclass
class HoldoutInputs:
    """Everything a final run needs. Holdout sales are reachable ONLY through `read_actuals`."""

    aggregate_history: pd.Series              # development sales only
    item_histories: Dict[str, pd.Series]      # development sales only
    calendar: Optional[pd.DataFrame]          # calendar facts known in advance (events, SNAP); contains no sales
    read_actuals: Callable[[], HoldoutActuals]
    horizon: int = 28


def load_holdout_inputs(config: Optional[Mapping[str, Any]] = None) -> HoldoutInputs:
    """Real inputs: development series now, holdout sales only when `read_actuals()` is called."""
    from src.models.xgboost_model import load_event_calendar

    config = load_config() if config is None else config
    horizon = int(config["backtest"]["horizon"])
    holdout_days = int(config["backtest"]["holdout_days"])
    series, _ = load_aggregate_development(config)
    items = load_selected_item_series(config)
    panel_path = resolve_path(Path(config["data"]["interim_dir"]) / config["output"]["panel_filename"])
    calendar = load_event_calendar(panel_path)           # calendar columns only; `sales` is never loaded here

    def read_actuals() -> HoldoutActuals:
        table = _read_aggregate_table(aggregate_series_path(config))
        tail = table.iloc[-holdout_days:]
        aggregate = pd.Series(tail["sales"].to_numpy(dtype=float), index=pd.DatetimeIndex(tail["date"]), name="aggregate_sales")
        daily = pd.read_parquet(
            resolve_path(Path(config["data"]["processed_dir"]) / config["output"]["selected_items_daily_sales_filename"])
        )
        item_actuals: Dict[str, pd.Series] = {}
        for item_id, group in daily.groupby("item_id"):
            group = group.sort_values("date")
            part = group.iloc[-holdout_days:]
            item_actuals[str(item_id)] = pd.Series(
                part["sales"].to_numpy(dtype=float), index=pd.DatetimeIndex(part["date"]), name=str(item_id)
            )
        return HoldoutActuals(aggregate, item_actuals)

    return HoldoutInputs(series, items, calendar, read_actuals, horizon)


# ----------------------------------------------------------------------------------------------------- run

def _forecast_dates(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def _predict_aggregate(factory: Factory, frozen_model: str, history: pd.Series, horizon: int) -> Tuple[np.ndarray, bool, Optional[str]]:
    model = factory()
    if getattr(model, "name", None) != frozen_model:
        raise FinalHoldoutError(
            f"Candidate was frozen as {frozen_model!r} but now builds as {getattr(model, 'name', None)!r}. "
            "Settings changed after the freeze; restore config.yaml or re-freeze."
        )
    try:
        values = np.asarray(model.fit(history).predict(horizon), dtype=float)
        if values.shape != (horizon,) or not np.all(np.isfinite(values)):
            raise ValueError(f"expected {horizon} finite forecasts, got shape {values.shape}")
        return values, True, None
    except Exception as exc:                                  # a failed candidate is reported, never replaced
        return np.full(horizon, np.nan), False, f"{type(exc).__name__}: {exc}"


def _predict_item_level(
    factory: Factory, frozen_model: str, histories: Mapping[str, pd.Series], horizon: int
) -> Dict[str, Tuple[np.ndarray, bool, Optional[str]]]:
    first = factory()
    if getattr(first, "name", None) != frozen_model:
        raise FinalHoldoutError(
            f"Candidate was frozen as {frozen_model!r} but now builds as {getattr(first, 'name', None)!r}. "
            "Settings changed after the freeze; restore config.yaml or re-freeze."
        )
    out: Dict[str, Tuple[np.ndarray, bool, Optional[str]]] = {}
    if isinstance(first, BaseForecaster):                      # one model per item
        for index, (item_id, history) in enumerate(histories.items()):
            model = first if index == 0 else factory()
            try:
                values = np.asarray(model.fit(history).predict(horizon), dtype=float)
                if values.shape != (horizon,) or not np.all(np.isfinite(values)):
                    raise ValueError(f"expected {horizon} finite forecasts, got shape {values.shape}")
                out[item_id] = (values, True, None)
            except Exception as exc:
                out[item_id] = (np.full(horizon, np.nan), False, f"{type(exc).__name__}: {exc}")
        return out
    try:                                                       # one pooled model for all items
        predictions = first.fit(dict(histories)).predict(horizon)
        for item_id in histories:
            values = np.asarray(predictions[item_id], dtype=float)
            if values.shape != (horizon,) or not np.all(np.isfinite(values)):
                raise ValueError(f"expected {horizon} finite forecasts for {item_id}, got shape {values.shape}")
            out[item_id] = (values, True, None)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        out = {item_id: (np.full(horizon, np.nan), False, error) for item_id in histories}
    return out


def _check_refusals(config: Mapping[str, Any], confirm: Optional[str], override_existing: bool) -> Dict[str, Any]:
    if confirm != CONFIRMATION_PHRASE:
        raise FinalHoldoutError(
            "Refusing to evaluate the final holdout without explicit confirmation. This is a one-time action: the "
            f"result can no longer be treated as unseen afterwards. Re-run with:  --confirm {CONFIRMATION_PHRASE}"
        )
    target = frozen_path(config)
    if not target.exists():
        raise FinalHoldoutError(
            f"No frozen candidates found ('{_display(target)}'). Freeze first: python -m src.evaluation.final_holdout freeze"
        )
    frozen = read_json(target)
    if int(frozen.get("schema_version", -1)) != SCHEMA_VERSION:
        raise FinalHoldoutError("The frozen file has an unsupported schema version; re-freeze.")
    if frozen.get("config_sha256") != config_section_hash(config, FROZEN_CONFIG_SECTIONS):
        raise FinalHoldoutError(
            "config.yaml (model, backtest or final_holdout sections) has changed since the candidates were frozen. "
            "Restore it, or re-freeze with --overwrite and review the new freeze before evaluating."
        )
    if (results_path(config).exists() or record_path(config).exists()) and not override_existing:
        raise FinalHoldoutError(
            "The final holdout has already been evaluated, and a second look at it is not an unseen estimate any "
            "more. If you must repeat it, pass --override-existing; the old files are archived and the repeat is recorded."
        )
    return frozen


def _archive_existing(config: Mapping[str, Any], evaluation_count: int) -> List[str]:
    archived = []
    for path in (results_path(config), forecasts_path(config), record_path(config)):
        if path.exists():
            target = path.with_name(f"{path.stem}.superseded-{evaluation_count}{path.suffix}")
            path.replace(target)
            archived.append(target.name)
    return archived


def _score(forecasts: pd.DataFrame, scales: pd.DataFrame, group: List[str]) -> pd.DataFrame:
    return metrics_by_group(forecasts, group, scales)


def evaluate_frozen_candidates(
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
    inputs: HoldoutInputs,
    factories: Mapping[str, Mapping[str, Factory]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each frozen candidate on development data, forecast, and only then read and score the holdout.

    Returns `(forecast_table, results_table)`. The holdout actuals are read exactly once, after every candidate has
    produced (or failed to produce) its forecasts.
    """
    horizon = inputs.horizon
    produced: List[Dict[str, Any]] = []                      # forecasts and failures, before any actual is known
    for scope in SCOPES:
        for entry in frozen["scopes"][scope]["candidates"]:
            short, full = entry["candidate"], entry["model"]
            if short not in factories[scope]:
                raise FinalHoldoutError(f"No constructor is available for {scope} candidate {short!r}.")
            if scope == "aggregate":
                values, ok, error = _predict_aggregate(factories[scope][short], full, inputs.aggregate_history, horizon)
                dates = _forecast_dates(inputs.aggregate_history, horizon)
                produced.append({"scope": scope, "model": full, "item_id": None, "dates": dates,
                                 "predicted": values, "success": ok, "error": error})
            else:
                per_item = _predict_item_level(factories[scope][short], full, inputs.item_histories, horizon)
                for item_id, (values, ok, error) in per_item.items():
                    dates = _forecast_dates(inputs.item_histories[item_id], horizon)
                    produced.append({"scope": scope, "model": full, "item_id": item_id, "dates": dates,
                                     "predicted": values, "success": ok, "error": error})

    actuals = inputs.read_actuals()                           # <- the only moment holdout sales are read

    rows: List[Dict[str, Any]] = []
    for part in produced:
        source = actuals.aggregate if part["scope"] == "aggregate" else actuals.items.get(part["item_id"])
        if source is None:
            raise FinalHoldoutError(f"Holdout actuals are missing for item {part['item_id']!r}.")
        if not pd.DatetimeIndex(source.index).equals(pd.DatetimeIndex(part["dates"])):
            raise FinalHoldoutError(
                f"Holdout dates for {part['scope']} {part['item_id'] or ''} do not match the forecast dates "
                f"({source.index[0].date()}..{source.index[-1].date()} vs {part['dates'][0].date()}..{part['dates'][-1].date()})."
            )
        for day, actual, predicted in zip(part["dates"], source.to_numpy(dtype=float), part["predicted"]):
            rows.append({"scope": part["scope"], "model": part["model"], "item_id": part["item_id"], "fold": 1,
                         "date": day, "actual": float(actual), "predicted": float(predicted),
                         "success": part["success"], "error": part["error"]})
    forecast_table = pd.DataFrame(rows)

    lag = int(config["comparison"]["mase_seasonal_lag"])
    headline = {scope: frozen["scopes"][scope]["headline_model"] for scope in SCOPES}
    results: List[pd.DataFrame] = []

    aggregate_rows = forecast_table[forecast_table["scope"] == "aggregate"]
    scale = seasonal_naive_scale(inputs.aggregate_history.to_numpy(dtype=float), lag)
    metrics = _score(aggregate_rows, pd.DataFrame([{"fold": 1, "mase_scale": scale}]), ["model", "fold"])
    metrics["scope"] = "aggregate"
    metrics["n_series"] = 1
    results.append(metrics)

    item_rows = forecast_table[forecast_table["scope"] == "item_level"]
    scale_table = pd.DataFrame(
        [{"item_id": item, "fold": 1, "mase_scale": seasonal_naive_scale(series.to_numpy(dtype=float), lag)}
         for item, series in inputs.item_histories.items()]
    )
    per_item = _score(item_rows, scale_table, ["item_id", "model", "fold"])
    macro_rows = []
    for model, group in per_item.groupby("model", sort=False):
        ok = bool(group["success"].astype(bool).all())
        record: Dict[str, Any] = {"model": model, "fold": 1, "success": ok, "n": int(group["n"].sum()),
                                  "n_series": int(len(group))}
        for metric in ("MAE", "RMSE", "MAPE", "WAPE", "MASE", "bias"):
            record[metric] = float(group[metric].mean()) if ok and group[metric].notna().any() else float("nan")
        record["mape_n_excluded"] = int(group["mape_n_excluded"].fillna(0).sum()) if ok else float("nan")
        macro_rows.append(record)
    macro = pd.DataFrame(macro_rows)
    macro["scope"] = "item_level"
    results.append(macro)

    table = pd.concat(results, ignore_index=True)
    table["headline"] = [model == headline[scope] for scope, model in zip(table["scope"], table["model"])]
    errors = forecast_table.dropna(subset=["error"]).groupby(["scope", "model"])["error"].first()
    table["error"] = [errors.get((scope, model)) for scope, model in zip(table["scope"], table["model"])]
    columns = ["scope", "model", "headline", "n_series", "n", "MAE", "RMSE", "MAPE", "mape_n_excluded", "WAPE",
               "MASE", "bias", "success", "error"]
    table = table[columns].rename(columns={"n": "n_days_scored"})
    table["averaging"] = np.where(table["scope"] == "item_level", "equal-weight mean over items", "single aggregate series")
    return forecast_table, table


def run(
    config: Optional[Mapping[str, Any]] = None,
    *,
    confirm: Optional[str] = None,
    override_existing: bool = False,
    inputs: Optional[HoldoutInputs] = None,
    factories: Optional[Mapping[str, Mapping[str, Factory]]] = None,
    now: Optional[datetime] = None,
) -> pd.DataFrame:
    """The one-time evaluation. See the module docstring for the refusals it enforces. Returns the results table."""
    config = load_config() if config is None else config
    frozen = _check_refusals(config, confirm, override_existing)
    inputs = load_holdout_inputs(config) if inputs is None else inputs
    factories = default_factories(config, inputs.calendar) if factories is None else factories

    forecast_table, results = evaluate_frozen_candidates(frozen, config, inputs, factories)

    previous_count = 0
    record_file = record_path(config)
    if record_file.exists():
        previous_count = int(read_json(record_file).get("evaluation_count", 1))
    elif results_path(config).exists():
        previous_count = 1
    count = previous_count + 1
    archived = _archive_existing(config, previous_count) if previous_count else []

    target_dir = final_dir(config)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_forecasts = forecast_table.copy()
    out_forecasts["date"] = pd.to_datetime(out_forecasts["date"]).dt.strftime("%Y-%m-%d")
    out_forecasts.drop(columns=["fold"]).to_csv(forecasts_path(config), index=False, lineterminator="\n")
    results.to_csv(results_path(config), index=False, lineterminator="\n")
    metadata = build_run_metadata(
        "final_holdout", config, FROZEN_CONFIG_SECTIONS,
        {
            "evaluation_count": count,
            "override_used": bool(previous_count),
            "archived_previous_files": archived,
            "frozen_file_sha256": file_sha256(frozen_path(config)),
            "frozen_at_utc": frozen["frozen_at_utc"],
            "holdout_window": frozen["reserved_holdout"],
            "headline": {scope: frozen["scopes"][scope]["headline_model"] for scope in SCOPES},
            "scoring": "MAE, RMSE, zero-protected MAPE, WAPE, bias, MASE (lag-7 scale from the development window); "
                       "item-level numbers are equal-weight means over items; aggregate and item-level are not comparable.",
        },
    )
    metadata["evaluation_count"] = count
    metadata["warning"] = REPEAT_WARNING if count > 1 else None
    write_json(record_file, metadata)
    if count > 1:
        print(REPEAT_WARNING, file=sys.stderr)
    return results


# ----------------------------------------------------------------------------------------------------- command line

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.final_holdout", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show what is frozen and what is missing (read-only)")
    freeze_parser = sub.add_parser("freeze", help="record the candidates and the development-based headline model")
    freeze_parser.add_argument("--overwrite", action="store_true", help="replace a freeze that has not been evaluated yet")
    run_parser = sub.add_parser("run", help="evaluate the frozen candidates on the reserved holdout (one time)")
    run_parser.add_argument("--confirm", default=None, help=f"must be exactly {CONFIRMATION_PHRASE}")
    run_parser.add_argument("--override-existing", action="store_true", help="repeat an evaluation that already exists")
    args = parser.parse_args(argv)
    config = load_config()
    try:
        if args.command == "status":
            print(status(config).to_text())
        elif args.command == "freeze":
            record = freeze(config, overwrite=args.overwrite)
            print(f"Frozen {sum(len(s['candidates']) for s in record['scopes'].values())} candidates into "
                  f"'{_display(frozen_path(config))}'. No holdout value was read.")
            for scope in SCOPES:
                print(f"  {scope}: headline {record['scopes'][scope]['headline_model']} "
                      f"(lowest mean development-fold MAE); all {len(record['scopes'][scope]['candidates'])} are reported.")
            print(f"Next: python -m src.evaluation.final_holdout run --confirm {CONFIRMATION_PHRASE}")
        else:
            results = run(config, confirm=args.confirm, override_existing=args.override_existing)
            print(results.to_string(index=False))
            print(f"Written to '{_display(final_dir(config))}'. The holdout has now been evaluated; do not tune on it.")
    except FinalHoldoutError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
