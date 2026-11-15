"""Seasonally distributed, fold-local classical model-selection experiment.

The experiment reads only the development series returned by
``load_aggregate_development``. A quarterly manifest is written before any fit;
each outer 28-day window is disjoint, and ARIMA/SARIMA/Holt-Winters settings are
selected using a 28-day validation tail contained strictly inside that fold's
training prefix. Existing Phase 2 outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.aggregate_series import load_aggregate_development
from src.evaluation.metrics import compute_metrics
from src.evaluation.forecast_artifacts import seasonal_naive_scale
from src.models.arima import ARIMAForecaster
from src.models.baseline import SeasonalNaive
from src.models.holt_winters import HoltWintersForecaster
from src.models.sarima import SARIMAForecaster
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT, resolve_path
from src.utils.provenance import canonical_json, file_sha256, write_json


OUTPUT_DIR = Path("artifacts/expanded_seasonal")
CHECKPOINT_DIR = Path("artifacts/checkpoints/expanded_seasonal")
MANIFEST_NAME = "fold_manifest.json"
MANIFEST_CSV_NAME = "fold_manifest.csv"
FORECAST_NAME = "forecasts.csv"
METRICS_NAME = "fold_metrics.csv"
SUMMARY_NAME = "model_summary.csv"
COMPARISON_NAME = "legacy_vs_fold_local.csv"
METADATA_NAME = "run_metadata.json"


def seasonal_origins(
    dates: pd.DatetimeIndex,
    *,
    years: Sequence[int] = (2012, 2013, 2014, 2015),
    horizon: int = 28,
    min_train_days: int = 365,
) -> List[Dict[str, Any]]:
    """Create a reproducible quarter-spread set of disjoint forecast windows."""
    dates = pd.DatetimeIndex(dates)
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("dates must be unique and ascending")
    if len(dates) < min_train_days + horizon:
        raise ValueError("development period is too short for the requested history and horizon")
    if horizon <= 0 or min_train_days <= 0:
        raise ValueError("horizon and min_train_days must be positive")

    rows: List[Dict[str, Any]] = []
    q_starts = ((1, 1), (4, 1), (7, 1), (10, 1))
    first_allowed = dates[min_train_days]
    for year in years:
        for quarter, (month, day) in enumerate(q_starts, start=1):
            nominal = pd.Timestamp(year=int(year), month=month, day=day)
            candidate = max(nominal, first_allowed)
            q_end = nominal + pd.offsets.QuarterEnd(startingMonth=12)
            if candidate > q_end:
                continue
            start_pos = int(dates.searchsorted(candidate))
            if start_pos >= len(dates):
                continue
            start = dates[start_pos]
            end_pos = start_pos + horizon
            if end_pos > len(dates) or dates[end_pos - 1] > q_end:
                continue
            rows.append(
                {
                    "fold": len(rows) + 1,
                    "year": int(year),
                    "quarter": quarter,
                    "train_start": dates[0].date().isoformat(),
                    "train_end": dates[start_pos - 1].date().isoformat(),
                    "training_days": start_pos,
                    "forecast_start": start.date().isoformat(),
                    "forecast_end": dates[end_pos - 1].date().isoformat(),
                    "test_start_pos": start_pos,
                    "test_end_pos": end_pos,
                    "horizon": horizon,
                    "minimum_training_days": min_train_days,
                }
            )
    if len(rows) < 12:
        raise ValueError(f"only {len(rows)} valid seasonal origins were generated; at least 12 are required")
    for previous, current in zip(rows, rows[1:]):
        if current["test_start_pos"] < previous["test_end_pos"]:
            raise AssertionError("seasonal forecast windows overlap")
    return rows


def _development_digest(series: pd.Series) -> str:
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes())
    return hasher.hexdigest()


def _settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    section = config["expanded_seasonal"]
    return {
        "horizon": int(section["horizon"]),
        "years": [int(x) for x in section["years"]],
        "min_train_days": int(section["min_train_days"]),
        "inner_validation_days": int(section["inner_validation_days"]),
        "arima_orders": section["arima_orders"],
        "sarima_candidates": section["sarima_candidates"],
        "holt_winters_trends": section["holt_winters_trends"],
        "legacy": config["models"],
        "season_length": int(config["backtest"]["season_length"]),
    }


def _paths(config: Mapping[str, Any]) -> Tuple[Path, Path]:
    return resolve_path(OUTPUT_DIR), resolve_path(CHECKPOINT_DIR)


def prepare_manifest(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Persist the fold plan and its development-only fingerprint before model execution."""
    config = load_config() if config is None else config
    series, window = load_aggregate_development(config)
    settings = _settings(config)
    folds = seasonal_origins(
        series.index,
        years=settings["years"],
        horizon=settings["horizon"],
        min_train_days=settings["min_train_days"],
    )
    code_path = Path(__file__)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "development_window": {
            "start": window.dev_start.date().isoformat(),
            "end": window.dev_end.date().isoformat(),
            "n_days": window.n_development_days,
        },
        "data_scope": "aggregate CA_1 / FOODS_3; development prefix only",
        "development_series_sha256": _development_digest(series),
        "config_sha256": hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest(),
        "code_sha256": file_sha256(code_path),
        "settings": settings,
        "folds": folds,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}).encode("utf-8")
    ).hexdigest()
    output_dir, _ = _paths(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_NAME
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise FileExistsError(
                f"A different fold manifest already exists at {path}; archive it explicitly before preparing a new plan."
            )
        return old
    write_json(path, manifest)
    pd.DataFrame(folds).drop(columns=["test_start_pos", "test_end_pos"]).to_csv(
        output_dir / MANIFEST_CSV_NAME, index=False, lineterminator="\n"
    )
    return manifest


def _model(family: str, spec: Mapping[str, Any], config: Mapping[str, Any]):
    if family == "SeasonalNaive":
        return SeasonalNaive(season_length=int(config["backtest"]["season_length"]))
    if family == "ARIMA":
        return ARIMAForecaster(order=tuple(int(x) for x in spec["order"]))
    if family == "SARIMA":
        return SARIMAForecaster(
            order=tuple(int(x) for x in spec["order"]),
            seasonal_order=tuple(int(x) for x in spec["seasonal_order"]),
        )
    if family == "HoltWinters":
        return HoltWintersForecaster(trend=spec.get("trend"), seasonal_periods=int(spec["seasonal_periods"]))
    raise ValueError(f"unsupported model family {family!r}")


def _candidate_specs(config: Mapping[str, Any], family: str) -> List[Dict[str, Any]]:
    section = config["expanded_seasonal"]
    if family == "ARIMA":
        return [{"order": [int(x) for x in order]} for order in section["arima_orders"]]
    if family == "SARIMA":
        return [
            {"order": [int(x) for x in row["order"]], "seasonal_order": [int(x) for x in row["seasonal_order"]]}
            for row in section["sarima_candidates"]
        ]
    if family == "HoltWinters":
        return [
            {"trend": trend, "seasonal_periods": int(config["models"]["holt_winters_seasonal_periods"])}
            for trend in section["holt_winters_trends"]
        ]
    return []


def _select_parameters(
    family: str, train: pd.Series, config: Mapping[str, Any]
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Choose settings by MAE on the training prefix's last chronological validation tail."""
    validation_days = int(config["expanded_seasonal"]["inner_validation_days"])
    if len(train) <= validation_days + 28:
        raise ValueError("training prefix is too short for inner validation")
    inner_train, validation = train.iloc[:-validation_days], train.iloc[-validation_days:]
    rows: List[Dict[str, Any]] = []
    for spec in _candidate_specs(config, family):
        result: Dict[str, Any] = {"parameters": spec, "success": False, "validation_mae": None, "error": None}
        try:
            model = _model(family, spec, config).fit(inner_train)
            prediction = np.asarray(model.predict(validation_days), dtype=float)
            if prediction.shape != (validation_days,) or not np.isfinite(prediction).all():
                raise ValueError("inner validation prediction has invalid shape or non-finite values")
            result["validation_mae"] = float(np.mean(np.abs(validation.to_numpy(dtype=float) - prediction)))
            result["success"] = True
        except Exception as exc:  # retain candidate failures in the tuning record
            result["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(result)
    valid = [r for r in rows if r["success"]]
    if not valid:
        raise RuntimeError(f"all {family} inner candidates failed")
    winner = min(valid, key=lambda r: (r["validation_mae"], canonical_json(r["parameters"])))
    return dict(winner["parameters"]), float(winner["validation_mae"]), rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def run_experiment(config: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """Run or resume the expanded seasonal comparison without touching prior artifacts."""
    config = load_config() if config is None else config
    output_dir, checkpoint_dir = _paths(config)
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError("fold manifest missing; run `python -m src.evaluation.expanded_seasonal prepare` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    series, _ = load_aggregate_development(config)
    if _development_digest(series) != manifest["development_series_sha256"]:
        raise RuntimeError("development series changed after the fold manifest was created")
    if file_sha256(Path(__file__)) != manifest["code_sha256"]:
        raise RuntimeError("experiment code changed after the fold manifest was created")
    settings = _settings(config)
    if hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest() != manifest["config_sha256"]:
        raise RuntimeError("experiment settings changed after the fold manifest was created")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    legacy = config["models"]
    plan: List[Tuple[str, str, Dict[str, Any]]] = [
        ("SeasonalNaive", "fixed_baseline", {}),
        ("ARIMA", "legacy_fixed", {"order": list(legacy["arima_order"])}),
        ("SARIMA", "legacy_fixed", {"order": list(legacy["sarima_order"]), "seasonal_order": list(legacy["sarima_seasonal_order"])}),
        ("HoltWinters", "legacy_fixed", {"trend": "add", "seasonal_periods": int(legacy["holt_winters_seasonal_periods"])}),
    ]
    for family in ("ARIMA", "SARIMA", "HoltWinters"):
        plan.append((family, "fold_local_selected", {}))

    records: List[Dict[str, Any]] = []
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        train = series.iloc[: int(fold["test_start_pos"])]
        test = series.iloc[int(fold["test_start_pos"]): int(fold["test_end_pos"])]
        for family, selection, fixed_spec in plan:
            checkpoint = checkpoint_dir / f"fold-{fold_id:02d}--{_slug(family + '-' + selection)}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("manifest_sha256") != manifest["manifest_sha256"]:
                    raise RuntimeError(f"checkpoint {checkpoint.name} belongs to a different fold manifest")
                records.append(saved)
                continue
            selected_spec = fixed_spec
            inner_mae: Optional[float] = None
            tuning: List[Dict[str, Any]] = []
            success, error = True, None
            try:
                if selection == "fold_local_selected":
                    selected_spec, inner_mae, tuning = _select_parameters(family, train, config)
                forecaster = _model(family, selected_spec, config).fit(train)
                predicted = np.asarray(forecaster.predict(len(test)), dtype=float)
                if predicted.shape != (len(test),) or not np.isfinite(predicted).all():
                    raise ValueError("outer forecast has invalid shape or non-finite values")
                actual = test.to_numpy(dtype=float)
                metric = compute_metrics(actual, predicted)
                scale = seasonal_naive_scale(train.to_numpy(dtype=float), int(config["comparison"]["mase_seasonal_lag"]))
                wape = float(np.abs(actual - predicted).sum() / np.abs(actual).sum()) if np.abs(actual).sum() else None
                mase = float(metric.mae / scale) if scale is not None and np.isfinite(scale) and scale > 0 else None
                metrics = {
                    "MAE": metric.mae, "RMSE": metric.rmse, "MAPE": metric.mape,
                    "mape_n_excluded": metric.mape_n_excluded, "WAPE": wape,
                    "MASE": mase, "bias": float(np.mean(predicted - actual)),
                }
            except Exception as exc:
                success, error = False, f"{type(exc).__name__}: {exc}"
                predicted = np.full(len(test), np.nan)
                metrics = {key: None for key in ("MAE", "RMSE", "MAPE", "mape_n_excluded", "WAPE", "MASE", "bias")}
            saved = {
                "manifest_sha256": manifest["manifest_sha256"], "fold": fold_id,
                "year": int(fold["year"]), "quarter": int(fold["quarter"]),
                "model_family": family, "selection": selection,
                "parameters": selected_spec, "inner_validation_mae": inner_mae,
                "tuning_candidates": tuning, "success": success, "error": error,
                "origin_date": train.index[-1].date().isoformat(),
                "forecast_start": test.index[0].date().isoformat(), "forecast_end": test.index[-1].date().isoformat(),
                "n_train": len(train), "n_scored": len(test), "metrics": metrics,
                "dates": [d.date().isoformat() for d in test.index],
                "actual": test.to_numpy(dtype=float).tolist(), "predicted": predicted.tolist(),
            }
            _atomic_json(checkpoint, saved)
            records.append(saved)

    metrics_rows: List[Dict[str, Any]] = []
    forecast_rows: List[Dict[str, Any]] = []
    for record in records:
        metrics_rows.append({k: v for k, v in record.items() if k not in {"manifest_sha256", "metrics", "dates", "actual", "predicted", "tuning_candidates"}} | record["metrics"])
        for date, actual, predicted in zip(record["dates"], record["actual"], record["predicted"]):
            forecast_rows.append({
                "model_family": record["model_family"], "selection": record["selection"],
                "fold": record["fold"], "origin_date": record["origin_date"], "date": date,
                "actual": actual, "predicted": predicted, "success": record["success"], "error": record["error"],
            })
    metrics_table = pd.DataFrame(metrics_rows)
    forecasts = pd.DataFrame(forecast_rows)
    summary_rows = []
    for (family, selection), group in metrics_table.groupby(["model_family", "selection"], sort=False):
        successful = group[group["success"].astype(bool)]
        row = {"model_family": family, "selection": selection, "folds_total": len(group),
               "folds_scored": len(successful), "folds_failed": len(group) - len(successful)}
        for metric in ("MAE", "RMSE", "WAPE", "MASE", "bias"):
            row[f"{metric}_mean"] = float(successful[metric].mean()) if successful[metric].notna().any() else np.nan
            row[f"{metric}_std"] = float(successful[metric].std(ddof=1)) if successful[metric].notna().sum() > 1 else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    comparison = summary[summary["selection"].isin(["legacy_fixed", "fold_local_selected"])].copy()
    comparison["comparison_scope"] = "same 16 quarterly forecast windows; lower MAE is descriptive, not an inferential claim"

    final_files = [output_dir / n for n in (FORECAST_NAME, METRICS_NAME, SUMMARY_NAME, COMPARISON_NAME, METADATA_NAME)]
    if any(path.exists() for path in final_files):
        raise FileExistsError("expanded seasonal result files already exist; refusing to overwrite saved results")
    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_dir / FORECAST_NAME, index=False, lineterminator="\n")
    metrics_table.to_csv(output_dir / METRICS_NAME, index=False, lineterminator="\n")
    summary.to_csv(output_dir / SUMMARY_NAME, index=False, lineterminator="\n")
    comparison.to_csv(output_dir / COMPARISON_NAME, index=False, lineterminator="\n")
    write_json(output_dir / METADATA_NAME, {
        "component": "expanded_seasonal_classical_selection",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": manifest["manifest_sha256"],
        "development_series_sha256": manifest["development_series_sha256"],
        "folds": len(manifest["folds"]), "horizon_days": settings["horizon"],
        "models": ["SeasonalNaive", "ARIMA", "SARIMA", "HoltWinters"],
        "information_set": "historical sales only",
        "selection_metric": "inner chronological validation MAE",
        "failed_folds": int((~metrics_table["success"].astype(bool)).sum()),
        "holdout_accessed": False,
    })
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_manifest()
        print(f"Prepared {len(manifest['folds'])} folds; manifest SHA-256 {manifest['manifest_sha256']}.")
    else:
        summary = run_experiment()
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
