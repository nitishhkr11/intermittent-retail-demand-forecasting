"""Checkpointed full-panel XGBoost feature ablations on the shared seasonal folds.

All feature rows are built from a fold's training prefix. Forecasts are recursive;
actual sales in a scored fold never enter a lag or rolling statistic. The loader
uses Parquet date predicates so reserved-period sales are not read by this study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.demand_classification import STANDARD_CATEGORIES, classify_series
from src.evaluation.expanded_seasonal import OUTPUT_DIR as SEASONAL_DIR, seasonal_origins
from src.evaluation.forecast_artifacts import seasonal_naive_scale
from src.evaluation.intermittent_experiment import load_all_item_development_sales, panel_path
from src.models.xgboost_model import (
    PooledXGBoostForecaster, XGBoostConfig, XGBoostForecaster, load_event_calendar,
)
from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.provenance import canonical_json, file_sha256, write_json


OUTPUT_DIR = Path("artifacts/expanded_xgboost")
CHECKPOINT_DIR = Path("artifacts/checkpoints/expanded_xgboost")
SAMPLE_SEED = 4207
ITEM_SAMPLE_PER_CLASS = 50
PER_ITEM_WORKERS = 8

FEATURE_VARIANTS: tuple[dict[str, Any], ...] = (
    {"key": "lags_only", "rolling": False, "calendar": False, "events": False,
     "objective": "reg:squarederror", "scope": "pooled"},
    {"key": "lags_rolling", "rolling": True, "calendar": False, "events": False,
     "objective": "reg:squarederror", "scope": "pooled"},
    {"key": "lags_calendar", "rolling": False, "calendar": True, "events": False,
     "objective": "reg:squarederror", "scope": "pooled"},
    {"key": "lags_calendar_events", "rolling": True, "calendar": True, "events": True,
     "objective": "reg:squarederror", "scope": "pooled"},
    {"key": "lags_calendar_events_poisson", "rolling": True, "calendar": True, "events": True,
     "objective": "count:poisson", "scope": "pooled"},
    {"key": "per_item_lags_calendar_events", "rolling": True, "calendar": True, "events": True,
     "objective": "reg:squarederror", "scope": "per_item"},
)


def _digest_frame(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def stratified_item_sample(
    wide: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    *,
    per_class: int = ITEM_SAMPLE_PER_CLASS,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """Choose a fixed sample using only the earliest fold's training prefix."""
    if per_class < 1 or not folds:
        raise ValueError("per_class and folds must be non-empty and positive")
    first = min(folds, key=lambda row: int(row["test_start_pos"]))
    end = int(first["test_start_pos"])
    train = wide.iloc[:end]
    if train.index[-1].date().isoformat() != str(first["train_end"]):
        raise AssertionError("sample class labels do not match the first fold's training cutoff")
    labels = {str(item): classify_series(train[item]) ["demand_class"] for item in train.columns}
    rng = np.random.default_rng(seed)
    rows = []
    for label in STANDARD_CATEGORIES:
        available = sorted(item for item, item_class in labels.items() if item_class == label)
        count = min(per_class, len(available))
        picked = sorted(rng.choice(available, size=count, replace=False).tolist()) if count else []
        rows.extend({"item_id": item, "sample_class_at_first_origin": label,
                     "sample_origin": str(first["train_end"]), "sample_seed": seed,
                     "class_available": len(available), "sampled_from_class": count} for item in picked)
    return pd.DataFrame(rows)


def _settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    section = config["expanded_seasonal"]
    xgb = XGBoostConfig.from_config(config)
    return {
        "horizon": int(section["horizon"]), "years": [int(v) for v in section["years"]],
        "minimum_training_days": int(section["min_train_days"]), "fold_manifest_sha256": None,
        "feature_variants": [dict(row) for row in FEATURE_VARIANTS],
        "xgboost_hyperparameters": xgb.regressor_params(), "lags": list(xgb.lags),
        "rolling_windows": list(xgb.rolling_mean_windows), "sample_seed": SAMPLE_SEED,
        "per_item_sample_per_class": ITEM_SAMPLE_PER_CLASS,
        "per_item_workers": PER_ITEM_WORKERS,
    }


def prepare_manifest(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Write fold and per-item sample plans before fitting any ablation model."""
    config = load_config() if config is None else config
    wide, window = load_all_item_development_sales(config)
    seasonal_path = resolve_path(SEASONAL_DIR / "fold_manifest.json")
    if not seasonal_path.exists():
        raise FileNotFoundError("prepare the shared 16-fold seasonal manifest first")
    seasonal = json.loads(seasonal_path.read_text(encoding="utf-8"))
    if seasonal["development_window"]["start"] != window.dev_start.date().isoformat() or seasonal["development_window"]["end"] != window.dev_end.date().isoformat():
        raise RuntimeError("item panel and aggregate seasonal development windows differ")
    folds = seasonal_origins(
        wide.index, years=config["expanded_seasonal"]["years"],
        horizon=int(config["expanded_seasonal"]["horizon"]),
        min_train_days=int(config["expanded_seasonal"]["min_train_days"]),
    )
    sample = stratified_item_sample(wide, folds)
    settings = _settings(config)
    settings["fold_manifest_sha256"] = seasonal["manifest_sha256"]
    payload = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_scope": "CA_1 / FOODS_3 all 823 item series; development values through 2016-04-24",
        "development_window": {"start": window.dev_start.date().isoformat(), "end": window.dev_end.date().isoformat(),
                               "days": window.n_development_days},
        "development_panel_sha256": _digest_frame(wide),
        "calendar_sha256": hashlib.sha256(pd.util.hash_pandas_object(
            load_event_calendar(panel_path(config), last_date=window.dev_end), index=True).to_numpy().tobytes()).hexdigest(),
        "source_code_sha256": {"experiment": file_sha256(Path(__file__)),
                               "model": file_sha256(Path("src/models/xgboost_model.py")),
                               "data_loader": file_sha256(Path("src/evaluation/intermittent_experiment.py"))},
        "settings": settings, "folds": folds, "per_item_sample": sample.to_dict(orient="records"),
        "per_item_sample_n": len(sample), "holdout_sales_used": False,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    out = resolve_path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "fold_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("manifest_sha256") != payload["manifest_sha256"]:
            raise FileExistsError(f"a different XGBoost ablation plan already exists: {manifest_path}")
        return old
    write_json(manifest_path, payload)
    sample.to_csv(out / "per_item_sample.csv", index=False, lineterminator="\n")
    pd.DataFrame(folds).drop(columns=["test_start_pos", "test_end_pos"]).to_csv(
        out / "folds.csv", index=False, lineterminator="\n")
    return payload


def _variant_config(base: XGBoostConfig, spec: Mapping[str, Any]) -> XGBoostConfig:
    from dataclasses import replace
    return replace(base, use_rolling_features=bool(spec["rolling"]),
                   use_calendar_features=bool(spec["calendar"]),
                   use_event_calendar=bool(spec["events"]), objective=str(spec["objective"]))


def _checkpoint_path(key: str, fold: int, item_id: Optional[str] = None) -> Path:
    safe = "" if item_id is None else "_" + hashlib.sha1(item_id.encode()).hexdigest()[:12]
    return resolve_path(CHECKPOINT_DIR) / f"{key}{safe}_fold{fold}.json"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)


def _item_metrics(item_id: str, label: str, fold_id: int, year: int, quarter: int,
                  origin: str, dates: pd.DatetimeIndex, actual: np.ndarray,
                  predicted: Optional[np.ndarray], train: np.ndarray, model: str,
                  error: Optional[str] = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ok = predicted is not None and len(predicted) == len(actual) and np.isfinite(predicted).all()
    if ok:
        abs_error = np.abs(actual - predicted)
        scale = seasonal_naive_scale(train, lag=7)
        wape = float(abs_error.sum() / np.abs(actual).sum()) if np.abs(actual).sum() else None
        mase = float(abs_error.mean() / scale) if scale is not None and np.isfinite(scale) and scale > 0 else None
        metrics = {"MAE": float(abs_error.mean()), "RMSE": float(np.sqrt(np.mean((actual-predicted)**2))),
                   "WAPE": wape, "MASE": mase, "bias": float(np.mean(predicted-actual)),
                   "forecast_zero_share": float(np.mean(predicted == 0))}
    else:
        metrics = {key: None for key in ("MAE", "RMSE", "WAPE", "MASE", "bias", "forecast_zero_share")}
    row = {"item_id": item_id, "fold": fold_id, "year": year, "quarter": quarter,
           "fold_local_class": label, "model": model, "origin_date": origin,
           "forecast_start": dates[0].date().isoformat(), "forecast_end": dates[-1].date().isoformat(),
           "n_train": len(train), "n_scored": len(actual), "success": bool(ok), "error": error, **metrics}
    forecasts = [{"item_id": item_id, "fold": fold_id, "year": year, "quarter": quarter,
                  "fold_local_class": label, "model": model, "origin_date": origin,
                  "date": date.date().isoformat(), "actual": float(obs),
                  "predicted": float(pred) if ok else None, "success": bool(ok), "error": error}
                 for date, obs, pred in zip(dates, actual, predicted if ok else np.full(len(actual), np.nan))]
    return row, forecasts


def _fit_pooled(spec: Mapping[str, Any], fold: Mapping[str, Any], wide: pd.DataFrame,
                calendar: pd.DataFrame, base: XGBoostConfig, manifest_sha: str) -> Dict[str, Any]:
    fold_id, start, end = int(fold["fold"]), int(fold["test_start_pos"]), int(fold["test_end_pos"])
    key = str(spec["key"])
    path = _checkpoint_path(key, fold_id)
    train_wide, test_wide = wide.iloc[:start], wide.iloc[start:end]
    fp = hashlib.sha256(canonical_json({"manifest": manifest_sha, "model": key, "fold": fold_id,
                                        "train_hash": _digest_frame(train_wide)}).encode()).hexdigest()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fp:
            raise RuntimeError(f"XGBoost checkpoint fingerprint mismatch: {path}")
        return saved["result"]
    histories = {str(c): pd.Series(train_wide[c].to_numpy(float), index=train_wide.index, name=str(c))
                 for c in train_wide.columns}
    cfg = _variant_config(base, spec)
    name = f"XGBoost(pooled,{key})"
    if spec["objective"] == "count:poisson":
        name = f"XGBoost(pooled,{key},poisson)"
    started = time.perf_counter()
    metrics, forecasts, importances, diagnostics = [], [], [], None
    try:
        model = PooledXGBoostForecaster(cfg, event_calendar=calendar, name=name).fit(histories)
        predicted = model.predict(len(test_wide))
        diagnostics = model.fit_diagnostics()
        importance = model.feature_importance("gain")
        importances = [{"model": name, "fold": fold_id, "feature": str(feature), "gain": float(value)}
                       for feature, value in importance.items()]
        actual_dates = test_wide.index
        for item_id in test_wide.columns:
            train = train_wide[item_id].to_numpy(float)
            actual = test_wide[item_id].to_numpy(float)
            label = model.item_class_table().set_index("item_id").loc[str(item_id), "demand_class"]
            row, fc = _item_metrics(str(item_id), str(label), fold_id, int(fold["year"]), int(fold["quarter"]),
                                    str(fold["train_end"]), actual_dates, actual,
                                    np.asarray(predicted[str(item_id)], float), train, name)
            metrics.append(row); forecasts.extend(fc)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for item_id in test_wide.columns:
            train = train_wide[item_id].to_numpy(float); actual = test_wide[item_id].to_numpy(float)
            label = classify_series(pd.Series(train))["demand_class"]
            row, fc = _item_metrics(str(item_id), str(label), fold_id, int(fold["year"]), int(fold["quarter"]),
                                    str(fold["train_end"]), test_wide.index, actual, None, train, name, error)
            metrics.append(row); forecasts.extend(fc)
        diagnostics = {"fit_seconds": time.perf_counter()-started, "error": error}
    result = {"model": name, "variant": key, "scope": "pooled", "fold": fold_id,
              "success": bool(all(r["success"] for r in metrics)), "fit_seconds": time.perf_counter()-started,
              "metrics": metrics, "forecasts": forecasts, "feature_importance": importances,
              "diagnostics": diagnostics}
    _atomic_json(path, {"fingerprint": fp, "result": result})
    return result


def _fit_per_item(spec: Mapping[str, Any], fold: Mapping[str, Any], item_id: str,
                  wide: pd.DataFrame, calendar: pd.DataFrame, base: XGBoostConfig,
                  manifest_sha: str) -> Dict[str, Any]:
    fold_id, start, end = int(fold["fold"]), int(fold["test_start_pos"]), int(fold["test_end_pos"])
    key = str(spec["key"]); path = _checkpoint_path(key, fold_id, item_id)
    train, actual = wide[item_id].iloc[:start], wide[item_id].iloc[start:end]
    fp = hashlib.sha256(canonical_json({"manifest": manifest_sha, "model": key, "fold": fold_id,
                                        "item": item_id, "train_hash": _digest_frame(train.to_frame())}).encode()).hexdigest()
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fp:
            raise RuntimeError(f"XGBoost checkpoint fingerprint mismatch: {path}")
        return saved["result"]
    label = classify_series(train)["demand_class"]
    model_name = "XGBoost(per-item,lags_calendar_events)"
    started = time.perf_counter()
    try:
        cfg = _variant_config(base, spec)
        model = XGBoostForecaster(cfg, event_calendar=calendar)
        prediction = model.fit(train).predict(len(actual))
        diag = model.fit_diagnostics()
        importance = model.feature_importance("gain")
        row, forecasts = _item_metrics(item_id, label, fold_id, int(fold["year"]), int(fold["quarter"]),
                                       str(fold["train_end"]), actual.index, actual.to_numpy(float),
                                       np.asarray(prediction, float), train.to_numpy(float), model_name)
        error = None
        importance_rows = [{"model": model_name, "fold": fold_id, "item_id": item_id,
                            "feature": str(feature), "gain": float(value)} for feature, value in importance.items()]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        row, forecasts = _item_metrics(item_id, label, fold_id, int(fold["year"]), int(fold["quarter"]),
                                       str(fold["train_end"]), actual.index, actual.to_numpy(float),
                                       None, train.to_numpy(float), model_name, error)
        diag = {"fit_seconds": time.perf_counter()-started, "error": error}
        importance_rows = []
    result = {"model": model_name, "variant": key, "scope": "per_item", "fold": fold_id,
              "item_id": item_id, "success": row["success"], "fit_seconds": time.perf_counter()-started,
              "metric": row, "forecasts": forecasts, "feature_importance": importance_rows,
              "diagnostics": diag}
    _atomic_json(path, {"fingerprint": fp, "result": result})
    return result


def _load_saved_results(manifest: Mapping[str, Any], wide: pd.DataFrame) -> List[Dict[str, Any]]:
    results = []
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        for spec in FEATURE_VARIANTS:
            if spec["scope"] == "pooled":
                path = _checkpoint_path(str(spec["key"]), fold_id)
                saved = json.loads(path.read_text(encoding="utf-8"))["result"]
                results.append(saved)
            else:
                for item_id in manifest["per_item_sample"]:
                    path = _checkpoint_path(str(spec["key"]), fold_id, item_id["item_id"])
                    saved = json.loads(path.read_text(encoding="utf-8"))["result"]
                    results.append(saved)
    return results


def run_experiment(config: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    config = load_config() if config is None else config
    out, checkpoints = resolve_path(OUTPUT_DIR), resolve_path(CHECKPOINT_DIR)
    manifest_path = out / "fold_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run `python -m src.evaluation.expanded_xgboost prepare` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wide, window = load_all_item_development_sales(config)
    if _digest_frame(wide) != manifest["development_panel_sha256"] or window.dev_end.date().isoformat() != manifest["development_window"]["end"]:
        raise RuntimeError("development data no longer matches the predeclared XGBoost study")
    if manifest.get("source_code_sha256", {}).get("experiment") != file_sha256(Path(__file__)):
        raise RuntimeError("XGBoost experiment source changed after manifest creation; create a new versioned study")
    sample_ids = {str(row["item_id"]) for row in manifest["per_item_sample"]}
    if not sample_ids.issubset(set(wide.columns)):
        raise RuntimeError("predeclared per-item sample no longer matches the panel")
    calendar = load_event_calendar(panel_path(config), last_date=window.dev_end)
    base = XGBoostConfig.from_config(config)
    for fold in manifest["folds"]:
        for spec in FEATURE_VARIANTS:
            if spec["scope"] == "pooled":
                result = _fit_pooled(spec, fold, wide, calendar, base, manifest["manifest_sha256"])
                print(f"{spec['key']} fold {fold['fold']}: {'ok' if result['success'] else 'failed'} ({result['fit_seconds']:.1f}s)", flush=True)
            else:
                tasks = []
                with ThreadPoolExecutor(max_workers=PER_ITEM_WORKERS) as pool:
                    for row in manifest["per_item_sample"]:
                        tasks.append(pool.submit(_fit_per_item, spec, fold, str(row["item_id"]), wide, calendar,
                                                 base, manifest["manifest_sha256"]))
                    finished = 0
                    for task in as_completed(tasks):
                        task.result()
                        finished += 1
                        if finished % 25 == 0 or finished == len(tasks):
                            print(f"per-item full-feature fold {fold['fold']}: {finished}/{len(tasks)} products checkpointed", flush=True)

    results = _load_saved_results(manifest, wide)
    metric_rows, forecast_rows, importance_rows, diagnostic_rows = [], [], [], []
    for result in results:
        if result["scope"] == "pooled":
            metric_rows.extend(result["metrics"]); forecast_rows.extend(result["forecasts"])
            importance_rows.extend(result["feature_importance"])
            diagnostic_rows.append({"model": result["model"], "fold": result["fold"],
                                    "scope": result["scope"], **(result.get("diagnostics") or {})})
        else:
            metric_rows.append(result["metric"]); forecast_rows.extend(result["forecasts"])
            importance_rows.extend(result["feature_importance"])
            diag = result.get("diagnostics") or {}
            diagnostic_rows.append({"model": result["model"], "fold": result["fold"],
                                    "item_id": result["item_id"], "scope": result["scope"], **diag})
    metrics = pd.DataFrame(metric_rows)
    forecasts = pd.DataFrame(forecast_rows)
    importance = pd.DataFrame(importance_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = metrics.groupby(["model"], dropna=False).agg(
        item_fold_cells=("success", "size"), item_fold_success=("success", "sum"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE_mean=("RMSE", "mean"),
        RMSE_sd=("RMSE", "std"), WAPE_mean=("WAPE", "mean"), MASE_mean=("MASE", "mean"),
        bias_mean=("bias", "mean"), forecast_zero_share_mean=("forecast_zero_share", "mean"),
    ).reset_index()
    fold_summary = metrics.groupby(["model", "fold"], dropna=False).agg(
        item_fold_cells=("success", "size"), item_fold_success=("success", "sum"),
        MAE_mean=("MAE", "mean"), RMSE_mean=("RMSE", "mean"), WAPE_mean=("WAPE", "mean"),
        MASE_mean=("MASE", "mean"), bias_mean=("bias", "mean"),
    ).reset_index()
    class_summary = metrics.groupby(["model", "fold_local_class"], dropna=False).agg(
        items=("item_id", "nunique"), item_fold_cells=("success", "size"), item_fold_success=("success", "sum"),
        MAE_mean=("MAE", "mean"), RMSE_mean=("RMSE", "mean"), WAPE_mean=("WAPE", "mean"),
        MASE_mean=("MASE", "mean"), bias_mean=("bias", "mean"),
    ).reset_index()
    importance_summary = importance.groupby(["model", "feature"], dropna=False).agg(
        folds_reported=("fold", "nunique"), gain_mean=("gain", "mean"), gain_sd=("gain", "std"),
    ).reset_index() if len(importance) else importance
    outputs = {
        "item_fold_metrics.csv": metrics, "fold_metrics.csv": fold_summary,
        "class_fold_metrics.csv": class_summary, "forecasts.csv": forecasts,
        "feature_importance_by_fold.csv": importance, "feature_importance_summary.csv": importance_summary,
        "fit_diagnostics.csv": diagnostics, "model_summary.csv": summary,
    }
    for filename, frame in outputs.items():
        path = out / filename
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing XGBoost output: {path}")
        frame.to_csv(path, index=False, lineterminator="\n")
    write_json(out / "run_metadata.json", {
        "component": "expanded_xgboost_feature_ablation", "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": manifest["manifest_sha256"], "development_panel_sha256": manifest["development_panel_sha256"],
        "calendar_sha256": manifest["calendar_sha256"], "source_code_sha256": manifest["source_code_sha256"],
        "folds": 16, "horizon_days": 28, "products": int(wide.shape[1]),
        "per_item_sample_products": len(sample_ids), "variants": [dict(spec) for spec in FEATURE_VARIANTS],
        "pooled_variant_folds_planned": sum(spec["scope"] == "pooled" for spec in FEATURE_VARIANTS)*16,
        "per_item_item_fold_fits_planned": len(sample_ids)*16,
        "failed_item_fold_rows": int((~metrics["success"].astype(bool)).sum()),
        "holdout_sales_used": False, "fit_checkpointed": True,
    })
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_manifest()
        print(f"prepared {len(manifest['folds'])} folds and {manifest['per_item_sample_n']} stratified per-item series")
    else:
        print(run_experiment().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
