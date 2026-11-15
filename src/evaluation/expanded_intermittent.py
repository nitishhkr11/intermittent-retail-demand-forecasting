"""Seasonally distributed intermittent-demand experiment on the complete item panel.

Evaluates Seasonal Naive, Croston, SBA, and TSB on every available CA_1/FOODS_3
item using the predeclared expanded seasonal fold manifest. Existing 20-item
historical benchmark files are preserved. Results are chunk-checkpointed by item
so an interrupted run can resume without repeating completed products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.demand_classification import classify_series
from src.evaluation.metrics import compute_metrics
from src.evaluation.forecast_artifacts import seasonal_naive_scale
from src.evaluation.intermittent_experiment import load_all_item_development_sales
from src.models.baseline import SeasonalNaive
from src.models.croston import CrostonForecaster
from src.models.sba import SBAForecaster
from src.models.tsb import TSBForecaster
from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.provenance import canonical_json, file_sha256, write_json


OUTPUT_DIR = Path("artifacts/expanded_intermittent")
CHECKPOINT_DIR = Path("artifacts/checkpoints/expanded_intermittent")
CHUNK_SIZE = 25


def _series_digest(wide: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(wide, index=True).to_numpy().tobytes()).hexdigest()


def _factories(config: Mapping[str, Any]):
    model_config = config["models"]
    season = int(config["backtest"]["season_length"])
    return {
        "SeasonalNaive": lambda: SeasonalNaive(season_length=season),
        f"Croston(alpha={float(model_config['croston_alpha'])})": lambda: CrostonForecaster(alpha=float(model_config["croston_alpha"])),
        f"SBA(alpha={float(model_config['sba_alpha'])})": lambda: SBAForecaster(alpha=float(model_config["sba_alpha"])),
        f"TSB(alpha={float(model_config['tsb_alpha'])},beta={float(model_config['tsb_beta'])})": lambda: TSBForecaster(
            alpha=float(model_config["tsb_alpha"]), beta=float(model_config["tsb_beta"])
        ),
    }


def prepare_manifest(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = load_config() if config is None else config
    wide, window = load_all_item_development_sales(config)
    seasonal_path = resolve_path(Path("artifacts/expanded_seasonal/fold_manifest.json"))
    if not seasonal_path.exists():
        raise FileNotFoundError("prepare the shared expanded seasonal fold manifest first")
    seasonal = json.loads(seasonal_path.read_text(encoding="utf-8"))
    if seasonal["development_window"]["start"] != window.dev_start.date().isoformat() or seasonal["development_window"]["end"] != window.dev_end.date().isoformat():
        raise RuntimeError("aggregate and item development windows do not match")
    model_names = list(_factories(config))
    config_values = {
        "fold_manifest_sha256": seasonal["manifest_sha256"],
        "models": model_names,
        "horizon": int(config["expanded_seasonal"]["horizon"]),
        "n_items": int(wide.shape[1]),
        "chunk_size": CHUNK_SIZE,
    }
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_scope": "all CA_1 / FOODS_3 item series, development prefix only",
        "development_start": window.dev_start.date().isoformat(),
        "development_end": window.dev_end.date().isoformat(),
        "development_days": window.n_development_days,
        "development_panel_sha256": _series_digest(wide),
        "fold_manifest_sha256": seasonal["manifest_sha256"],
        "settings_sha256": hashlib.sha256(canonical_json(config_values).encode("utf-8")).hexdigest(),
        "code_sha256": file_sha256(Path(__file__)),
        "n_items": int(wide.shape[1]),
        "item_ids": [str(value) for value in wide.columns],
        "models": model_names,
        "folds": seasonal["folds"],
        "horizon_days": int(config["expanded_seasonal"]["horizon"]),
        "metrics": ["MAE", "RMSE", "WAPE", "MASE", "bias", "forecast_zero_share"],
        "holdout_accessed": False,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    output_dir = resolve_path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fold_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != payload["manifest_sha256"]:
            raise FileExistsError(f"different expanded intermittent manifest already exists: {path}")
        return existing
    write_json(path, payload)
    return payload


def _json_scalar(value: Any):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _run_item_chunk(
    item_ids: Sequence[str], wide: pd.DataFrame, folds: Sequence[Mapping[str, Any]],
    factories: Mapping[str, Any], manifest_sha256: str, lag: int,
) -> Dict[str, Any]:
    forecasts: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    for item_id in item_ids:
        values = wide[item_id].to_numpy(dtype=float)
        for fold in folds:
            fold_id = int(fold["fold"])
            start, end = int(fold["test_start_pos"]), int(fold["test_end_pos"])
            train, actual = values[:start], values[start:end]
            train_series = pd.Series(train, index=wide.index[:start], name=item_id)
            dates = wide.index[start:end]
            label = classify_series(train_series)["demand_class"]
            scale = seasonal_naive_scale(train, lag)
            for name, factory in factories.items():
                ok, error = True, None
                try:
                    model = factory().fit(train_series)
                    predicted = np.asarray(model.predict(len(actual)), dtype=float)
                    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
                        raise ValueError("forecast has invalid shape or non-finite values")
                    score = compute_metrics(actual, predicted)
                    wape = float(np.abs(actual - predicted).sum() / np.abs(actual).sum()) if np.abs(actual).sum() else None
                    mase = float(score.mae / scale) if scale is not None and np.isfinite(scale) and scale > 0 else None
                    values_metrics = {
                        "MAE": score.mae, "RMSE": score.rmse, "MAPE": score.mape,
                        "mape_n_excluded": score.mape_n_excluded, "WAPE": wape, "MASE": mase,
                        "bias": float(np.mean(predicted - actual)),
                        "forecast_zero_share": float(np.mean(predicted == 0.0)),
                        "all_zero_forecast": bool(np.all(predicted == 0.0)),
                    }
                except Exception as exc:
                    ok, error = False, f"{type(exc).__name__}: {exc}"
                    predicted = np.full(len(actual), np.nan)
                    values_metrics = {key: None for key in (
                        "MAE", "RMSE", "MAPE", "mape_n_excluded", "WAPE", "MASE", "bias",
                        "forecast_zero_share", "all_zero_forecast",
                    )}
                scores.append({
                    "item_id": item_id, "fold": fold_id, "year": int(fold["year"]),
                    "quarter": int(fold["quarter"]), "fold_local_class": label,
                    "model": name, "origin_date": fold["train_end"],
                    "forecast_start": fold["forecast_start"], "forecast_end": fold["forecast_end"],
                    "n_train": len(train), "n_scored": len(actual), "success": ok, "error": error,
                    **values_metrics,
                })
                for date, obs, pred in zip(dates, actual, predicted):
                    forecasts.append({
                        "item_id": item_id, "fold": fold_id, "year": int(fold["year"]),
                        "quarter": int(fold["quarter"]), "fold_local_class": label,
                        "model": name, "origin_date": fold["train_end"],
                        "date": date.date().isoformat(), "actual": float(obs),
                        "predicted": _json_scalar(pred), "success": ok, "error": error,
                    })
    return {"manifest_sha256": manifest_sha256, "item_ids": list(item_ids), "scores": scores, "forecasts": forecasts}


def run_experiment(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = load_config() if config is None else config
    output_dir = resolve_path(OUTPUT_DIR)
    checkpoint_dir = resolve_path(CHECKPOINT_DIR)
    manifest_path = output_dir / "fold_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("expanded intermittent manifest missing; run `prepare` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wide, _ = load_all_item_development_sales(config)
    if _series_digest(wide) != manifest["development_panel_sha256"]:
        raise RuntimeError("development item panel changed after manifest preparation")
    if file_sha256(Path(__file__)) != manifest["code_sha256"]:
        raise RuntimeError("expanded intermittent code changed after manifest preparation")
    seasonal = json.loads(resolve_path(Path("artifacts/expanded_seasonal/fold_manifest.json")).read_text(encoding="utf-8"))
    model_names = list(_factories(config))
    settings = {
        "fold_manifest_sha256": seasonal["manifest_sha256"],
        "models": model_names,
        "horizon": int(config["expanded_seasonal"]["horizon"]),
        "n_items": int(wide.shape[1]),
        "chunk_size": CHUNK_SIZE,
    }
    if hashlib.sha256(canonical_json(settings).encode("utf-8")).hexdigest() != manifest["settings_sha256"]:
        raise RuntimeError("expanded intermittent settings changed after manifest preparation")
    if seasonal["manifest_sha256"] != manifest["fold_manifest_sha256"]:
        raise RuntimeError("shared seasonal fold manifest changed after intermittent plan preparation")
    output_names = (
        "forecasts.csv", "fold_metrics.csv", "fold_class_summary.csv", "class_summary.csv",
        "model_summary.csv", "run_metadata.json",
    )
    if any((output_dir / name).exists() for name in output_names):
        raise FileExistsError("expanded intermittent outputs already exist; refusing to overwrite saved results")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    folds = manifest["folds"]
    # Use precisely the same positions as the aggregate study.
    for fold in folds:
        start, end = int(fold["test_start_pos"]), int(fold["test_end_pos"])
        if end - start != int(manifest["horizon_days"]) or end > len(wide):
            raise ValueError("item panel does not cover a complete manifest test window")
    factories = _factories(config)
    logger = logging.getLogger("src.evaluation.expanded_intermittent")
    logger.setLevel(logging.INFO)
    item_ids = [str(item_id) for item_id in wide.columns]
    chunk_paths: List[Path] = []
    for chunk_number, offset in enumerate(range(0, len(item_ids), CHUNK_SIZE), start=1):
        chunk = item_ids[offset:offset + CHUNK_SIZE]
        path = checkpoint_dir / f"chunk-{chunk_number:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("manifest_sha256") != manifest["manifest_sha256"] or saved.get("item_ids") != chunk:
                raise RuntimeError(f"checkpoint {path.name} does not match the prepared manifest")
        else:
            saved = _run_item_chunk(chunk, wide, folds, factories, manifest["manifest_sha256"], int(config["comparison"]["mase_seasonal_lag"]))
            path.write_text(json.dumps(saved, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        chunk_paths.append(path)
        logger.info("Completed item chunk %d/%d (%d/%d items)", chunk_number, (len(item_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE, min(offset + CHUNK_SIZE, len(item_ids)), len(item_ids))

    score_frames: List[pd.DataFrame] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / "forecasts.csv"
    header_written = False
    for path in chunk_paths:
        chunk_data = json.loads(path.read_text(encoding="utf-8"))
        score_frames.append(pd.DataFrame(chunk_data["scores"]))
        pd.DataFrame(chunk_data["forecasts"]).to_csv(
            forecast_path, mode="a" if header_written else "w", header=not header_written,
            index=False, lineterminator="\n",
        )
        header_written = True
    metrics = pd.concat(score_frames, ignore_index=True)
    if len(metrics) != len(item_ids) * len(folds) * len(factories):
        raise AssertionError("expanded intermittent metric row count is incomplete")
    fold_class = (
        metrics[metrics["success"].astype(bool)]
        .groupby(["fold", "year", "quarter", "fold_local_class", "model"], sort=False)
        .agg(n_items=("item_id", "nunique"), MAE_macro_mean=("MAE", "mean"),
             RMSE_macro_mean=("RMSE", "mean"), WAPE_macro_mean=("WAPE", "mean"),
             MASE_macro_mean=("MASE", "mean"), bias_macro_mean=("bias", "mean"),
             forecast_zero_share_mean=("forecast_zero_share", "mean"),
             all_zero_forecast_items=("all_zero_forecast", "sum"))
        .reset_index()
    )
    class_summary = (
        fold_class.groupby(["fold_local_class", "model"], sort=False)
        .agg(n_folds=("fold", "nunique"), mean_items_per_fold=("n_items", "mean"),
             MAE_mean_over_folds=("MAE_macro_mean", "mean"), MAE_across_folds_std=("MAE_macro_mean", "std"),
             RMSE_mean_over_folds=("RMSE_macro_mean", "mean"), WAPE_mean_over_folds=("WAPE_macro_mean", "mean"),
             MASE_mean_over_folds=("MASE_macro_mean", "mean"), bias_mean_over_folds=("bias_macro_mean", "mean"),
             forecast_zero_share=("forecast_zero_share_mean", "mean"),
             all_zero_forecast_items_mean=("all_zero_forecast_items", "mean"))
        .reset_index()
    )
    model_summary = (
        metrics.groupby("model", sort=False)
        .agg(n_items=("item_id", "nunique"), folds=("fold", "nunique"),
             attempts=("success", "size"), failures=("success", lambda values: int((~values.astype(bool)).sum())),
             MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"),
             WAPE_mean=("WAPE", "mean"), MASE_mean=("MASE", "mean"), bias_mean=("bias", "mean"),
             forecast_zero_share=("forecast_zero_share", "mean"),
             all_zero_forecast_item_folds=("all_zero_forecast", "sum"))
        .reset_index()
    )
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    fold_class.to_csv(output_dir / "fold_class_summary.csv", index=False, lineterminator="\n")
    class_summary.to_csv(output_dir / "class_summary.csv", index=False, lineterminator="\n")
    model_summary.to_csv(output_dir / "model_summary.csv", index=False, lineterminator="\n")
    metadata = {
        "component": "expanded_intermittent_all_items",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": manifest["manifest_sha256"],
        "fold_manifest_sha256": manifest["fold_manifest_sha256"],
        "development_panel_sha256": manifest["development_panel_sha256"],
        "n_items": len(item_ids), "n_folds": len(folds), "horizon_days": manifest["horizon_days"],
        "models": list(factories), "metric_rows": len(metrics),
        "failed_model_item_folds": int((~metrics["success"].astype(bool)).sum()),
        "classification": "recomputed independently from each training prefix",
        "forecast_zero_behavior": "zero forecast share and all-zero forecast flags saved per item/fold/model",
        "holdout_accessed": False,
    }
    write_json(output_dir / "run_metadata.json", metadata)
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_manifest()
        print(f"Prepared full-panel manifest for {manifest['n_items']} products and {len(manifest['folds'])} folds.")
    else:
        print(json.dumps(run_experiment(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
