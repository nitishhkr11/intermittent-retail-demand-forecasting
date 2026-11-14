"""Development-only Prophet backtest on the shared 16 seasonal windows.

Point metrics and nominal 80% interval diagnostics are checkpointed per fold and
variant. The first eight chronological folds calibrate an additive split-conformal
interval offset; the last eight are held out for that calibration check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.aggregate_series import load_aggregate_development
from src.evaluation.expanded_seasonal import _development_digest, OUTPUT_DIR as SEASONAL_DIR
from src.evaluation.metrics import compute_metrics
from src.models.prophet_model import ProphetForecaster, ProphetSettings, variants_from_config
from src.models.xgboost_model import load_event_calendar
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT, resolve_path
from src.utils.provenance import file_sha256, write_json

OUTPUT_DIR = Path("artifacts/expanded_prophet_block_conformal")
CHECKPOINT_DIR = Path("artifacts/checkpoints/expanded_prophet")


def conformal_interval_offset(actual: Sequence[float], lower: Sequence[float], upper: Sequence[float], coverage: float = 0.8) -> float:
    """Finite-sample split-conformal additive expansion for two-sided intervals."""
    y, lo, hi = (np.asarray(v, dtype=float) for v in (actual, lower, upper))
    if y.ndim != 1 or y.shape != lo.shape or y.shape != hi.shape or len(y) == 0:
        raise ValueError("actual and interval bounds must be non-empty equal-length vectors")
    if not (np.isfinite(y).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("conformal calibration inputs must be finite")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    scores = np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)])
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * coverage)))
    return float(np.partition(scores, rank - 1)[rank - 1])


def block_conformal_interval_offset(blocks: Sequence[tuple[Sequence[float], Sequence[float], Sequence[float]]], coverage: float = 0.8) -> float:
    """Calibrate simultaneous horizon coverage using one max score per 28-day block."""
    if not blocks:
        raise ValueError("at least one calibration block is required")
    scores = []
    for actual, lower, upper in blocks:
        y, lo, hi = (np.asarray(v, dtype=float) for v in (actual, lower, upper))
        if y.ndim != 1 or y.shape != lo.shape or y.shape != hi.shape or not len(y):
            raise ValueError("each block must contain equal-length non-empty vectors")
        if not (np.isfinite(y).all() and np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError("calibration blocks must be finite")
        scores.append(float(np.maximum.reduce([lo - y, y - hi, np.zeros_like(y)]).max()))
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * coverage)))
    return float(np.partition(np.asarray(scores), rank - 1)[rank - 1])


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def run_experiment(config: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    config = load_config() if config is None else config
    root = resolve_path(OUTPUT_DIR)
    checkpoints = resolve_path(CHECKPOINT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    fold_path = resolve_path(SEASONAL_DIR / "fold_manifest.json")
    if not fold_path.is_file():
        raise FileNotFoundError("prepare the shared quarterly seasonal manifest first")
    manifest = json.loads(fold_path.read_text(encoding="utf-8"))
    series, window = load_aggregate_development(config)
    if _development_digest(series) != manifest["development_series_sha256"]:
        raise RuntimeError("development series does not match the shared seasonal manifest")
    folds = manifest["folds"]
    if len(folds) < 12:
        raise ValueError("expanded Prophet evaluation requires at least 12 shared seasonal folds")
    variants = variants_from_config(config)
    settings = ProphetSettings.from_config(config, horizon=int(manifest["settings"]["horizon"]))
    panel_path = resolve_path(Path(config["data"]["interim_dir"]) / config["output"]["panel_filename"])
    calendar = load_event_calendar(panel_path, last_date=window.dev_end)
    settings_hash = hashlib.sha256(json.dumps({
        "variants": [v.__dict__ for v in variants], "settings": settings.__dict__,
        "fold_manifest_sha256": manifest["manifest_sha256"],
    }, sort_keys=True, default=list).encode()).hexdigest()
    meta_path = root / "run_metadata.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("settings_sha256") != settings_hash:
            raise FileExistsError("a different Prophet experiment already exists; refusing to overwrite it")

    records: List[Dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        test_start = int(fold["test_start_pos"])
        test_end = int(fold["test_end_pos"])
        train, test = series.iloc[:test_start], series.iloc[test_start:test_end]
        for variant in variants:
            slug = variant.name.lower().replace(" ", "-").replace("(", "-").replace(")", "").replace(",", "")
            checkpoint = checkpoints / f"fold-{fold_id:02d}--{slug}.json"
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text(encoding="utf-8"))
                if row.get("settings_sha256") != settings_hash:
                    raise RuntimeError(f"checkpoint {checkpoint.name} belongs to different experiment settings")
                records.append(row)
                continue
            success, error = True, None
            try:
                model = ProphetForecaster(variant, settings, calendar=calendar).fit(train)
                point = np.asarray(model.predict(len(test)), dtype=float)
                lower, upper = (np.asarray(x, dtype=float) for x in model.predict_interval(len(test)))
                if any(x.shape != (len(test),) or not np.isfinite(x).all() for x in (point, lower, upper)):
                    raise ValueError("forecast or interval returned non-finite values or an invalid shape")
                metric = compute_metrics(test.to_numpy(dtype=float), point)
                raw_coverage = float(np.mean((test.to_numpy() >= lower) & (test.to_numpy() <= upper)))
                raw_width = float(np.mean(upper - lower))
                arrays = {"predicted": point.tolist(), "lower": lower.tolist(), "upper": upper.tolist()}
                diagnostics = model.fit_diagnostics()
                del model
            except Exception as exc:
                success, error = False, f"{type(exc).__name__}: {exc}"
                metric = None
                raw_coverage = raw_width = float("nan")
                arrays = {k: [None] * len(test) for k in ("predicted", "lower", "upper")}
                diagnostics = {}
            row = {
                "settings_sha256": settings_hash, "fold": fold_id,
                "year": int(fold["year"]), "quarter": int(fold["quarter"]),
                "model": variant.name, "success": success, "error": error,
                "origin_date": train.index[-1].date().isoformat(),
                "forecast_start": test.index[0].date().isoformat(),
                "forecast_end": test.index[-1].date().isoformat(),
                "n_train": len(train), "n_scored": len(test),
                "MAE": None if metric is None else metric.mae,
                "RMSE": None if metric is None else metric.rmse,
                "bias": None if metric is None else float(np.mean(np.asarray(arrays["predicted"]) - test.to_numpy())),
                "coverage_80_nominal": None if not success else raw_coverage,
                "interval_width_mean": None if not success else raw_width,
                "actual": test.to_numpy(dtype=float).tolist(), **arrays,
                "diagnostics": diagnostics,
            }
            _atomic_json(checkpoint, row)
            records.append(row)

    records.sort(key=lambda r: (r["fold"], r["model"]))
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in {"actual", "predicted", "lower", "upper", "diagnostics", "settings_sha256"}} for r in records])
    success = table[table.success.astype(bool)]
    summary = success.groupby("model", sort=False).agg(
        folds_scored=("fold", "count"), MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), bias_mean=("bias", "mean"),
        coverage_80_mean=("coverage_80_nominal", "mean"), interval_width_mean=("interval_width_mean", "mean"),
    ).reset_index()
    summary["folds_total"] = len(folds)
    summary["folds_failed"] = len(folds) - summary["folds_scored"]

    # Chronological split: folds 1..8 calibrate the interval adjustment; 9..16 evaluate it.
    calibration_end = len(folds) // 2
    adjustments: List[Dict[str, Any]] = []
    calibrated_rows: List[Dict[str, Any]] = []
    for variant in variants:
        model_rows = [r for r in records if r["model"] == variant.name and r["success"]]
        calib = [r for r in model_rows if r["fold"] <= calibration_end]
        validation = [r for r in model_rows if r["fold"] > calibration_end]
        if not calib or not validation:
            continue
        offset = block_conformal_interval_offset([(r["actual"], r["lower"], r["upper"]) for r in calib], 0.8)
        adjustments.append({"model": variant.name, "calibration_folds": [r["fold"] for r in calib],
                            "validation_folds": [r["fold"] for r in validation], "offset_units": offset})
        for row in validation:
            actual = np.asarray(row["actual"], dtype=float)
            lower, upper = np.asarray(row["lower"], dtype=float), np.asarray(row["upper"], dtype=float)
            calibrated_rows.append({
                "model": variant.name, "fold": row["fold"], "calibration_offset": offset,
                "coverage_80_calibrated": float(np.mean((actual >= lower - offset) & (actual <= upper + offset))),
                "simultaneous_coverage_80_calibrated": float(np.all((actual >= lower - offset) & (actual <= upper + offset))),
                "interval_width_calibrated": float(np.mean(upper - lower + 2 * offset)),
            })
    calibrated = pd.DataFrame(calibrated_rows)
    if not calibrated.empty:
        cal_summary = calibrated.groupby("model", sort=False).agg(
            validation_folds=("fold", "count"), coverage_80_calibrated_mean=("coverage_80_calibrated", "mean"),
            simultaneous_coverage_80_calibrated=("simultaneous_coverage_80_calibrated", "mean"),
            interval_width_calibrated_mean=("interval_width_calibrated", "mean"), offset_units=("calibration_offset", "first"),
        ).reset_index()
    else:
        cal_summary = pd.DataFrame(columns=["model", "validation_folds", "coverage_80_calibrated_mean", "simultaneous_coverage_80_calibrated", "interval_width_calibrated_mean", "offset_units"])

    outputs = {
        "fold_metrics.csv": table,
        "model_summary.csv": summary,
        "interval_calibration_validation.csv": calibrated,
        "interval_calibration_summary.csv": cal_summary,
    }
    for name, frame in outputs.items():
        path = root / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite completed Prophet output: {path}")
        frame.to_csv(path, index=False, lineterminator="\n")
    write_json(meta_path, {
        "component": "expanded_prophet_seasonal_backtest", "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fold_manifest_sha256": manifest["manifest_sha256"], "settings_sha256": settings_hash,
        "code_sha256": file_sha256(Path(__file__)), "development_series_sha256": _development_digest(series),
        "folds": len(folds), "horizon_days": settings.horizon, "variants": [v.name for v in variants],
        "information_set": "historical sales; published calendar regressors where configured", "failed_fits": int((~table.success.astype(bool)).sum()),
        "interval_calibration": {"method": "two-sided block split-conformal additive expansion using each calibration fold's maximum nonconformity score", "target_coverage": 0.8,
                                 "calibration_folds": list(range(1, calibration_end + 1)), "validation_folds": list(range(calibration_end + 1, len(folds) + 1)),
                                 "adjustments": adjustments},
        "holdout_accessed": False,
    })
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.parse_args(argv)
    print(run_experiment().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
