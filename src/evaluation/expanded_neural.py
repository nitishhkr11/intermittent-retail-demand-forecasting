"""Checkpointed LSTM/GRU calendar ablation on the shared seasonal folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.aggregate_series import load_aggregate_development
from src.evaluation.expanded_seasonal import _development_digest, OUTPUT_DIR as SEASONAL_DIR
from src.evaluation.metrics import compute_metrics
from src.models.lstm_gru import (
    NeuralConfig, NeuralSequenceForecaster, checkpoint_fingerprint, load_checkpoint, save_checkpoint,
    variants_from_config,
)
from src.models.xgboost_model import load_event_calendar
from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.provenance import canonical_json, file_sha256, write_json

OUTPUT_DIR = Path("artifacts/expanded_neural")
CHECKPOINT_DIR = Path("artifacts/checkpoints/expanded_neural")


def _calendar_digest(calendar: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(calendar, index=True).to_numpy().tobytes()).hexdigest()


def run_experiment(config: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    config = load_config() if config is None else config
    output = resolve_path(OUTPUT_DIR)
    checkpoints = resolve_path(CHECKPOINT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    fold_path = resolve_path(SEASONAL_DIR / "fold_manifest.json")
    if not fold_path.is_file():
        raise FileNotFoundError("prepare the shared quarterly seasonal manifest first")
    folds_manifest = json.loads(fold_path.read_text(encoding="utf-8"))
    series, window = load_aggregate_development(config)
    if _development_digest(series) != folds_manifest["development_series_sha256"]:
        raise RuntimeError("development series does not match the shared seasonal manifest")
    panel = resolve_path(Path(config["data"]["interim_dir"]) / config["output"]["panel_filename"])
    calendar = load_event_calendar(panel, last_date=window.dev_end)
    calendar_sha = _calendar_digest(calendar)
    folds = folds_manifest["folds"]
    model_config = NeuralConfig.from_config(config, horizon=int(folds_manifest["settings"]["horizon"]))
    variants = variants_from_config(config)
    experiment_sha = hashlib.sha256(canonical_json({
        "fold_manifest_sha256": folds_manifest["manifest_sha256"], "development_series_sha256": _development_digest(series),
        "calendar_sha256": calendar_sha, "config": {"neural": config["neural"], "horizon": model_config.horizon},
        "code_sha256": file_sha256(Path(__file__)),
    }).encode()).hexdigest()
    metadata_path = output / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("experiment_sha256") != experiment_sha:
            raise FileExistsError("a different neural experiment is already saved; refusing to overwrite it")

    rows: List[Dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        test_start, test_end = int(fold["test_start_pos"]), int(fold["test_end_pos"])
        train, test = series.iloc[:test_start], series.iloc[test_start:test_end]
        for variant in variants:
            base_fp = checkpoint_fingerprint(config, "expanded:" + variant.name, fold_id, train.to_numpy(dtype=float))
            fingerprint = hashlib.sha256((base_fp + experiment_sha).encode()).hexdigest()
            saved = load_checkpoint(checkpoints, "expanded:" + variant.name, fold_id, fingerprint)
            if saved is not None and saved.get("result", {}).get("experiment_sha256") == experiment_sha:
                rows.append(saved["result"])
                continue
            started = time.perf_counter()
            model = NeuralSequenceForecaster(variant.cell, variant.features, model_config, calendar=calendar).fit(train)
            members = np.asarray(model.seed_predictions(), dtype=float)
            prediction = members.mean(axis=0)
            diagnostics = model.fit_diagnostics()
            history = model.training_history()
            actual = test.to_numpy(dtype=float)
            if members.shape != (len(model_config.seeds), len(test)) or not np.isfinite(members).all():
                raise ValueError(f"{variant.name} fold {fold_id}: seed predictions failed shape/finite checks")
            ensemble_metric = compute_metrics(actual, prediction)
            seed_metrics = []
            for seed, seed_prediction in zip(model_config.seeds, members):
                metric = compute_metrics(actual, seed_prediction)
                seed_metrics.append({"seed": int(seed), "MAE": metric.mae, "RMSE": metric.rmse,
                                     "bias": float(np.mean(seed_prediction - actual))})
            record: Dict[str, Any] = {
                "experiment_sha256": experiment_sha, "fold": fold_id, "year": int(fold["year"]),
                "quarter": int(fold["quarter"]), "model": variant.name, "success": True, "error": None,
                "origin_date": train.index[-1].date().isoformat(), "forecast_start": test.index[0].date().isoformat(),
                "forecast_end": test.index[-1].date().isoformat(), "n_train": len(train), "n_scored": len(test),
                "MAE": ensemble_metric.mae, "RMSE": ensemble_metric.rmse,
                "bias": float(np.mean(prediction - actual)), "fit_seconds": time.perf_counter() - started,
                "dates": [d.date().isoformat() for d in test.index], "actual": actual.tolist(),
                "predicted_ensemble": prediction.tolist(), "predicted_by_seed": members.tolist(),
                "seed_metrics": seed_metrics, "diagnostics": diagnostics,
                "training_history": history.to_dict(orient="records"),
            }
            save_checkpoint(checkpoints, "expanded:" + variant.name, fold_id, fingerprint, record, None, record["fit_seconds"])
            rows.append(record)

    rows.sort(key=lambda row: (row["fold"], row["model"]))
    fold_rows, seed_rows, forecast_rows, diagnostics_rows, history_rows = [], [], [], [], []
    for row in rows:
        fold_rows.append({k: v for k, v in row.items() if k not in {
            "experiment_sha256", "dates", "actual", "predicted_ensemble", "predicted_by_seed", "seed_metrics", "diagnostics", "training_history"
        }})
        for seed_metric, member in zip(row["seed_metrics"], row["predicted_by_seed"]):
            seed_rows.append({"model": row["model"], "fold": row["fold"], **seed_metric})
        for date, actual, point, spread in zip(row["dates"], row["actual"], row["predicted_ensemble"], np.std(row["predicted_by_seed"], axis=0).tolist()):
            forecast_rows.append({"model": row["model"], "fold": row["fold"], "date": date, "actual": actual,
                                  "predicted_ensemble": point, "predicted_std_across_seeds": spread})
        diag = row["diagnostics"]
        for seed_i, seed in enumerate(diag["seeds"]):
            diagnostics_rows.append({"model": row["model"], "fold": row["fold"], "seed": seed,
                "n_train_windows": diag["n_train_windows"], "n_validation_windows": diag["n_validation_windows"],
                "scaler_mean": diag["scaler_mean"], "scaler_scale": diag["scaler_scale"],
                "scaler_fit_end_date": diag["scaler_fit_end_date"], "epochs_run": diag["epochs_run"][seed_i],
                "best_epoch": diag["best_epoch"][seed_i], "best_val_loss": diag["best_val_loss"][seed_i],
                "stopped_early": diag["stopped_early"][seed_i], "hit_max_epochs": diag["hit_max_epochs"][seed_i],
                "fit_seconds": diag["member_fit_seconds"][seed_i]})
        for training_row in row["training_history"]:
            history_rows.append({"model": row["model"], "fold": row["fold"], **training_row})

    fold_table, seed_table = pd.DataFrame(fold_rows), pd.DataFrame(seed_rows)
    summary_rows = []
    for name, group in fold_table.groupby("model", sort=False):
        summary_rows.append({"model": name, "folds_total": len(folds), "folds_scored": int(group.success.sum()),
            "folds_failed": int((~group.success.astype(bool)).sum()), "MAE_mean": group.MAE.mean(), "MAE_sd": group.MAE.std(ddof=1),
            "RMSE_mean": group.RMSE.mean(), "RMSE_sd": group.RMSE.std(ddof=1), "bias_mean": group.bias.mean(),
            "seed_level_MAE_mean": seed_table[seed_table.model == name].MAE.mean(),
            "seed_level_MAE_sd": seed_table[seed_table.model == name].MAE.std(ddof=1)})
    summary = pd.DataFrame(summary_rows)
    outputs = {
        "fold_metrics.csv": fold_table, "seed_metrics.csv": seed_table,
        "forecasts.csv": pd.DataFrame(forecast_rows), "training_diagnostics.csv": pd.DataFrame(diagnostics_rows),
        "training_history.csv": pd.DataFrame(history_rows), "model_summary.csv": summary,
    }
    for name, frame in outputs.items():
        path = output / name
        if path.exists():
            raise FileExistsError(f"refusing to overwrite completed neural output: {path}")
        frame.to_csv(path, index=False, lineterminator="\n")
    write_json(metadata_path, {
        "component": "expanded_neural_seasonal_backtest", "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_sha256": experiment_sha, "fold_manifest_sha256": folds_manifest["manifest_sha256"],
        "development_series_sha256": _development_digest(series), "calendar_sha256": calendar_sha,
        "code_sha256": file_sha256(Path(__file__)), "folds": len(folds), "horizon_days": model_config.horizon,
        "items": 1, "scope": "aggregate CA_1 / FOODS_3", "variants": [v.name for v in variants],
        "seeds_per_variant_fold": len(model_config.seeds), "device": "cpu", "checkpointed": True,
        "selection_or_tuning": "none; config.yaml settings retained", "holdout_accessed": False,
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
