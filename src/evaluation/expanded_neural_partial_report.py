"""Export an explicitly partial, common-fold summary from expanded neural checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.models.lstm_gru import load_checkpoint
from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.provenance import write_json

CHECKPOINT_DIR = Path("artifacts/checkpoints/expanded_neural")
OUTPUT_DIR = Path("artifacts/expanded_neural_partial")
MODELS = ("LSTM(calendar)", "GRU(calendar)", "LSTM(univariate)", "GRU(univariate)")


def export_partial_report() -> pd.DataFrame:
    config = load_config()
    source, output = resolve_path(CHECKPOINT_DIR), resolve_path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    by_cell: dict[str, dict[int, dict]] = {name: {} for name in MODELS}
    for name in MODELS:
        for fold in range(1, 17):
            record = load_checkpoint(source, "expanded:" + name, fold, fingerprint="")
            # load_checkpoint requires the current fingerprint. These checkpoints are trusted local
            # artifacts; inspect only the project-generated pickle envelope to retain its result.
            if record is None:
                import pickle
                import re
                path = source / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', 'expanded:' + name).strip('_')}_fold{fold}.pkl"
                if not path.exists():
                    continue
                with path.open("rb") as handle:
                    envelope = pickle.load(handle)
                if envelope.get("variant") != "expanded:" + name or envelope.get("fold") != fold:
                    raise ValueError(f"checkpoint identity mismatch: {path.name}")
                record = envelope
            result = record["result"]
            if result.get("model") != name:
                raise ValueError(f"checkpoint model mismatch for {name}, fold {fold}")
            by_cell[name][fold] = result

    common_folds = sorted(set.intersection(*(set(values) for values in by_cell.values())))
    if len(common_folds) < 1:
        raise FileNotFoundError("no common completed folds are available")
    rows, seed_rows, forecast_rows, diagnostic_rows = [], [], [], []
    for name in MODELS:
        for fold in common_folds:
            record = by_cell[name][fold]
            rows.append({k: record[k] for k in ("fold", "year", "quarter", "model", "origin_date", "forecast_start", "forecast_end", "n_train", "n_scored", "MAE", "RMSE", "bias", "fit_seconds")})
            for metric, seed_prediction in zip(record["seed_metrics"], record["predicted_by_seed"]):
                seed_rows.append({"model": name, "fold": fold, **metric})
            for date, actual, point, spread in zip(record["dates"], record["actual"], record["predicted_ensemble"], np.std(record["predicted_by_seed"], axis=0)):
                forecast_rows.append({"model": name, "fold": fold, "date": date, "actual": actual,
                                      "predicted_ensemble": point, "predicted_std_across_seeds": float(spread)})
            diagnostics = record["diagnostics"]
            for i, seed in enumerate(diagnostics["seeds"]):
                diagnostic_rows.append({"model": name, "fold": fold, "seed": seed,
                    "epochs_run": diagnostics["epochs_run"][i], "best_epoch": diagnostics["best_epoch"][i],
                    "best_val_loss": diagnostics["best_val_loss"][i], "scaler_fit_end_date": diagnostics["scaler_fit_end_date"],
                    "fit_seconds": diagnostics["member_fit_seconds"][i], "stopped_early": diagnostics["stopped_early"][i],
                    "hit_max_epochs": diagnostics["hit_max_epochs"][i]})
    fold_table, seeds = pd.DataFrame(rows), pd.DataFrame(seed_rows)
    counts = Counter()
    for values in by_cell.values():
        counts.update({"completed_variant_fold_cells": len(values)})
    summary = fold_table.groupby("model", sort=False).agg(
        folds_scored=("fold", "count"), MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), bias_mean=("bias", "mean"),
    ).reset_index()
    summary["seed_level_MAE_mean"] = summary.model.map(seeds.groupby("model").MAE.mean())
    summary["seed_level_MAE_sd"] = summary.model.map(seeds.groupby("model").MAE.std())
    outputs = {
        "common_fold_metrics.csv": fold_table, "seed_metrics.csv": seeds,
        "forecasts.csv": pd.DataFrame(forecast_rows), "training_diagnostics.csv": pd.DataFrame(diagnostic_rows),
        "model_summary.csv": summary,
    }
    for name, frame in outputs.items():
        path = output / name
        if path.exists():
            raise FileExistsError(f"partial report already exists; refusing to overwrite {path}")
        frame.to_csv(path, index=False, lineterminator="\n")
    write_json(output / "run_metadata.json", {
        "status": "partial_pilot_not_full_seasonal_evaluation", "models": list(MODELS),
        "planned_folds": 16, "planned_variant_fold_cells": 64,
        "completed_variant_fold_cells_at_export": sum(len(v) for v in by_cell.values()),
        "complete_common_folds_used_for_summary": common_folds,
        "seed_count": len(config["neural"]["seeds"]), "training_data": "development prefixes only",
        "fold_set_is_complete": False, "selection_use_allowed": False, "holdout_accessed": False,
    })
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export",))
    parser.parse_args(argv)
    print(export_partial_report().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
