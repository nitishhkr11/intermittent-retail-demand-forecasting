"""Produce a separate SBA forecast artifact on the canonical five development folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.demand_classification import classify_series
from src.data.aggregate_series import load_aggregate_development
from src.evaluation.forecast_artifacts import expected_fold_windows, load_selected_item_series
from src.models.sba import SBAForecaster
from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.provenance import file_sha256, write_json


def run() -> Path:
    config = load_config()
    filename = str(config["phase3"]["sba_forecasts_filename"])
    output = resolve_path(Path(config["phase3"]["forecasts_dir"]) / filename)
    metadata = output.with_suffix(".metadata.json")
    if output.exists() or metadata.exists():
        raise FileExistsError(f"refusing to overwrite an existing SBA development artifact: {output}")
    series = load_selected_item_series(config)
    dev, window = load_aggregate_development(config)
    horizon, n_folds = int(config["backtest"]["horizon"]), int(config["backtest"]["n_folds"])
    folds = expected_fold_windows(dev.index, horizon, n_folds)
    alpha = float(config["models"]["sba_alpha"])
    rows = []
    for item_id, values in series.items():
        for fold in folds.itertuples(index=False):
            train = values.loc[:fold.origin_date]
            dates = pd.date_range(fold.forecast_start, periods=horizon, freq="D")
            actual = values.reindex(dates)
            if actual.isna().any():
                raise ValueError(f"incomplete development forecast window for {item_id}, fold {fold.fold}")
            demand_class = classify_series(values)["demand_class"]
            error = ""
            try:
                prediction = np.asarray(SBAForecaster(alpha=alpha).fit(train).predict(horizon), dtype=float)
                if prediction.shape != (horizon,) or not np.isfinite(prediction).all() or (prediction < -1e-9).any():
                    raise ValueError("SBA returned invalid predictions")
                success = True
            except Exception as exc:
                prediction = np.full(horizon, np.nan)
                success, error = False, f"{type(exc).__name__}: {exc}"
            for date, observed, predicted in zip(dates, actual.to_numpy(dtype=float), prediction):
                rows.append({"item_id": item_id, "demand_class": demand_class,
                             "model": f"SBA(alpha={alpha})", "fold": int(fold.fold),
                             "date": date.date().isoformat(), "actual": observed,
                             "predicted": predicted, "success": success, "error": error})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, lineterminator="\n")
    write_json(metadata, {
        "artifact": output.relative_to(resolve_path(Path('.'))).as_posix(),
        "artifact_sha256": file_sha256(output), "model": "SBA", "alpha": alpha,
        "scope": "20 selected items; five canonical development folds; 28-day horizon",
        "development_start": window.dev_start.date().isoformat(),
        "development_end": window.dev_end.date().isoformat(), "holdout_accessed": False,
    })
    return output


if __name__ == "__main__":
    print(run())
