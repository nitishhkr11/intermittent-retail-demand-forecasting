"""Development-only service-level calibration using the expanded item forecast panel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.inventory.reorder import service_level_backtest, summarize_service_level_backtest
from src.utils.paths import resolve_path
from src.utils.provenance import file_sha256, write_json


FORECAST_PATH = Path("artifacts/expanded_intermittent/forecasts.csv")
OUTPUT_DIR = Path("artifacts/expanded_inventory")


def run(window_days: int = 7, service_level: float = 0.95) -> pd.DataFrame:
    source = resolve_path(FORECAST_PATH)
    if not source.exists():
        raise FileNotFoundError(f"missing expanded forecasts: {source}")
    output = resolve_path(OUTPUT_DIR)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing expanded inventory results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_csv(source, parse_dates=["date"])
    models = list(forecasts["model"].drop_duplicates())
    cells = []
    summaries = []
    for model in models:
        result = service_level_backtest(
            forecasts, window_days, service_levels=(service_level,), model=model, min_calibration_folds=2
        )
        cells.append(result)
        summary = summarize_service_level_backtest(result, n_resamples=1000, seed=42)
        summary.insert(0, "model", model)
        summaries.append(summary)
    cell_table = pd.concat(cells, ignore_index=True)
    summary_table = pd.concat(summaries, ignore_index=True)
    cell_table.to_csv(output / "service_level_cells.csv", index=False, lineterminator="\n")
    summary_table.to_csv(output / "service_level_summary.csv", index=False, lineterminator="\n")
    write_json(output / "run_metadata.json", {
        "source": FORECAST_PATH.as_posix(), "source_sha256": file_sha256(source),
        "window_days": window_days, "target_cycle_service": service_level,
        "calibration": "earlier folds only; at least two earlier folds",
        "uncertainty": "cluster bootstrap over item-fold cells; overlapping windows remain within cells",
        "holdout_accessed": False,
    })
    return summary_table


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--service-level", type=float, default=0.95)
    args = parser.parse_args(argv)
    print(run(args.window_days, args.service_level).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
