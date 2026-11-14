"""Produce model- and demand-class-specific summaries for inventory calibration v2."""

from pathlib import Path

import pandas as pd

from src.inventory.reorder import summarize_service_level_backtest
from src.utils.paths import resolve_path


ROOT = Path("artifacts/expanded_inventory_calibration_v2")


def _summarize(cells: pd.DataFrame, keys: list[str], n_resamples: int) -> pd.DataFrame:
    rows = []
    for values, group in cells.groupby(keys, dropna=False, sort=True):
        result = summarize_service_level_backtest(group, n_resamples=n_resamples, seed=4207)
        values = values if isinstance(values, tuple) else (values,)
        rows.extend({**dict(zip(keys, values)), **record} for record in result.to_dict(orient="records"))
    return pd.DataFrame(rows)


def main() -> None:
    root = resolve_path(ROOT)
    cells = pd.read_csv(root / "calibration_cells.csv")
    cells = cells[cells["status"] == "ok"]
    _summarize(cells, ["method", "model"], 1000).to_csv(root / "calibration_model_summary.csv", index=False, lineterminator="\n")
    _summarize(cells, ["method", "model", "fold_local_class"], 1000).to_csv(
        root / "calibration_model_class_summary.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
