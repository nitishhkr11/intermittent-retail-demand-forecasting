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
    all_cells = pd.read_csv(root / "calibration_cells.csv")
    cells = all_cells[all_cells["status"] == "ok"].copy()
    _summarize(cells, ["method", "model"], 1000).to_csv(root / "calibration_model_summary.csv", index=False, lineterminator="\n")
    _summarize(cells, ["method", "model", "fold_local_class"], 1000).to_csv(
        root / "calibration_model_class_summary.csv", index=False, lineterminator="\n")

    counts = (all_cells.assign(is_failed=all_cells["status"].eq("test fold failed or missing"))
              .groupby("method", sort=True)
              .agg(n_status_rows=("status", "size"), n_failed_test_cells=("is_failed", "sum"))
              .reset_index())
    operating = (cells.groupby("method", sort=True)
                 .agg(n_products=("item_id", "nunique"), n_evaluation_folds=("fold", "nunique"),
                      n_cells=("item_id", "size"), n_windows=("n_test_windows", "sum"),
                      calibration_samples_mean=("n_calibration_samples", "mean"),
                      calibration_samples_min=("n_calibration_samples", "min"),
                      calibration_samples_max=("n_calibration_samples", "max"),
                      mean_safety_stock=("safety_stock", "mean"),
                      mean_reorder_point=("mean_reorder_point", "mean"))
                 .reset_index())
    operating = operating.rename(columns={"n_cells": "eligible_cells", "n_windows": "evaluation_windows"})
    headline = pd.read_csv(root / "calibration_summary.csv")
    overall = headline.merge(operating, on="method", validate="one_to_one").merge(
        counts, on="method", validate="one_to_one")
    overall["n_not_evaluable_cells"] = overall["n_status_rows"] - overall["n_cells"]
    overall.to_csv(root / "calibration_method_summary.csv", index=False, lineterminator="\n")

    fold_ok = (cells.groupby(["method", "fold"], sort=True)
               .agg(n_products=("item_id", "nunique"), n_cells=("item_id", "size"),
                    n_windows=("n_test_windows", "sum"), n_stockout_windows=("n_stockout_windows", "sum"),
                    calibration_samples_mean=("n_calibration_samples", "mean"),
                    safety_stock_mean=("safety_stock", "mean"),
                    reorder_point_mean=("mean_reorder_point", "mean"))
               .reset_index())
    fold_ok["achieved_cycle_service"] = 1 - fold_ok["n_stockout_windows"] / fold_ok["n_windows"]
    fold_status = (all_cells[all_cells["status"].ne("ok")]
                   .groupby(["method", "fold"], sort=True).size().rename("n_not_evaluable_cells").reset_index())
    fold_summary = fold_ok.merge(fold_status, on=["method", "fold"], how="left", validate="one_to_one")
    fold_summary["n_not_evaluable_cells"] = fold_summary["n_not_evaluable_cells"].fillna(0).astype(int)
    fold_summary.to_csv(root / "calibration_fold_summary.csv", index=False, lineterminator="\n")

    class_summary = (cells.groupby(["method", "model", "fold_local_class"], dropna=False, sort=True)
                     .agg(n_products=("item_id", "nunique"), n_cells=("item_id", "size"),
                          n_windows=("n_test_windows", "sum"), n_stockout_windows=("n_stockout_windows", "sum"),
                          calibration_samples_mean=("n_calibration_samples", "mean"),
                          safety_stock_mean=("safety_stock", "mean"),
                          reorder_point_mean=("mean_reorder_point", "mean"))
                     .reset_index())
    class_summary["achieved_cycle_service"] = 1 - class_summary["n_stockout_windows"] / class_summary["n_windows"]
    class_summary.to_csv(root / "calibration_class_operating_summary.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
