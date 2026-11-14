"""Compare causal inventory-error calibration methods on saved development forecasts.

Every evaluation fold uses calibration forecasts from strictly earlier folds. Methods share
the same evaluated item/fold cells and protection interval; failures and insufficient history
remain explicit. This does not use the reserved final period.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.inventory.reorder import _fold_windows, summarize_service_level_backtest
from src.utils.paths import resolve_path
from src.utils.provenance import file_sha256, write_json


SOURCE = Path("artifacts/expanded_intermittent/forecasts.csv")
OUT = Path("artifacts/expanded_inventory_calibration_v2")
TARGET = 0.95
WINDOW = 7
MIN_FOLDS = 2
METHODS = ("empirical_all", "recent_two_folds", "recency_weighted", "class_normalized")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or values.size != weights.size or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weighted quantile requires matching values and positive weights")
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cutoff = q * weights.sum()
    return float(values[min(np.searchsorted(np.cumsum(weights), cutoff, side="left"), len(values)-1)])


def compare_calibrations(forecasts: pd.DataFrame, *, window_days: int = WINDOW,
                         target: float = TARGET, min_folds: int = MIN_FOLDS) -> pd.DataFrame:
    required = {"model", "item_id", "fold", "date", "actual", "predicted", "success", "fold_local_class"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecast table missing required columns: {sorted(missing)}")
    if not (0.5 <= target < 1.0) or window_days < 1 or min_folds < 1:
        raise ValueError("invalid target, protection interval, or minimum history")
    table = forecasts.copy()
    table["date"] = pd.to_datetime(table["date"])
    errors: dict[tuple[str, str, int], np.ndarray] = {}
    info: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, group in table.groupby(["model", "item_id", "fold"], sort=False):
        ok = group["success"].astype(bool).all() and np.isfinite(group["predicted"].to_numpy(float)).all()
        if not ok:
            continue
        ordered = group.sort_values("date")
        err = _fold_windows(ordered, window_days)
        errors[(str(key[0]), str(key[1]), int(key[2]))] = err
        info[(str(key[0]), str(key[1]), int(key[2]))] = {
            "class": str(ordered["fold_local_class"].iloc[0]),
            "scale": max(1.0, float(ordered["actual"].mean())),
            "forecast_windows": np.convolve(ordered["predicted"].to_numpy(float), np.ones(window_days), mode="valid"),
            "test_windows": err,
        }

    class_normalized: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for (model, item, fold), arr in errors.items():
        detail = info[(model, item, fold)]
        class_normalized.setdefault((model, fold, detail["class"]), []).append(arr / detail["scale"])

    rows: list[dict[str, Any]] = []
    models = sorted(table["model"].astype(str).unique())
    for model in models:
        model_table = table[table["model"].astype(str) == model]
        for item, item_rows in model_table.groupby("item_id", sort=False):
            folds = sorted(int(value) for value in item_rows["fold"].unique())
            for fold in folds:
                test_key = (model, str(item), fold)
                test = item_rows[item_rows["fold"] == fold].sort_values("date")
                earlier = [f for f in folds if f < fold]
                for method in METHODS:
                    base = {"model": model, "item_id": str(item), "fold": fold,
                            "fold_local_class": str(test["fold_local_class"].iloc[0]),
                            "method": method, "target_cycle_service": target,
                            "window_days": window_days}
                    if len(earlier) < min_folds:
                        rows.append({**base, "status": f"fewer than {min_folds} earlier folds"})
                        continue
                    if test_key not in info:
                        rows.append({**base, "status": "test fold failed or missing"})
                        continue
                    prior = [(f, errors.get((model, str(item), f))) for f in earlier]
                    prior = [(f, a) for f, a in prior if a is not None]
                    if method == "recent_two_folds":
                        prior = prior[-2:]
                    if method == "class_normalized":
                        label = info[test_key]["class"]
                        normalized_parts = [arr for hist_fold in earlier
                                            for arr in class_normalized.get((model, hist_fold, label), [])]
                        if not normalized_parts:
                            rows.append({**base, "status": "no prior same-class calibration samples"})
                            continue
                        normalized_values = np.concatenate(normalized_parts)
                        normalized = _weighted_quantile(normalized_values, np.ones(len(normalized_values)), target)
                        # The target scale must also be known at the order origin. Use the
                        # most recent successful *earlier* fold, never test-fold actuals.
                        scale_folds = [f for f in earlier if (model, str(item), f) in info]
                        if not scale_folds:
                            rows.append({**base, "status": "no prior item scale available"})
                            continue
                        target_scale = info[(model, str(item), max(scale_folds))]["scale"]
                        safety = max(0.0, normalized * target_scale)
                        n_cal = len(normalized_values)
                    else:
                        if not prior:
                            rows.append({**base, "status": "no usable prior calibration folds"})
                            continue
                        vals = np.concatenate([arr for _, arr in prior])
                        if method == "recency_weighted":
                            weight_parts = [np.full(len(arr), 2.0 ** (-(fold - f) / 2.0)) for f, arr in prior]
                            weights = np.concatenate(weight_parts)
                            safety = max(0.0, _weighted_quantile(vals, weights, target))
                        else:
                            safety = max(0.0, float(np.quantile(vals, target, method="higher")))
                        n_cal = len(vals)
                    errs = info[test_key]["test_windows"]
                    fcast = info[test_key]["forecast_windows"]
                    stockouts = int((errs > safety).sum())
                    rows.append({**base, "status": "ok", "n_calibration_samples": n_cal,
                                 "calibration_folds": len(prior), "mean_expected_demand": float(fcast.mean()),
                                 "safety_stock": safety, "mean_reorder_point": float(fcast.mean() + safety),
                                 "n_test_windows": len(errs), "n_stockout_windows": stockouts,
                                 "achieved_cycle_service": 1.0 - stockouts / len(errs)})
    return pd.DataFrame(rows)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = resolve_path(SOURCE)
    if not source.is_file():
        raise FileNotFoundError(source)
    out = resolve_path(OUT)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite calibration study: {out}")
    forecasts = pd.read_csv(source)
    cells = compare_calibrations(forecasts)
    summaries = []
    for method, group in cells.groupby("method", sort=False):
        summary = summarize_service_level_backtest(group.rename(columns={"method": "calibration_method"}),
                                                    n_resamples=1000, seed=4207)
        summary.insert(0, "method", method)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "calibration_cells.csv", index=False, lineterminator="\n")
    summary.to_csv(out / "calibration_summary.csv", index=False, lineterminator="\n")
    cells.groupby(["method", "fold", "fold_local_class"], dropna=False).agg(
        cells=("status", "size"), successful=("achieved_cycle_service", "count"),
        achieved_cycle_service=("achieved_cycle_service", "mean"),
        safety_stock_mean=("safety_stock", "mean"),
        calibration_n_mean=("n_calibration_samples", "mean"),
    ).reset_index().to_csv(out / "fold_class_summary.csv", index=False, lineterminator="\n")
    write_json(out / "run_metadata.json", {
        "component": "alternative_temporally_valid_inventory_calibration",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_sha256": file_sha256(source), "window_days": WINDOW, "target": TARGET,
        "minimum_prior_folds": MIN_FOLDS, "methods": list(METHODS),
        "evaluation": "same later fold, item, and overlapping 7-day windows for each method; calibration uses strictly earlier folds",
        "recency_weight": "2^(-fold_age/2)",
        "class_normalization": "historical cumulative errors divided by item-fold mean actual demand, pooled only within the historical/current fold-local class, then rescaled by target item's last completed fold mean demand",
        "uncertainty": "cluster bootstrap by item-fold cell; overlapping windows remain within cells",
        "holdout_evaluated": False,
    })
    return cells, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    _, summary = run()
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
