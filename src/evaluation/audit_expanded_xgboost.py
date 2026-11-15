"""Independent row-level audit of the saved expanded XGBoost forecasts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import resolve_path
from src.utils.provenance import file_sha256, write_json


ROOT = Path("artifacts/expanded_xgboost")


def audit() -> dict[str, Any]:
    root = resolve_path(ROOT)
    metric = pd.read_csv(root / "item_fold_metrics.csv")
    meta = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "fold_manifest.json").read_text(encoding="utf-8"))
    key = ["model", "item_id", "fold"]
    expected = int(metric.shape[0]) * int(meta["horizon_days"])
    count = 0
    dates_max = None
    accum: dict[tuple[str, str, int], list[float]] = {}
    for chunk in pd.read_csv(root / "forecasts.csv", chunksize=250_000, parse_dates=["date"]):
        count += len(chunk)
        chunk = chunk[chunk["success"].astype(bool)]
        dates_max = max(dates_max, chunk["date"].max()) if dates_max is not None else chunk["date"].max()
        # Compute metric totals directly from each raw prediction group, independently of runner metrics.
        for group_key, rows in chunk.groupby(key, sort=False):
            actual = rows["actual"].to_numpy(float)
            pred = rows["predicted"].to_numpy(float)
            state = accum.setdefault((str(group_key[0]), str(group_key[1]), int(group_key[2])), [0., 0., 0., 0.])
            state[0] += len(rows)
            state[1] += float(np.abs(actual-pred).sum())
            state[2] += float(((actual-pred)**2).sum())
            state[3] += float((pred-actual).sum())
    check = metric[metric["success"].astype(bool)].copy()
    mismatch = []
    for row in check.itertuples(index=False):
        state = accum.get((str(row.model), str(row.item_id), int(row.fold)))
        if state is None or state[0] != int(row.n_scored):
            mismatch.append({"model": row.model, "item_id": row.item_id, "fold": int(row.fold), "issue": "missing/short forecast group"})
            continue
        n, sae, sse, sbias = state
        actual_mae = sae/n
        actual_rmse = np.sqrt(sse/n)
        actual_bias = sbias/n
        if not (np.isclose(actual_mae, row.MAE, atol=1e-9) and
                np.isclose(actual_rmse, row.RMSE, atol=1e-9) and
                np.isclose(actual_bias, row.bias, atol=1e-9)):
            mismatch.append({"model": row.model, "item_id": row.item_id, "fold": int(row.fold),
                             "issue": "metric mismatch", "MAE_delta": actual_mae-row.MAE,
                             "RMSE_delta": actual_rmse-row.RMSE, "bias_delta": actual_bias-row.bias})
    planned = (5 * 823 + int(meta["per_item_sample_products"])) * 16
    successes_by_model = metric.groupby("model")["success"].agg(["size", "sum"]).to_dict(orient="index")
    artifacts = sorted(p for p in root.iterdir() if p.is_file() and p.name not in {"independent_audit.json"})
    result = {
        "independent_row_recalculation": "MAE, RMSE, bias recomputed from raw forecast rows by model/item/fold",
        "item_fold_rows": int(len(metric)), "planned_item_fold_rows": planned,
        "forecast_rows": count, "planned_forecast_rows": expected,
        "max_forecast_date": None if dates_max is None else dates_max.date().isoformat(),
        "last_development_date": manifest["development_window"]["end"],
        "success_counts_by_model": successes_by_model,
        "mismatch_count": len(mismatch), "mismatches_first_20": mismatch[:20],
        "artifact_sha256": {p.name: file_sha256(p) for p in artifacts},
        "manifest_sha256": manifest["manifest_sha256"],
        "holdout_evaluated": False,
    }
    result["passed"] = len(mismatch) == 0 and len(metric) == planned and count == expected and dates_max is not None and dates_max <= pd.Timestamp(manifest["development_window"]["end"])
    write_json(root / "independent_audit.json", result)
    if not result["passed"]:
        raise AssertionError(json.dumps({k: result[k] for k in ("item_fold_rows", "planned_item_fold_rows", "forecast_rows", "planned_forecast_rows", "mismatch_count", "max_forecast_date", "passed")}, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
