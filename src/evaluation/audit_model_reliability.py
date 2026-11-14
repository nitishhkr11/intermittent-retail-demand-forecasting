"""Summarize saved fit failures, fold coverage and interval reliability for the final report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.paths import resolve_path
from src.utils.provenance import file_sha256, write_json


def audit() -> dict:
    paths = {
        "neural": Path("artifacts/expanded_neural/fold_metrics.csv"),
        "neural_diagnostics": Path("artifacts/expanded_neural/training_diagnostics.csv"),
        "prophet": Path("artifacts/expanded_prophet_block_conformal/fold_metrics.csv"),
        "xgboost": Path("artifacts/expanded_xgboost/item_fold_metrics.csv"),
        "intermittent": Path("artifacts/expanded_intermittent/fold_metrics.csv"),
    }
    frames = {name: pd.read_csv(resolve_path(path)) for name, path in paths.items()}
    neural, prophet, xgb, intermittent = (frames[k] for k in ("neural", "prophet", "xgboost", "intermittent"))
    neural_diagnostics = pd.read_csv(resolve_path(paths["neural_diagnostics"]))
    prophet_coverage = prophet.groupby("model")["coverage_80_nominal"].mean().to_dict()
    xgb_failures = int((~xgb["success"].astype(bool)).sum())
    intermittent_failures = intermittent.groupby("model")["success"].apply(lambda x: int((~x.astype(bool)).sum())).to_dict()
    result = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "neural": {"cells": int(len(neural)), "successful": int(neural["success"].astype(bool).sum()),
                   "failed": int((~neural["success"].astype(bool)).sum()),
                   "early_stopped_seeds": int(neural_diagnostics["stopped_early"].astype(bool).sum()),
                   "planned_seeds": int(len(neural_diagnostics))},
        "prophet": {"fits": int(len(prophet)), "successful": int(prophet["success"].astype(bool).sum()),
                    "nominal_80_day_coverage_by_model": prophet_coverage,
                    "all_variants_below_80_percent": bool(all(float(v) < 0.8 for v in prophet_coverage.values()))},
        "xgboost": {"item_fold_cells": int(len(xgb)), "successful": int(xgb["success"].astype(bool).sum()),
                    "failed": xgb_failures},
        "intermittent_failures_by_model": intermittent_failures,
        "input_sha256": {name: file_sha256(resolve_path(path)) for name, path in paths.items()},
        "limitations": ["Prophet nominal 80% intervals under-cover in this development sample.",
                        "Cycle service is not fill rate; failures and missing calibration cells are not imputed."],
        "passed": (len(neural) == 64 and neural["success"].astype(bool).all()
                   and len(prophet) == 48 and prophet["success"].astype(bool).all() and xgb_failures == 0),
    }
    write_json(resolve_path(Path("reports/final/model_reliability_audit.json")), result)
    if not result["passed"]:
        raise AssertionError("model reliability evidence has missing or failed planned seasonal fits")
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(audit(), indent=2))
