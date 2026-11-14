"""Cross-artifact integrity check for the saved development research runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.paths import resolve_path
from src.utils.provenance import file_sha256, write_json


FILES = {
    "seasonal": "artifacts/expanded_seasonal/model_summary.csv",
    "intermittent": "artifacts/expanded_intermittent/fold_metrics.csv",
    "neural": "artifacts/expanded_neural/model_summary.csv",
    "prophet": "artifacts/expanded_prophet_block_conformal/model_summary.csv",
    "xgboost": "artifacts/expanded_xgboost/independent_audit.json",
    "inventory": "artifacts/expanded_inventory_calibration_v2/calibration_model_summary.csv",
    "selection": "artifacts/final/FROZEN_CANDIDATES.json",
    "comparison": "artifacts/metrics/comparison_artifact_audit.csv",
}


def audit() -> dict:
    out = {}
    errors = []
    for name, rel in FILES.items():
        path = resolve_path(Path(rel))
        if not path.is_file():
            errors.append(f"missing {rel}")
            continue
        out[name] = {"path": rel, "sha256": file_sha256(path), "bytes": path.stat().st_size}
    if not errors:
        xgb = json.loads(resolve_path(Path(FILES["xgboost"])).read_text(encoding="utf-8"))
        if not xgb.get("passed"):
            errors.append("XGBoost independent audit is not passed")
        neural = pd.read_csv(resolve_path(Path(FILES["neural"])))
        if len(neural) != 4 or (neural["folds_scored"] != 16).any() or (neural["folds_failed"] != 0).any():
            errors.append("neural seasonal summary is not 4 variants x 16 successful folds")
        prophet = pd.read_csv(resolve_path(Path(FILES["prophet"])))
        if len(prophet) != 3 or (prophet["folds_scored"] != 16).any() or (prophet["folds_failed"] != 0).any():
            errors.append("Prophet seasonal summary is not 3 variants x 16 successful folds")
        panel = pd.read_csv(resolve_path(Path(FILES["intermittent"])), usecols=["fold", "item_id", "model", "success"])
        if panel["item_id"].nunique() != 823 or panel["fold"].nunique() != 16:
            errors.append("expanded intermittent metrics do not cover 823 items x 16 folds")
        comparison = pd.read_csv(resolve_path(Path(FILES["comparison"])))
        if (comparison["status"] != "ok").any():
            errors.append("legacy comparison artifact audit contains missing/invalid sources")
    result = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "files": out, "errors": errors, "passed": not errors,
              "final_holdout_evaluated": False}
    write_json(resolve_path(Path("reports/final/development_artifact_audit.json")), result)
    if errors:
        raise AssertionError("development artifact audit failed: " + "; ".join(errors))
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
