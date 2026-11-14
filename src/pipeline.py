"""Repeatable development workflow orchestration.

Run ``python -m src.pipeline --help``. The default workflow is development-only;
the protected final holdout is deliberately absent from every pipeline stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import zipfile
import hashlib
from pathlib import Path
from typing import Sequence

from src.utils.paths import PROJECT_ROOT, resolve_path
from src.utils.config import load_config
from src.utils.provenance import file_sha256


STAGES = {
    "data": ("src.data.preprocessor",),
    "classify": ("src.analysis.demand_classification",),
    "seasonal": ("src.evaluation.expanded_seasonal", "prepare", "src.evaluation.expanded_seasonal", "run"),
    "intermittent": ("src.evaluation.expanded_intermittent", "prepare", "src.evaluation.expanded_intermittent", "run"),
    "xgboost": ("src.evaluation.expanded_xgboost", "prepare", "src.evaluation.expanded_xgboost", "run"),
    "neural": ("src.evaluation.expanded_neural", "run"),
    "prophet": ("src.evaluation.expanded_prophet", "run"),
    "comparison": ("src.evaluation.audit_development_outputs",),
    "reliability": ("src.evaluation.audit_model_reliability",),
    "inventory": ("src.evaluation.expanded_inventory", "--window-days", "7", "--service-level", "0.95"),
    "inventory-alternatives": ("src.evaluation.expanded_inventory_calibration",),
    "selection": ("src.evaluation.select_development_candidates",),
    "reports": ("reports",),
    "dashboard-data": ("src.deployment.dashboard_package",),
}

OUTPUTS = {
    "data": Path("data/interim/panel_ca1_foods3.parquet"),
    "classify": Path("data/processed/demand_classification.parquet"),
    "seasonal": Path("artifacts/expanded_seasonal/model_summary.csv"),
    "intermittent": Path("artifacts/expanded_intermittent/model_summary.csv"),
    "inventory": Path("artifacts/expanded_inventory/service_level_summary.csv"),
    "xgboost": Path("artifacts/expanded_xgboost/independent_audit.json"),
    "neural": Path("artifacts/expanded_neural/completion_verification.json"),
    "prophet": Path("artifacts/expanded_prophet_block_conformal/model_summary.csv"),
    "comparison": Path("reports/final/development_artifact_audit.json"),
    "reliability": Path("reports/final/model_reliability_audit.json"),
    "inventory-alternatives": Path("artifacts/expanded_inventory_calibration_v2/calibration_summary.csv"),
    "selection": Path("artifacts/final/DEVELOPMENT_SELECTION.json"),
    "reports": Path("reports/phase3/expanded_xgboost_pooled_ablation_mae.png"),
    "dashboard-data": Path("reports/final/dashboard_data_bundle_final_evaluation.zip"),
}


def _invoke(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def validate_raw_inputs() -> dict[str, dict[str, str | int]]:
    """Validate configured raw-file presence and schema headers without loading sales values."""
    config = load_config()
    data = config["data"]
    root = resolve_path(Path(data["raw_dir"]))
    expected = {
        data["sales_file"]: {"id", "item_id", "store_id", "state_id"},
        data["calendar_file"]: {"date", "wm_yr_wk", "d"},
        data["prices_file"]: {"store_id", "item_id", "wm_yr_wk", "sell_price"},
    }
    result = {}
    for name, required in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required raw input is missing: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            header = set(next(csv.reader(stream)))
        missing = required - header
        if missing:
            raise ValueError(f"{name} is missing required column(s): {sorted(missing)}")
        result[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size, "columns": len(header)}
    logger = logging.getLogger("src.pipeline")
    logger.info("Validated %d configured raw data files without reading target values.", len(result))
    return result


def run_stage(stage: str, *, resume: bool = False) -> str:
    if stage not in STAGES:
        raise ValueError(f"unknown development stage: {stage}")
    output = resolve_path(OUTPUTS[stage])
    if resume and output.exists() and _verified_existing(stage):
        print(f"= {stage}: verified saved output exists, leaving it unchanged ({output})", flush=True)
        if stage != "intermittent":
            return "skipped"
        sba_path = resolve_path(Path("artifacts/forecasts") / load_config()["phase3"]["sba_forecasts_filename"])
        if sba_path.exists():
            return "skipped"
        _invoke("src.evaluation.sba_backtest")
        return "completed missing SBA artifact"
    commands = STAGES[stage]
    if stage == "xgboost" and not resolve_path(Path("artifacts/expanded_xgboost/fold_manifest.json")).exists():
        _invoke("src.evaluation.expanded_xgboost", "prepare")
        _invoke("src.evaluation.expanded_xgboost", "run")
    elif stage in {"seasonal", "intermittent"}:
        for i in range(0, len(commands), 2):
            _invoke(commands[i], commands[i + 1])
    elif stage == "reports":
        _invoke("src.evaluation.summarize_expanded_xgboost")
        _invoke("src.evaluation.summarize_expanded_inventory_calibration")
    else:
        _invoke(commands[0], *commands[1:])
    if stage == "intermittent":
        _invoke("src.evaluation.sba_backtest")
    return "complete"


def _verified_existing(stage: str) -> bool:
    """Only skip expensive result stages when their sidecar and core outputs agree."""
    root = resolve_path(Path("artifacts"))
    output = resolve_path(OUTPUTS[stage])
    if not output.is_file():
        return False
    if stage == "xgboost":
        try:
            audit = json.loads((root / "expanded_xgboost/independent_audit.json").read_text(encoding="utf-8"))
            metadata = json.loads((root / "expanded_xgboost/run_metadata.json").read_text(encoding="utf-8"))
            hashes_match = all(file_sha256(root / "expanded_xgboost" / name) == digest
                               for name, digest in audit.get("artifact_sha256", {}).items())
            return bool(hashes_match and audit.get("passed") and audit.get("mismatch_count") == 0
                        and audit.get("item_fold_rows") == audit.get("planned_item_fold_rows")
                        and metadata.get("failed_item_fold_rows") == 0)
        except (OSError, ValueError):
            return False
    if stage == "neural":
        try:
            proof = json.loads((root / "expanded_neural/completion_verification.json").read_text(encoding="utf-8"))
            hashes_match = all(file_sha256(root / "expanded_neural" / name) == digest
                               for name, digest in proof.get("output_sha256", {}).items())
            return bool(hashes_match and proof.get("status_counts", {}).get("success") == 64
                        and proof.get("status_counts", {}).get("failed") == 0)
        except (OSError, ValueError):
            return False
    if stage == "prophet":
        try:
            meta = json.loads((root / "expanded_prophet_block_conformal/run_metadata.json").read_text(encoding="utf-8"))
            return int(meta.get("failed_fits", -1)) == 0 and int(meta.get("folds", 0)) == 16
        except (OSError, ValueError):
            return False
    if stage == "inventory-alternatives":
        try:
            meta = json.loads((root / "expanded_inventory_calibration_v2/run_metadata.json").read_text(encoding="utf-8"))
            source = resolve_path(Path("artifacts/expanded_intermittent/forecasts.csv"))
            return meta.get("source_sha256") == file_sha256(source)
        except (OSError, ValueError):
            return False
    if stage == "selection":
        try:
            manifest = json.loads((root / "final/DEVELOPMENT_SELECTION.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "DEVELOPMENT_SELECTION_FROZEN_BEFORE_FINAL_HOLDOUT":
                return False
            if manifest.get("config_sha256") != file_sha256(resolve_path(Path("config.yaml"))):
                return False
            artifacts_match = all(file_sha256(resolve_path(Path(path))) == expected
                                  for path, expected in manifest.get("artifact_sha256", {}).items())
            source_match = all(file_sha256(resolve_path(Path(path))) == expected
                               for path, expected in manifest.get("selection_source_code_sha256", {}).items())
            return bool(artifacts_match and source_match)
        except (OSError, ValueError):
            return False
    if stage == "comparison":
        try:
            audit = json.loads(resolve_path(Path("reports/final/development_artifact_audit.json")).read_text(encoding="utf-8"))
            return bool(audit.get("passed"))
        except (OSError, ValueError):
            return False
    if stage == "reliability":
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
            return bool(report.get("passed") and report.get("xgboost", {}).get("failed") == 0
                        and report.get("neural", {}).get("successful") == 64)
        except (OSError, ValueError):
            return False
    if stage == "reports":
        return (resolve_path(Path("reports/phase3/expanded_xgboost_pooled_ablation_mae.png")).is_file()
                and resolve_path(Path("artifacts/expanded_inventory_calibration_v2/calibration_model_summary.csv")).is_file())
    if stage == "dashboard-data":
        try:
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("dashboard_data_manifest.json"))
                for name, expected in manifest["files"].items():
                    digest = hashlib.sha256(archive.read(name)).hexdigest()
                    if digest != expected:
                        return False
            return True
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            return False
    return output.exists()


def run_development(*, resume: bool = False) -> dict[str, str]:
    """Run the reproducible data and expanded-model development stages in order."""
    validate_raw_inputs()
    outcomes: dict[str, str] = {"validate": "complete"}
    for stage in ("data", "classify", "seasonal", "intermittent", "xgboost", "neural", "prophet",
                  "comparison", "reliability", "inventory", "inventory-alternatives", "selection", "reports", "dashboard-data"):
        outcomes[stage] = run_stage(stage, resume=resume)
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES) + ("validate", "preprocess", "experiments", "development", "all-development"))
    parser.add_argument("--resume", action="store_true", help="skip a stage whose declared summary output already exists")
    args = parser.parse_args(argv)
    if args.stage == "validate":
        print(validate_raw_inputs())
    elif args.stage in {"development", "all-development"}:
        print(run_development(resume=args.resume))
    elif args.stage == "preprocess":
        print(f"preprocess: {run_stage('data', resume=args.resume)}")
    elif args.stage == "experiments":
        outcomes = {name: run_stage(name, resume=args.resume) for name in ("seasonal", "intermittent", "xgboost", "neural", "prophet")}
        print(outcomes)
    elif args.stage == "dashboard-data":
        print(f"dashboard-data: {run_stage('dashboard-data', resume=args.resume)}")
    else:
        print(f"{args.stage}: {run_stage(args.stage, resume=args.resume)}")
    print("Final holdout status remains unchanged; use src.evaluation.final_holdout for its separately gated workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
