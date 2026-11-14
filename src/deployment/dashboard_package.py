"""Build a small raw-data-free archive containing the files read by RiverDemand's dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Sequence

from src.evaluation.model_comparison import aggregate_sources, item_sources
from src.deployment.development_data import (
    AGGREGATE_NAME, ALL_ITEMS_NAME, MANIFEST_NAME as DEV_MANIFEST_NAME,
    OUTPUT_DIR as DEV_DATA_DIR, SELECTED_ITEMS_NAME, build_development_data,
)
from src.utils.config import load_config
from src.utils.paths import resolve_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(output: Path | None = None) -> Path:
    config = load_config()
    build_development_data(config)
    required = [Path("config.yaml")]
    required.extend(DEV_DATA_DIR / name for name in (AGGREGATE_NAME, ALL_ITEMS_NAME, SELECTED_ITEMS_NAME, DEV_MANIFEST_NAME))
    required.extend(Path(config["phase3"]["forecasts_dir"]) / s.filename for s in aggregate_sources(config) + item_sources(config))
    metrics_dir = Path(config["phase3"]["metrics_dir"])
    required.extend(metrics_dir / path.name for path in sorted(resolve_path(metrics_dir).iterdir()) if path.is_file())
    inventory_dir = Path(config["phase3"]["inventory_dir"])
    required.extend(inventory_dir / path.name for path in sorted(resolve_path(inventory_dir).iterdir()) if path.is_file())
    # The expanded comparison page only needs compact summaries and fold tables, not the 175 MB daily forecast file.
    required.extend(Path("artifacts/expanded_seasonal") / name for name in ("model_summary.csv", "legacy_vs_fold_local.csv"))
    required.extend(Path("artifacts/expanded_neural") / name for name in (
        "model_summary.csv", "fold_metrics.csv", "training_diagnostics.csv", "forecasts.csv",
        "seed_metrics.csv", "training_history.csv", "run_metadata.json", "completion_verification.json",
    ))
    required.extend(Path("artifacts/expanded_xgboost") / name for name in (
        "pooled_model_summary.csv", "matched_sample_model_summary.csv", "paired_fold_comparisons.csv",
        "feature_importance_summary.csv", "pooled_fold_metrics.csv",
        "item_fold_metrics.csv", "fold_manifest.json", "independent_audit.json",
    ))
    xgb_audit = resolve_path(Path("artifacts/expanded_xgboost/independent_audit.json"))
    if xgb_audit.is_file():
        audited_files = json.loads(xgb_audit.read_text(encoding="utf-8")).get("artifact_sha256", {})
        required.extend(Path("artifacts/expanded_xgboost") / name for name in audited_files if name != "forecasts.csv")
    required.extend(Path("artifacts/expanded_intermittent") / name for name in (
        "model_summary.csv", "class_summary.csv", "fold_metrics.csv", "fold_class_summary.csv",
    ))
    required.extend(Path("artifacts/expanded_prophet_block_conformal") / name for name in (
        "model_summary.csv", "fold_metrics.csv", "interval_calibration_summary.csv", "interval_calibration_validation.csv",
    ))
    expanded_inventory = resolve_path(Path("artifacts/expanded_inventory/service_level_summary.csv"))
    if expanded_inventory.exists():
        required.append(Path("artifacts/expanded_inventory/service_level_summary.csv"))
    alternative_inventory = resolve_path(Path("artifacts/expanded_inventory_calibration_v2/calibration_summary.csv"))
    if alternative_inventory.exists():
        required.extend(Path("artifacts/expanded_inventory_calibration_v2") / name for name in (
            "calibration_summary.csv", "fold_class_summary.csv", "calibration_model_summary.csv",
            "calibration_model_class_summary.csv",
        ))
    final_dir = resolve_path(Path(config["phase3"]["final_dir"]))
    development_selection = final_dir / "DEVELOPMENT_SELECTION.json"
    frozen_candidates = final_dir / "FROZEN_CANDIDATES.json"
    if development_selection.is_file():
        required.append(Path(config["phase3"]["final_dir"]) / "DEVELOPMENT_SELECTION.json")
        selection_manifest = json.loads(development_selection.read_text(encoding="utf-8"))
        required.extend(Path(name) for name in selection_manifest.get("artifact_sha256", {}))
    if frozen_candidates.is_file():
        required.append(Path(config["phase3"]["final_dir"]) / "FROZEN_CANDIDATES.json")
    final_summary_names = (config["final_holdout"]["results_filename"],
                           config["final_holdout"].get("class_results_filename", "final_holdout_class_results.csv"),
                           config["final_holdout"]["record_filename"])
    required.extend(Path(config["phase3"]["final_dir"]) / name for name in final_summary_names if (final_dir / name).is_file())
    missing = [path.as_posix() for path in required if not resolve_path(path).is_file()]
    if missing:
        raise FileNotFoundError("dashboard package is incomplete: " + ", ".join(missing))
    if any("raw" in path.parts for path in required):
        raise ValueError("raw M5 inputs must not be included in the dashboard package")
    output = output or resolve_path(Path("reports/final/dashboard_data_bundle_final_evaluation.zip"))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {path.as_posix(): _sha256(resolve_path(path)) for path in sorted(set(required))}
    manifest_bytes = json.dumps({"schema_version": 1, "raw_data_included": False, "files": manifest}, indent=2).encode("utf-8")
    readme = (
        "RiverDemand dashboard data package\n\n"
        "Extract this archive at the repository root. It contains processed development data, saved dashboard artifacts, "
        "and when present, compact final-holdout summary metrics and the evaluation record. It contains no raw M5 inputs "
        "or row-level final-holdout forecasts. Verify file hashes using dashboard_data_manifest.json.\n"
    ).encode("utf-8")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for path in sorted(set(required)):
            archive.write(resolve_path(path), arcname=path.as_posix())
        archive.writestr("dashboard_data_manifest.json", manifest_bytes)
        archive.writestr("DASHBOARD_DATA_PACKAGE_README.txt", readme)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    path = package(args.output)
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
