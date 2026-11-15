"""Select development-only aggregate and demand-class candidates using predeclared rules.

The output is a research freeze candidate manifest, not a final-holdout evaluation record.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.paths import resolve_path
from src.utils.provenance import file_sha256, write_json


GROUPS = ("aggregate", "Smooth", "Erratic", "Intermittent", "Lumpy")
CLASSES = GROUPS[1:]
FOLDS = tuple(range(1, 17))
MIN_CELL_SUCCESS = 0.95
PRIMARY = "MAE"
SECONDARY = ("RMSE", "WAPE", "MASE", "abs_bias")


def _item_class_table() -> pd.DataFrame:
    src = resolve_path(Path("artifacts/expanded_intermittent/fold_metrics.csv"))
    base = pd.read_csv(src)
    xgb = pd.read_csv(resolve_path(Path("artifacts/expanded_xgboost/item_fold_metrics.csv")))
    item = pd.concat([
        base[["model", "fold", "item_id", "fold_local_class", "success", "MAE", "RMSE", "WAPE", "MASE", "bias"]],
        xgb[["model", "fold", "item_id", "fold_local_class", "success", "MAE", "RMSE", "WAPE", "MASE", "bias"]],
    ], ignore_index=True)
    records = []
    for (model, fold, cls), group in item.groupby(["model", "fold", "fold_local_class"], sort=False):
        ok = group["success"].astype(bool)
        record = {"model": str(model), "fold": int(fold), "fold_local_class": str(cls),
                  "n_items": int(group["item_id"].nunique()), "n_success": int(ok.sum()),
                  "cell_success_rate": float(ok.mean())}
        for metric in ("MAE", "RMSE", "WAPE", "MASE", "bias"):
            record[metric] = float(group.loc[ok, metric].mean()) if ok.any() else np.nan
        records.append(record)
    return pd.DataFrame(records)


def _aggregate_candidates() -> pd.DataFrame:
    rows = []
    seasonal = pd.read_csv(resolve_path(Path("artifacts/expanded_seasonal/model_summary.csv")))
    for row in seasonal.itertuples(index=False):
        rows.append({"model": f"{row.model_family}({row.selection})", "source": "expanded_seasonal/model_summary.csv",
                     "folds": int(row.folds_scored), "failures": int(row.folds_failed),
                     "MAE": float(row.MAE_mean), "RMSE": float(row.RMSE_mean), "WAPE": float(row.WAPE_mean),
                     "MASE": float(row.MASE_mean), "bias": float(row.bias_mean),
                     "information_set": "aggregate lags only; fold-local classical settings where marked"})
    neural = pd.read_csv(resolve_path(Path("artifacts/expanded_neural/model_summary.csv")))
    for row in neural.itertuples(index=False):
        rows.append({"model": row.model, "source": "expanded_neural/model_summary.csv",
                     "folds": int(row.folds_scored), "failures": int(row.folds_failed),
                     "MAE": float(row.MAE_mean), "RMSE": float(row.RMSE_mean), "WAPE": np.nan,
                     "MASE": np.nan, "bias": float(row.bias_mean),
                     "information_set": "aggregate lag history; calendar/event features for calendar variants"})
    prophet = pd.read_csv(resolve_path(Path("artifacts/expanded_prophet_block_conformal/model_summary.csv")))
    for row in prophet.itertuples(index=False):
        rows.append({"model": row.model, "source": "expanded_prophet_block_conformal/model_summary.csv",
                     "folds": int(row.folds_scored), "failures": int(row.folds_failed),
                     "MAE": float(row.MAE_mean), "RMSE": float(row.RMSE_mean), "WAPE": np.nan,
                     "MASE": np.nan, "bias": float(row.bias_mean),
                     "information_set": "aggregate history; Prophet calendar/event regressors per variant"})
    table = pd.DataFrame(rows)
    table = table[(table["folds"] == 16) & (table["failures"] == 0)].copy()
    return table.sort_values(["MAE", "RMSE", "WAPE", "MASE"], na_position="last").reset_index(drop=True)


def _class_candidates() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    table = _item_class_table()
    tables: dict[str, pd.DataFrame] = {}
    selected = []
    for cls in CLASSES:
        sub = table[table["fold_local_class"] == cls]
        rows = []
        for model, group in sub.groupby("model", sort=True):
            group = group.sort_values("fold")
            full = set(group["fold"].astype(int)) == set(FOLDS)
            reliable = bool((group["cell_success_rate"] >= MIN_CELL_SUCCESS).all())
            complete_classes = bool((group["n_items"] >= 5).all())
            full_population = not str(model).startswith("XGBoost(per-item")
            row = {"model": model, "selection_group": cls, "folds": int(group["fold"].nunique()),
                   "mean_items_per_fold": float(group["n_items"].mean()),
                   "minimum_fold_success_rate": float(group["cell_success_rate"].min()),
                   "MAE": float(group["MAE"].mean()), "MAE_fold_sd": float(group["MAE"].std(ddof=1)),
                   "RMSE": float(group["RMSE"].mean()), "WAPE": float(group["WAPE"].mean()),
                   "MASE": float(group["MASE"].mean()), "bias": float(group["bias"].mean()),
                   "eligible": bool(full and reliable and complete_classes and full_population),
                   "eligibility_note": "fixed 183-item XGBoost sample excluded from whole-population class selection" if not full_population else "",
                   "reliability_rule": "all 16 folds; at least 5 items/class/fold; at least 95% successful item-fold cells in every fold; full 823-item panel for selection"}
            rows.append(row)
        result = pd.DataFrame(rows).sort_values(["eligible", "MAE", "RMSE", "WAPE", "MASE"],
                                                ascending=[False, True, True, True, True]).reset_index(drop=True)
        candidates = result[result["eligible"]]
        if candidates.empty:
            raise RuntimeError(f"No eligible model for the {cls} selection group")
        winner = candidates.iloc[0]
        result["selected"] = result["model"].eq(winner["model"])
        selected.append(winner)
        tables[cls] = result
    return pd.DataFrame(selected), tables


def _settings(model: str, config: dict[str, Any], xgb_manifest: dict[str, Any]) -> dict[str, Any]:
    if model.startswith("XGBoost(pooled,"):
        spec = next(v for v in xgb_manifest["settings"]["feature_variants"] if v["key"] in model)
        params = dict(xgb_manifest["settings"]["xgboost_hyperparameters"])
        params["objective"] = spec["objective"]
        return {"scope": "pooled", "features": dict(spec), "hyperparameters": params,
                "lags": xgb_manifest["settings"]["lags"], "rolling_windows": xgb_manifest["settings"]["rolling_windows"]}
    models = config["models"]
    if model.startswith("TSB"):
        return {"alpha": models["tsb_alpha"], "beta": models["tsb_beta"]}
    if model.startswith("SBA"):
        return {"alpha": models["sba_alpha"]}
    if model.startswith("Croston"):
        return {"alpha": models["croston_alpha"]}
    if model == "SeasonalNaive":
        return {"season_length": config["backtest"]["season_length"]}
    raise ValueError(f"No settings recorder for selected item candidate {model}")


def run(output: Path | None = None) -> dict[str, Any]:
    config = load_config()
    aggregate = _aggregate_candidates()
    if aggregate.empty:
        raise RuntimeError("no complete aggregate candidate")
    aggregate_winner = aggregate.iloc[0]
    class_winners, class_tables = _class_candidates()
    xgb_manifest = json.loads(resolve_path(Path("artifacts/expanded_xgboost/fold_manifest.json")).read_text(encoding="utf-8"))
    config_path = resolve_path(Path("config.yaml"))
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = "unavailable"
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    files = [
        "config.yaml", "artifacts/expanded_seasonal/model_summary.csv", "artifacts/expanded_seasonal/fold_manifest.json",
        "artifacts/expanded_neural/model_summary.csv", "artifacts/expanded_neural/forecasts.csv",
        "artifacts/expanded_prophet_block_conformal/model_summary.csv", "artifacts/expanded_xgboost/independent_audit.json",
        "artifacts/expanded_xgboost/item_fold_metrics.csv", "artifacts/expanded_xgboost/fold_manifest.json",
        "artifacts/expanded_intermittent/fold_metrics.csv", "artifacts/expanded_intermittent/fold_class_summary.csv",
        "artifacts/expanded_inventory_calibration_v2/calibration_model_summary.csv",
        "artifacts/expanded_inventory_calibration_v2/calibration_model_class_summary.csv",
    ]
    hashes = {p: file_sha256(resolve_path(Path(p))) for p in files}
    class_manifest = []
    for row in class_winners.itertuples(index=False):
        slug = re.sub(r"[^a-z0-9]+", "_", row.model.lower()).strip("_")
        class_manifest.append({"candidate_id": f"{row.selection_group.lower()}_{slug}",
                               "selection_group": row.selection_group, "model": row.model,
                               "settings": _settings(row.model, config, xgb_manifest),
                               "horizon_days": 28, "development_folds": list(FOLDS),
                               "mean_items_per_fold": row.mean_items_per_fold,
                               "minimum_fold_success_rate": row.minimum_fold_success_rate,
                               "development_metrics": {k: getattr(row, k) for k in ("MAE", "MAE_fold_sd", "RMSE", "WAPE", "MASE", "bias")},
                               "selection_criteria": {"primary": PRIMARY, "secondary": list(SECONDARY),
                                                      "fold_stability": "fold MAE SD reported; lower MAE fold counts retained in class comparison CSV"},
                               "selection_rationale": "eligible under 16-fold/95% reliability rule and lowest mean class-specific MAE; item-level macros are not compared with aggregate errors",
                               "information_set": "item sales history; XGBoost candidates additionally use calendar/event flags where specified"})
    # Items with no positive sales in the available development history are not
    # treated as an estimable intermittent-demand class. Route them to the
    # zero-producing seasonal-naive baseline using development-only labels.
    class_manifest.append({"candidate_id": "no_demand_seasonalnaive", "selection_group": "NoDemand",
                           "model": "SeasonalNaive", "settings": _settings("SeasonalNaive", config, xgb_manifest),
                           "horizon_days": 28, "development_folds": list(FOLDS),
                           "development_metrics": {"status": "deterministic fallback; not performance-selected"},
                           "selection_criteria": {"primary": "not applicable", "fallback": "no positive development demand"},
                           "selection_rationale": "NoDemand items receive the seasonal-naive zero-demand fallback; this is not ranked as a fifth performance-selected demand class.",
                           "information_set": "development sales history only"})
    chosen_aggregate = {"candidate_id": "aggregate_" + str(aggregate_winner["model"]).lower().replace(" ", "_").replace("(", "_").replace(")", ""),
                       "selection_group": "aggregate", "model": str(aggregate_winner["model"]),
                       "settings": {"cell": "lstm", "features": "calendar", **{k: v for k, v in config["neural"].items() if k != "variants"}},
                       "horizon_days": 28, "development_folds": list(FOLDS),
                       "development_metrics": {k: float(aggregate_winner[k]) for k in ("MAE", "RMSE", "bias")},
                       "reliability": {"folds_scored": 16, "folds_failed": 0, "seed_members_per_fold": 5},
                       "selection_criteria": {"primary": PRIMARY, "secondary": list(SECONDARY),
                                              "eligibility": "16/16 common folds and zero failed fits"},
                       "selection_rationale": "lowest mean MAE among complete aggregate candidates across the shared 2012–2015 quarterly windows; descriptive only",
                       "information_set": "aggregate history plus calendar/event regressors; richer than lag-only classical models"}
    manifest = {
        "schema_version": 1, "status": "DEVELOPMENT_SELECTION_FROZEN_BEFORE_FINAL_HOLDOUT",
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "holdout_status": "not evaluated; prior loader transient-read caveat documented in reports/final/HOLDOUT_DATA_BOUNDARY_AUDIT.md",
        "training_window": {"start": "2011-01-29", "end": "2016-04-24"}, "horizon_days": 28,
        "git_sha": git_sha, "config_sha256": file_sha256(config_path),
        "working_tree_dirty_at_selection": dirty,
        "selection_source_code_sha256": {p: file_sha256(resolve_path(Path(p))) for p in (
            "src/evaluation/select_development_candidates.py", "src/evaluation/expanded_xgboost.py",
            "src/evaluation/expanded_neural.py", "src/evaluation/expanded_prophet.py",
            "src/evaluation/expanded_seasonal.py", "src/evaluation/expanded_intermittent.py",
            "src/evaluation/expanded_inventory_calibration.py", "src/inventory/reorder.py",
            "src/models/xgboost_model.py", "src/models/lstm_gru.py",
        )},
        "dataset_sha256": {"aggregate": json.loads(resolve_path(Path("artifacts/expanded_seasonal/fold_manifest.json")).read_text(encoding="utf-8"))["development_series_sha256"],
                           "item_panel": xgb_manifest["development_panel_sha256"]},
        "artifact_sha256": hashes,
        "selection_rules": {"primary": PRIMARY, "secondary": list(SECONDARY),
                            "aggregate_eligibility": "16 common quarterly folds, zero fit failures",
                            "class_eligibility": "all 16 folds; at least 5 class items per fold; at least 95% successful item-fold cells in every fold",
                            "selection": "lowest mean fold-level macro MAE among eligible models, then RMSE, WAPE, MASE; bias and fold SD disclosed",
                            "limitations": "descriptive fold comparisons only; information sets differ; no statistical significance claimed"},
        "candidates": [chosen_aggregate, *class_manifest],
        "class_candidate_map": {**{str(c["selection_group"]): str(c["model"]) for c in class_manifest},
                                "NoDemand": "SeasonalNaive"},
        "dashboard_calibration": {"target_cycle_service": 0.95, "selected_method": "empirical_all",
                                  "selection_rationale": "simple per-model empirical calibration has more historical cells and near-tied development service; no method reaches 95%, and class-normalized slight gain does not establish robust superiority"},
    }
    out = resolve_path(output or Path("artifacts/final/DEVELOPMENT_SELECTION.json"))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection manifest: {out}")
    write_json(out, manifest)
    aggregate.to_csv(resolve_path(Path("artifacts/final/aggregate_selection_comparison.csv")), index=False, lineterminator="\n")
    class_table = pd.concat(class_tables.values(), ignore_index=True)
    class_table.to_csv(resolve_path(Path("artifacts/final/class_selection_comparison.csv")), index=False, lineterminator="\n")
    lines = ["# Development candidate selection and freeze", "",
             "Selection used development-only folds before any final-period scoring. Fold means are descriptive; no significance claim is made.", "",
             "## Rules", "",
             "- Primary metric: MAE. Secondary metrics: RMSE, WAPE, MASE and bias; report fold SD and failures.",
             "- Aggregate candidates require all 16 common quarterly folds and zero failed fits.",
             "- Demand-class candidates require all 16 folds, at least five items per fold, and at least 95% successful item-fold cells in each fold.",
             "- Compare equal-weight class/item-fold results only within the item scope. Do not combine with aggregate units.",
             "- Calendar/event features are a richer information set and are identified in the manifest.", "",
             "## Selected candidates", "",
             "| Group | Candidate | MAE | RMSE | Fold coverage | Rationale |", "|---|---|---:|---:|---:|---|"]
    a = manifest["candidates"][0]
    lines.append(f"| Aggregate | {a['model']} | {a['development_metrics']['MAE']:.3f} | {a['development_metrics']['RMSE']:.3f} | 16/16 | Lowest MAE among complete aggregate candidates |")
    for c in manifest["candidates"][1:]:
        if c["selection_group"] == "NoDemand":
            continue
        m = c["development_metrics"]
        lines.append(f"| {c['selection_group']} | {c['model']} | {m['MAE']:.3f} | {m['RMSE']:.3f} | 16/16 | Lowest eligible class-specific mean MAE |")
    lines.append("| NoDemand | SeasonalNaive | — | — | development-only fallback | Deterministic zero-demand fallback; not performance-ranked |")
    lines += ["", "The development selection manifest is `artifacts/final/DEVELOPMENT_SELECTION.json`. It records exact settings, data/config/artifact hashes and selection rationale. The protected `FROZEN_CANDIDATES.json` is written separately by the final-holdout freeze command. Neither step evaluates the holdout.", "",
              "## Inventory calibration", "",
              "All-history empirical error quantiles remain the conservative default: recency weighting and a two-fold history were worse; class-normalized calibration was close but did not prove a robust gain. Every evaluated method remained below the 95% target overall. Model and fold-local class tables are in `artifacts/expanded_inventory_calibration_v2/`.", "",
              "## Limits", "",
              "The demand classes are fold-local and the number of items per class changes by origin. XGBoost has known calendar/event information that some baselines do not. The per-item XGBoost sample is not selected for whole-class deployment because it covers a fixed 183-item subset only. Inventory service is cycle service, not fill rate. The final holdout remains not evaluated; earlier transient reads are disclosed in `HOLDOUT_DATA_BOUNDARY_AUDIT.md`.", ""]
    report = resolve_path(Path("reports/final/MODEL_SELECTION_AND_FREEZE_REPORT.md"))
    report.write_text("\n".join(lines), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "selected": [x["model"] for x in result["candidates"]]}, indent=2))
