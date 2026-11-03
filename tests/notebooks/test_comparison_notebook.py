"""Notebook 09 (unified comparison): delivered-file checks and an independent re-derivation of what it saved.

The notebook computes every number from saved forecast files. These tests recompute the headline numbers a second way, straight from
the forecast CSVs, so a bug in the comparison layer cannot confirm itself. Artifacts under `artifacts/` are git-ignored and rebuilt by
running the notebook; tests that need them are skipped (with the command to create them) when they are absent.
"""

from __future__ import annotations

import json
import re

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.evaluation.forecast_artifacts import forecasts_dir, metrics_by_group
from src.evaluation.model_comparison import aggregate_sources, item_sources
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "09_model_comparison.ipynb"
CONFIG = load_config()
PHASE3 = CONFIG["phase3"]
METRICS = PROJECT_ROOT / PHASE3["metrics_dir"]
RUN_COMMAND = "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/09_model_comparison.ipynb"
AGG_MODELS = {
    "SeasonalNaive", "ARIMA(6, 1, 1)", "SARIMA(6, 1, 1)x(0, 0, 2, 7)", "HoltWinters(trend=add)",
    "LSTM(calendar)", "GRU(calendar)", "LSTM(univariate)", "GRU(univariate)",
    "Prophet(additive)", "Prophet(multiplicative)", "Prophet(additive,no-regressors)",
}
ITEM_MODELS = {
    "SeasonalNaive", "Croston(alpha=0.1)", "TSB(alpha=0.1,beta=0.1)",
    "XGBoost(pooled)", "XGBoost(per-item)", "XGBoost(pooled,no-events)", "XGBoost(pooled,poisson)",
}


def _text(nb) -> str:
    parts = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts.append(out.get("text", ""))
            elif out.get("output_type") in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
                parts.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(parts)


def _markdown(nb) -> str:
    return "\n".join(c.source for c in nb.cells if c.cell_type == "markdown").lower()


def _need(name):
    path = METRICS / name
    if not path.exists():
        pytest.skip(f"{name} not found. Create it with: {RUN_COMMAND}")
    return path


# --------------------------------------------------------------------------------------------------------------- the notebook
def test_notebook_is_the_executed_deliverable_not_the_placeholder():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    assert len(code) >= 12 and all(c.execution_count for c in code)
    assert "not yet implemented" not in _markdown(nb)


def test_notebook_defines_its_measures_and_handles_zeros_explicitly():
    prose = _markdown(nbformat.read(NOTEBOOK_PATH, as_version=4))
    for phrase in ("mase", "wape", "mape", "zero", "lag-7", "macro", "mae", "fold", "scale"):
        assert phrase in prose, f"notebook 09 never explains {phrase!r}"
    assert re.search(r"mape.{0,200}zero|zero.{0,200}mape", prose, flags=re.S), "MAPE and its zero handling are never discussed together"


def test_notebook_never_ranks_aggregate_and_item_level_models_together():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    combined = (_markdown(nb) + "\n" + _text(nb)).lower()
    assert "never ranked" in combined or "not comparable" in combined or "never compared" in combined
    assert "aggregate series: 11 models compared; item level: 7 models compared" in combined


def test_notebook_reports_missing_artifacts_explicitly_and_substitutes_nothing():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    text = _text(nb)
    assert "Unavailable artifacts: 0" in text
    assert "nothing is substituted" in (_markdown(nb) + text.lower()) or "absent from the comparison" in (_markdown(nb) + text.lower())
    assert "Failed folds in the saved tables: 0 of 55" in text and "Failed item-model folds in the saved tables: 0 of 700" in text


def test_notebook_claims_no_significance_and_no_universal_winner_and_never_uses_the_holdout():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    combined = (_markdown(nb) + "\n" + _text(nb)).lower()
    assert "no significance test" in combined
    assert "never loaded here" in combined or "not used" in combined
    for phrase in ("universal winner", "significantly better", "best overall"):
        for line in combined.splitlines():
            if phrase in line:
                assert re.search(r"\bnot?\b|\bnothing\b|\bwithout\b", line), f"unnegated claim {phrase!r}: {line.strip()[:160]}"


def test_notebook_outputs_contain_no_errors_or_machine_paths():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    assert not [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not re.search(r"/home/|/tmp/|/Users/|[A-Za-z]:\\\\", _text(nb))


# ----------------------------------------------------------------------------------------------------------- saved artifacts
@pytest.fixture(scope="module")
def agg_summary():
    return pd.read_csv(_need(PHASE3["comparison_aggregate_summary_filename"]))


@pytest.fixture(scope="module")
def agg_folds():
    return pd.read_csv(_need(PHASE3["comparison_aggregate_fold_filename"]))


def test_aggregate_summary_has_every_model_five_folds_and_the_required_measures(agg_summary):
    assert set(agg_summary["model"]) == AGG_MODELS and len(agg_summary) == 11
    assert agg_summary["folds_scored"].eq(5).all() and agg_summary["folds_failed"].eq(0).all()
    for measure in ("MAE", "RMSE", "MAPE", "WAPE", "MASE", "bias"):
        for stat in ("mean", "std", "min", "max"):
            assert f"{measure}_{stat}" in agg_summary.columns
    assert (agg_summary["MAE_min"] <= agg_summary["MAE_mean"]).all() and (agg_summary["MAE_mean"] <= agg_summary["MAE_max"]).all()
    assert (agg_summary["MAE_std"] > 0).all()
    assert (agg_summary["mape_zero_days_excluded"] >= 0).all(), "MAPE zero handling must be reported"


def test_fold_level_values_are_saved_for_every_model_and_fold(agg_folds):
    assert len(agg_folds) == 55 and set(agg_folds["model"]) == AGG_MODELS
    assert agg_folds.groupby("model")["fold"].apply(lambda s: sorted(s)).apply(lambda v: v == [1, 2, 3, 4, 5]).all()
    assert agg_folds["success"].all()


def _forecast_files_changed_since_notebook_09_ran() -> list[str]:
    """Names of saved forecast files whose SHA-256 no longer matches the one notebook 09 recorded in its run metadata."""
    import hashlib

    meta_path = PROJECT_ROOT / "artifacts" / "metrics" / "comparison_run_metadata.json"
    recorded = json.loads(meta_path.read_text(encoding="utf-8"))["details"]["artifact_sha256_prefix"]
    changed = []
    for name, prefix in recorded.items():
        path = forecasts_dir(CONFIG) / name
        if not path.exists() or not hashlib.sha256(path.read_bytes()).hexdigest().startswith(prefix):
            changed.append(name)
    return changed


def test_every_forecast_file_used_by_the_comparison_has_a_recorded_hash():
    meta = json.loads((PROJECT_ROOT / "artifacts" / "metrics" / "comparison_run_metadata.json").read_text(encoding="utf-8"))
    recorded = set(meta["details"]["artifact_sha256_prefix"])
    expected = {source.filename for source in aggregate_sources(CONFIG)} | {source.filename for source in item_sources(CONFIG)}
    assert expected <= recorded, f"no recorded hash for {sorted(expected - recorded)}"


def test_comparison_mae_equals_mae_recomputed_directly_from_the_forecast_files(agg_folds):
    config = CONFIG
    changed = _forecast_files_changed_since_notebook_09_ran()
    if changed:
        pytest.skip(
            f"{', '.join(changed)} changed after notebook 09 ran (for example because the notebook-04 execution test "
            "re-estimated SARIMA), so the saved comparison cannot be checked against it. Re-run notebook 09 to refresh: "
            "python -m jupyter nbconvert --to notebook --execute --inplace notebooks/09_model_comparison.ipynb"
        )
    frames = []
    for source in aggregate_sources(config):
        path = forecasts_dir(config) / source.filename
        if not path.exists():
            pytest.skip(f"{source.filename} not found")
        frames.append(pd.read_csv(path))
    raw = pd.concat(frames, ignore_index=True)
    direct = metrics_by_group(raw, ["model", "fold"])
    merged = agg_folds.merge(direct, on=["model", "fold"], suffixes=("_cmp", "_direct"))
    assert len(merged) == 55
    np.testing.assert_allclose(merged["MAE_cmp"], merged["MAE_direct"], rtol=1e-9)
    np.testing.assert_allclose(merged["RMSE_cmp"], merged["RMSE_direct"], rtol=1e-9)


def test_summary_mean_and_sample_std_follow_from_the_fold_table(agg_summary, agg_folds):
    by_model = agg_folds.groupby("model")["MAE"].agg(["mean", "std"])
    merged = agg_summary.set_index("model").join(by_model)
    np.testing.assert_allclose(merged["MAE_mean"], merged["mean"], rtol=1e-9)
    np.testing.assert_allclose(merged["MAE_std"], merged["std"], rtol=1e-9)  # std with ddof=1 (sample), as the notebook states


def test_wape_and_mase_are_defined_the_way_the_notebook_says(agg_folds):
    assert (agg_folds["WAPE"] > 0).all() and (agg_folds["WAPE"] < 1).all(), "WAPE = sum|error| / sum(actual) must be a share"
    assert (agg_folds["MASE"] > 0).all()
    # MASE is MAE divided by one positive training-window scale per fold, so for a fixed fold it must preserve the MAE ordering
    for _, part in agg_folds.groupby("fold"):
        order_mae = part.sort_values("MAE")["model"].tolist()
        order_mase = part.sort_values("MASE")["model"].tolist()
        assert order_mae == order_mase


def test_item_level_results_are_in_a_separate_scale_and_file():
    item = pd.read_csv(_need(PHASE3["comparison_item_class_filename"]))
    summary = pd.read_csv(_need("comparison_item_level_model_summary.csv"))
    agg = pd.read_csv(_need(PHASE3["comparison_aggregate_summary_filename"]))
    assert set(summary["model"]) == ITEM_MODELS and len(summary) == 7
    assert summary["MAE_mean"].max() < 20 < agg["MAE_mean"].min(), "item-level MAE (units per item per day) must not share a scale with aggregate MAE"
    assert set(item["demand_class"]) <= {"Smooth", "Erratic", "Intermittent", "Lumpy"}
    assert item["n_items"].sum() == 20 * len(ITEM_MODELS) or item.groupby("model")["n_items"].sum().eq(20).all()


def test_item_level_macro_average_gives_every_item_equal_weight():
    folds = pd.read_csv(_need(PHASE3["comparison_item_fold_filename"]))
    summary = pd.read_csv(_need("comparison_item_level_model_summary.csv")).set_index("model")
    assert len(folds) == 700 and folds["success"].all()
    per_item = folds.groupby(["model", "item_id"])["MAE"].mean()
    macro = per_item.groupby("model").mean()
    np.testing.assert_allclose(summary.loc[macro.index, "MAE_mean"], macro, rtol=1e-9)


def test_artifact_audit_lists_every_source_with_a_status():
    audit = pd.read_csv(_need(PHASE3["comparison_artifact_audit_filename"] if "comparison_artifact_audit_filename" in PHASE3 else "comparison_artifact_audit.csv"))
    expected = {s.filename for s in aggregate_sources(CONFIG)} | {s.filename for s in item_sources(CONFIG)}
    assert expected <= set(audit["artifact"])
    assert set(audit["status"]) == {"ok"}


def test_run_metadata_records_the_inputs_and_the_absence_of_significance_tests():
    meta = json.loads(_need(PHASE3["comparison_metadata_filename"]).read_text(encoding="utf-8"))
    assert meta["details"]["significance_tests"] in (False, "none", [], 0)
    assert meta["details"]["artifacts_unavailable"] in ([], 0)
    assert re.fullmatch(r"[0-9a-f]{64}", meta["config_sha256"])
    assert not re.search(r"/home/|/tmp/|/Users/|[A-Za-z]:\\\\", json.dumps(meta))


def test_the_four_comparison_figures_exist():
    for key in ("comparison_aggregate_plot_filename", "comparison_aggregate_folds_plot_filename", "comparison_item_plot_filename", "comparison_item_rmse_plot_filename"):
        path = PROJECT_ROOT / CONFIG["reports"]["phase3_dir"] / PHASE3[key]
        if not path.exists():
            pytest.skip(f"{path.name} not found. Create it with: {RUN_COMMAND}")
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 5_000
