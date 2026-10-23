"""Notebook load + real end-to-end execution test for Phase 3 Step 1 (XGBoost).

`test_notebook_executes_end_to_end_against_real_data` runs
notebooks/06_xgboost.ipynb for real, against the real CA_1/FOODS_3 development
data (not a stub series), via nbclient. It executes a fresh in-memory copy:
it neither overwrites the delivered notebook file nor depends on that file
already having been executed.

It checks structure and integrity facts the notebook prints -- row counts,
zero failed folds, identical evaluation grid, bit-identical repeat run, the
holdout-perturbation check -- and deliberately does NOT assert any metric
value. Last-digit floating-point results can differ between operating
systems, CPUs and xgboost versions, and a model-quality number is not a
property this test should pin.

Slow by nature (about two minutes: one pooled fit per fold for four model
variants, 100 per-item fits, three more pooled runs for the integrity
checks, and the Phase 2 baselines re-run for comparison). The test is
skipped when the real data files are not present (they are git-ignored and
rebuilt by the Phase 1 pipeline).
"""

from __future__ import annotations

import nbformat
import pytest
from nbclient import NotebookClient

from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_xgboost.ipynb"


def _all_text(nb) -> str:
    """Everything a reader of the executed notebook sees: printed output and displayed results."""
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


def _error_outputs(nb):
    return [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]


def test_notebook_loads_as_valid_notebook():
    assert NOTEBOOK_PATH.exists(), f"{NOTEBOOK_PATH} not found."
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(nb)
    assert any(c.cell_type == "code" for c in nb.cells)
    markdown = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Phase 3" in markdown and "XGBoost" in markdown
    assert "not complete" in markdown.lower()      # the notebook must not present Phase 3 as finished
    assert "Walk-forward protocol" in markdown and "Final holdout" in markdown


def test_delivered_notebook_was_executed_and_raised_no_errors():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.get("outputs") for c in code_cells), "a code cell in the delivered notebook has no stored output"
    assert _error_outputs(nb) == []


def test_notebook_executes_end_to_end_against_real_data():
    config = load_config()
    required = [
        PROJECT_ROOT / config["data"]["processed_dir"] / config["output"]["selected_items_daily_sales_filename"],
        PROJECT_ROOT / config["data"]["interim_dir"] / config["output"]["panel_filename"],
        PROJECT_ROOT / config["phase2"]["forecasts_dir"] / config["phase2"]["item_level_long_results_filename"],
        PROJECT_ROOT / config["phase2"]["metrics_dir"] / config["phase2"]["item_level_fold_summary_filename"],
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(f"real project data / Phase 2 artifacts not present: {missing}")

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()  # raises CellExecutionError on any failing cell

    assert _error_outputs(nb) == []
    text = _all_text(nb)

    # Real run, exact sizes: 20 items x 5 folds x 28 days, one row set per model variant.
    assert "Long results table: 2800 rows (expected 2800)" in text
    assert "Fold summary table: 100 rows (expected 100)" in text
    assert "Failed (item, fold) combinations: 0" in text
    assert "NaN / infinite / negative predictions in the saved forecasts: 0" in text
    assert "Failed folds across every XGBoost variant: 0" in text

    # Same folds, items and days as the Phase 2 baselines, which themselves reproduce.
    assert "Fold dates are identical to the Phase 2 item-level artifacts" in text
    assert "reproduces the saved Phase 2 fold metrics" in text
    assert "Same evaluation grid for baselines and every XGBoost variant: 2800 (item, fold, date) cells" in text

    # Leakage checks shown on real data, not just asserted in prose.
    assert "Hand check passed" in text and "Leakage probe passed" in text
    assert "Determinism: a second run with the same seed gives bit-identical forecasts." in text
    assert (
        "Holdout perturbation: replacing all 560 holdout sales values by 1,000,000 changes no forecast and no actual."
        in text
    )
    assert "Confirmed: no date on or after the final holdout start was used anywhere in this notebook." in text
    assert "byte-for-byte unchanged by this notebook" in text

    # What the notebook is and is not claiming.
    assert "Phase 3 is NOT complete" in text
    assert "Phase 3 is complete" not in text
    assert "not a final ranking" in text
    assert "must not be compared" in "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")

    # The deliverables were written, and none of them is a Phase 2 file name.
    p3 = config["phase3"]
    forecasts_dir = PROJECT_ROOT / p3["forecasts_dir"]
    metrics_dir = PROJECT_ROOT / p3["metrics_dir"]
    reports_dir = PROJECT_ROOT / config["reports"]["phase3_dir"]
    expected = [
        forecasts_dir / p3["xgboost_long_results_filename"],
        metrics_dir / p3["xgboost_fold_summary_filename"],
        metrics_dir / p3["xgboost_per_item_summary_filename"],
        metrics_dir / p3["xgboost_category_summary_filename"],
        metrics_dir / p3["xgboost_comparison_per_item_filename"],
        metrics_dir / p3["xgboost_comparison_category_filename"],
        metrics_dir / p3["xgboost_fold_diagnostics_filename"],
        metrics_dir / p3["xgboost_fold_item_classes_filename"],
        metrics_dir / p3["xgboost_feature_importance_filename"],
        reports_dir / p3["xgboost_category_mae_plot_filename"],
        reports_dir / p3["xgboost_category_rmse_plot_filename"],
        reports_dir / p3["xgboost_example_forecasts_plot_filename"],
        reports_dir / p3["xgboost_feature_importance_plot_filename"],
    ]
    for path in expected:
        assert path.exists() and path.stat().st_size > 0, f"missing deliverable: {path}"
    phase2_names = {v for k, v in config["phase2"].items() if isinstance(v, str) and "." in v}
    assert not ({p.name for p in expected} & phase2_names), "a Phase 3 output reuses a Phase 2 file name"
