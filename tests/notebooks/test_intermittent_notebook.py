"""Notebook 05 (Seasonal Naive / Croston / TSB): delivered-file checks, saved-artifact audit, real execution.

Artifacts under `artifacts/` are git-ignored and rebuilt by running the notebook, so tests that need them are
skipped (with the command that creates them) when they are absent. The slow test re-executes the notebook for
real, in memory, and asserts integrity facts it prints -- never a metric value.
"""

from __future__ import annotations

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.evaluation.forecast_artifacts import (
    audit_forecast_table,
    forecasts_dir,
    load_selected_item_series,
    metrics_by_group,
    metrics_dir,
    read_forecast_table,
)
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_intermittent_models.ipynb"
CONFIG = load_config()
PHASE3 = CONFIG["phase3"]
LONG = forecasts_dir(CONFIG) / PHASE3["intermittent_long_filename"]
RUN_COMMAND = "jupyter nbconvert --to notebook --execute --inplace notebooks/05_intermittent_models.ipynb"


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


def test_notebook_teaches_the_topic_and_states_its_limits():
    prose = _markdown(nbformat.read(NOTEBOOK_PATH, as_version=4))
    for phrase in (
        "intermittent", "croston", "tsb", "seasonal naive", "adi", "cv²", "zero run", "probability",
        "mape", "wape", "mase", "macro", "fold-local", "twenty items", "limitation", "holdout",
        "no significance test", "zero-forecast",
    ):
        assert phrase in prose, f"notebook 05 never mentions {phrase!r}"


def test_notebook_never_claims_a_universal_winner():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    combined = (_markdown(nb) + "\n" + _text(nb)).lower()
    for line in combined.splitlines():
        if "universal winner" in line:
            assert "no universal winner" in line or " not " in line or "does not" in line, line
    assert "best model overall" not in combined and "always best" not in combined


def test_executed_notebook_shows_the_integrity_facts():
    text = _text(nbformat.read(NOTEBOOK_PATH, as_version=4))
    assert "reproduced: True" in text, "the Phase 2 item-level results were not reproduced"
    assert "holdout days in the table: 0" in text
    assert "FAIL -" not in text
    assert "Hand-written recursion equals the library classes" in text
    assert "20 selected items, 1913 development days each" in text
    assert "823 items x 1913 development days" in text


@pytest.fixture(scope="module")
def saved_forecasts():
    if not LONG.exists():
        pytest.skip(f"{LONG.name} not found. Create it with: {RUN_COMMAND}")
    return read_forecast_table(LONG, kind="item")


def test_saved_forecast_table_passes_the_full_audit(saved_forecasts):
    series = load_selected_item_series(CONFIG)
    index = next(iter(series.values())).index
    checks = audit_forecast_table(saved_forecasts, kind="item", item_series=series, dev_dates=index)
    failed = [(c.name, c.detail) for c in checks if not c.passed]
    assert failed == [], failed


def test_saved_forecasts_cover_three_models_twenty_items_five_folds(saved_forecasts):
    table = saved_forecasts
    assert set(table["model"]) == {"SeasonalNaive", "Croston(alpha=0.1)", "TSB(alpha=0.1,beta=0.1)"}
    assert table["item_id"].nunique() == 20 and table["fold"].nunique() == 5
    assert len(table) == 3 * 20 * 5 * 28
    assert pd.to_datetime(table["date"]).max() <= pd.Timestamp("2016-04-24")  # final holdout never appears


def test_saved_fold_metrics_equal_the_metrics_recomputed_from_the_saved_forecasts(saved_forecasts):
    path = metrics_dir(CONFIG) / PHASE3["intermittent_fold_summary_filename"]
    if not path.exists():
        pytest.skip(f"{path.name} not found. Create it with: {RUN_COMMAND}")
    saved = pd.read_csv(path)
    recomputed = metrics_by_group(saved_forecasts, ["item_id", "demand_class", "model", "fold"])
    keys = ["item_id", "model", "fold"]
    merged = saved.merge(recomputed, on=keys, suffixes=("_saved", "_new"))
    assert len(merged) == len(saved) == 300
    for metric in ("MAE", "RMSE"):
        np.testing.assert_allclose(merged[f"{metric}_saved"], merged[f"{metric}_new"], rtol=1e-9, atol=1e-12)


def test_saved_phase2_item_level_files_were_not_overwritten_by_this_notebook():
    # Notebook 05 writes only `intermittent_*` files; the completed Phase 2 item-level files keep their names.
    names = {PHASE3[key] for key in PHASE3 if key.startswith("intermittent_")}
    phase2_names = {CONFIG["phase2"][key] for key in CONFIG["phase2"] if key.startswith("item_level_")}
    assert names.isdisjoint(phase2_names)
    assert all(name.startswith("intermittent_") for name in names)


@pytest.mark.slow
def test_notebook_executes_end_to_end():
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    NotebookClient(
        nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}}
    ).execute()
    errors = [
        out for cell in nb.cells if cell.cell_type == "code" for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert errors == []
    text = _text(nb)
    assert "reproduced: True" in text and "holdout days in the table: 0" in text and "FAIL -" not in text
