"""Notebook 10 (inventory decision support): delivered-file checks, saved-artifact checks, real execution.

Artifacts under `artifacts/` are git-ignored and rebuilt by running the notebook, so tests that need them are
skipped (with the command that creates them) when they are absent.
"""

from __future__ import annotations

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "10_inventory_decision_support.ipynb"
CONFIG = load_config()
PHASE3 = CONFIG["phase3"]
INVENTORY_DIR = PROJECT_ROOT / PHASE3["inventory_dir"]
RUN_COMMAND = "jupyter nbconvert --to notebook --execute --inplace notebooks/10_inventory_decision_support.ipynb"


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


def _artifact(key: str) -> pd.DataFrame:
    path = INVENTORY_DIR / PHASE3[key]
    if not path.exists():
        pytest.skip(f"{path.name} not found. Create it with: {RUN_COMMAND}")
    return pd.read_csv(path)


def test_notebook_explains_the_decision_logic_and_its_limits():
    prose = _markdown(nbformat.read(NOTEBOOK_PATH, as_version=4))
    for phrase in (
        "lead time", "review period", "protection interval", "safety stock", "reorder point", "service level",
        "fill rate", "cycle service level", "empirical", "illustrative", "no cost", "forecast vs decision",
        "stationar", "censored", "limitation", "holdout",
    ):
        assert phrase in prose, f"notebook 10 never mentions {phrase!r}"


def test_notebook_claims_no_savings_and_no_live_forecast():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    combined = (_markdown(nb) + "\n" + _text(nb)).lower()
    assert "no savings" in combined or "no cost" in combined
    for phrase in ("we saved", "will save", "cost reduction of", "% saving", "profit increase"):
        assert phrase not in combined, phrase
    assert "no forecast beyond 2016-04-24" in combined or "no forecast after 2016-04-24" in combined


def test_executed_notebook_shows_the_audit_and_the_sanity_check():
    text = _text(nbformat.read(NOTEBOOK_PATH, as_version=4))
    assert text.count("all checks pass") >= 2
    assert "In-sample coverage reaches the target" in text
    assert "FAILED" not in text


def test_saved_reorder_points_are_non_negative_and_consistent():
    demo = _artifact("inventory_demo_filename")
    ok = demo[demo["status"] == "ok"]
    assert len(ok) > 0
    assert (ok["reorder_point"] >= 0).all() and (ok["safety_stock"] >= 0).all() and (ok["expected_demand"] >= 0).all()
    np.testing.assert_allclose(ok["reorder_point"], ok["expected_demand"] + ok["safety_stock"], rtol=1e-9, atol=1e-9)
    assert (ok["safety_stock_ci_low"] <= ok["safety_stock_ci_high"] + 1e-12).all()
    assert (ok["n_error_samples"] > 0).all() and ok["service_level"].between(0.5, 0.999).all()
    assert (ok["protection_days"] <= 28).all()


def test_saved_backtest_summary_is_internally_consistent():
    summary = _artifact("inventory_backtest_summary_filename")
    assert {"model", "target_cycle_service", "achieved_cycle_service", "ci_low", "ci_high", "gap", "n_cells", "n_windows"} <= set(summary.columns)
    assert summary["achieved_cycle_service"].between(0, 1).all()
    assert (summary["ci_low"] <= summary["achieved_cycle_service"] + 1e-9).all()
    assert (summary["achieved_cycle_service"] <= summary["ci_high"] + 1e-9).all()
    np.testing.assert_allclose(summary["gap"], summary["achieved_cycle_service"] - summary["target_cycle_service"], atol=1e-9)


def test_saved_backtest_only_calibrates_on_earlier_folds():
    cells = _artifact("inventory_backtest_filename")
    ok = cells[cells["status"] == "ok"]
    assert (ok["fold"] >= 3).all()  # two calibration folds are required, and the folds are 1..5
    window = ok["window_days"].iloc[0]
    assert (ok["n_calibration_samples"] == (ok["fold"] - 1) * (28 - window + 1)).all()
    skipped = cells[cells["status"] != "ok"]
    assert skipped["achieved_cycle_service"].isna().all()


def test_metadata_records_the_assumptions_as_illustrative():
    import json

    path = INVENTORY_DIR / PHASE3["inventory_metadata_filename"]
    if not path.exists():
        pytest.skip(f"{path.name} not found. Create it with: {RUN_COMMAND}")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["component"] == "inventory"
    assumptions = record["details"]["assumptions"]
    assert "illustrative" in assumptions and "no cost" in assumptions
    assert not {"hostname", "node", "user", "username"} & set(record["environment"])  # no machine or person identifiers


@pytest.mark.slow
def test_notebook_executes_end_to_end():
    from nbclient import NotebookClient

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}}
    ).execute()
    errors = [
        out for cell in nb.cells if cell.cell_type == "code" for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert errors == []
    assert "In-sample coverage reaches the target" in _text(nb)
