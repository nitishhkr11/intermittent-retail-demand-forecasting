"""Notebook 08 (Prophet): delivered-file checks, saved-artifact audit, and a real end-to-end execution.

* The first group reads the DELIVERED notebook and the artifacts it saved. Artifacts under
  `artifacts/` are git-ignored and rebuilt by running the notebook, so tests that need them are
  skipped (with the command to create them) when they are absent.
* `test_notebook_executes_end_to_end` re-executes the notebook for real in memory (about a minute;
  `@pytest.mark.slow`). It asserts integrity facts the notebook prints, never a metric value:
  last-digit results can differ between operating systems and Prophet/Stan builds.
"""

from __future__ import annotations

import nbformat
import numpy as np
import pandas as pd
import pytest

from src.data.aggregate_series import load_aggregate_development_series
from src.evaluation.forecast_artifacts import (
    audit_forecast_table,
    forecasts_dir,
    metrics_by_group,
    read_forecast_table,
)
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "08_prophet.ipynb"
CONFIG = load_config()
PHASE3 = CONFIG["phase3"]
FORECASTS = forecasts_dir(CONFIG) / PHASE3["prophet_forecasts_filename"]
RUN_COMMAND = "jupyter nbconvert --to notebook --execute --inplace notebooks/08_prophet.ipynb"


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


def test_notebook_explains_prophet_and_states_its_limits():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    prose = _markdown(nb)
    for phrase in (
        "trend", "seasonality", "changepoint", "regressor", "additive", "multiplicative",
        "interval", "no-regressors", "holdout", "limitation", "aggregate",
    ):
        assert phrase in prose, f"notebook 08 never mentions {phrase!r}"


def test_notebook_shows_the_leakage_and_determinism_checks_and_the_fixed_variants():
    text = _text(nbformat.read(NOTEBOOK_PATH, as_version=4))
    assert "Perturbation test passed" in text
    assert "bit for bit" in text
    for name in ("Prophet(additive)", "Prophet(multiplicative)", "Prophet(additive,no-regressors)"):
        assert name in text, f"{name} is missing from the executed notebook"
    assert "Failed fits: 0 of 15" in text


def test_notebook_compares_only_like_with_like_and_claims_no_universal_winner():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    combined = _markdown(nb) + "\n" + _text(nb).lower()
    assert "item-level" in combined  # it says item-level errors are a different scale
    assert "no significance test" in combined
    assert "best overall" not in combined
    # "universal winner" may only appear in a negated sentence
    for line in combined.splitlines():
        if "universal winner" in line:
            assert "not" in line or "no universal winner" in line, line


@pytest.fixture(scope="module")
def saved_forecasts():
    if not FORECASTS.exists():
        pytest.skip(f"{FORECASTS.name} not found. Create it with: {RUN_COMMAND}")
    return read_forecast_table(FORECASTS, kind="aggregate")


def test_saved_forecast_table_passes_the_full_audit(saved_forecasts):
    dev_series = load_aggregate_development_series(CONFIG)
    checks = audit_forecast_table(saved_forecasts, kind="aggregate", dev_series=dev_series)
    failed = [c for c in checks if not c.passed]
    assert failed == [], f"audit failures: {[(c.name, c.detail) for c in failed]}"


def test_saved_forecasts_have_five_folds_per_variant_and_valid_intervals(saved_forecasts):
    table = saved_forecasts
    assert set(table["model"]) == {
        "Prophet(additive)", "Prophet(multiplicative)", "Prophet(additive,no-regressors)",
    }
    assert table.groupby("model")["fold"].nunique().eq(5).all()
    assert table["success"].all() and (table["predicted"] >= 0).all()
    assert {"lower", "upper"} <= set(table.columns)
    assert (table["lower"] <= table["predicted"] + 1e-9).all() and (table["predicted"] <= table["upper"] + 1e-9).all()
    assert (table["lower"] >= 0).all()


def test_saved_metrics_equal_the_metrics_recomputed_from_the_saved_forecasts(saved_forecasts):
    metrics_path = PROJECT_ROOT / PHASE3["metrics_dir"] / PHASE3["prophet_metrics_filename"]
    if not metrics_path.exists():
        pytest.skip(f"{metrics_path.name} not found. Create it with: {RUN_COMMAND}")
    saved = pd.read_csv(metrics_path)
    recomputed = metrics_by_group(saved_forecasts, ["model", "fold"])
    merged = saved.merge(recomputed, on=["model", "fold"], suffixes=("_saved", "_recomputed"))
    assert len(merged) == len(saved) == 15
    np.testing.assert_allclose(merged["MAE_saved"], merged["MAE_recomputed"], rtol=1e-9)
    np.testing.assert_allclose(merged["RMSE_saved"], merged["RMSE_recomputed"], rtol=1e-9)


def test_saved_forecasts_never_reach_into_the_reserved_holdout(saved_forecasts):
    window_end = pd.Timestamp("2016-04-24")
    assert pd.to_datetime(saved_forecasts["date"]).max() <= window_end


@pytest.mark.slow
def test_notebook_executes_end_to_end():
    pytest.importorskip("prophet", reason="Prophet is optional; install with: python -m pip install -r requirements-phase3.txt")
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
    assert "Failed fits: 0 of 15" in text
    assert "Perturbation test passed" in text
    assert "bit for bit" in text
