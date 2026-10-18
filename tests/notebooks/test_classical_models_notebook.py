"""Notebook load + real end-to-end execution test for Phase 2 Steps 5-7.

`test_notebook_executes_end_to_end_against_real_data` is the "successful
notebook load+execution" test the Step 5 brief asks for (Step 6
re-asks for it -- "execute the updated notebook on the real development
data" -- and Step 7 again -- "Execute the notebook on the real data"):
it runs notebooks/04_classical_models.ipynb for real, against the real
CA_1/FOODS_3 project data (not a mock/stub series), via the same
nbclient mechanism used to produce the delivered, already-executed
notebook. It executes a fresh in-memory copy -- it does not overwrite
the delivered notebook file on disk, and does not depend on it already
having been executed.

Slow by nature (fits ARIMA/SARIMA/Holt-Winters across 5 folds each, plus
Section 11's SARIMA fold-3 optimizer sensitivity check, which re-fits
that one fold 3 more times -- about 5-8 minutes total): this is a full
real run, not a shortcut. Section 12 (Croston/TSB, Phase 2 Step 7) adds
only a few seconds -- neither method uses an iterative optimizer.
"""

from __future__ import annotations

import nbformat
import pytest
from nbclient import NotebookClient

from src.utils.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_classical_models.ipynb"


def _all_stream_text(nb) -> str:
    """Every bit of text a human reading the executed notebook would
    see: `print()` output (`stream`) and the auto-displayed repr of a
    cell's last expression, e.g. a bare DataFrame (`execute_result`'s
    `text/plain`) -- Section 11's sensitivity-check table is shown the
    second way, not printed, so both are needed here.
    """
    parts = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts.append(out.get("text", ""))
            elif out.get("output_type") == "execute_result":
                text = out.get("data", {}).get("text/plain", "")
                parts.append("".join(text) if isinstance(text, list) else text)
    return "\n".join(parts)


def test_notebook_loads_as_valid_notebook():
    assert NOTEBOOK_PATH.exists(), f"{NOTEBOOK_PATH} not found."
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    assert len(nb.cells) > 0
    assert any(c.cell_type == "code" for c in nb.cells)
    assert any(
        c.cell_type == "markdown" and "Holt-Winters" in c.source for c in nb.cells
    ), "Expected a Holt-Winters section (Phase 2 Step 5) in the notebook."


def test_notebook_executes_end_to_end_against_real_data():
    if not NOTEBOOK_PATH.exists():
        pytest.skip(f"{NOTEBOOK_PATH} not found.")

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,  # per-cell ceiling; the real run finishes in ~8 minutes total
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()  # raises CellExecutionError on any cell failure

    error_outputs = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert error_outputs == [], f"Notebook raised {len(error_outputs)} error output(s)."

    all_text = _all_stream_text(nb)

    # Structural content checks -- not just "ran without raising", but
    # "produced the Step 5 content it's supposed to produce".
    assert "ARIMA order (p,d,q) = (6, 1, 1)" in all_text
    assert "SARIMA order (p,d,q) = (6, 1, 1), seasonal_order (P,D,Q,m) = (0, 0, 2, 7)" in all_text
    assert "Holt-Winters trend=add vs trend=None" in all_text

    # The two requirements this project treats as non-negotiable:
    # the frozen Seasonal Naive baseline is unchanged, and the final
    # holdout was never touched by any model fit/selection above.
    assert "SeasonalNaive   MAE=  256.55  RMSE=  329.05  MAPE=12.08%" in all_text
    assert "was NOT used anywhere in this notebook for fitting" in all_text

    # Phase 2 Step 6 -- Section 11 content checks: residual diagnostics
    # actually ran (not skipped/stubbed), the SARIMA fold-3 investigation
    # recorded its real, non-fabricated convergence verdict, the
    # sensitivity check ran as a separate result, the Holt-Winters
    # trend-selection caveat was stated explicitly, and Section 11's own
    # holdout-exclusion confirmation printed.
    assert "SARIMA fold 3" in all_text and "converged:    False" in all_text
    # The sensitivity table is a displayed DataFrame (not printed), and
    # pandas' default repr truncates long string cells (~50 chars) --
    # this substring is short enough to survive that truncation.
    assert "ORIGINAL (canonical, used throughout this note" in all_text
    assert "sensitivity: lbfgs, maxiter=2000" in all_text
    assert "NOT an independent out-of-sample validation of that choice" in all_text
    assert "NOT used anywhere in Section 11 for residual computation" in all_text

    # Phase 2 Step 7 -- Section 12 content checks: the real 20-item
    # backtest actually ran (exact expected row counts, not placeholders),
    # no (item, model, fold) combination failed, the holdout-leakage
    # check passed explicitly, and the section does not declare a
    # universal winner or conflate item-level and aggregate-level error
    # scales.
    assert "Long results table: 8400 rows (expected 8400)" in all_text
    assert "Fold summary table: 300 rows (expected 300)" in all_text
    assert "Failed (item, model, fold) combinations: 0" in all_text
    assert (
        "Confirmed: no date on or after the final holdout start was used "
        "anywhere in Section 12." in all_text
    )
    assert "does not declare a universal winning model" in all_text
    assert "must not be compared to them directly" in all_text
    # The "next step" marker is a markdown cell (not code-cell output),
    # so it's invisible to _all_stream_text() -- checked against the
    # markdown source directly instead. It was rewritten for Step 7:
    # Croston/TSB must no longer be listed as not-yet-implemented, but
    # the still-unimplemented Phase 3 methods must still be.
    markdown_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "Croston's method — NEXT STEP — NOT YET IMPLEMENTED" not in markdown_text
    assert "TSB — NEXT STEP — NOT YET IMPLEMENTED" not in markdown_text
    assert "XGBoost — NEXT STEP — NOT YET IMPLEMENTED" in markdown_text
