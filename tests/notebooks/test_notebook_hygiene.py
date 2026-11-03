"""Structure and hygiene checks for the Phase 3 notebooks (05, 07, 08, 09, 10).

These tests read the DELIVERED notebook files (they do not execute them; each
notebook has its own execution test). A notebook that is still a placeholder, was
never executed, raised an error, leaks a machine-specific path or lacks the
sections a reader needs fails here with a message naming the notebook.
"""

from __future__ import annotations

import re

import nbformat
import pytest

from src.utils.paths import PROJECT_ROOT

NOTEBOOKS = [
    "05_intermittent_models.ipynb",
    "07_lstm_gru.ipynb",
    "08_prophet.ipynb",
    "09_model_comparison.ipynb",
    "10_inventory_decision_support.ipynb",
]
PLACEHOLDER_PHRASES = (
    "not yet implemented",
    "arrives in phase",
    "placeholder",
    "todo",
    "tbd",
    "coming soon",
    "lorem ipsum",
)
#: Machine-specific locations that must never appear in a delivered notebook's outputs.
PATH_PATTERNS = (
    r"/home/", r"/tmp/", r"/root/", r"/Users/", r"/mnt/", r"/usr/", r"/opt/", r"/var/",
    r"site-packages", r"\b[A-Za-z]:\\", r"\\\\Users\\\\",
)
REQUIRED_SECTIONS = ("objective", "method", "result", "limitation", "conclusion")
MAX_NOTEBOOK_BYTES = 12 * 1024 * 1024


def _load(name: str):
    path = PROJECT_ROOT / "notebooks" / name
    assert path.exists(), f"notebooks/{name} does not exist"
    return path, nbformat.read(path, as_version=4)


def _output_text(nb) -> str:
    parts = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                parts.append(out.get("text", ""))
            elif kind in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
                parts.append("".join(text) if isinstance(text, list) else text)
            elif kind == "error":
                parts.append("\n".join(out.get("traceback", [])))
    return "\n".join(parts)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_valid_and_not_a_placeholder(name):
    path, nb = _load(name)
    nbformat.validate(nb)
    assert path.stat().st_size <= MAX_NOTEBOOK_BYTES, "notebook is too large to ship (embedded images?)"
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 8, f"{name} has only {len(code_cells)} code cells; it looks like a placeholder"
    visible = "\n".join(c.source for c in nb.cells).lower()
    found = [phrase for phrase in PLACEHOLDER_PHRASES if re.search(rf"\b{re.escape(phrase)}\b", visible)]
    assert not found, f"{name} still contains placeholder wording: {found}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_was_executed_without_errors(name):
    _, nb = _load(name)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.get("execution_count") is not None for c in code_cells), f"{name}: a code cell was never executed"
    errors = [
        out.get("ename")
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert errors == [], f"{name} stored error output(s): {errors}"
    assert "Traceback (most recent call last)" not in _output_text(nb)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_outputs_leak_no_machine_specific_paths(name):
    _, nb = _load(name)
    text = _output_text(nb)
    for pattern in PATH_PATTERNS:
        match = re.search(pattern, text)
        assert match is None, f"{name}: output contains a machine-specific path ({pattern!r}) near: {text[max(0, match.start() - 60):match.end() + 60]!r}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_has_the_sections_a_reader_needs(name):
    _, nb = _load(name)
    headings = " ".join(
        line.lower()
        for c in nb.cells
        if c.cell_type == "markdown"
        for line in c.source.splitlines()
        if line.lstrip().startswith("#")
    )
    missing = [section for section in REQUIRED_SECTIONS if section not in headings]
    assert not missing, f"{name} has no heading mentioning: {missing}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_never_presents_the_holdout_as_evaluated(name):
    _, nb = _load(name)
    combined = "\n".join(c.source for c in nb.cells).lower() + "\n" + _output_text(nb).lower()
    for phrase in ("holdout mae", "holdout rmse", "final-holdout result", "final holdout result:"):
        assert phrase not in combined, f"{name} appears to report a final-holdout score ({phrase!r})"
