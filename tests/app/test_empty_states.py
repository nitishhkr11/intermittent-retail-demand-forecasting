"""Missing artifacts: every page must explain what is absent and print the exact command that creates it - never crash, never invent.

The check runs in a fresh interpreter against a copy of the CODE only (app/, src/, config.yaml) in a temporary folder, so it sees a
repository with no data and no saved results, whatever the real checkout contains. Nothing in the real repository is touched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]

RUNNER = textwrap.dedent(
    '''
    import json, sys
    from streamlit.testing.v1 import AppTest

    PAGES = json.loads(sys.argv[1])
    out = {}
    for theme_name in ("light", "dark"):
        at = AppTest.from_file("app/app.py", default_timeout=180)
        at.session_state["rd_theme"] = theme_name
        at.run()
        result = {"entry": {"exceptions": [str(e.value)[:400] for e in at.exception],
                            "sidebar": " ".join(str(m.value) for m in at.sidebar.markdown)}}
        for page in PAGES:
            at.switch_page(page).run()
            body = " ".join(str(m.value) for m in at.main.markdown)
            result[page] = {"exceptions": [str(e.value)[:400] for e in at.exception], "text": body,
                            "plots": len(at.get("plotly_chart")), "tables": body.count('<table class="rd-table">')}
        out[theme_name] = result
    print("@@JSON@@" + json.dumps(out))
    '''
)


def _copy_code(destination: Path) -> Path:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for name in ("app", "src", ".streamlit"):
        shutil.copytree(ROOT / name, destination / name, ignore=ignore)
    shutil.copy(ROOT / "config.yaml", destination / "config.yaml")
    return destination


def _run(repo: Path):
    from tests.app.helpers import PAGE_PATHS

    (repo / "_runner.py").write_text(RUNNER, encoding="utf-8")
    proc = subprocess.run([sys.executable, "_runner.py", json.dumps(PAGE_PATHS)], cwd=repo, capture_output=True, text=True, timeout=600,
                          env={**__import__("os").environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("@@JSON@@"))
    return json.loads(line[len("@@JSON@@"):])


@pytest.fixture(scope="module")
def empty_repo(tmp_path_factory):
    repo = _copy_code(tmp_path_factory.mktemp("empty_repo"))
    assert not (repo / "data").exists() and not (repo / "artifacts").exists()
    return _run(repo)


def test_every_page_runs_without_exception_when_nothing_has_been_produced(empty_repo):
    for theme_name, result in empty_repo.items():
        assert result["entry"]["exceptions"] == [], (theme_name, result["entry"]["exceptions"])
        for page, info in result.items():
            if page != "entry":
                assert info["exceptions"] == [], (theme_name, page, info["exceptions"])


def test_missing_results_are_explained_with_exact_powershell_commands(empty_repo):
    texts = {page: info["text"] for page, info in empty_repo["light"].items() if page != "entry"}
    guided = [page for page, text in texts.items() if "Create it from the project root (PowerShell)" in text]
    assert len(guided) >= 4, f"only {guided} explain how to create what is missing"
    joined = " ".join(texts.values())
    assert "python -m src.data.preprocessor" in joined and "python -m src.analysis.demand_classification" in joined, "the commands that build the development data are not shown"
    assert "src.pipeline" not in joined, "src/pipeline.py is a placeholder and must never be offered as a command"
    assert "aggregate_ca1_foods3.parquet" in joined and "not found" in joined, "the missing file is not named"


def test_nothing_is_fabricated_when_artifacts_are_missing(empty_repo):
    for theme_name, result in empty_repo.items():
        for page, info in result.items():
            if page == "entry":
                continue
            assert info["plots"] == 0, f"{theme_name}/{page}: a chart is drawn although no saved result exists"
            if not page.endswith("6_About_Methodology.py"):
                assert info["tables"] == 0, f"{theme_name}/{page}: a table of numbers is shown although no saved result exists"


def test_shell_and_holdout_state_survive_an_empty_repository(empty_repo):
    entry = empty_repo["dark"]["entry"]
    assert "Not evaluated" in entry["sidebar"] and "Final holdout" in entry["sidebar"]
    assert "River" in entry["sidebar"]


def test_empty_repository_output_contains_no_machine_paths(empty_repo):
    blob = json.dumps(empty_repo)
    assert "/home/" not in blob and "\\\\Users\\\\" not in blob and "/tmp/" not in blob
