"""Helpers shared by the dashboard tests (imported as `tests.app.helpers`).

The dashboard modules are imported by the entry script as top-level modules (`import theme`, `import data_access`) because the entry
file is named `app.py`; the tests do the same so both see one module object.
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
APP_FILE = APP_DIR / "app.py"
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("streamlit", reason="streamlit is not installed (pip install -r requirements.txt)")
from streamlit.testing.v1 import AppTest  # noqa: E402

from components.sidebar import PAGES  # noqa: E402

PAGE_PATHS: List[str] = [p for p, _, _ in PAGES]
PAGE_LABELS = {p: label for p, label, _ in PAGES}
HOLDOUT_START, HOLDOUT_END = "2016-04-25", "2016-05-22"


def start_app(theme_name: str = "light") -> AppTest:
    """Run the entry script once with the session theme preset."""
    at = AppTest.from_file(str(APP_FILE), default_timeout=180)
    at.session_state["rd_theme"] = theme_name
    return at.run()


def open_page(theme_name: str, path: str) -> AppTest:
    at = start_app(theme_name)
    assert not at.exception, [e.value for e in at.exception]
    return at.switch_page(path).run()


def texts(at: AppTest, *, sidebar: bool = False) -> List[str]:
    """Every piece of text a person can read in the main area (or the sidebar): markdown, captions, headers, widget labels, notices."""
    block = at.sidebar if sidebar else at.main
    out: List[str] = []
    for kind in ("markdown", "caption", "header", "subheader", "title", "text", "code", "info", "warning", "error", "success"):
        out.extend(str(el.value) for el in getattr(block, kind, []))
    for kind in ("selectbox", "slider", "number_input", "checkbox", "radio", "toggle", "multiselect", "text_input", "button"):
        out.extend(str(el.label) for el in getattr(block, kind, []))
    return out


def chart_specs(at: AppTest) -> List[dict]:
    return [json.loads(el.proto.spec) for el in at.get("plotly_chart")]


def flat_values(obj) -> Iterable:
    """Walk a nested Plotly spec and yield every scalar."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from flat_values(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from flat_values(v)
    else:
        yield obj


def raw_options(widget) -> list:
    """The values a select box was given. AppTest only exposes the displayed labels, so labels that stand for integers
    ("28", "Fold 3: ...") are turned back into the integer the page script compares against."""
    out = []
    for label in widget.options:
        text = str(label)
        match = re.match(r"^(?:Fold\s+)?(\d+)\b", text)
        out.append(int(match.group(1)) if match else text)
    return out


def decode_array(values) -> "np.ndarray":
    """Plotly >= 6 may ship numeric arrays as {"dtype": ..., "bdata": base64}; plain lists are returned as arrays too."""
    if isinstance(values, dict) and "bdata" in values:
        raw = base64.b64decode(values["bdata"])
        return np.frombuffer(raw, dtype=np.dtype(values["dtype"]).newbyteorder("<")).astype(float)
    return np.asarray(values, dtype=float)
