"""Shared path-resolution helpers.

config.yaml writes paths as repo-root-relative strings (`data/raw`,
`data/interim`, ...) for readability. Joining them with a bare
`Path(...)` resolves against the CURRENT WORKING DIRECTORY at runtime
-- which is only the repo root when a script happens to be launched
from there. It is NOT reliably the repo root: a Jupyter kernel's
starting directory varies by how Jupyter is launched (often the
notebook's own directory, sometimes the launch directory), so the same
bare-relative-path code that works when running
`python -m src.data.preprocessor` from the repo root can silently look
in the wrong place from a notebook.

PROJECT_ROOT is anchored to this file's own location on disk via
`__file__`, which is reliable regardless of the caller's cwd.
`resolve_path` is what every module (preprocessor.py,
demand_classification.py, eda.py) should use instead of
`Path(config["data"]["..."])` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

# src/utils/paths.py -> src/utils -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a repo-root-relative path (as written in config.yaml) to
    an absolute Path, regardless of the current working directory.
    """
    return PROJECT_ROOT / relative_path
